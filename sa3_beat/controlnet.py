"""A real ControlNet for SA3's DiT, driven by beat conditioning.

What this adds over the ``modular_local_cond`` seam already in ``build.py``:

    modular_local_cond          this
    ------------------          ----
    Linear->SiLU->Linear        trainable copy of the first N backbone blocks,
    per frame, per block        initialised from the pretrained weights
    ~21M params                 ~half the backbone
    never sees x_t              takes x_t, so control can depend on content
    no frame interaction        full self-attention across frames

The second and fourth differences are the point.  A per-frame map cannot
represent "this beat arrives late relative to the last three" -- it sees one
frame at a time -- and rubato is exactly that.  Metrical structure needs
frame-to-frame interaction, which is what copying attention blocks buys.

Injection topology is unchanged from ControlNet and from the existing seam:
per-block, additive, zero-init output projections, so the model is bit-identical
to the published checkpoint until trained.  The backbone stays frozen; only the
copy and its output projections learn.

Implemented with forward hooks rather than by overriding
``ContinuousTransformer.forward``.  That method does a lot of setup -- project_in,
rotary embeddings, memory tokens, padding masks, varlen metadata -- and
reimplementing it would fork behaviour that upstream is free to change.  Hooks
observe the real call and add to its result.
"""

import copy

import torch
from torch import nn

from .conditioners import BeatConditioner


class BeatControlNet(nn.Module):
    """Trainable copy of the first ``n_blocks`` DiT blocks, steered by beats.

    ``transformer`` is the ``ContinuousTransformer`` inside SA3's DiT.  Its
    blocks are deep-copied (weights included -- a ControlNet starts from the
    pretrained model, it does not learn control from scratch) and the copy runs
    alongside the frozen original, contributing a zero-init residual to each of
    the first ``n_blocks`` backbone layers.
    """

    def __init__(self, transformer, n_blocks: int = None,
                 include_downbeat: bool = True, cond_id: str = "beat",
                 scale: float = 1.0):
        super().__init__()
        self.cond_id = cond_id
        # Inference-time strength of the control signal. Because the branch
        # contributes additive residuals, this is a free knob -- no retraining,
        # no extra forward pass -- and it is the ControlNet analogue of
        # `controlnet_conditioning_scale`. It matters here because SA3's CFG
        # cannot amplify beat adherence: `modular_local_cond` is never dropped
        # in training and is absent from the negative branch, so the beat term
        # cancels out of the guidance direction. This knob also decouples
        # adherence from cfg_scale, which is what lets guidance stay low for
        # distributional coverage while beat-following stays strong.
        self.scale = scale
        total = len(transformer.layers)
        # ControlNet copies the "encoder half" of a U-Net; a flat DiT has no
        # such split, so half the depth is the closest analogue and the knob is
        # left explicit.
        self.n_blocks = total // 2 if n_blocks is None else n_blocks
        if not 0 < self.n_blocks <= total:
            raise ValueError(f"n_blocks must be in 1..{total}, got {self.n_blocks}")

        self.control_layers = nn.ModuleList(
            copy.deepcopy(transformer.layers[i]) for i in range(self.n_blocks))

        dim = transformer.dim
        self.beat_dim = BeatConditioner(include_downbeat=include_downbeat).cond_dim
        # NOT zero-init: the output projections already guarantee identity, and
        # zeroing here as well would leave the control branch blind to the beat
        # signal until the output projections happen to move first.
        self.cond_in = nn.Linear(self.beat_dim, dim)
        nn.init.normal_(self.cond_in.weight, std=0.02)
        nn.init.zeros_(self.cond_in.bias)

        # One zero-init projection per copied block: this is what makes the
        # whole branch a no-op at step 0 while keeping gradients alive.
        self.out_projs = nn.ModuleList()
        for _ in range(self.n_blocks):
            proj = nn.Linear(dim, dim)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.out_projs.append(proj)

        # Beat-sensitivity probe. The diffusion loss is a poor progress signal:
        # it is dominated by noise prediction, it falls simply because 244M new
        # parameters can fit corpus statistics, and it contains no contrast
        # between one grid and another. This measures the thing that actually
        # matters -- how much the branch's injected residual DEPENDS on the beat
        # grid -- by re-running the branch on a phase-shifted grid and comparing.
        # It is 0 if the conditioning is ignored, regardless of the loss.
        # Rubato guidance: extrapolate away from the metronomic twin of the
        # requested grid. The model collapses irregular grids toward a steady
        # pulse (measured: rubato adherence +0.002 vs metronomic +0.350), so
        # amplifying the difference between the true grid and its constant-tempo
        # twin counteracts exactly that collapse. Both endpoints are
        # in-distribution, so this is not asking for unseen conditioning.
        #   residual = res(twin) + guidance * (res(true) - res(twin))
        # guidance=1 is ordinary behaviour; >1 exaggerates the deviation.
        self.guidance = 1.0
        self.twin_beat = None

        self.probe_every = 0
        self.last_sensitivity = None
        self._probe_calls = 0

        self._transformer = [transformer]   # not a submodule: keep it frozen
        self._handles = []
        self._residuals = {}
        self._beat = None

    # -- wiring ----------------------------------------------------------

    def attach(self):
        """Install hooks on the backbone. Idempotent."""
        if self._handles:
            return self
        tf = self._transformer[0]
        self._handles.append(
            tf.layers[0].register_forward_pre_hook(self._run_branch,
                                                   with_kwargs=True))
        for i in range(self.n_blocks):
            self._handles.append(
                tf.layers[i].register_forward_hook(self._add_residual(i),
                                                  with_kwargs=True))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return self

    def set_beat(self, beat):
        """``beat`` is ``(batch, frames, beat_dim)``, or None to disable."""
        if beat is not None and beat.shape[-1] != self.beat_dim:
            raise ValueError(
                f"beat has {beat.shape[-1]} channels, expected {self.beat_dim}")
        self._beat = beat
        return self

    # -- hooks -----------------------------------------------------------

    def _run_branch(self, module, args, kwargs):
        """Run the whole control branch once, before backbone block 0."""
        self._residuals = {}

        beat = self._beat
        if beat is None:
            # Fall back to whatever beat conditioning is already flowing to the
            # DiT, so the branch needs no plumbing in train.py or generate.py.
            mlc = kwargs.get("modular_local_cond")
            if isinstance(mlc, dict):
                beat = mlc.get(self.cond_id)
        if beat is None:
            return None

        x = args[0]
        self._residuals = self._branch(x, beat, kwargs)

        if self.twin_beat is not None and self.guidance != 1.0:
            kw = dict(kwargs)
            mlc = kw.get("modular_local_cond")
            if isinstance(mlc, dict) and self.cond_id in mlc:
                mlc = dict(mlc); mlc[self.cond_id] = self.twin_beat
                kw["modular_local_cond"] = mlc
            twin = self._branch(x, self.twin_beat, kw)
            g = self.guidance
            self._residuals = {i: twin[i] + g * (self._residuals[i] - twin[i])
                               for i in self._residuals}

        if self.probe_every:
            self._probe_calls += 1
            if self._probe_calls % self.probe_every == 0:
                self._measure_sensitivity(x, beat, kwargs)
        return None

    def _branch(self, x, beat, kwargs):
        h = x + self._pad_to_match(self.cond_in(beat), x)
        out = {}
        for i, layer in enumerate(self.control_layers):
            h = layer(h, **kwargs)
            out[i] = self.scale * self.out_projs[i](h)
        return out

    @torch.no_grad()
    def _measure_sensitivity(self, x, beat, kwargs):
        """Relative change in the injected residual under a phase-wrong grid.

        The grid is rolled along time, which preserves every marginal statistic
        and changes only WHERE the beats are -- the same logic as the half-beat
        control in the adherence verifier. A branch that has learned to place
        beats must respond; one using the beat channels as generic extra
        capacity will not.
        """
        shift = max(1, beat.shape[-2] // 3)
        rolled = torch.roll(beat, shifts=shift, dims=-2)
        # The grid reaches the branch by TWO routes: cond_in on the input, and
        # each copied block's own modular_local_embeds via kwargs. Rolling only
        # the first measures the weaker path (~0.24% of branch input) while the
        # blocks still see the true grid -- which reads as near-zero sensitivity
        # even when the branch has learned to use the beats.
        kw = dict(kwargs)
        mlc = kw.get("modular_local_cond")
        if isinstance(mlc, dict) and self.cond_id in mlc:
            mlc = dict(mlc)
            mlc[self.cond_id] = torch.roll(mlc[self.cond_id], shifts=shift, dims=-2)
            kw["modular_local_cond"] = mlc
        other = self._branch(x, rolled, kw)
        num = sum((self._residuals[i] - other[i]).pow(2).sum() for i in other)
        den = sum(self._residuals[i].pow(2).sum() for i in other) + 1e-12
        self.last_sensitivity = float((num / den).sqrt())

    def _add_residual(self, i):
        def hook(module, args, kwargs, output):
            res = self._residuals.get(i)
            return output if res is None else output + res
        return hook

    @staticmethod
    def _pad_to_match(cond, x):
        """Left-pad the conditioning to the backbone's sequence length.

        The DiT prepends memory/global tokens, so the conditioning is shorter
        than ``x``.  SAT's own local conditioning left-pads for the same reason;
        matching that keeps beat frame k aligned with latent frame k.
        """
        n = x.shape[-2] - cond.shape[-2]
        if n == 0:
            return cond
        if n < 0:
            raise ValueError(f"conditioning longer than sequence by {-n}")
        pad = cond.new_zeros(cond.shape[0], n, cond.shape[-1])
        return torch.cat([pad, cond], dim=-2)

    # -- training helpers ------------------------------------------------

    def freeze_backbone(self):
        for p in self._transformer[0].parameters():
            p.requires_grad = False
        for p in self.parameters():
            p.requires_grad = True
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
