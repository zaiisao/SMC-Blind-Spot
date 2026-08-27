
import copy

import torch
from torch import nn

from .conditioners import BeatConditioner


class BeatControlNet(nn.Module):

    def __init__(self, transformer, n_blocks: int = None,
                 include_downbeat: bool = True, cond_id: str = "beat",
                 scale: float = 1.0):
        super().__init__()
        self.cond_id = cond_id
        self.scale = scale
        total = len(transformer.layers)
        self.n_blocks = total // 2 if n_blocks is None else n_blocks
        if not 0 < self.n_blocks <= total:
            raise ValueError(f"n_blocks must be in 1..{total}, got {self.n_blocks}")

        self.control_layers = nn.ModuleList(
            copy.deepcopy(transformer.layers[i]) for i in range(self.n_blocks))

        dim = transformer.dim
        self.beat_dim = BeatConditioner(include_downbeat=include_downbeat).cond_dim
        self.cond_in = nn.Linear(self.beat_dim, dim)
        nn.init.normal_(self.cond_in.weight, std=0.02)
        nn.init.zeros_(self.cond_in.bias)

        self.out_projs = nn.ModuleList()
        for _ in range(self.n_blocks):
            proj = nn.Linear(dim, dim)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.out_projs.append(proj)

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
        if beat is not None and beat.shape[-1] != self.beat_dim:
            raise ValueError(
                f"beat has {beat.shape[-1]} channels, expected {self.beat_dim}")
        self._beat = beat
        return self

    # -- hooks -----------------------------------------------------------

    def _run_branch(self, module, args, kwargs):
        self._residuals = {}

        beat = self._beat
        if beat is None:
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
        shift = max(1, beat.shape[-2] // 3)
        rolled = torch.roll(beat, shifts=shift, dims=-2)
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
