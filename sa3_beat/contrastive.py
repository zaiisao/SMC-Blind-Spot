"""Contrastive beat conditioning for ControlNet adapter training.

THE PROBLEM. Training pairs are (real audio, that audio's true grid). The audio
already contains its beats, so the beat conditioning is redundant and the
noise-prediction loss can be minimised while ignoring it. At generation time
there is no audio to copy from and the conditioning is the only timing source --
so the model generates a coherent pulse unrelated to the grid. Listening
confirmed this: clips have a clear beat, the ticks land elsewhere, one clip was
audibly on the offbeat.

THE FIX. Make the conditioning non-redundant. Each batch item is duplicated with
a CORRUPTED grid; the pair shares the same latent, the same noise and the same
timestep, so their losses differ only by the conditioning. A hinge then requires
the true grid to fit better by a margin:

    L = L_true + lam * mean(relu(margin - (L_neg - L_true)))

The hinge is zero wherever the model already prefers the true grid, so all
gradient goes to the cases where conditioning is being ignored.

Corruptions (sa3_beat/negatives.py) cover every misalignment class observed --
including jitter, drift and dropouts, since 15.7% of tracker errors on SMC match
no clean ratio -- but 50% of the weight sits on half / double / offbeat, which is
what the listening test surfaced.
"""
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from stable_audio_tools.inference.sampling import get_alphas_sigmas
from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper
from stable_audio_tools.models.inpainting import random_inpaint_mask

from .negatives import corrupt

BEAT_KEY = "beat"


class ContrastiveBeatWrapper(DiffusionCondTrainingWrapper):
    """Adds the hinge term. Everything else follows the parent's recipe."""

    def __init__(self, *a, contrast_lambda=1.0, contrast_margin=0.02, contrast_seed=0, **kw):
        super().__init__(*a, **kw)
        self.contrast_lambda = contrast_lambda
        self.contrast_margin = contrast_margin
        self._rng = np.random.RandomState(contrast_seed)
        self._kind_counts = {}
        self._kind_gap = {}      # kind -> recent (l_neg - l_pos) values
        self._n_dead = 0         # steps where the loss did not depend on params

    def _ensure_state(self):
        """train.py re-classes an already-built wrapper, so __init__ never runs.
        Anything this class adds has to be created on first use, not in __init__."""
        for name, default in (("_kind_counts", dict), ("_kind_gap", dict)):
            if name not in self.__dict__:
                self.__dict__[name] = default()
        if "_n_dead" not in self.__dict__:
            self.__dict__["_n_dead"] = 0
        if "_rng" not in self.__dict__:
            self.__dict__["_rng"] = np.random.RandomState(0)

    def _negative_metadata(self, metadata):
        self._ensure_state()
        neg, kinds = [], []
        n = len(metadata)
        for i, md in enumerate(metadata):
            m = dict(md)
            b = md.get(BEAT_KEY)
            if isinstance(b, dict) and b.get("beats") is not None and len(b["beats"]) >= 4:
                other = metadata[(i + 1) % n].get(BEAT_KEY, {}).get("beats") if n > 1 else None
                secs = float(md.get("seconds_total", 0)) or (float(np.max(b["beats"])) + 1.0)
                nb, kind = corrupt(b["beats"], secs, self._rng, other=other)
                m[BEAT_KEY] = {**b, "beats": nb,
                               "downbeats": None}   # corrupted grid has no valid downbeats
                self._kind_counts[kind] = self._kind_counts.get(kind, 0) + 1
            else:
                kind = "none"
            kinds.append(kind)
            neg.append(m)
        return neg, kinds

    def training_step(self, batch, batch_idx):
        reals, metadata = batch
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]
        B = reals.shape[0]

        neg_md, neg_kinds = self._negative_metadata(metadata)
        cond_pos = self.diffusion.conditioner(metadata, self.device)
        cond_neg = self.diffusion.conditioner(neg_md, self.device)
        cond = {k: [torch.cat([cond_pos[k][0], cond_neg[k][0]], dim=0),
                    None if cond_pos[k][1] is None
                    else torch.cat([cond_pos[k][1], cond_neg[k][1]], dim=0)]
                for k in cond_pos}

        x = reals
        if self.diffusion.pretransform is not None and self.pre_encoded:
            if getattr(self.diffusion.pretransform, "scale", 1.0) != 1.0:
                x = x / self.diffusion.pretransform.scale
        x2 = torch.cat([x, x], dim=0)                       # same latent for both halves

        from stable_audio_tools.inference.sampling import truncated_logistic_normal_rescaled
        t = 1 - truncated_logistic_normal_rescaled(B).to(self.device)
        if self.diffusion.dist_shift is not None:
            t = self.diffusion.dist_shift.shift(t, x.shape[2])
        t2 = torch.cat([t, t], dim=0)                       # SAME timestep for the pair

        if self.diffusion_objective in ["v"]:
            alphas, sigmas = get_alphas_sigmas(t2)
        else:
            alphas, sigmas = 1 - t2, t2
        alphas, sigmas = alphas[:, None, None], sigmas[:, None, None]

        noise = torch.randn_like(x)
        noise2 = torch.cat([noise, noise], dim=0)           # SAME noise for the pair
        noised = x2 * alphas + noise2 * sigmas
        targets = (noise2 * alphas - x2 * sigmas) if self.diffusion_objective == "v" \
                  else (noise2 - x2)

        # The published checkpoint is diffusion_cond_inpaint: the DiT always expects
        # the inpaint pair, and the parent's step draws a RANDOM mask per batch and
        # scores only the masked region. Draw ONE mask and tile it, so the pair stays
        # comparable, and score on the same region the parent would.
        loss_mask = None
        if self.inpainting_config is not None:
            pad = torch.ones(B, x.shape[-1], dtype=torch.bool, device=self.device)
            masked_input, inp_mask = random_inpaint_mask(
                x, padding_masks=pad, mask_padding=bool(self.mask_padding_attention),
                **self.inpaint_mask_kwargs)
            cond["inpaint_mask"] = [torch.cat([inp_mask, inp_mask], dim=0), None]
            cond["inpaint_masked_input"] = [torch.cat([masked_input, masked_input], dim=0), None]
            # inpaint_mask is 1 where audio is GIVEN, 0 where it must be generated.
            # Score the GENERATED region: that is where the beat grid is the only
            # timing source. The parent does the same (diffusion.py:713,
            # loss_mask &= ~inpaint_mask). Scoring the given region instead makes
            # the task trivial, and under FULL_MASK (p=0.8) leaves it empty, so the
            # loss stops depending on the parameters and the step contributes nothing.
            loss_mask = ~torch.cat([inp_mask, inp_mask], dim=0).squeeze(1).to(torch.bool)
        else:
            z = torch.zeros(2 * B, 1, x.shape[-1], device=self.device)
            cond["inpaint_mask"] = [z, None]
            cond["inpaint_masked_input"] = [torch.zeros_like(x2), None]

        # cfg_dropout would randomly blank the beat conditioning and make the pair
        # incomparable, so it is disabled on this step.
        output = self.diffusion(noised, t2, cond=cond, cfg_dropout_prob=0.0)

        sq = F.mse_loss(output, targets, reduction="none")
        if loss_mask is not None:
            m = loss_mask.unsqueeze(1).expand_as(sq)
            per = (sq * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp(min=1)
        else:
            per = sq.mean(dim=(1, 2))
        l_pos, l_neg = per[:B], per[B:]
        loss_main = l_pos.mean()

        # A FIXED margin is a bar the model clears once and then coasts past:
        # runs A and B cleared theirs on all eight corruption kinds by step ~1900
        # and plateaued at a 0.9% / 2.0% preference for the true grid. Since the
        # hinge is zero above the margin, lambda cannot push further -- the margin
        # is the destination, lambda only sets the speed. A RELATIVE margin asks
        # for a fraction of the current loss instead, so the demand scales with
        # the task and does not go slack as mse_loss falls.
        rel = getattr(self, "contrast_margin_rel", 0.0)
        margin = rel * loss_main.detach() if rel else self.contrast_margin
        hinge = F.relu(margin - (l_neg - l_pos)).mean()
        loss = loss_main + self.contrast_lambda * hinge

        gaps = (l_neg - l_pos).detach()
        for k, gv in zip(neg_kinds, gaps.tolist()):
            buf = self._kind_gap.setdefault(k, [])
            buf.append(gv)
            if len(buf) > 400:
                del buf[:-400]

        # Gap by NOISE LEVEL, as a fraction of that bucket's loss. A large average
        # gap can be a false positive if it lives entirely at low noise, where the
        # audio is nearly clean and the model can simply check the grid against
        # onsets it can already see. Generation starts from pure noise, and tempo
        # and phase are decided in the HIGH-noise steps -- so a gap concentrated
        # at low t predicts nothing about generation.
        if "_t_gap" not in self.__dict__:
            self.__dict__["_t_gap"] = {b: [] for b in ("hi", "mid", "lo")}
        tv = t.detach().tolist()
        lp = l_pos.detach().tolist()
        for tt, gv, pv in zip(tv, gaps.tolist(), lp):
            b = "hi" if tt > 0.66 else ("mid" if tt > 0.33 else "lo")
            buf = self._t_gap[b]
            buf.append(100.0 * gv / max(pv, 1e-9))
            if len(buf) > 400:
                del buf[:-400]
        # A step whose loss does not depend on the parameters teaches nothing.
        # This is what the inverted loss mask caused: 46% of steps silently dead.
        if float(loss_main.detach()) == 0.0:
            self._n_dead += 1

        extra = {f"gap/{k}": float(np.mean(v)) for k, v in self._kind_gap.items()}
        for b, v in self._t_gap.items():
            if v:
                extra[f"tgap/{b}"] = float(np.median(v))   # median: robust to a heavy tail
        extra["train/dead_steps"] = float(self._n_dead)
        # The gap as a share of the loss is the scale-free version of margin_gap,
        # and the only number comparable across runs with different margins.
        extra["train/margin"] = float(margin)
        extra["train/gap_pct"] = float(100.0 * (l_neg - l_pos).mean().detach()
                                       / loss_main.detach().clamp(min=1e-9))
        self.log_dict(extra, on_step=True, prog_bar=False)
        self.log_dict({
            "train/loss": loss.detach(),
            "train/mse_loss": loss_main.detach(),
            "train/hinge": hinge.detach(),
            "train/margin_gap": (l_neg - l_pos).mean().detach(),
            "train/frac_wrong_order": (l_neg < l_pos).float().mean().detach(),
        }, prog_bar=True, on_step=True)
        return loss
