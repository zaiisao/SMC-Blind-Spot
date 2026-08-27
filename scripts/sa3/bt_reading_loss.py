"""Contrastive METRICAL-READING loss for Beat This.

Motivation, in order of evidence:
- SMC's strongest failure predictors are ambiguity of metrical level (t=-3.38)
  and beat phase (t=-2.98), fold-honest over 216 tracks.
- The stock ShiftTolerantBCELoss penalizes energy at offbeat/subdivision frames
  at 1/20th the weight of a missed beat (pos_weight=20, negatives weight 1), so
  the two SMC-dominant errors are the cheapest ones to make.
- Framewise-independent probabilities cannot represent COMPETING readings at all
  (the ISMIR'26 tatum-grid submission makes the same argument): 70 vs 143 bpm is
  a choice between hypotheses, not a property of any single frame.

So add a term that compares readings as wholes. For each excerpt, score the
annotated grid and corrupted versions of it (offbeat, double, two half phases)
by the mean beat logit along each grid, then demand the true reading win a
softmax:

    L_read = -log  exp(s_true/T) / sum_g exp(s_g/T)

Gradient flows only where readings compete -- confident excerpts contribute ~0.
Unlike the ControlNet contrastive loss (refuted 2026-08-24), the quantity shaped
here IS the quantity evaluated: there is no training/deployment gap to fall into.

Why not bounded by the +0.028 oracle: the oracle regrids FIXED activations;
this changes what the activations are.
"""
import numpy as np
import torch
import torch.nn.functional as F

# widened candidate set, toggled by ReadingLoss(wide=True)
WIDE = False


def _grid_scores(logits, beat_frames, n_frames, tol_pool):
    """Scores for [true, offbeat, double, half_a, half_b] given beat FRAME indices."""
    if len(beat_frames) < 6:
        return None
    b = beat_frames
    mid = ((b[:-1] + b[1:]) // 2)
    cands = [
        b,                                        # true reading
        mid,                                      # offbeat: right tempo, wrong phase
        np.sort(np.concatenate([b, mid])),        # double: subdivision level
        b[0::2],                                  # half, phase a
        b[1::2],                                  # half, phase b
    ]
    if WIDE:
        # A softmax can only move mass onto candidates it contains. With only the
        # five above, suppressing `double` pushed probability onto UNCOVERED
        # readings instead of onto the truth (measured: doubled 29.6->11.1% but
        # level_ok also 25.9->11.1%). Non-integer ratios matter -- 15.7% of SMC
        # tracker errors match no clean ratio.
        ibi = float(np.median(np.diff(b)))
        span0, span1 = float(b[0]), float(b[-1])
        for r in (1.5, 2 / 3, 3.0, 1 / 3, 4 / 3, 3 / 4):
            step = ibi / r
            if step < 3:
                continue
            g = np.arange(span0, span1 + 1e-6, step).astype(np.int64)
            if len(g) >= 4:
                cands.append(g)
        # third-of-a-beat phase offsets: phase errors that are not half-beat
        for frac in (1 / 3, 2 / 3):
            cands.append((b + int(round(frac * ibi))).astype(np.int64))
    pooled = tol_pool(logits[None, None, :])[0, 0]      # +/-1 frame tolerance
    out = []
    for c in cands:
        c = c[(c >= 0) & (c < n_frames)]
        if len(c) < 3:
            return None
        # Score = on-grid energy MINUS energy at the positions this reading
        # claims are non-beats (its own midpoints). Plain on-grid mean is broken
        # for half readings: they sample a SUBSET of true beats and tie the true
        # reading by construction (verified: perfect activations scored loss
        # 1.10 = log 3, a three-way tie with both half phases). Energy at the
        # skipped beats is exactly what refutes a half reading.
        cm = (c[:-1] + c[1:]) // 2
        on = pooled[torch.as_tensor(c, device=logits.device, dtype=torch.long)].mean()
        off = pooled[torch.as_tensor(cm, device=logits.device, dtype=torch.long)].mean()
        out.append(on - off)
    return torch.stack(out)


class ReadingLoss(torch.nn.Module):
    def __init__(self, temperature=1.0, wide=False):
        super().__init__()
        self.temperature = temperature
        self.wide = wide
        self.tol_pool = torch.nn.MaxPool1d(3, stride=1, padding=1)

    def forward(self, beat_logits, truth_beat, padding_mask):
        """beat_logits (B,T) raw logits; truth_beat (B,T) 0/1; padding_mask (B,T)."""
        losses = []
        for i in range(beat_logits.shape[0]):
            valid = int(padding_mask[i].sum())
            t = truth_beat[i, :valid]
            frames = torch.nonzero(t > 0.5).squeeze(-1).cpu().numpy()
            global WIDE
            WIDE = self.wide
            s = _grid_scores(beat_logits[i, :valid], frames, valid, self.tol_pool)
            if s is None:
                continue
            losses.append(F.cross_entropy(
                (s / self.temperature)[None, :],
                torch.zeros(1, dtype=torch.long, device=s.device)))
        if not losses:
            return beat_logits.sum() * 0.0
        return torch.stack(losses).mean()


def attach(pl_model, lam, temperature=1.0, wide=False):
    """Add the reading loss to a PLBeatThis instance's _compute_loss in place."""
    reading = ReadingLoss(temperature, wide=wide)
    reading = reading.to(next(pl_model.parameters()).device)
    orig = pl_model._compute_loss

    def compute(batch, model_prediction):
        losses = orig(batch, model_prediction)
        lr_ = reading(model_prediction["beat"], batch["truth_beat"].float(),
                      batch["padding_mask"])
        losses["reading"] = lr_
        losses["total"] = losses["total"] + lam * lr_
        return losses

    pl_model._compute_loss = compute
    pl_model._reading_loss_mod = reading    # keep a reference; move with .to()
    return pl_model
