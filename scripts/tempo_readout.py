"""Differentiable soft tempo readout from beat activations, for the equivariance loss.

The equivariance identity we want to train on,
    f(stretch(x, s)) = f(x) - log2 s,
needs f to be differentiable. The probe measured f by peak-picking and taking the median inter-beat
interval, which has no usable gradient. This module supplies a soft replacement:

  1. mean-centre the beat activation a[t] (so the ACF measures periodicity, not DC level)
  2. unbiased autocorrelation r[l] over candidate lags l (lag = beat period in frames)
  3. soft-select over LOG lag:   w = softmax(tau * r),   f = sum_l w[l] * log2(l)

Reading out in log2 space is what makes the loss natural: the target relation is additive there, so
the equivariance residual is simply (f_s - f_1) - log2 s with no Jacobian correction.

WHY MEAN-CENTRING MATTERS. Without it r[l] is dominated by the squared mean of the activation and
is nearly flat in l, which both flattens the gradient and biases the soft-argmax toward the middle
of the lag grid -- it would manufacture exactly the pull-to-the-prior artefact we are trying to
measure.

OCTAVE AMBIGUITY IS DELIBERATELY NOT RESOLVED HERE. The ACF has peaks at the period and at its
multiples, and the measured failure IS an octave collapse (median defect -0.966 at s=2.0, i.e. the
model reports half the true period). A readout that silently snapped octaves would hide the very
quantity under study. tau controls how sharply the readout commits to the dominant peak.
"""
import numpy as np
import torch

FPS = 50.0


def lag_grid(min_bpm=30.0, max_bpm=300.0, fps=FPS):
    """Candidate beat periods in frames, inclusive, as an integer lag grid."""
    lo = max(2, int(np.floor(60.0 * fps / max_bpm)))
    hi = int(np.ceil(60.0 * fps / min_bpm))
    return torch.arange(lo, hi + 1, dtype=torch.long)


def soft_log_period(act, lags, tau=30.0):
    """act [B, T] beat activation in [0,1] -> (f [B] soft log2 period in frames, w [B, L] weights).

    Uses the unbiased ACF estimate r[l] = sum_t a'[t] a'[t+l] / (T - l): the plain sum falls off
    linearly with l purely from the shrinking overlap, which would bias the readout toward SHORT
    lags -- i.e. toward reporting music as too fast, the exact direction of the failure.
    """
    if act.dim() == 1:
        act = act[None, :]
    B, T = act.shape
    a = act - act.mean(dim=1, keepdim=True)
    r = []
    for l in lags.tolist():
        if T - l < 8:
            r.append(torch.full((B,), -1e9, device=act.device, dtype=act.dtype))
        else:
            r.append((a[:, :T - l] * a[:, l:]).sum(dim=1) / float(T - l))
    r = torch.stack(r, dim=1)                                   # [B, L]
    denom = (a * a).sum(dim=1, keepdim=True) / float(T) + 1e-8  # normalise: scale-free ACF
    r = r / denom
    w = torch.softmax(tau * r, dim=1)
    logl = torch.log2(lags.to(act.device).to(act.dtype))[None, :]
    return (w * logl).sum(dim=1), w


def equivariance_residual(f_a, f_b, s_a, s_b):
    """Residual of f(stretch(x,s)) = f(x) - log2 s between two views of the SAME source.

    Both views are stretches of one clip, so the unknown absolute log-period of the source cancels:
        f_b - f_a  should equal  log2(s_b / s_a)
    (our convention: s > 1 means SLOWER, hence a LONGER period, hence f increases with s). No tempo
    annotation is required anywhere -- the supervision is the ratio we chose ourselves.
    """
    target = torch.log2(torch.as_tensor(s_b, dtype=f_a.dtype, device=f_a.device)) - \
             torch.log2(torch.as_tensor(s_a, dtype=f_a.dtype, device=f_a.device))
    return (f_b - f_a) - target
