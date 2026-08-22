"""Shared MMI pieces. ONE implementation of the numerator mask, imported by both the gate and the
trainer.

WHY A SHARED MODULE. The CRF work died partly because the training path and the evaluation path
were separate implementations that silently drifted apart. Here the numerator mask is defined once;
if the gate certifies it, the trainer is using the certified thing by construction.

THE OBJECTIVE.
    L_MMI = -(log Z_num - log Z_den)
with both partition functions produced by the SAME certified
StructuredBarPointerDP.forward_log_likelihood, differing only by the additive mask below. The
gradient is the occupancy difference gamma_den - gamma_num, obtained for free by autograd on log Z
(d log Z / d log_emission[t, c] = P(class(z_t) = c | obs), the same identity R2/R4.5 use for their
E-steps), and is therefore bounded by 1 in absolute value -- which the gate checks.

THE MASK (variant B of scripts/diagnostics/gate_mmi_numerator.py, which measured F 0.9916 / CMLt 0.9802 on the
183 representable tracks):
  * forbid the BEAT class at frames farther than TOL_FRAMES from every annotation, and
  * forbid the NO_BEAT class at each annotation's own frame, which forces a beat AT the annotation.
The first alone is NOT sufficient -- measured F 0.9103 / CMLt 0.7661 over 217 tracks, because the
model is free to choose a slow tempo that satisfies the mask while landing on only some annotations.

PER-BEAT REPRESENTABILITY, and why this is not a per-track exclusion. A beat gap outside the state
space's tempo grid has no state to occupy, so no path can honour those two annotations and the
numerator is infeasible there. 34 of 217 SMC tracks contain at least one such gap -- 30 of them
dipping below the 30 BPM floor -- but the MEDIAN such track has only 5.2% of its beats affected,
and they are enriched exactly in the failure modes this project exists to fix (expressive timing
x1.56, tempo incontinuity x2.13, pauses x1.65, ternary meter x2.13). Those excursions are momentary
-- a ritardando or a pause -- not nominal tempo; only 44% are even tagged "slow tempo". Dropping the
whole track would therefore discard our hardest and most relevant material over a twentieth of its
beats. Instead the unrepresentable SPAN is left unconstrained: the numerator becomes the set of
paths consistent with the REPRESENTABLE annotations, which is still a valid (if slightly weaker)
numerator, and all 217 tracks stay in.

A finite penalty is used rather than -inf: -inf can leave a logsumexp row with no finite entry,
whose backward is NaN. The tempo transition already carries true -inf on ~half its entries by
design, so the risk is not hypothetical.
"""
import numpy as np
import torch

from rungs.bar_pointer.state_space import BEAT

NO_BEAT = 0
TOL_FRAMES = 3          # 3 frames @ 50 fps = 60 ms, inside the 70 ms metric tolerance
PENALTY = -1e4


def annotation_frames(beat_times, fps, first_frame, num_frames):
    """Annotation times (s) -> frame indices inside a threshold-cropped activation."""
    fr = np.round(np.asarray(beat_times, float) * fps).astype(int) - first_frame
    return fr[(fr >= 0) & (fr < num_frames)]


def numerator_mask(frames, num_frames, interval_frames, device, dtype=torch.float64,
                   tol_frames=TOL_FRAMES, penalty=PENALTY):
    """Additive [num_frames, 3] mask turning forward_log_likelihood into log Z_num.

    frames: annotation frame indices (sorted, already cropped).
    interval_frames: the state space's tempo grid, in frames per beat -- used to decide which beat
    gaps are representable at all.

    Returns (mask, n_anchored, n_spans_freed).
    """
    fr = np.asarray(frames, np.int64)
    gap = np.diff(fr).astype(float)
    lo, hi = float(interval_frames[0]), float(interval_frames[-1])
    bad = (gap < lo) | (gap > hi)                       # gap i spans fr[i] -> fr[i+1]

    allowed = np.zeros(num_frames, bool)
    for f in fr:                                        # the +-tol windows around annotations
        allowed[max(0, f - tol_frames):min(num_frames, f + tol_frames + 1)] = True
    for i in np.nonzero(bad)[0]:                        # unrepresentable spans: leave free
        allowed[fr[i]:min(num_frames, fr[i + 1] + 1)] = True

    # An annotation is ANCHORED (a beat is forced there) only if neither adjacent gap is bad --
    # forcing a beat at one end of an unspannable gap would make the numerator infeasible.
    anchored = np.ones(len(fr), bool)
    for i in np.nonzero(bad)[0]:
        anchored[i] = anchored[i + 1] = False

    mask = torch.zeros((num_frames, 3), dtype=dtype, device=device)
    mask[torch.from_numpy(~allowed).to(device), BEAT] = penalty
    if anchored.any():
        mask[torch.from_numpy(fr[anchored]).to(device), NO_BEAT] = penalty
    return mask, int(anchored.sum()), int(bad.sum())


def mmi_loss(dp, log_initial, log_transition, log_emission, state_to_class, mask):
    """-(log Z_num - log Z_den). Both terms from the SAME certified forward."""
    zden = dp.forward_log_likelihood(log_initial, log_transition, log_emission,
                                     state_to_class=state_to_class)
    znum = dp.forward_log_likelihood(log_initial, log_transition, log_emission + mask,
                                     state_to_class=state_to_class)
    return -(znum - zden), znum, zden


def mmi_loss_batched(dp, log_initial, log_transition, log_emission, state_to_class, mask,
                     lengths=None):
    """Batched -(log Z_num - log Z_den) -> [B]. Same objective, one kernel launch per frame
    for the whole batch instead of one per track.

    The DP is dispatch-bound, not arithmetic-bound (measured: 3x the states costs 0% more time),
    so per-track calls waste ~94% of each training epoch. gate_batched_dp.py certifies this path
    against the unbatched one at 0.000e+00 on log Z AND on d logZ/d emission.

    log_emission / mask: [B, T, num_classes]. All SMC clips are 40 s so training batches are
    rectangular and `lengths` is unnecessary; it exists for the ragged corpus case.
    """
    from batched_dp import forward_log_likelihood_batched as fwd
    zden = fwd(dp, log_initial, log_transition, log_emission,
               state_to_class=state_to_class, lengths=lengths)
    znum = fwd(dp, log_initial, log_transition, log_emission + mask,
               state_to_class=state_to_class, lengths=lengths)
    return -(znum - zden), znum, zden
