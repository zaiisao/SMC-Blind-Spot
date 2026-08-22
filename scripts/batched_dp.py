"""Batched forward for the bar-pointer DP.

WHY. The frame loop is Python-dispatch and kernel-launch bound, not arithmetic bound: measured on an
A6000, tripling the state count (2921 -> 8763) costs 0% more wall-clock, and float64 vs float32
costs 12%. 2000 frames x ~6 ops is ~12k tiny kernel launches with the GPU idle between them. So
widening each launch across a BATCH of tracks is close to free, while the launch count is unchanged.

Stage 1 (fold 0, ~140 SMC tracks) does not need this. Stage 2 does: 3116 corpus tracks at ~2 s each
is 1.7 h per epoch unbatched, versus roughly 6 min batched.

DELIBERATELY NOT A MODIFICATION OF structured_dp.py. That file carries a certificate chain (dense DP
-> hmmlearn / torch-struct, and R1 == madmom event-for-event). This is an additive function taking
the DP object, so the certified path is untouched and remains available as the reference the gate
compares against.

VARIABLE LENGTH. `lengths` records log Z at each track's own final frame. Padding with zeros would
NOT be harmless: a zero emission is log p = 0, i.e. probability 1, so the pointer would keep
advancing through the padding and accumulate transition mass, inflating Z. SMC clips are all 40 s so
stage 1 is rectangular, but the corpus is not.
"""
import torch


def forward_log_likelihood_batched(dp, log_initial_distribution, log_tempo_transition,
                                   log_emission, state_to_class=None, lengths=None):
    """Exact log p(obs) for a batch. Differentiable.

    log_emission: [B, T, num_classes] with state_to_class, else [B, T, num_states].
    lengths: optional [B] long; log Z is taken at each track's own final frame.
    Returns [B].
    """
    B, T = log_emission.shape[0], log_emission.shape[1]
    V, M = dp.num_tempi, dp.beats_per_bar

    def emission_row(frame):
        row = log_emission[:, frame]                              # [B, C] or [B, K]
        return row if state_to_class is None else row[:, state_to_class]

    log_scores = log_initial_distribution.unsqueeze(0) + emission_row(0)      # [B, K]
    if lengths is None:
        out = None
    else:
        out = torch.zeros(B, dtype=log_scores.dtype, device=log_scores.device)
        done = (lengths <= 1)
        if done.any():
            out = out.masked_scatter(done, torch.logsumexp(log_scores[done], dim=1))

    for frame in range(1, T):
        kernel = dp._kernel_at(log_tempo_transition, frame)
        advanced = log_scores[:, dp.predecessor_states]                       # [B, K]
        at_beat_end = log_scores[:, dp.last_state_indices_of_previous_beat].reshape(B, M, V)
        boundary = at_beat_end.unsqueeze(3) + kernel.unsqueeze(0).unsqueeze(0)  # [B, M, from, to]
        mixed = torch.logsumexp(boundary, dim=2)                              # [B, M, to]
        log_scores = advanced.index_copy(1, dp.first_state_indices, mixed.reshape(B, -1))
        log_scores = log_scores + emission_row(frame)
        if lengths is not None:
            done = (lengths == frame + 1)
            if done.any():
                out = out.masked_scatter(done, torch.logsumexp(log_scores[done], dim=1))

    return torch.logsumexp(log_scores, dim=1) if lengths is None else out
