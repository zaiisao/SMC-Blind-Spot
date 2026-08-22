"""Exact inference on a discrete HMM (log-space). The readable REFERENCE implementation.

WHAT THIS IS FOR. This dense form cannot run our real model and is not meant to: at our real state
count (num_states=16756 at 4/4, 86.13 fps) a dense transition is 1.12 GB and 280,763,536 ops/frame,
i.e. 4.4 trillion ops per 3-minute song. rungs/bar_pointer/structured_dp.py is the engine; this file is the
plain-English statement of what that engine computes, and the oracle that certifies it
(tests/test_structured_dp.py builds the same model both ways and compares).

It earns its place by being the only differentiable dense reference we have, so it can certify the
FORWARD/gradient path that R2+ train through -- hmmlearn and librosa produce a path but do not give
gradients. The time-varying form below is likewise a reference: R3's audio-conditioned transition
will have to be a structured [num_frames-1, num_tempi, num_tempi] tempo transition, because a dense
[num_frames-1, num_states, num_states] would be 17.41 TB for one song.

Factors (the standard HMM symbols are given so this maps onto any textbook, but the code spells the
names out -- `log_A` and `log_B` are impossible to tell apart at a glance):

    log_initial_distribution : [num_states]                  log p(z_1)                        (pi)
    log_transition           : [num_states, num_states]      log p(z_t = j | z_{t-1} = i),      (A)
                               indexed [from_state, to_state]
                            or [num_frames - 1, num_states, num_states]
                               for a TIME-VARYING (audio-conditioned) transition; R3+ need that
                               form, R1/R2 use the 2-D one.
    log_emission             : [num_frames, num_states]      log p(obs_t | z_t = k)             (B)

forward_log_likelihood is differentiable -- it is the training objective for R2+ (maximize the exact
marginal by gradient ascent). viterbi is the deployment path finder. Both are EXACT because the model is
finite + Markov + factorized; that is the whole premise of the ladder.

WHY NOT A LIBRARY? Not for lack of one, and not for correctness: FOUR independent libraries compute
exactly our answer -- hmmlearn, librosa, torch-struct and pomegranate's SparseHMM all agree with our
structured DP on 100% of frames (and so does madmom, at the real size). Measured, per 3-min song:

    madmom (sparse, Cython)        1.04 s       agrees 100%, but gives NO gradient  <- FASTEST
    ours (structured, GPU)         1.95 s
    hmmlearn (dense, CPU)          4.9 hours    agrees 100%
    librosa  (dense, CPU)          8.9 hours    agrees 100%
    pomegranate SparseHMM (torch)  4.9 hours    agrees 100%; IS sparse AND differentiable
    torch-struct (dense, GPU)      17.41 TB of potentials -> OOMs at 40 frames

Note madmom BEATS us 1.9x, so speed is NOT why this exists -- gradients are. Our cost is not the
compute: viterbi takes a flat ~122 us/frame whether num_states is 952 or 16756 (17.6x the ops, 1.00x
the time), i.e. we are bound by Python dispatch + ~10 kernel launches per frame, not arithmetic.
Autograd is not the culprit either (viterbi is @torch.no_grad; building the graph costs 1.16x on
forward only). That also means the headroom is large and untapped: a 17.6x wider tensor is free, so
batching songs -- which madmom's single-song C loop cannot do -- should amortize the launch cost
almost entirely.

The real obstacle is narrower than "dense vs sparse". A GENERIC sparse HMM needs a SEGMENT-MAX (a
scatter_max over each state's incoming arcs), and core PyTorch has no scatter_max -- so pomegranate
falls back to a Python `for j in range(n_states)` loop per frame, which its own source comment flags
as a placeholder. That is 260M Python iterations for one song.

Our topology is not generically sparse, it is STRUCTURED, and that is the whole point: Krebs's +1
advance makes the transition exactly one gather plus one dense
[beats_per_bar, num_tempi, num_tempi] max. The segment-max that forces every generic library onto a
slow path never arises for us -- a 4x71x71 dense max replaces it. So this is ~40 lines not because
nobody wrote Viterbi, but because our structure dissolves the operation that makes generic sparse
Viterbi hard. (k2 attacks the same problem from the other end, shipping CUDA segment-max kernels for
arbitrary FSAs. It has a wheel for our exact torch build and is a genuine alternative we have NOT yet
evaluated.)

We keep the libraries where they belong: tests/test_inference.py certifies this code against BOTH
hmmlearn AND torch-struct as independent oracles (both agree to ~1e-14).
"""
import torch


def forward_log_likelihood(log_initial_distribution: torch.Tensor, log_transition: torch.Tensor,
                           log_emission: torch.Tensor) -> torch.Tensor:
    """Forward algorithm. Returns scalar log p(obs) = log sum_z p(obs, z). Differentiable."""
    transition_is_time_varying = log_transition.dim() == 3
    # The forward variable (alpha): log p(obs_1..t, z_t = k) for every state k.
    log_forward_scores = log_initial_distribution + log_emission[0]
    for frame in range(1, log_emission.shape[0]):
        log_transition_now = log_transition[frame - 1] if transition_is_time_varying else log_transition
        # log_forward_scores'[j] = logsumexp_i( log_forward_scores[i] + log_transition_now[i, j] )
        log_forward_scores = (torch.logsumexp(log_forward_scores.unsqueeze(1) + log_transition_now, dim=0)
                              + log_emission[frame])
    return torch.logsumexp(log_forward_scores, dim=0)


@torch.no_grad()
def viterbi(log_initial_distribution: torch.Tensor, log_transition: torch.Tensor,
            log_emission: torch.Tensor) -> torch.Tensor:
    """Max-product + backtracking. Returns the exact MAP state path: [num_frames] (long)."""
    transition_is_time_varying = log_transition.dim() == 3
    num_frames, num_states = log_emission.shape
    # The Viterbi variable (delta): log-score of the best path that ends in each state.
    log_best_score = log_initial_distribution + log_emission[0]
    backpointers = torch.empty((num_frames, num_states), dtype=torch.long, device=log_emission.device)
    for frame in range(1, num_frames):
        log_transition_now = log_transition[frame - 1] if transition_is_time_varying else log_transition
        scores_by_previous_state = log_best_score.unsqueeze(1) + log_transition_now   # [from, to]
        best_previous_state = scores_by_previous_state.argmax(dim=0)                  # [num_states]
        log_best_score = (scores_by_previous_state.gather(0, best_previous_state.unsqueeze(0)).squeeze(0)
                          + log_emission[frame])
        backpointers[frame] = best_previous_state
    state_path = torch.empty(num_frames, dtype=torch.long, device=log_emission.device)
    state_path[-1] = log_best_score.argmax()
    for frame in range(num_frames - 1, 0, -1):
        state_path[frame - 1] = backpointers[frame, state_path[frame]]
    return state_path
