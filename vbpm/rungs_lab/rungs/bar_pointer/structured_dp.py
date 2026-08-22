"""Structured DP for the Krebs bar-pointer: O(num_states + beats_per_bar * num_tempi^2) per frame
instead of O(num_states^2).

The transition is not a general matrix. It is exactly two things (Krebs 2015):
  * inside a tempo's block, a DETERMINISTIC +1 advance                    -> a gather
  * at beat boundaries only, a tempo mix (last_states of beat b-1 -> first_states of beat b),
    with madmom's exponential_transition                                  -> one [num_tempi, num_tempi]
                                                                            logsumexp
So we never materialize [num_states, num_states]. Measured at our real num_states=16756 (4/4,
86.13 fps, num_tempi=71):

    dense       1.12 GB, K^2       = 280,763,536 ops/frame   -> 4.4 trillion ops per 3-min song
    structured  no matrix, K+M*V^2 =      36,920 ops/frame   -> 7,605x less work

which is the difference between ~2 s and ~5-9 HOURS per song (hmmlearn/librosa, dense, both of which
agree with this code on 100% of frames at that size -- see rungs/bar_pointer/inference.py for the numbers).

Speed is NOT why this file exists: madmom's Cython HMM runs the same topology in 1.04 s vs our
1.95 s. GRADIENTS are why. And our 1.9x deficit is not intrinsic -- viterbi costs a flat ~122
us/frame at num_states=952 and at num_states=16756 alike (17.6x the ops, 1.00x the time), so we are
bound by Python dispatch and kernel launches, not arithmetic.

This file exists because our topology is STRUCTURED, not merely sparse. A generic sparse HMM needs a
segment-max over each state's incoming arcs, and core PyTorch has no scatter_max -- pomegranate's
SparseHMM (which IS sparse and differentiable, and agrees with this code on 100% of frames) therefore
loops `for j in range(n_states)` per frame in Python and takes ~4.9 hours per song. Here the +1
advance makes the transition one gather plus one dense [M, V, V] max, so the segment-max never
arises. madmom's Cython HMM runs this topology fast but gives no gradient, which R2-R4 need.
See rungs/bar_pointer/inference.py for the full library comparison.

Certified against the dense DP in tests/test_structured_dp.py; the dense DP is itself certified
against hmmlearn and torch-struct. Chain of certificates.
"""
import numpy as np
import torch


class StructuredBarPointerDP:
    """Precomputes the index structure once; then forward/viterbi are cheap per-frame ops."""

    def __init__(self, state_space, device: str = "cuda", dtype: torch.dtype = torch.float64):
        """dtype: float64 for decoding (R1 must match madmom's MAP exactly -- float32 accumulation
        over ~20k frames loses it; measured 0.12 nats worse than madmom's path on a val song, i.e. a
        strictly suboptimal Viterbi path). float32 is fine for R2+ TRAINING, where 0.12 nats is noise.
        """
        self.state_space = state_space
        self.device = device
        self.dtype = dtype
        self.num_states = state_space.num_states
        self.beats_per_bar = state_space.beats_per_bar
        self.num_tempi = state_space.num_tempi

        # Predecessor of every state = state - 1. Correct for all NON-first states; the first state of
        # each beat has no in-block predecessor and is overwritten by the boundary mix below.
        predecessor_states = np.arange(self.num_states, dtype=np.int64) - 1
        predecessor_states[0] = 0
        self.predecessor_states = torch.from_numpy(predecessor_states).to(device)

        # The boundary mix reads from the last states of the PREVIOUS beat and writes to the first
        # states of this beat. Both flattened to [beats_per_bar * num_tempi] for index_copy.
        # Beat 0 reads from beat -1 (= the last beat), which is what wraps the bar.
        self.first_state_indices = torch.from_numpy(state_space.first_states.reshape(-1)).to(device)
        last_states_of_previous_beat = state_space.last_states[np.arange(self.beats_per_bar) - 1]
        self.last_state_indices_of_previous_beat = torch.from_numpy(
            last_states_of_previous_beat.reshape(-1)).to(device)

        # Inverse of first_state_indices: for each state, its slot in the flattened
        # [beats_per_bar, num_tempi] first-state grid, or -1 if it is not a first state. Viterbi
        # backtracking uses this to tell "look up a stored backpointer" from "predecessor is
        # state - 1", which is what lets us store only the first states' backpointers.
        first_state_slot = np.full(self.num_states, -1, dtype=np.int64)
        first_state_slot[state_space.first_states.reshape(-1)] = np.arange(self.beats_per_bar
                                                                           * self.num_tempi)
        self.first_state_slot = first_state_slot                       # numpy: backtracking is scalar

    def build_log_tempo_transition(self, transition_lambda: float) -> torch.Tensor:
        """[num_tempi, num_tempi] log p(tempo_to | tempo_from) at a beat boundary.

        The zeros are load-bearing. madmom's exponential_transition THRESHOLDS improbable tempo
        jumps to exactly 0 -- at transition_lambda=100 that is 2504 of 5041 entries -- and a zero
        means FORBIDDEN, so it must map to -inf. This line used to read log(p + 1e-30), which mapped
        every forbidden jump to a merely-expensive -69.1 instead. Viterbi then bought them: on a
        Liszt val song our Viterbi path took a transition madmom assigns probability zero (61.5% path
        agreement, an illegal jump at frame 8861, path score -inf under madmom's own factors).
        With true -inf, R1 reproduces madmom's path and score exactly.

        -inf is safe here: every column keeps its diagonal (ratio 1.0 -> prob 1.0 -> never
        thresholded), so no state is left unreachable and no logsumexp sees an all--inf row.
        """
        probabilities = self.state_space.tempo_transition_probabilities(transition_lambda)
        log_probabilities = torch.log(torch.from_numpy(probabilities).to(self.dtype))
        return log_probabilities.to(self.device)                  # p == 0 -> -inf, as intended

    @staticmethod
    def _kernel_at(log_tempo_transition, frame):
        """The tempo kernel governing the boundary mix INTO `frame`. A 2-D [V, V] kernel is fixed
        (R1/R2); a 3-D [num_frames, V, V] kernel is per-frame audio-conditioned (R3) -- indexed at
        frame-1, the boundary being crossed. The 2-D path is byte-identical to before (R1 cert)."""
        return log_tempo_transition[frame - 1] if log_tempo_transition.dim() == 3 \
            else log_tempo_transition

    def _advance_one_frame(self, log_scores, log_tempo_transition, combine_over_source_tempi):
        """One transition step, WITHOUT the emission.

        combine_over_source_tempi reduces the source-tempo axis of a
        [beats_per_bar, tempo_from, tempo_to] tensor and returns (values, argmax_or_None):
        logsumexp for the forward algorithm, max for Viterbi.
        """
        advanced = log_scores[self.predecessor_states]                            # deterministic +1
        log_scores_at_beat_end = log_scores[self.last_state_indices_of_previous_beat].reshape(
            self.beats_per_bar, self.num_tempi)
        boundary_scores = (log_scores_at_beat_end.unsqueeze(2)
                           + log_tempo_transition.unsqueeze(0))                   # [beat, from, to]
        mixed_scores, source_tempo_index = combine_over_source_tempi(boundary_scores)
        advanced = advanced.index_copy(0, self.first_state_indices, mixed_scores.reshape(-1))
        return advanced, source_tempo_index

    def _emission_row(self, log_emission, frame, state_to_class):
        """One frame's per-state emission [num_states].

        With state_to_class [num_states] (long, on device), log_emission is the COMPACT
        [num_frames, num_classes] class-density table and we gather on the fly -- the full
        [num_frames, num_states] array never exists. At Böck's 3 classes that is 372 KB instead of
        2.08 GB per 3-min song (5,585x), and it is what makes batched decoding fit in GPU memory.
        Without state_to_class, log_emission is already per-state [num_frames, num_states].

        The gather is differentiable (backward scatter-adds into the [num_frames, num_classes]
        table), so R2+ can train straight through the compact form -- and their gradient buffer
        shrinks by the same factor.
        """
        row = log_emission[frame]
        return row if state_to_class is None else row[state_to_class]

    def forward_log_likelihood(self, log_initial_distribution, log_tempo_transition,
                               log_emission, state_to_class=None) -> torch.Tensor:
        """Exact log p(obs). Differentiable -- the R2+ training objective.

        log_emission: [num_frames, num_states], or [num_frames, num_classes] with state_to_class
        (see _emission_row).
        """
        log_scores = log_initial_distribution + self._emission_row(log_emission, 0, state_to_class)
        for frame in range(1, log_emission.shape[0]):
            log_scores, _ = self._advance_one_frame(
                log_scores, self._kernel_at(log_tempo_transition, frame),
                lambda scores: (torch.logsumexp(scores, dim=1), None))
            log_scores = log_scores + self._emission_row(log_emission, frame, state_to_class)
        return torch.logsumexp(log_scores, dim=0)

    @torch.no_grad()
    def viterbi(self, log_initial_distribution, log_tempo_transition, log_emission,
                state_to_class=None, return_log_score: bool = False):
        """Exact MAP state path [num_frames] (long); with return_log_score, (path, MAP log score).

        The score is what multi-meter decoding compares: madmom's multi-pattern model stacks one
        bar per meter block-diagonally with NO cross-meter transitions, so decoding the union
        equals decoding each meter separately and keeping the higher-scoring path -- PROVIDED both
        runs use the same initial-distribution constant (see r1's shared -log(total states)).

        log_emission: [num_frames, num_states], or [num_frames, num_classes] with state_to_class
        (see _emission_row).

        Stores backpointers ONLY for the first state of each beat. Every other state's predecessor is
        deterministically state - 1 -- that is the Krebs +1 advance, the same property that makes the
        forward cheap -- so a full [num_frames, num_states] backpointer table would be 98% a stored
        copy of "state - 1": 2.08 GB for a 3-min song and 6.07 GB for the longest song in our val set.
        This table is [num_frames, beats_per_bar, num_tempi] instead, 59x smaller (35 MB / 103 MB),
        which is also what makes batched decoding fit at all.
        """
        num_frames = log_emission.shape[0]
        log_best_score = log_initial_distribution + self._emission_row(log_emission, 0,
                                                                       state_to_class)
        last_states_by_beat = self.last_state_indices_of_previous_beat.reshape(
            self.beats_per_bar, self.num_tempi)
        chosen_previous = torch.empty((num_frames, self.beats_per_bar, self.num_tempi),
                                      dtype=torch.long, device=self.device)
        for frame in range(1, num_frames):
            log_best_score, source_tempo_index = self._advance_one_frame(
                log_best_score, self._kernel_at(log_tempo_transition, frame),
                lambda scores: torch.max(scores, dim=1))
            log_best_score = log_best_score + self._emission_row(log_emission, frame,
                                                                 state_to_class)
            # For each first state, which last-state of the previous beat did the argmax pick?
            chosen_previous[frame] = last_states_by_beat.gather(1, source_tempo_index)

        # Backtracking is inherently scalar, so run it in numpy: T tiny GPU indexing ops cost far
        # more than the transfer of a 35 MB table.
        chosen_previous_flat = chosen_previous.cpu().numpy().reshape(num_frames, -1)
        state_path = np.empty(num_frames, dtype=np.int64)
        state = int(log_best_score.argmax())
        state_path[-1] = state
        for frame in range(num_frames - 1, 0, -1):
            slot = self.first_state_slot[state]
            state = int(chosen_previous_flat[frame, slot]) if slot >= 0 else state - 1
            state_path[frame - 1] = state
        state_path = torch.from_numpy(state_path).to(self.device)
        if return_log_score:
            return state_path, float(log_best_score.max())
        return state_path

    def dense_transition(self, log_tempo_transition) -> torch.Tensor:
        """The SAME transition written out as a dense [num_states, num_states] -- only for
        certification on small instances (num_states^2 blows up at real sizes).
        dense[i, j] = log p(z_t = j | z_{t-1} = i)."""
        dense = torch.full((self.num_states, self.num_states), -float("inf"),
                           dtype=log_tempo_transition.dtype, device=self.device)
        is_non_first_state = torch.ones(self.num_states, dtype=torch.bool, device=self.device)
        is_non_first_state[self.first_state_indices] = False
        non_first_states = torch.arange(self.num_states, device=self.device)[is_non_first_state]
        dense[non_first_states - 1, non_first_states] = 0.0                      # deterministic advance
        last_states_by_beat = self.last_state_indices_of_previous_beat.reshape(
            self.beats_per_bar, self.num_tempi)
        first_states_by_beat = self.first_state_indices.reshape(self.beats_per_bar, self.num_tempi)
        for beat in range(self.beats_per_bar):
            dense[last_states_by_beat[beat].unsqueeze(1),
                  first_states_by_beat[beat].unsqueeze(0)] = log_tempo_transition
        return dense
