"""R1 -- Baseline A in OUR framework: the Böck 2016 DBN, hand-set factors, our inference.

The same model R0 runs (madmom's DBN), rebuilt on our own code and run through our own DP. THE
CERTIFICATE: if R1 reproduces R0, our inference is validated on a real model and the ladder's anchor
is set -- after which R0 -> R4 is a clean "hand-set vs learned bar-pointer, same inference" axis.
R2+ change ONLY how the factors are produced.

Status (re-certified 2026-07-17, live Beat This 50 fps, squeeze bounding, 30 songs across
ballroom/beatles/gtzan/hainsworth/hjdb): event-identical to R0 30/30 shipped AND 30/30 bare;
per-meter Viterbi path agreement 1.0 against madmom's own hmm.viterbi; scores match to ~1e-4
(madmom computes observation densities in float32, we in float64 -- ~6e-8/frame). That float gap
is visible ONLY on degenerate inputs: constant-background synthetic click tracks create huge
score-tie plateaus where the two implementations may pick equal-score paths shifted by one frame.
Real activations break the ties; the certificate is a real-song claim.
The re-certification caught and fixed two real divergences: (1) the correct=False read-out
emitted a beat at frame 0 when the path starts inside a beat region (madmom's np.diff read-out
cannot fire there -- see rungs/bar_pointer/readout.py); (2) _log_class_densities hardcoded clip,
diverging from R0 under bounding="squeeze".
madmom's DBNDownBeatTrackingProcessor wraps the model in three
deployment heuristics -- num_tempi=60 (a coarser log-spaced tempo grid), threshold=0.05 (crop to the
main above-threshold segment) and correct=True (report the activation peak inside a beat region
rather than the region entry) -- and R1's DEFAULTS now match those shipped values, so out of the
box R1 == R0 event-for-event. They are deployment heuristics, not the model (~+0.006 beat F, almost all
of it threshold=0.05 cropping Ballroom's fade-ins/outs); the BARE model -- what the certificate
pins to a matched madmom and what rung-to-rung comparisons should run -- is the explicit opt-out
num_tempi=None, threshold=0.0, correct=False.

Everything here is HAND-SET; nothing is learned:
  state space : Krebs 2015 -- a tempo of i frames-per-beat owns exactly i states, so the pointer
                advances exactly +1 state per frame (see rungs/bar_pointer/state_space.py for why a uniform
                grid fails). Multi-meter (madmom's default [3, 4]) as in Böck 2016: one bar per
                meter, no cross-meter transitions, best Viterbi score wins (see __init__).
  transition  : deterministic advance inside a beat; madmom's exponential tempo mix at beat
                boundaries, exp(-transition_lambda |ratio - 1|), row-normalized.
  emission    : Böck 2016's mutually-exclusive {no-beat, beat, downbeat} observation model on the
                DECORRELATED activation (max(beat - downbeat, eps), downbeat) -- the same
                decorrelation R0 uses, and required because that model assumes the columns sum to <=1.

Everything is torch (GPU + autograd-ready), which is what R2-R4 need and what madmom cannot give.
"""
from typing import Optional

import numpy as np
import torch

from rungs.deployment import threshold_crop
from rungs.bar_pointer.readout import state_path_to_events
from rungs.bar_pointer.state_space import BarPointerStateSpace
from rungs.bar_pointer.structured_dp import StructuredBarPointerDP
from rungs.base import Rung


class DBN2016(Rung):
    def __init__(self, fps: float, min_bpm: float = 55.0, max_bpm: float = 215.0,
                 beats_per_bar=(3, 4), observation_lambda: int = 16,
                 transition_lambda: int = 100, bounding: str = "clip", eps: float = 1e-5,
                 dtype: torch.dtype = torch.float64, device: str = "cuda",
                 num_tempi: Optional[int] = 60, threshold: float = 0.05, correct: bool = True):
        """dtype defaults to float64 because R1's whole job is to reproduce madmom's MAP path. In
        float32 the score accumulated over ~20k frames drifts enough to lose it (measured: 0.12 nats
        worse than madmom on a Beethoven val song -- a strictly suboptimal Viterbi path, which Viterbi is
        not allowed to be). The cost is small: this DP is kernel-launch bound, not compute bound.

        num_tempi / threshold / correct default to madmom's SHIPPED values (60 / 0.05 / True), so a
        default R1 reproduces R0 out of the box. The bare model -- every integer interval, no crop,
        no peak snap; what the certificate runs and what rung comparisons should use -- is
        num_tempi=None, threshold=0.0, correct=False (see rungs/bar_pointer/state_space.py and
        rungs/deployment.py). threshold / correct set the INSTANCE defaults for predict()'s
        deployment options; predict()'s own arguments still override per call.

        beats_per_bar: an int (single meter) or a collection -- madmom's shipped default is [3, 4].
        Multiple meters replicate madmom's MultiPatternStateSpace exactly: the union HMM is
        block-diagonal with NO cross-meter transitions, so decoding it equals decoding each meter
        separately and keeping the higher-scoring Viterbi path. The one wiring subtlety: madmom's
        initial distribution is uniform over the UNION, so every meter's Viterbi search must use the SAME
        constant -log(total states across meters) -- per-meter uniform would bias the comparison
        toward the smaller (3-beat) space by log(num_states ratio).
        """
        super().__init__(fps=fps, bounding=bounding, eps=eps)
        self.threshold, self.correct = threshold, correct

        if isinstance(beats_per_bar, (int, np.integer)):
            beats_per_bar = (int(beats_per_bar),)

        self.beats_per_bar = tuple(beats_per_bar)
        if not self.beats_per_bar:
            raise ValueError("beats_per_bar must name at least one meter, got an empty collection")

        self.dtype, self.device = dtype, device

        self.state_spaces = [
            BarPointerStateSpace(fps, min_bpm, max_bpm, bpb, observation_lambda, num_tempi=num_tempi)
            for bpb in self.beats_per_bar
        ]

        self.dynamic_programs = [
            StructuredBarPointerDP(space, device=device, dtype=dtype)
            for space in self.state_spaces
        ]

        # Uniform over the union of all meters' states (madmom's initial distribution) -- the shared
        # constant that makes cross-meter score comparison valid.
        total_states = sum(space.num_states for space in self.state_spaces)
        self.log_initial_distributions = [
            torch.full((space.num_states,), -float(np.log(total_states)), dtype=dtype, device=device)
            for space in self.state_spaces
        ]

        # One meter's tempo grid == every meter's (it depends only on the tempo range), so the
        # tempo transition is shared; built from the first DP.
        self.log_tempo_transition = \
            self.dynamic_programs[0].build_log_tempo_transition(transition_lambda)

        # The DP gathers each state's class density on the fly from this map -- the per-state
        # [num_frames, num_states] emission (2.08 GB per 3-min song) never exists; only the
        # [num_frames, 3] class table (372 KB) does.
        self.state_to_classes = [
            torch.from_numpy(space.position_classes).to(device)
            for space in self.state_spaces
        ]

        self.observation_lambda = observation_lambda

    def _log_class_densities(self, beat_activation: np.ndarray,
                             downbeat_activation: np.ndarray) -> torch.Tensor:
        """Böck 2016's 3-class densities on the decorrelated activation.

        Returns [num_frames, 3] -- one column per observation class (NO_BEAT, BEAT, DOWNBEAT). Every
        state emits through exactly one class, so this table plus state_to_class IS the emission.
        """
        # Bound with the instance's convention (was a hardcoded clip -- which silently diverged
        # from R0 under bounding="squeeze": measured, single-meter events differed on real songs).
        beat_activation, downbeat_activation, decorrelation_floor = self._bound(
            beat_activation, downbeat_activation)

        # Decorrelate: the model needs mutually-exclusive classes, but the frontend's beat channel
        # fires on downbeats too, so beat + downbeat ~ 2 there and the no-beat density goes negative.
        beat_not_downbeat_activation = np.maximum(
            beat_activation - downbeat_activation, decorrelation_floor)

        # The no-beat probability is shared out over the observation_lambda - 1 non-beat states.
        num_non_beat_states = self.observation_lambda - 1
        no_beat_probability = 1.0 - beat_not_downbeat_activation - downbeat_activation

        # Deliberately NOT clipped, and madmom does not clip it either. A zero here means the frame
        # CANNOT be a non-beat state; flooring it to log(eps/15) = -14.2 downgrades an impossibility
        # to an improbability, and Viterbi will buy it -- measured on a Ballroom val song where this
        # fires on just 5 frames: 92.07% path agreement and a 0.0187-nat SUBOPTIMAL Viterbi path. Safe
        # because decorrelation guarantees no_beat_probability >= 0 (so the worst case is
        # log(0) = -inf, never log(negative) = nan), and the beat/downbeat classes are floored at
        # eps and therefore always finite, so no frame can be -inf in all three classes.
        with np.errstate(divide="ignore"):
            log_class_densities = np.stack(                              # [num_frames, 3]
                [np.log(no_beat_probability / num_non_beat_states),
                 np.log(beat_not_downbeat_activation), np.log(downbeat_activation)], axis=1)
        return torch.from_numpy(log_class_densities).to(dtype=self.dtype, device=self.device)

    # Training-side utilities: unused by predict(), shared by the learned rungs.
    def log_class_densities(self, activations: torch.Tensor) -> torch.Tensor:
        """Differentiable mirror of _log_class_densities: [num_frames, 2] probability TENSOR ->
        [num_frames, 3].

        One deliberate difference: no_beat is floored at 1e-12 rather than reaching log(0) = -inf,
        which would make the gradient NaN. The floor only bites where the numpy path is -inf.
        """
        beat, downbeat, decorrelation_floor = self._bound_tensor(
            activations[:, 0], activations[:, 1])
        beat_not_downbeat = (beat - downbeat).clamp(min=decorrelation_floor)
        no_beat = (1.0 - beat_not_downbeat - downbeat).clamp(min=1e-12)
        num_non_beat_states = self.observation_lambda - 1
        return torch.stack([torch.log(no_beat / num_non_beat_states),
                            torch.log(beat_not_downbeat), torch.log(downbeat)], dim=1)

    def annotated_state_path(self, beat_frames: np.ndarray, beat_in_bar: np.ndarray,
                             beats_per_bar) -> Optional[tuple]:
        """(state_path[int64], meter_index) for the span beat_frames[0]..beat_frames[-1], or None
        if unrepresentable (meter absent / gap outside the tempo grid / non-consecutive bar
        positions). Between annotated beats f_i -> f_{i+1} the pointer occupies the tempo block
        whose interval equals the frame gap, advancing +1 per frame, then takes the beat-boundary
        transition. beats_per_bar is the SONG meter -- a crop may not span a full bar."""
        meters = self.beats_per_bar
        bpb = int(beats_per_bar)
        if bpb not in meters:
            return None
        meter_index = meters.index(bpb)
        space = self.state_spaces[meter_index]
        min_interval = int(space.interval_frames[0])
        max_interval = int(space.interval_frames[-1])
        gaps = np.diff(beat_frames)
        if len(gaps) == 0 or gaps.min() < min_interval or gaps.max() > max_interval:
            return None
        if not np.all((beat_in_bar[1:] - beat_in_bar[:-1]) % bpb == 1):
            return None
        path = np.concatenate([
            space.first_states[beat_in_bar[i], gap - min_interval] + np.arange(gap)
            for i, gap in enumerate(gaps)])
        return path.astype(np.int64), meter_index

    @torch.no_grad()
    def _predict_features(self, activations: np.ndarray, threshold: Optional[float] = None,
                            correct: Optional[bool] = None) -> dict:
        """activations: [num_frames, 2] probabilities (beat, downbeat).

        threshold / correct: madmom's shipped deployment lessons (crop dead air before decoding /
        report beats at activation peaks), behavior-copied, named as madmom names them. None falls
        back to the instance defaults (constructor), which are madmom's SHIPPED values; the bare
        model -- what the certificate pins to madmom and what rung comparisons use -- is the
        explicit opt-out (threshold=0.0, correct=False). See rungs/deployment.py.
        """
        threshold = self.threshold if threshold is None else threshold
        correct = self.correct if correct is None else correct

        # The crop must see what the prediction sees: the decorrelated pair (madmom thresholds the
        # combined activation it runs on, not the raw channels). _bound reads the base Rung
        # bounding/eps (frontend-wired by the Tracker, "clip" standalone) -- same rule as every rung.
        beat_activation, downbeat_activation, decorrelation_floor = self._bound(
            activations[:, 0], activations[:, 1]
        )

        decorrelated_activations = self._decorrelate(
            beat_activation, downbeat_activation, decorrelation_floor)
        decorrelated_activations, first_frame = threshold_crop(decorrelated_activations, threshold)

        if not len(decorrelated_activations):
            return self._empty_events()

        activations = activations[first_frame:first_frame + len(decorrelated_activations)]
        log_class_densities = self._log_class_densities(activations[:, 0], activations[:, 1])

        # One Viterbi per meter; keep the best-scoring path (== decoding madmom's block-diagonal
        # multi-pattern union, see __init__). Scores are comparable because every meter shares the
        # union-uniform initial constant.
        best = None
        for space, dp, log_init, state_to_class in zip(
            self.state_spaces, self.dynamic_programs,
            self.log_initial_distributions, self.state_to_classes
        ):
            state_path, log_score = dp.viterbi(
                log_init, self.log_tempo_transition, log_class_densities,
                state_to_class=state_to_class, return_log_score=True
            )

            if best is None or log_score > best[0]:
                best = (log_score, state_path.cpu().numpy(), space)

        assert best is not None      # beats_per_bar is validated non-empty in __init__
        _, state_path, state_space = best

        return state_path_to_events(state_path, state_space, self.fps,
                                    snap_to_activations=decorrelated_activations if correct else None,
                                    first_frame=first_frame)


if __name__ == "__main__":
    # Synthetic 120 BPM 4/4, 10 s. NOTE: this test passed even with the BROKEN uniform grid, because
    # at fps=50/120 BPM the fractional advance happened to be 0.96 (~1). Real data at 86 fps is what
    # caught it. Kept as a smoke test only -- the real gate is R1 vs R0.
    fps = 50.0
    num_frames = int(round(10 * fps))
    beat_period_frames = fps * 60.0 / 120.0
    activations = np.zeros((num_frames, 2))
    activations[:, 0], activations[:, 1] = 0.05, 0.02
    beat_frames = np.round(np.arange(0.0, num_frames, beat_period_frames)).astype(int)
    beat_frames = beat_frames[beat_frames < num_frames]
    activations[beat_frames, 0] = 0.9
    activations[beat_frames[::4], 1] = 0.92

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DBN2016(fps=fps, device=device)
    for space in model.state_spaces:
        print(space)
    events = model.predict(activations)
    print(f"synthetic 120 BPM 4/4, 10 s (fps={fps:.1f}), device={device}:")
    print(f"  beats {len(events['beats'])} (expect ~20) | "
          f"downbeats {len(events['downbeats'])} (expect ~5)")
    if len(events["beats"]) > 1:
        mean_inter_beat_interval = float(np.diff(events["beats"]).mean())
        print(f"  mean IBI {mean_inter_beat_interval:.3f}s -> "
              f"{60.0 / mean_inter_beat_interval:.1f} BPM (expect ~120)")
