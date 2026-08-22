"""Krebs 2015 efficient bar-pointer state space: a tempo of i frames-per-beat gets EXACTLY i states.

The key property: because a tempo's state count equals its interval, the pointer advances exactly
ONE state per frame. No interpolation, no self-loop bias.

That is not an efficiency trick -- it is what makes the model correct. A uniform (position x tempo)
grid, which is what the earlier bar-pointer models used, makes the advance fractional: at our 86 fps a
uniform 96-bin grid advances 0.26-0.96 bins/frame, so the two-bin split turns a deterministic advance
into a stay-vs-advance coin flip, and below ~120 BPM Viterbi simply prefers to STAY. The pointer
stalls. We measured that: -0.14 beat F vs madmom. Replacing the uniform grid is the entire
contribution of the 2015 paper.

Layout (mirrors madmom's BeatStateSpace / BarStateSpace):
  * a tempo is stored as an INTERVAL: how many frames one beat lasts. Slow tempo = long interval.
  * one beat  : sum(interval_frames) states, laid out as contiguous per-tempo BLOCKS
  * one bar   : beats_per_bar stacked copies -> num_states = beats_per_bar * sum(interval_frames)
  * positions : bar position in BEAT units, in [0, beats_per_bar)

References
----------
Krebs, Böck & Widmer, "An Efficient State Space Model for Joint Tempo and Meter Tracking",
ISMIR 2015 (state space + transition).
Böck, Krebs & Widmer, "Joint Beat and Downbeat Tracking with Recurrent Neural Networks",
ISMIR 2016 (the {no-beat, beat, downbeat} observation classes).
"""
from typing import Optional

import numpy as np

# The three Böck 2016 observation classes. Every state emits via exactly one of them, so these index
# the columns of the [num_frames, 3] density array -- see rungs/r1_2016_dbn.py.
NO_BEAT = 0
BEAT = 1
DOWNBEAT = 2


class BarPointerStateSpace:
    def __init__(self, fps: float, min_bpm: float = 55.0, max_bpm: float = 215.0,
                 beats_per_bar: int = 4, observation_lambda: int = 16,
                 num_tempi: Optional[int] = None):
        """num_tempi: None (default) models EVERY integer interval -- the exact model, and what the
        R1 certificate runs. An integer replicates madmom's shipped grid: log-spaced, quantized to
        unique integers (madmom's default is 60). Worth +0.002..+0.005 F in deployment; a model-
        design lesson we keep available rather than throw away, but off for rung comparisons.
        """
        self.fps = fps
        self.beats_per_bar = beats_per_bar
        self.observation_lambda = observation_lambda

        # A fast tempo (max_bpm) means FEW frames per beat, hence the crossed pairing.
        shortest_interval_frames = 60.0 * fps / max_bpm
        longest_interval_frames = 60.0 * fps / min_bpm
        interval_frames = np.arange(round(shortest_interval_frames),
                                    round(longest_interval_frames) + 1)
        if num_tempi is not None and num_tempi < len(interval_frames):
            # madmom's BeatStateSpace quantization: widen the log grid until, after rounding to
            # unique integers, exactly num_tempi intervals survive. Copied behavior, not re-derived.
            num_log_tempi = num_tempi
            interval_frames = []
            while len(interval_frames) < num_tempi:
                interval_frames = np.logspace(np.log2(shortest_interval_frames),
                                              np.log2(longest_interval_frames),
                                              num_log_tempi, base=2)
                interval_frames = np.unique(np.round(interval_frames))
                num_log_tempi += 1
        self.interval_frames = np.ascontiguousarray(interval_frames, dtype=np.int64)
        self.num_tempi = len(self.interval_frames)

        # --- one beat: contiguous per-tempo blocks, a tempo of i frames owning i states -----------
        num_states_per_beat = int(self.interval_frames.sum())
        first_state_per_tempo = np.cumsum(np.r_[0, self.interval_frames[:-1]]).astype(np.int64)
        last_state_per_tempo = (np.cumsum(self.interval_frames) - 1).astype(np.int64)
        position_within_beat = np.concatenate(
            [np.linspace(0, 1, interval, endpoint=False) for interval in self.interval_frames])
        interval_of_each_state = np.concatenate(
            [np.full(interval, interval, dtype=np.int64) for interval in self.interval_frames])

        # --- one bar: beats_per_bar stacked copies of the beat space -----------------------------
        # first_states/last_states are [beats_per_bar, num_tempi]: entry [b, v] is where tempo v's
        # block starts/ends inside beat b. The tempo transition connects last -> first across beats.
        self.num_states = beats_per_bar * num_states_per_beat
        self.first_states = np.stack([first_state_per_tempo + beat * num_states_per_beat
                                      for beat in range(beats_per_bar)])
        self.last_states = np.stack([last_state_per_tempo + beat * num_states_per_beat
                                     for beat in range(beats_per_bar)])
        self.state_positions = np.concatenate([position_within_beat + beat
                                               for beat in range(beats_per_bar)])
        self.state_interval_frames = np.tile(interval_of_each_state, beats_per_bar)

        # --- Böck 2016 observation classes over bar position (beat units) -----------------------
        # A state counts as "on the beat" if it sits in the first 1/observation_lambda of a beat.
        beat_region_width = 1.0 / observation_lambda
        position_classes = np.full(self.num_states, NO_BEAT, dtype=np.int64)
        position_classes[(self.state_positions % 1.0) < beat_region_width] = BEAT
        position_classes[self.state_positions < beat_region_width] = DOWNBEAT   # overwrites BEAT at position ~0
        self.position_classes = position_classes

    def tempo_transition_probabilities(self, transition_lambda: float,
                                       threshold: float = np.spacing(1)) -> np.ndarray:
        """madmom's exponential_transition, as [num_tempi, num_tempi] rows=from, cols=to.

        p(interval_to | interval_from) ~ exp(-transition_lambda * |ratio - 1|), row-normalized, so a
        higher transition_lambda prefers holding the tempo from one beat to the next.
        """
        interval_ratio = (self.interval_frames.astype(float)
                          / self.interval_frames.astype(float)[:, np.newaxis])
        probabilities = np.exp(-transition_lambda * np.abs(interval_ratio - 1.0))
        probabilities[probabilities <= threshold] = 0.0
        probabilities /= probabilities.sum(axis=1)[:, np.newaxis]
        return probabilities

    def __repr__(self):
        return (f"BarPointerStateSpace(Krebs2015: num_states={self.num_states} = {self.beats_per_bar} "
                f"beats x sum(interval_frames {self.interval_frames[0]}..{self.interval_frames[-1]}), "
                f"num_tempi={self.num_tempi}, "
                f"{60 * self.fps / self.interval_frames[-1]:.0f}-"
                f"{60 * self.fps / self.interval_frames[0]:.0f} BPM)")
