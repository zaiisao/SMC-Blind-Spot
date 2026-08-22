"""R0 -- Baseline A: the madmom bar-pointer DBN (Krebs/Böck 2015).

Self-contained: it calls the official madmom DBNDownBeatTrackingProcessor directly and formats the
input with the standard decorrelation convention (Böck et al.):

    decorrelated = [ max(beat - downbeat, floor),  downbeat ]

Why decorrelate: madmom's downbeat observation model computes the "no-beat" log-density as roughly
    log( (1 - (beat + downbeat)) / normalizer ),
so it assumes beat + downbeat <= 1. A neural frontend whose beat channel fires on every beat
(downbeats included) gives beat + downbeat ~ 2 at each downbeat -> log(negative), which madmom would
silently clamp. Decorrelating turns the two columns into a valid [P(beat-not-downbeat), P(downbeat)]
pair that sums to <= 1, so the observation model is well-formed and nothing is clamped. (Measured on
our val cache: decorrelating is worth ~+0.06 F vs feeding raw activations -- not optional; the raw
form's log(negative) nan corrupts the whole Viterbi path.) R0 is the literal, correct DBN baseline --
the certificate R1 (our own PyTorch bar-pointer DBN) must reproduce.

This follows the Beat This / Beat Transformer paradigm -- the joint bar-pointer DBN + Böck
decorrelation that both SOTA systems independently converged on. WaveBeat's separate-tracker approach
is a different model class, considered and NOT adopted (unproven on our data).

Input contract (uniform across all rungs, see rungs/base.py): predict() receives PROBABILITY
activations. A logit-emitting frontend is sigmoided by the caller (the Tracker in a pairing; by
hand standalone) -- sigmoid is sigmoid, so where it happens changes nothing. The one knob that
does replicate each published system's recipe is `bounding` (a base Rung parameter, wired from the
frontend by the Tracker); per our ablation the conventions are near-equivalent in F but NOT always
event-identical, so fidelity requires the published one:

    bounding   : "clip"    np.clip(x, eps, 1-eps)          (ours, WaveBeat)                [DEFAULT]
                 "squeeze" x*(1-eps) + eps/2               (Beat This)
                 "none"    no bounding (Beat Transformer). Unsafe on probs with exact 0/1
    eps        : bounds values off 0/1 and floors the decorrelation; 1e-5 (Beat This/ours) is a hair
                 better than 1e-8 (WaveBeat), but its exact value barely matters.

We deliberately leave madmom's OWN defaults untouched, because Beat This and Beat Transformer also
call the processor with them -- so R0-as-published includes all three. They are deployment heuristics
wrapped around the DBN, and R1 implements none of them:
    num_tempi=60    a coarser log-spaced tempo grid (vs all 71 integer intervals here)
    threshold=0.05  crop to the main above-threshold segment before decoding
    correct=True    report the activation peak inside a beat region, not the region entry
Measured on our val set they are worth ~+0.006 beat F over the bare model, almost all of it
threshold=0.05 cropping fade-in/fade-out -- which only the Ballroom subset has (74/85 songs; every
other dataset ~0%). correct=True actually costs us -0.0014. So the R1-vs-R0 gap is these heuristics,
not our inference: against a MATCHED madmom, R1 produces an identical Viterbi path.

fps: REQUIRED and frontend-specific -- a property of the activations, never a global constant, and a
wrong fps mis-scales every DBN tempo. Pass record['fps'] from the cache. Native rates differ by model
(Beat This 50, MERT ~75, WaveBeat 172.27); our cache currently interpolates every frontend up to 86.13
(= 22050/256), so record['fps'] is 86.13 today -- but reading it keeps R0 correct the moment a frontend
is cached at its own native rate.
"""
import numpy as np
from madmom.features.downbeats import DBNDownBeatTrackingProcessor

from rungs.base import Rung

DOWNBEAT_POSITION_IN_BAR = 1        # madmom counts bar positions naturally: 1 = downbeat


class MadmomDBN(Rung):
    """R0: the official madmom joint bar-pointer DBN, fed the standard decorrelated activation.

    Follows the Beat This / Beat Transformer paradigm; `bounding` matches each system's exact
    recipe (see module docstring). beats_per_bar=[3,4], min_bpm=55, max_bpm=215,
    transition_lambda=100 are the settings both SOTA trackers use -- a faithful baseline, not re-tuned.
    """

    def __init__(
        self,
        fps: float,
        bounding: str = "clip",
        eps: float = 1e-5,
        min_bpm: float = 55.0,
        max_bpm: float = 215.0,
        beats_per_bar=(3, 4),
        transition_lambda: int = 100,
        observation_lambda: int = 16,
        num_tempi: int = 60,
        threshold: float = 0.05,
        correct: bool = True
    ):
        # num_tempi/threshold/correct default to madmom's shipped values (what Beat This / Beat
        # Transformer run) -- exposed explicitly because they are deployment heuristics, NOT the model:
        # measured ~+0.006..+0.024 beat F over the bare model depending on the sample. Set
        # num_tempi=None, threshold=0.0, correct=False to get the bare model, which R1 reproduces
        # path- and score-identically.
        super().__init__(fps=fps, bounding=bounding, eps=eps)
        self._madmom_dbn = DBNDownBeatTrackingProcessor(
            beats_per_bar=list(beats_per_bar),
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            fps=fps,
            transition_lambda=transition_lambda,
            observation_lambda=observation_lambda,
            num_tempi=num_tempi,
            threshold=threshold,
            correct=correct
        )

    def _predict_features(self, activations: np.ndarray) -> dict:
        # For this rung features ARE [T, 2] activations (INPUT_CHANNELS=2); named accordingly.
        """activations: [num_frames, 2] (beat, downbeat) PROBABILITIES (the uniform rung contract).

        R0 fulfils the rung contract via madmom's joint bar-pointer DBN.
        """
        beat_activation, downbeat_activation = activations[:, 0], activations[:, 1]

        # Decorrelate the two channels to satisfy madmom's observation model, which assumes
        # beat + downbeat <= 1.0. Each feature extractor satisfies this constraint differently; thus,
        # self._bound returns a bounded beat and downbeat activation pair based on the given
        # bounding convention (the base Rung parameter, frontend-wired by the Tracker) and eps.
        beat_activation, downbeat_activation, decorrelation_floor = self._bound(
            beat_activation, downbeat_activation
        )

        decorrelated_activations = self._decorrelate(
            beat_activation, downbeat_activation, decorrelation_floor)

        # madmom returns [num_beats, 2]: (time in seconds, position in bar)
        beats_with_bar_positions = np.asarray(self._madmom_dbn(decorrelated_activations))
        if not beats_with_bar_positions.size:
            return self._empty_events()

        is_downbeat = beats_with_bar_positions[:, 1] == 1

        beat_times = beats_with_bar_positions[:, 0]
        downbeat_times = beats_with_bar_positions[is_downbeat, 0]

        return {"beats": np.asarray(beat_times), "downbeats": np.asarray(downbeat_times)}


if __name__ == "__main__":
    import warnings

    # Smoke test: synthetic 120 BPM, 4/4, 10 s of PROBABILITY activations (beat channel fires on every
    # beat, downbeats included). Proves the wrapper runs end to end.
    fps = 50.0   # Beat This's native rate; the synthetic below mimics its channel form
    num_frames = int(round(10 * fps))
    beat_period_frames = fps * 60.0 / 120.0
    activations = np.zeros((num_frames, 2))
    activations[:, 0], activations[:, 1] = 0.05, 0.02    # distinct baselines (avoid equal channels)
    beat_frames = np.round(np.arange(0.0, num_frames, beat_period_frames)).astype(int)
    beat_frames = beat_frames[beat_frames < num_frames]
    activations[beat_frames, 0] = 0.9
    activations[beat_frames[::4], 1] = 0.92

    events = MadmomDBN(fps=fps).predict(activations)
    print(f"synthetic 120 BPM 4/4, 10 s (fps={fps:.1f}), probability activations (defaults):")
    print(f"  detected beats:     {len(events['beats']):2d}   (expect ~20)")
    print(f"  detected downbeats: {len(events['downbeats']):2d}   (expect ~5)")
    if len(events["beats"]) > 1:
        mean_inter_beat_interval = float(np.diff(events["beats"]).mean())
        print(f"  mean inter-beat interval: {mean_inter_beat_interval:.3f}s  ->  "
              f"{60.0 / mean_inter_beat_interval:.1f} BPM  (expect ~120)")

    # Verify the bounding conventions all run and land on ~the same answer (near-equivalent in F,
    # NOT always event-neutral -- which is why fidelity wires the published one). Rungs receive
    # PROBABILITIES by contract: a logit source is sigmoided by the caller (the Tracker in a
    # pairing; by hand here), so the logit row must reproduce the probability row exactly.
    # 'none' on probs with exact 0/1 can hit log(0) -> harmless -inf.
    bounded = np.clip(activations, 1e-6, 1 - 1e-6)
    logit_activations = np.log(bounded / (1 - bounded))
    sigmoided = 1.0 / (1.0 + np.exp(-logit_activations))
    print("\nbounding matrix (beats / downbeats -- should be ~equal across cells):")
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for source, source_activations in (("probability", activations),
                                           ("sigmoid(logit)", sigmoided)):
            for bounding in ("clip", "squeeze", "none"):
                events = MadmomDBN(fps=fps, bounding=bounding).predict(source_activations)
                print(f"  source={source:14s} bounding={bounding:7s} -> "
                      f"{len(events['beats']):2d} / {len(events['downbeats']):2d}")
