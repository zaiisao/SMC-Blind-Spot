"""The rung contract: every rung on the ladder is a Rung.

A rung is a deployable beat/downbeat tracker with ONE public entry point:

    predict(features, **predict_options) -> {'beats': seconds[N], 'downbeats': seconds[M]}

so evaluate.py can score every rung identically and rung comparisons are always apples-to-apples.
The base class owns what is common to ALL rungs -- the deployment interface, input coercion and
validation, the Böck 2016 decorrelation, and the empty-result shape -- and leaves the model itself
to the subclass:

    coercion      accept numpy or (possibly CUDA) torch [num_frames, 2] (beat, downbeat), hand the
                  subclass float64 numpy. Validated here so no rung ever sees a malformed input.
    decorrelation max(beat - downbeat, floor) -- the standard convention (Böck, Krebs & Widmer,
                  ISMIR 2016) that converts a modern two-sigmoid frontend (beat fires on downbeats
                  too, so beat + downbeat can reach ~2) into the mutually-exclusive
                  [P(beat-not-downbeat), P(downbeat)] pair the bar-pointer observation model
                  assumes (it computes the no-beat mass as 1 - beat - downbeat). Shared here
                  because EVERY rung that emits through that observation model needs the identical
                  transform -- R0 and R1 using different decorrelations would void the
                  certificate.

The input contract is uniform: activation-consuming rungs receive PROBABILITIES -- whoever calls
predict() converts logits first (the Tracker does it automatically in a pairing; sigmoid is
sigmoid, so where it happens is not a modeling choice). The one genuine frontend-fidelity knob,
`bounding` (clip/squeeze/none -- measured near-equivalent in F but NOT always event-neutral), is a
base constructor parameter wired from the frontend by the Tracker for every activations-mode
pairing. Rung-specific knobs stay in the subclasses (e.g. R1's threshold/correct deployment
options).
"""
from abc import ABC, abstractmethod

import numpy as np


class Rung(ABC):
    """Base class for every rung. Subclasses implement _predict_features; predict() is final."""

    # How many channels this rung's predict() consumes. The HMM family reads exactly [T, 2]
    # (beat, downbeat) activations -- the frontend's FINAL layer. A latent-variable rung that
    # conditions on rich penultimate features instead declares its own count (e.g. 512) or None
    # for "any". The Tracker checks this against the frontend's output mode at construction, so a
    # frontend cut at the wrong depth fails loudly instead of predicting on garbage.
    INPUT_CHANNELS = 2      # int, or None for "any"

    # A learned rung declares TRAIN_MODE and implements that mode's hooks:
    #     "em"        em_step(crops) -> statistic
    #     "gradient"  trainable_parameters(), training_step(observation, path, meter_index)
    # Both also provide decode(observation, deploy=...), the coercion-free evaluation path the
    # training harness calls instead of predict() (see fit).
    TRAIN_MODE = None        # "em" | "gradient" | None (not trainable)
    ARM_NAME = None          # config `rung.name` and checkpoint prefix
    TRAIN_DEFAULTS = {}      # this rung's idiomatic frozen/EM hyperparameters
    E2E_DEFAULTS = None      # overrides when the frontend trains too; None -> reuse TRAIN_DEFAULTS

    @classmethod
    def observes_features(cls) -> bool:
        """True for a rung conditioned on rich features rather than [num_frames, 2] activations."""
        return cls.INPUT_CHANNELS != 2

    @classmethod
    def from_config(cls, config):
        """Build this rung from the config's `rung:` section."""
        from training import data, harness
        return cls(fps=data.FPS, device=harness.device_of(config),
                   **harness.rung_kwargs(config))

    @classmethod
    def fit(cls, config):
        """Train this rung from a config (see configs/train.yaml).

        Named `fit`, not `train`: nn.Module.train(mode: bool) already exists, and an nn.Module rung
        (R3, CRF) would shadow one of the two.
        """
        if cls.TRAIN_MODE is None:
            raise TypeError(
                f"{cls.__name__} is not trainable: its factors are hand-set (R1) or opaque (R0). "
                f"A learned rung declares TRAIN_MODE ('em' or 'gradient').")
        from training import harness
        return harness.dispatch(cls, config)

    def __init__(self, fps: float, bounding: str = "clip", eps: float = 1e-5):
        """Shared construction, uniform across ALL rungs -- no per-class wiring declarations.

        fps: the frame rate of the incoming features (a property of the activations, never a
        constant). bounding/eps: how an activation-consuming rung nudges probabilities off exact
        0/1 before taking logs -- in a Tracker pairing, bounding is wired from the frontend's
        PUBLISHED convention ("clip"/"squeeze"/"none", see the _bound recipes). Feature-consuming
        rungs (INPUT_CHANNELS != 2) inherit the parameters and simply never use them.
        """
        if bounding not in ("clip", "squeeze", "none"):
            raise ValueError(f"bounding must be 'clip', 'squeeze' or 'none', got {bounding!r}")

        self.fps = fps
        self.bounding = bounding
        self.eps = eps

    def predict(self, features, **predict_options) -> dict:
        """features: [num_frames, INPUT_CHANNELS]. The common deployment interface.

        For the HMM family (INPUT_CHANNELS=2) that is (beat, downbeat) activations; a
        latent-variable rung consumes whatever width it declared. predict_options are forwarded to
        the subclass (e.g. R1's threshold/correct overrides); rungs without per-call options
        accept none.
        """
        features = self._coerce_features(features)
        return self._predict_features(features, **predict_options)

    @abstractmethod
    def _predict_features(self, features: np.ndarray, **predict_options) -> dict:
        """features is a validated float64 numpy [num_frames, INPUT_CHANNELS]. Return the events
        dict."""

    @classmethod
    def _coerce_features(cls, features) -> np.ndarray:
        """Accept a numpy array or a (possibly CUDA) torch tensor; validate the shape against the
        class's declared INPUT_CHANNELS (None = any channel count)."""
        if hasattr(features, "detach"):
            features = features.detach().cpu().numpy()
        features = np.ascontiguousarray(np.asarray(features, dtype=np.float64))
        if features.ndim != 2 or (cls.INPUT_CHANNELS is not None
                                  and features.shape[1] != cls.INPUT_CHANNELS):
            raise ValueError(f"expected [num_frames, {cls.INPUT_CHANNELS or 'any'}] features, "
                             f"got {features.shape}")
        return features

    def _bound(self, beat_activation, downbeat_activation):
        """Bound activations off exact 0/1 and return the decorrelation floor to pair with them.

        An activations-only helper (INPUT_CHANNELS=2 rungs). Reads self.bounding -- each published
        system's convention ("clip", "squeeze" or "none", see rungs/r0 for the per-system table)
        -- and self.eps, both declared and validated in the base constructor (the Tracker wires
        bounding from the frontend).
        """
        bounding, eps = self.bounding, self.eps

        if bounding == "clip":
            beat_activation = np.clip(beat_activation, eps, 1 - eps)
            downbeat_activation = np.clip(downbeat_activation, eps, 1 - eps)
            decorrelation_floor = eps
        elif bounding == "squeeze":
            beat_activation = beat_activation * (1 - eps) + eps / 2
            downbeat_activation = downbeat_activation * (1 - eps) + eps / 2
            decorrelation_floor = eps / 2
        else:
            # none (Beat Transformer); tiny floor for safety
            decorrelation_floor = 1e-12

        return beat_activation, downbeat_activation, decorrelation_floor

    def _bound_tensor(self, beat_activation, downbeat_activation):
        """Torch mirror of _bound, kept adjacent to it so the two recipes cannot drift.

        The training path must bound exactly as deployment does: a bounding="none" frontend
        (Beat Transformer) trained through clip-bounded densities disagrees with its own decode by
        up to 16 nats.
        """
        bounding, eps = self.bounding, self.eps

        if bounding == "clip":
            beat_activation = beat_activation.clamp(eps, 1 - eps)
            downbeat_activation = downbeat_activation.clamp(eps, 1 - eps)
            decorrelation_floor = eps
        elif bounding == "squeeze":
            beat_activation = beat_activation * (1 - eps) + eps / 2
            downbeat_activation = downbeat_activation * (1 - eps) + eps / 2
            decorrelation_floor = eps / 2
        else:
            decorrelation_floor = 1e-12

        return beat_activation, downbeat_activation, decorrelation_floor

    @staticmethod
    def _decorrelate(beat_activation: np.ndarray, downbeat_activation: np.ndarray,
                     floor: float) -> np.ndarray:
        """Böck 2016 decorrelation -> [num_frames, 2] (beat-not-downbeat, downbeat).

        Assumes the channels are already bounded off exact 0/1 (each rung bounds per its recipe);
        floor keeps the subtraction positive so the observation model's logs stay finite.
        """
        beat_not_downbeat_activation = np.maximum(beat_activation - downbeat_activation, floor)
        return np.stack([beat_not_downbeat_activation, downbeat_activation], axis=-1)

    @staticmethod
    def _empty_events() -> dict:
        return {"beats": np.array([]), "downbeats": np.array([])}
