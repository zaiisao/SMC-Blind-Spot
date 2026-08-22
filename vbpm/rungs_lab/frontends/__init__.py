"""Feature extractors (frontends): audio -> [num_frames, num_channels] activations/features.

One script per frontend (beat_this.py, later mert.py, ...). A frontend wraps the official upstream
model behind a small property surface -- WHAT it emits (`fps`, `output`, `ACTIVATION_FORM`) and HOW
to get it (`get_features(signal, sample_rate) -> [num_frames, num_channels]` -- frontends are
feature extractors, and the [T, 2] activations are just the most compressed feature). Selecting a
frontend by name (build_frontend) and pairing it with a bar-pointer model is tracker.py's job, one
level up -- this package only defines the interface and its implementations.

Output modes: a frontend can usually emit at more than one depth of its network. The classic cut is
the FINAL layer -- [T, 2] (beat, downbeat) activations, what the HMM-family rungs consume -- vs the
PENULTIMATE layer -- rich features (e.g. [T, 512]), what a latent-variable rung conditions on
(deleting the final linear compression). Each frontend class declares its modes in OUTPUT_MODES
(mode name -> num_channels) and is constructed in exactly one mode; the Tracker checks the emitted
channel count against the rung's declared INPUT_CHANNELS, and the config layer additionally demands
the frontend's `output` and the bar-pointer's `input` be declared together (see track.py).

Deliberately simple (a resurrected, slimmed version of the archived
data/feature_extractor.py + configs/frontends/*.yaml system): properties live on the wrapper class,
not in YAML, until we have enough frontends to need config files again.
"""


class Frontend:
    """Interface. A frontend turns audio into [num_frames, num_channels] in its output mode."""

    OUTPUT_MODES: dict = {"activations": 2}
    ACTIVATION_FORM: str = "probability"
    BOUNDING: str = "clip"

    output: str = "activations"
    fps: float

    @property
    def name(self) -> str:
        """Derived from the defining module (a frontend's identity IS its module under the
        dotted-path loader): frontends.beat_this -> "beat_this". Never declared per class.
        When the module runs as a script (__main__), fall back to its file stem."""
        module_name = type(self).__module__.rsplit(".", 1)[-1]

        if module_name == "__main__":
            import inspect
            from pathlib import Path
            try:
                return Path(inspect.getfile(type(self))).stem
            except TypeError:        # class defined interactively; nothing better to derive
                pass

        return module_name

    @property
    def num_channels(self) -> int:
        return self.OUTPUT_MODES[self.output]

    def get_features(self, signal, sample_rate: int):
        """[num_samples] mono audio -> [num_frames, num_channels] in the instance's output mode."""
        raise NotImplementedError
