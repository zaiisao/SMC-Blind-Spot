"""Corrupted beat grids for contrastive ControlNet training.

Listening (2026-08-24) established the generator's failure: the audio has a clear
pulse, but it is not the conditioned one -- one clip was audibly on the offbeat.
The released tracker scores our corpus 0.45-0.49 against its own labels while
scoring time-warped REAL audio 0.84, so the defect is that conditioning is being
ignored, not that the audio is unbeatable.

Why the model learned to ignore it: training pairs are (real audio, that audio's
true grid). The audio already contains the beats, so the conditioning is redundant
and the noise-prediction loss can be minimised without using it. At generation time
there is no audio to copy and the conditioning is the only timing source.

The fix is to make conditioning non-redundant: the model must fit the audio BETTER
under the true grid than under a corrupted one. Corruptions cover every misalignment
class observed, not just the metrical ones -- on 216 SMC tracks 15.7% of tracker
errors matched no clean ratio at all -- but weight is concentrated on the
half/double/offbeat confusions that dominate.
"""
import numpy as np

WEIGHTS = {
    "half": 0.17, "double": 0.17, "offbeat": 0.16,   # 50% -- the audible failures
    "ratio": 0.15,                                    # 1.5x, 2/3x, 3x, 1/3x
    "jitter": 0.15,                                   # right grid, sloppy placement
    "drift": 0.10,                                    # starts right, slides
    "dropout": 0.05,                                  # missing / extra beats
    "other": 0.05,                                    # another clip's grid entirely
}
_KINDS = list(WEIGHTS); _P = np.array([WEIGHTS[k] for k in _KINDS]); _P = _P / _P.sum()


def corrupt(beats, seconds, rng, other=None):
    """Return (corrupted_beats, kind). Never returns the input unchanged."""
    b = np.asarray(beats, dtype=np.float64)
    if len(b) < 4:
        return b, "none"
    ibi = float(np.median(np.diff(b)))
    kind = _KINDS[rng.choice(len(_KINDS), p=_P)]

    if kind == "half":
        out = b[::2]
    elif kind == "double":
        mid = b[:-1] + np.diff(b) / 2.0
        out = np.sort(np.concatenate([b, mid]))
    elif kind == "offbeat":
        out = b + ibi / 2.0
    elif kind == "ratio":
        r = rng.choice([1.5, 2 / 3, 3.0, 1 / 3])
        out = np.arange(b[0], b[-1] + 1e-9, ibi / r)
    elif kind == "jitter":
        out = np.sort(b + rng.normal(0, 0.10 * ibi, size=len(b)))
    elif kind == "drift":
        # tempo ramps by +/-15% across the clip: correct at the start, wrong by the end
        f = 1.0 + rng.uniform(-0.15, 0.15) * np.linspace(0, 1, len(b))
        out = b[0] + np.cumsum(np.concatenate([[0.0], np.diff(b) * f[1:]]))
    elif kind == "dropout":
        if rng.random() < 0.5:                      # drop
            keep = rng.random(len(b)) > 0.25
            out = b[keep] if keep.sum() >= 4 else b[::2]
        else:                                       # insert spurious beats
            k = max(1, len(b) // 5)
            extra = rng.uniform(b[0], b[-1], k)
            out = np.sort(np.concatenate([b, extra]))
    else:                                           # another clip's grid
        out = np.asarray(other, dtype=np.float64) if other is not None and len(other) >= 4 \
              else b + ibi / 2.0

    out = out[(out >= 0) & (out < seconds)]
    if len(out) < 4:
        out = b + ibi / 2.0
        kind = "offbeat"
    return out, kind
