"""Beat annotations -> SA3 training metadata, and the label-preserving time warp.

The whole point of editing rather than generating (HANDOVER §4) is that labels
transform *exactly* through a known warp, so nothing here estimates beats: it
reads annotations and maps them through closed-form time maps.

These are the pure pieces; ``sa3_beat.metadata`` adapts them to the library's
``custom_metadata_module`` contract.
"""

from pathlib import Path

import numpy as np

from .conditioners import LATENT_RATE


def load_beats(path):
    """Read a beat annotation, returning ``(beats, downbeats)``.

    Two formats are supported, distinguished by inspecting the file rather than
    by extension:

    ``<time>[\\t<beat_number>]``
        The plain form.  Downbeats are the times whose beat number is 1.  SMC is
        the degenerate case -- all 217 files are bare times with no beat numbers
        -- and yields ``downbeats=None``.

    ``<time>\\t<time>\\t<label>``
        ASAP, where the label is ``b`` or ``db`` (downbeats optionally carrying a
        time signature, e.g. ``db,12/8,6``).

    Distinguishing them matters more than it looks.  Read as the plain form, an
    ASAP line's second column is a *timestamp*, so ``int(float(parts[1])) == 1``
    silently marks every beat between 1.0 s and 2.0 s as a downbeat -- garbage
    that usually collapses to ``None`` further downstream and occasionally does
    not.  ``downbeats=None`` is always preferred over a guess.
    """
    lines = [ln.split() for ln in Path(path).read_text().split("\n")]
    lines = [p for p in lines if p]
    if not lines:
        return np.array([]), None

    asap = any(len(p) > 2 and p[2].split(",")[0] in ("b", "db") for p in lines)

    beats, downbeats = [], []
    numbered = False
    for parts in lines:
        t = float(parts[0])
        beats.append(t)
        if asap:
            if len(parts) > 2:
                numbered = True
                if parts[2].split(",")[0] == "db":
                    downbeats.append(t)
        elif len(parts) > 1:
            numbered = True
            if int(float(parts[1])) == 1:
                downbeats.append(t)
    return np.array(beats), (np.array(downbeats) if numbered and downbeats else None)


def warp_times(times, rate):
    """Constant-rate time warp. ``rate>1`` speeds up, so times compress."""
    return np.asarray(times, dtype=np.float64) / rate


def warp_times_curve(times, breakpoints, rates):
    """Piecewise-constant-rate warp, for rubato rather than a global stretch.

    ``breakpoints`` are source-timebase edges (ascending, len == len(rates)+1);
    each segment is stretched by its own rate.  The output stays monotonic and
    continuous, so beat order and label validity are preserved by construction.
    """
    breakpoints = np.asarray(breakpoints, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    if breakpoints.size != rates.size + 1:
        raise ValueError("breakpoints must have one more element than rates")
    if np.any(np.diff(breakpoints) <= 0):
        raise ValueError("breakpoints must be strictly ascending")

    seg_src = np.diff(breakpoints)
    edges_dst = np.concatenate([[0.0], np.cumsum(seg_src / rates)])
    times = np.asarray(times, dtype=np.float64)
    # np.interp on a monotonic map is exactly the piecewise-linear time map.
    return np.interp(times, breakpoints, edges_dst)


def beat_metadata(beats, downbeats, num_frames, frame_offset=0, prompt=None):
    """The dict BeatConditioner consumes, plus the text prompt SA3 needs."""
    md = {
        "beat": {
            "beats": np.asarray(beats, dtype=np.float64),
            "downbeats": None if downbeats is None else np.asarray(downbeats, dtype=np.float64),
            "num_frames": int(num_frames),
            "frame_offset": int(frame_offset),
        }
    }
    if prompt is not None:
        md["prompt"] = prompt
    return md
