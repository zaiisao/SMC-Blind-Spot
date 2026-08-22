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
    """Read a ``.beats`` annotation: ``<time>\\t<beat_number>`` per line.

    Returns ``(beats, downbeats)``; downbeats are the times whose beat number is
    1.  Files with no beat numbers yield ``downbeats=None`` rather than a guess.
    """
    beats, downbeats = [], []
    numbered = False
    for line in Path(path).read_text().split("\n"):
        parts = line.split()
        if not parts:
            continue
        t = float(parts[0])
        beats.append(t)
        if len(parts) > 1:
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
