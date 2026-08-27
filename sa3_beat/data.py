
from pathlib import Path

import numpy as np

from .conditioners import LATENT_RATE


def load_beats(path):
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
    return np.asarray(times, dtype=np.float64) / rate


def warp_times_curve(times, breakpoints, rates):
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
