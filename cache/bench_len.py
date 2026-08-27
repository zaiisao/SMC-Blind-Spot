"""Is stretching a long track less efficient per audio-second than a short one?

Throughput fell ~2x in audio-seconds/min just as the corpus crossed from short
clips (median 30 s) into long ones (median 167 s). If pedalboard scaled linearly
with duration that should not happen -- so measure it directly. A -60% stretch
turns a 240 s track into a 600 s output, ~26 M float samples, which is far past
any cache and may be memory-bandwidth bound rather than compute bound.

Reports audio-seconds of OUTPUT produced per wall-second, which is the unit that
should be flat if duration does not matter.
"""
import glob, os, sys, time
import numpy as np
import soundfile as sf
from pedalboard import time_stretch

SRC = "/disk3/jaehoon/bt_wide/audio_src"
buckets = [(20, 40), (60, 90), (150, 200), (300, 420)]
picks = []
files = sorted(glob.glob(os.path.join(SRC, "*.wav")))
for lo, hi in buckets:
    for f in files:
        try:
            d = sf.info(f).duration
        except Exception:
            continue
        if lo <= d <= hi:
            picks.append((d, f))
            break

print(f"{'input':>8}{'output':>9}{'wall':>8}{'out-sec/wall-sec':>18}")
for d, f in picks:
    y, sr = sf.read(f, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    t0 = time.time()
    o = time_stretch(y[None, :], sr, 1.0 / 2.5)[0]   # -60%, the worst case
    w = time.time() - t0
    outsec = o.shape[-1] / sr
    print(f"{d:7.0f}s{outsec:8.0f}s{w:7.2f}s{outsec/w:18.1f}")
