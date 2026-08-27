"""Single-process, single-track render benchmark.

Local shows 146 wav/min on a 32-physical-core Threadripper while titan shows 484
on a 16-physical-core i9. That inversion has to come from either per-core speed
or I/O, and the aggregate rate cannot separate them. This times the two halves
apart: pure CPU (pedalboard stretch, no writes) and then the writes.

Run identically on both machines.
"""
import os, sys, time
import numpy as np
import soundfile as sf
from pedalboard import time_stretch

src = sys.argv[1]
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)

y, sr = sf.read(src, dtype="float32", always_2d=True)
y = y.mean(axis=1)
print(f"src {os.path.basename(src)}  {len(y)/sr:.1f}s @ {sr}")

# CPU only: 10 stretches, nothing written
t0 = time.time()
outs = []
for p in range(-60, -20, 4):
    outs.append(time_stretch(y[None, :], sr, 1.0 / (1 + p / 100.0))[0])
t_cpu = time.time() - t0

# writes only
t0 = time.time()
for i, o in enumerate(outs):
    sf.write(os.path.join(outdir, f"b{i}.wav"), o, 22050, format="WAV")
t_io = time.time() - t0

print(f"  cpu   {t_cpu:6.2f}s for 10 stretches  ({t_cpu/10:.3f}s each)")
print(f"  write {t_io:6.2f}s for 10 files       ({t_io/10:.3f}s each)")
print(f"  total {t_cpu+t_io:6.2f}s")
