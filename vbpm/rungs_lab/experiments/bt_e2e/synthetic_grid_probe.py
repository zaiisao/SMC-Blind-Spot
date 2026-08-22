"""Synthetic grid probe: does the integer-interval state space FORCE tempo wobble on
perfectly equidistant beats?

Synthesize activations with metronomically exact beats at a chosen (possibly non-integer)
frame interval. Then:
  1. Viterbi-decode (lambda=100) and print the latent path's beat-to-beat interval sequence:
     a non-integer true interval CANNOT be represented, so the path must alternate
     (e.g. 21,22,21,21,22,...). Integer control should decode flat.
  2. Run exact EM on many phase-randomized crops of the SAME metronomic process and report
     lambda_MLE: any finite/moderate lambda on perfectly steady data is looseness
     manufactured by the grid alone.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_bt import FPS, BT_SHIPPED_DECODE
from rungs.r2_em_dbn import R2GenerativeLambda
from rungs.r1_2016_dbn import DBN2016

DEVICE = "cuda:1"
T, N_CROPS, EM_ITERS = 700, 50, 6
PEAK, FLOOR = 0.95, 0.02


def synth(interval, phase, T):
    """[T,2] activations: exact beats every `interval` frames starting at `phase` (floats).
    Peak mass split linearly between the two straddling frames (frontend-like sub-frame cue);
    every 4th beat is a downbeat."""
    a = np.full((T, 2), FLOOR, dtype=np.float64)
    k = 0
    t = phase
    while t < T - 1:
        lo, frac = int(np.floor(t)), t - np.floor(t)
        for f, w in ((lo, 1 - frac), (lo + 1, frac)):
            if 0 <= f < T:
                a[f, 0] = max(a[f, 0], FLOOR + (PEAK - FLOOR) * w)
                if k % 4 == 0:
                    a[f, 1] = max(a[f, 1], FLOOR + (PEAK - FLOOR) * w * 0.9)
        k += 1
        t += interval
    return a


def main():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    decode = DBN2016(fps=FPS, device=DEVICE, dtype=torch.float32, bounding="none",
                     transition_lambda=100.0, **BT_SHIPPED_DECODE)
    for interval in (21.0, 21.4, 21.5):
        print(f"\n=== true interval {interval} frames ({60*FPS/interval:.2f} bpm), "
              f"metronomically EXACT ===", flush=True)
        # 1. what does the latent path encode?
        a = synth(interval, phase=0.3, T=T)
        ev = decode.predict(a)
        bf = np.round(np.asarray(ev["beats"]) * FPS).astype(int)
        d = np.diff(bf)
        print(f"decoded interval sequence (lambda=100): {list(d[:24])}", flush=True)
        vals, cnts = np.unique(d, return_counts=True)
        print(f"interval histogram: {dict(zip(vals.tolist(), cnts.tolist()))}", flush=True)
        # 2. what lambda does EM think this metronome has?
        crops = [torch.from_numpy(synth(interval, rng.uniform(0, interval), T)).float().to(DEVICE)
                 for _ in range(N_CROPS)]
        r2 = R2GenerativeLambda(fps=FPS, device=DEVICE,
                                observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
        for it in range(EM_ITERS):
            lam = r2.em_step(crops)
            print(f"  EM iter {it}: lambda = {lam:.2f}", flush=True)
        print(f"interval {interval}: FINAL lambda_MLE = {r2.transition_lambda:.2f}", flush=True)


if __name__ == "__main__":
    main()
