"""FINAL acoustic-evidence filter for the generated (SDEdit) slow-tempo corpus.

What it measures
----------------
For each clip: how far the LABELLED inter-beat interval falls short of the
periodicity that is actually dominant in the audio.

    flux  = positive spectral flux of the log-mel spectrogram (no beat model)
    ac    = unbiased-normalised autocorrelation of the flux
    score = ac(labelled period, interpolated) - max ac over [0.35 P, 4 P]

score <= 0 always.  0 means the labelled beat period IS the dominant periodicity
of the audio; strongly negative means the generation put its pulse somewhere
else, so the inherited labels no longer describe the audio.

Why this one
------------
It is a WITHIN-CLIP CONTRAST: the labelled hypothesis is compared against a rival
hypothesis measured on the same clip, so the clip's overall periodic strength
(which is what "difficulty" is) cancels out.  Absolute evidence measures
(onset energy at beats, raw autocorrelation, comb response) have larger raw
correlations with damage but are easiness filters in disguise.

Pre-registered criterion, held-out REPORT half (n=483), independently
re-implemented and re-verified:
    rho_partial(score, damage | difficulty) = -0.436   (need |.| > 0.25)
    rho_diff   (score, difficulty)          = -0.086   (need |.| < 0.15)
Baseline onset-flux z-score for comparison: -0.285 / -0.277 (fails the guard).

CPU only.  No GPU, no beat model, no tracker output.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

BT = "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"
FPS = 50

# Threshold chosen on the SELECT half only: keeps ~47%, halves mean label damage
# (0.175 kept vs 0.350 dropped on SELECT) while leaving mean difficulty flat
# (0.231 vs 0.221), i.e. it removes damage without removing hard music.
THRESHOLD = -0.15


def flux(spect):
    d = np.diff(spect, axis=0)
    f = np.maximum(d, 0).sum(axis=1)
    return np.concatenate([[0.0], f])


def evidence(spect, beat_times):
    """Shortfall of the labelled period against the dominant periodicity."""
    beats = np.asarray(beat_times, dtype=float)
    if len(beats) < 4:
        return np.nan
    f = flux(spect)
    f = f - f.mean()
    s = f.std()
    if s < 1e-9:
        return np.nan
    f = f / s
    T = len(f)

    n = int(2 ** np.ceil(np.log2(2 * T)))
    F = np.fft.rfft(f, n)
    ac = np.fft.irfft(F * np.conj(F), n)[:T]
    ac = ac / (np.arange(T, 0, -1) * (ac[0] / T + 1e-12))   # unbiased, ac[0]=1

    P = np.median(np.diff(beats)) * FPS                     # labelled period, frames
    lo = int(np.floor(0.35 * P))
    hi = min(int(np.ceil(4.0 * P)), T - 2)
    if P >= T - 2 or hi <= lo + 2:
        return np.nan
    i0 = int(np.floor(P))
    fr = P - i0
    at_period = ac[i0] * (1.0 - fr) + ac[i0 + 1] * fr
    return float(at_period - ac[lo:hi + 1].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spect", default="data/audio/spectrograms/sdedit.npz")
    ap.add_argument("--ann", default="data/annotations/sdedit/annotations/beats")
    ap.add_argument("--out", default="/home/sogang/jaehoon/Analyze-SMC/cache/sdedit_evidence.csv")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    a = ap.parse_args()

    os.chdir(BT)
    sys.path.insert(0, BT)
    from beat_this.dataset.mmnpz import MemmappedNpzFile

    npz = MemmappedNpzFile(Path(a.spect))
    annd = Path(a.ann)
    stems = sorted({k.split("/")[0] for k in npz.files})

    scores = {}
    for st in stems:
        fa = annd / f"{st}.beats"
        if not fa.exists():
            continue
        ref = np.loadtxt(fa, ndmin=2)[:, 0]
        sp = np.asarray(npz[f"{st}/track"], dtype=np.float32)
        scores[st] = evidence(sp, ref)

    with open(a.out, "w") as fh:
        fh.write("stem,evidence_score\n")
        for st, v in scores.items():
            fh.write(f"{st},{v:.6f}\n" if np.isfinite(v) else f"{st},nan\n")

    v = np.array([x for x in scores.values() if np.isfinite(x)])
    keep = (v >= a.threshold).mean()
    print(f"scored {len(scores)} clips ({len(v)} finite) -> {a.out}")
    print(f"score percentiles  p05={np.percentile(v,5):.3f}  p25={np.percentile(v,25):.3f} "
          f" median={np.median(v):.3f}  p75={np.percentile(v,75):.3f}  max={v.max():.3f}")
    print(f"recommended threshold {a.threshold:+.3f} -> keeps {keep*100:.1f}% of clips")


if __name__ == "__main__":
    main()
