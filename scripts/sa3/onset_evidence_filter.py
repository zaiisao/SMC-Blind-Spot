"""Proof of concept: filter generated clips by ACOUSTIC EVIDENCE for their labels,
not by whether a beat model likes them.

Motivation. Filtering generated training data on "does the tracker follow it"
selects for what the model already knows, so the corpus gets easier and teaches
nothing. We want to separate two different things:

    label damage  -- the audio no longer supports the inherited beat times
    difficulty    -- the audio is hard but the labels are still right

Only the first should be filtered. The second is the whole point of the corpus.

Method. Spectral flux from the log-mel spectrogram (no beat model involved).
Score = how much more onset energy sits at the labelled beat frames than at the
same grid shifted to random phases. Expressed as a z-score against that null, so
it is comparable across clips of different density and loudness.

Validation. Against the paired generated-vs-source scores we already have:
    corr(evidence, damage)     should be strongly NEGATIVE  (low evidence -> damaged)
    corr(evidence, difficulty) should be near ZERO          (not an easiness filter)
If both hold, the filter is selective for validity rather than difficulty.

CPU only -- deliberately does not touch the GPUs while training runs.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

BT = "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"
FPS = 50
TOL = 3          # frames either side of a beat to look for its onset
N_NULL = 200     # random phase shifts for the null


def flux(spect):
    """Positive spectral flux per frame, from a log-mel spectrogram (T, F)."""
    d = np.diff(spect, axis=0)
    f = np.maximum(d, 0).sum(axis=1)
    f = np.concatenate([[0.0], f])
    m = np.median(f)
    s = f.std()
    return (f - m) / (s + 1e-9)


def evidence(spect, beat_times, rng):
    """z-score of onset energy at the labelled beats vs randomly phased grids."""
    f = flux(spect)
    T = len(f)
    idx = np.round(np.asarray(beat_times) * FPS).astype(int)
    idx = idx[(idx >= TOL) & (idx < T - TOL)]
    if len(idx) < 3:
        return np.nan, np.nan

    def score(offsets):
        # peak flux within +-TOL of each beat, averaged over beats
        w = np.arange(-TOL, TOL + 1)
        pos = (offsets[:, None] + w[None, :]).clip(0, T - 1)
        return f[pos].max(axis=1).mean()

    obs = score(idx)
    span = T - 2 * TOL
    null = np.empty(N_NULL)
    for i in range(N_NULL):
        shift = rng.integers(1, span)
        null[i] = score(((idx - TOL + shift) % span) + TOL)
    z = (obs - null.mean()) / (null.std() + 1e-9)
    return float(obs), float(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--paired", default="/home/sogang/jaehoon/Analyze-SMC/cache/sdedit_vs_source.csv")
    a = ap.parse_args()

    os.chdir(BT)
    sys.path.insert(0, BT)
    from beat_this.dataset.mmnpz import MemmappedNpzFile

    rng = np.random.default_rng(0)

    # ---- generated clips -------------------------------------------------
    npz = MemmappedNpzFile(Path("data/audio/spectrograms/sdedit.npz"))
    annd = Path("data/annotations/sdedit/annotations/beats")
    stems = sorted({k.split("/")[0] for k in npz.files})
    rows = {}
    for s in stems:
        ref = np.loadtxt(annd / f"{s}.beats", ndmin=2)[:, 0]
        sp = np.asarray(npz[f"{s}/track"], dtype=np.float32)
        obs, z = evidence(sp, ref, rng)
        rows[s] = dict(stem=s, obs=obs, z=z,
                       bpm=60.0 / np.median(np.diff(ref)) if len(ref) > 1 else np.nan)
    print(f"  scored {len(rows)} generated clips", flush=True)

    # ---- real slow music, as a reference distribution ---------------------
    real = []
    for ds in ["smc", "asap", "hainsworth", "simac", "rwc"]:
        p = Path(f"data/audio/spectrograms/{ds}.npz")
        ad = Path(f"data/annotations/{ds}/annotations/beats")
        if not p.exists():
            continue
        z2 = MemmappedNpzFile(p)
        st = sorted({k.split("/")[0] for k in z2.files})[:120]
        for s in st:
            fa = ad / f"{s}.beats"
            if not fa.exists():
                continue
            ref = np.loadtxt(fa, ndmin=2)
            ref = ref[:, 0] if ref.ndim == 2 else ref
            if len(ref) < 4:
                continue
            bpm = 60.0 / np.median(np.diff(ref))
            if bpm >= 70:
                continue
            try:
                sp = np.asarray(z2[f"{s}/track"], dtype=np.float32)
            except KeyError:
                continue
            _, zz = evidence(sp, ref, rng)
            real.append((ds, s, bpm, zz))
    print(f"  scored {len(real)} real slow tracks as reference", flush=True)

    # ---- join with the paired generated-vs-source scores ------------------
    paired = {}
    if Path(a.paired).exists():
        for r in csv.DictReader(open(a.paired)):
            paired[r["stem"]] = (float(r["F_gen"]), float(r["F_src"]),
                                 float(r["r_gen"]), float(r["r_src"]))

    with open(a.out, "w") as fh:
        fh.write("stem,bpm,obs,z,F_gen,F_src,r_gen,r_src\n")
        for s, d in rows.items():
            p = paired.get(s)
            fh.write(f"{s},{d['bpm']:.3f},{d['obs']:.4f},{d['z']:.4f},"
                     + (f"{p[0]:.6f},{p[1]:.6f},{p[2]:.4f},{p[3]:.4f}\n" if p else ",,,\n"))
    with open(a.out.replace(".csv", "_real.csv"), "w") as fh:
        fh.write("dataset,stem,bpm,z\n")
        for ds, s, bpm, zz in real:
            fh.write(f"{ds},{s},{bpm:.3f},{zz:.4f}\n")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
