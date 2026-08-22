"""GATE: does the differentiable soft tempo readout reproduce the hard-readout equivariance defect?

The equivariance loss will be computed on soft_log_period(). If that readout does not agree with the
peak-pick / median-IBI measurement that produced the defect table, then training on it optimises
something other than the defect we diagnosed, and any resulting gain would be uninterpretable.

Hard readout (scripts/diagnostics/probe_tempo_equivariance.py, 40 SMC tracks, fold-honest, phase-vocoder):
        s      0.50   0.70   0.85   1.00   1.20   1.50   2.00
    median d  +0.757 +0.029 +0.012 +0.000 -0.018 -0.411 -0.966

PASS REQUIRES, on the same tracks and stretches:
  G1  sign agreement of the median defect at every s where |hard| > 0.1
  G2  |soft - hard| <= 0.25 on the median defect at every s
  G3  the flat region (|median| <= 0.1) is the same set of s
A readout that passes measures the same phenomenon; one that fails is a different instrument.
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third-party" / "beat_this"))
from beat_this.inference import Audio2Frames
from beat_this.preprocessing import load_audio
from _harness import FPS, track_ids, load_gt, peakpick
from tempo_readout import lag_grid, soft_log_period

SMC_AUD = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
STORE = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
STRETCH = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
N_TRACKS = 40
TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

HARD = {0.5: +0.757, 0.7: +0.029, 0.85: +0.012, 1.0: 0.0, 1.2: -0.018, 1.5: -0.411, 2.0: -0.966}


def main():
    dev = "cuda:1"
    split = {l.split("\t")[0]: int(l.split("\t")[1])
             for l in open(STORE / "smc" / "8-folds.split") if l.strip()}
    ids = [t for t in track_ids() if (SMC_AUD / f"{t.upper()}.wav").exists()][:N_TRACKS]
    by_fold = {}
    for t in ids:
        by_fold.setdefault(split.get(t, 0), []).append(t)

    lags = lag_grid()
    soft, hard = {s: {} for s in STRETCH}, {s: {} for s in STRETCH}
    for fold, tracks in sorted(by_fold.items()):
        a2f = Audio2Frames(checkpoint_path=f"fold{fold}", device=dev)
        for t in tracks:
            sig, sr = load_audio(str(SMC_AUD / f"{t.upper()}.wav"))
            y = sig.mean(1) if sig.ndim > 1 else sig
            y = np.ascontiguousarray(y.astype(np.float32))
            for s in STRETCH:
                ys = y if s == 1.0 else librosa.effects.time_stretch(y, rate=1.0 / s)
                b, _ = a2f(ys, sr)
                p = torch.sigmoid(b.float()).cpu()
                f, _ = soft_log_period(p, lags, tau=TAU)
                soft[s][t] = float(f[0])                       # log2 frames
                bt = peakpick(p.numpy(), 0.5, FPS)
                hard[s][t] = (float(np.log2(np.median(np.diff(bt)) * FPS))
                              if len(bt) >= 4 else float("nan"))
        print(f"  fold {fold}: {len(tracks)} done", flush=True)

    def defect(tab):
        out = {}
        for s in STRETCH:
            d = [tab[s][t] - tab[1.0][t] - np.log2(s) for t in ids
                 if np.isfinite(tab[s].get(t, np.nan)) and np.isfinite(tab[1.0].get(t, np.nan))]
            out[s] = float(np.median(d)) if d else float("nan")
        return out

    ds, dh = defect(soft), defect(hard)
    print(f"\ntau={TAU}   defect delta(s), median over {len(ids)} tracks")
    print(f"{'s':>6} {'hard(ref)':>10} {'hard(now)':>10} {'soft':>8} {'|soft-hard|':>12}")
    g1 = g2 = g3 = True
    for s in STRETCH:
        gap = abs(ds[s] - dh[s])
        print(f"{s:>6.2f} {HARD[s]:>+10.3f} {dh[s]:>+10.3f} {ds[s]:>+8.3f} {gap:>12.3f}")
        if abs(dh[s]) > 0.1:
            g1 &= (np.sign(ds[s]) == np.sign(dh[s]))
        g2 &= (gap <= 0.25)
        g3 &= ((abs(ds[s]) <= 0.1) == (abs(dh[s]) <= 0.1))
    print(f"\n  G1 sign agreement      {'PASS' if g1 else 'FAIL'}")
    print(f"  G2 |soft-hard|<=0.25   {'PASS' if g2 else 'FAIL'}")
    print(f"  G3 flat region matches {'PASS' if g3 else 'FAIL'}")
    ok = g1 and g2 and g3
    print(f"  GATE {'PASS' if ok else 'FAIL'}")
    json.dump({"tau": TAU, "soft": ds, "hard": dh, "pass": bool(ok)},
              open(f"/tmp/overnight/gate_soft_readout_tau{TAU:g}.json", "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
