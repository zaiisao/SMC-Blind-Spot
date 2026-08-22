"""K fractional position-bin labels: learn the beat SHAPE instead of a step of width 1/obs_lambda.

The 2016 emission is a 3-class step: BEAT iff (position within beat) < 1/observation_lambda, one
integer controlling both the region width and the no-beat divisor. Here each state maps to

    bin = floor(position_within_beat * K)          (position in [0,1), so K bins per beat)

and the head emits a log-density per bin per frame. Classes are 2K: bins 0..K-1 for beats inside
bar-position 0 (downbeats) and K..2K-1 for all other beats, preserving the beat/downbeat
distinction the DBN needs.

WHY THIS IS THE RIGHT GENERALISATION.
  * TEMPO-NEUTRAL BY CONSTRUCTION. Bins are FRACTIONS of the beat, so each bin holds T/K frames at
    every tempo. That is the identity E(i) = n_bin * (per-bin density) with n_bin independent of i
    -- the same property that makes the 1/16 rule work and that any fixed-millisecond region
    provably violates (measured spreads: fractional 0.0 nats, fixed-W 1671 and 5134 nats).
  * COMPUTATIONALLY FREE. The emission table stays compact [T, 2K] and is gathered by a static
    state->class map, exactly as now. No [T, num_states] blowup.
  * STRICTLY GENERALISES 2016, which is the K~16 special case with a STEP shape.

Labels here are the bin index of the nearest annotated beat, for frames within that beat's span;
frames are always assigned (every frame lies inside some beat), so unlike the 3-class version
there is no dominant NO_BEAT class -- the "no beat" information is carried by the bins far from 0.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from build_emission_labels import load_beats, peak_frames, FPS

EMB = Path("/disk1/jaehoon/bt_emb")


def label_track_bins(beat_t, beat_n, T, start_frame, K):
    """[T] int in [0, 2K): bin of position within the enclosing beat, +K if that beat is a downbeat."""
    lab = np.zeros(T, dtype=np.int64)
    f = beat_t * FPS - start_frame
    if len(f) < 2:
        return lab
    for k in range(len(f) - 1):
        a, b = f[k], f[k + 1]
        if b <= 0 or a >= T or b <= a:
            continue
        lo, hi = max(0, int(np.ceil(a))), min(T, int(np.ceil(b)))
        if hi <= lo:
            continue
        pos = (np.arange(lo, hi) - a) / (b - a)            # position within beat, [0,1)
        bins = np.clip((pos * K).astype(np.int64), 0, K - 1)
        if beat_n[k] == 1:                                  # this beat is a downbeat
            bins = bins + K
        lab[lo:hi] = bins
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--datasets", default="gtzan,ballroom,hainsworth,beatles,asap")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    K = args.K
    out = Path(args.out or f"/disk1/jaehoon/bt_labels_bin{K}")
    out.mkdir(parents=True, exist_ok=True)

    offs, per_ds, hist = [], {}, np.zeros(2 * K, dtype=np.int64)
    for ds in [d.strip() for d in args.datasets.split(",")]:
        d = EMB / ds
        if not d.exists():
            continue
        od = out / ds; od.mkdir(parents=True, exist_ok=True)
        ds_off = []
        for f in sorted(d.glob("*.npz")):
            z = np.load(f)
            T, sf = int(z["emb"].shape[0]), int(z["start_frame"])
            bt, bn = load_beats(ds, f.stem)
            if bt is None or len(bt) < 2:
                continue
            lab = label_track_bins(bt, bn, T, sf, K)
            hist += np.bincount(lab, minlength=2 * K)
            # alignment gate: bin-0 onsets must sit on the annotated beats (systematic-shift check)
            pk = peak_frames(z["beat"])
            lb = np.rint(bt * FPS - sf).astype(int); lb = lb[(lb >= 0) & (lb < T)]
            if len(pk) >= 4 and len(lb) >= 4:
                med = float(np.median([pk[np.argmin(np.abs(pk - x))] - x for x in lb]))
                ds_off.append(med); offs.append(med)
            np.savez_compressed(od / f"{f.stem}.npz", label=lab.astype(np.int8),
                                fold=int(z["fold"]))
        if ds_off:
            per_ds[ds] = float(np.median(ds_off))
    med_all = float(np.median(offs)) if offs else float("nan")
    worst = max(abs(v) for v in per_ds.values()) if per_ds else float("nan")
    print(f"K={K}  classes={2*K}  tracks with labels: {int(hist.sum()/1):,} frames")
    print(f"  bin occupancy (fraction): "
          f"{np.round(hist / max(hist.sum(), 1), 4)[:K]} (non-downbeat) / "
          f"{np.round(hist / max(hist.sum(), 1), 4)[K:]} (downbeat)")
    print(f"  ALIGNMENT GATE median offset {med_all:+.2f} fr, worst dataset |median| {worst:.2f}")
    ok = abs(med_all) <= 0.5 and worst <= 0.5
    print(f"  GATE {'PASS' if ok else 'FAIL'}")
    (out / "_gate.json").write_text(json.dumps(
        {"K": K, "per_dataset_median_offset": per_ds, "median_offset": med_all,
         "bin_occupancy": (hist / max(hist.sum(), 1)).tolist(), "pass": ok}, indent=1))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
