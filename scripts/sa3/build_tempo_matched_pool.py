"""Build a source pool whose tempo distribution MATCHES SMC's.

Every corpus built so far (sdall, r3, tf1) drew from data/pools/slow_under70.json,
which is capped at 70 bpm. Measured against the target that is a mistake:

    corpus (tf1)   median 60.0   IQR [53, 66]   5-95% [35, 70]   sd 10.5
    SMC            median 70.9   IQR [57, 93]   5-95% [37,140]   sd 31.8

53% of SMC sits above 70 bpm, so the corpus covered less than half of what it was
aimed at, and its spread was 3x too narrow. That cap was inherited from the first
experiment and then held fixed for one-variable comparability across three arms --
all of which came back null.

This samples sources so the resulting pool reproduces SMC's tempo histogram,
using SMC's own non-held-out tracks as the reference. Sampling is with weights
per tempo bin, without replacement, so no source appears twice.

Fold 0 is excluded unconditionally: those recordings are the validation set.
SMC tracks themselves are excluded from the POOL by default (--allow-smc to keep
them) -- generating from the very corpus we evaluate on would be the most
flattering possible setup and the least informative.

Usage:
  build_tempo_matched_pool.py --out data/pools/tempo_matched.json --n 1200
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

# Use the FULL staged training annotations, one dir per dataset. The earlier
# bt_wide/ set is a 2678-track subset that happens to contain nothing above
# 127 bpm, which silently made a "flat" pool impossible at the fast end.
ANN = Path("/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this/data_tf1/annotations")
AUDIO = Path("/disk1/jaehoon/dataset_store/audio_by_stem")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1200,
                    help="pool size; generation cycles through it, so this only "
                         "needs to be large enough for variety")
    ap.add_argument("--exclude-folds", type=int, nargs="*", default=[0])
    ap.add_argument("--allow-smc", action="store_true")
    ap.add_argument("--min-beats", type=int, default=12)
    ap.add_argument("--bins", type=int, default=14)
    ap.add_argument("--flatten", type=float, default=1.0,
                    help="0 = reproduce SMC's tempo histogram, 1 = uniform "
                         "mass per log-tempo bin (over-represents slow and "
                         "fast relative to mid). Values between blend the two.")
    ap.add_argument("--span", type=float, nargs=2, default=None,
                    metavar=("LO","HI"),
                    help="bpm range to spread over; default is SMC's own range")
    ap.add_argument("--max-repeat", type=int, default=6,
                    help="max times one recording may appear in the pool; each\n"
                         "appearance becomes a different 20 s window at generation")
    ap.add_argument("--inverse-real", action="store_true",
                    help="U-shape: target mass INVERSELY proportional to the "
                         "real training set's density in each tempo bin, so the "
                         "corpus fills the gaps rather than mirroring the peak")
    ap.add_argument("--inv-power", type=float, default=0.5,
                    help="how hard to invert: 1.0 = full inverse (puts ~46%% in "
                         "the scarcest bin, too extreme), 0 = uniform. ~0.5 "
                         "lifts the tails without collapsing onto one bin.")
    ap.add_argument("--inv-floor", type=float, default=0.005,
                    help="floor on a bin's real share, so an empty bin does not "
                         "take unbounded mass")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cands, smc_bpm = [], []
    for dsdir in sorted(ANN.iterdir()):
        ds = dsdir.name
        if ds.startswith("sdedit") or ds == "gtzan":
            continue
        bdir = dsdir / "annotations" / "beats"
        split = dsdir / "8-folds.split"
        if not bdir.is_dir() or not split.exists():
            continue
        folds = {}
        for line in open(split):
            q = line.split()
            if len(q) == 2:
                folds[q[0]] = int(q[1])
        for f in sorted(bdir.glob("*.beats")):
            stem = f.stem
            try:
                b = np.loadtxt(f, ndmin=2)[:, 0]
            except Exception:
                continue
            if len(b) < a.min_beats:
                continue
            bpm = 60.0 / np.median(np.diff(b))
            if not np.isfinite(bpm) or not (20 <= bpm <= 300):
                continue
            fold = folds.get(stem)
            if ds == "smc" and fold not in a.exclude_folds:
                smc_bpm.append(bpm)
            if fold in a.exclude_folds:
                continue
            if ds == "smc" and not a.allow_smc:
                continue
            wav = next((p for p in (AUDIO / ds / f"{stem}{e}"
                                    for e in (".wav", ".flac")) if p.exists()), None)
            if wav is None:
                continue
            cands.append(dict(name=f"{ds}__{stem}", corpus=ds, stem=stem,
                              audio=str(wav), beats=str(f), bpm=round(float(bpm), 2),
                              n_beats=int(len(b)), fold=fold))

    smc_bpm = np.array(smc_bpm)
    print(f"  candidates {len(cands)}   SMC reference n={len(smc_bpm)} "
          f"median {np.median(smc_bpm):.1f}", flush=True)
    if len(cands) == 0 or len(smc_bpm) == 0:
        raise SystemExit("no candidates or no SMC reference")

    # Target histogram in log-bpm (perceptually even bins).
    #
    # --flatten blends SMC's own shape toward UNIFORM. Matching SMC exactly
    # reproduces its peak, but training data should cover where the model is
    # WEAK, and the weakness is in the tails: SMC is unimodal around 70 bpm
    # while the failures concentrate below ~60 and above ~120. flatten=1.0
    # gives equal mass per tempo bin, i.e. deliberately over-representing slow
    # and fast relative to mid.
    lo, hi = np.log(smc_bpm.min()), np.log(smc_bpm.max())
    if a.span:                       # widen beyond SMC's own observed range
        lo, hi = np.log(a.span[0]), np.log(a.span[1])
    edges = np.linspace(lo, hi, a.bins + 1)
    # real training density per bin (all candidates = the real training set)
    _rh, _ = np.histogram(np.log([c["bpm"] for c in cands]), bins=edges)
    a._real_share = _rh / max(_rh.sum(), 1)

    shape, _ = np.histogram(np.log(smc_bpm), bins=edges)
    shape = shape / max(shape.sum(), 1)
    unif = np.full(a.bins, 1.0 / a.bins)
    if a.inverse_real:
        # U-SHAPE. Flat corpus + peaked real data still leaves a peaked TOTAL:
        # at 8% dose a flat corpus moves the 40-46bpm share of training from
        # 0.54% to 1.15%. Weighting inversely to the real training density
        # instead puts the corpus where the data is not: same dose reaches
        # 9.5% for that bin. `real_share` is filled in by the caller below.
        inv = (1.0 / np.maximum(a._real_share, a.inv_floor)) ** a.inv_power
        tgt = inv / inv.sum()
    else:
        tgt = (1 - a.flatten) * shape + a.flatten * unif
    tgt = tgt / tgt.sum()

    cb = np.log(np.array([c["bpm"] for c in cands]))
    idx_by_bin = [np.where((cb >= edges[i]) & (cb < edges[i + 1] if i < a.bins - 1
                                               else cb <= edges[i + 1]))[0]
                  for i in range(a.bins)]

    # Slow bins are genuinely scarce (8-23 sources below 51 bpm in the whole
    # training set), so a flat pool cannot be built from unique sources. Repeat
    # them instead: sa3_sdedit_bulk maps job i -> pool[i % len(pool)] with the
    # window drawn from Random(i), so a source appearing k times in the pool
    # yields k DIFFERENT windows of that recording, not k copies of one clip.
    # --max-repeat bounds how hard any one recording is leaned on.
    rng = np.random.default_rng(a.seed)
    want = np.floor(tgt * a.n).astype(int)
    picked, short = [], 0
    for i, k in enumerate(want):
        pool_i = idx_by_bin[i]
        if len(pool_i) == 0 or k == 0:
            short += k
            continue
        cap = len(pool_i) * a.max_repeat
        take = min(k, cap)
        short += k - take
        # even coverage first, then random extras, so no source is starved
        full = take // len(pool_i)
        picked += list(np.tile(pool_i, full))
        rem = take - full * len(pool_i)
        if rem:
            picked += list(rng.choice(pool_i, rem, replace=False))

    out = [cands[i] for i in picked]
    rng.shuffle(out)
    got = np.array([c["bpm"] for c in out])
    print(f"  pool {len(out)}  median {np.median(got):.1f}  "
          f"IQR [{np.percentile(got,25):.1f},{np.percentile(got,75):.1f}]  "
          f"5-95% [{np.percentile(got,5):.1f},{np.percentile(got,95):.1f}]  sd {got.std():.1f}")
    print(f"  SMC target      median {np.median(smc_bpm):.1f}  "
          f"IQR [{np.percentile(smc_bpm,25):.1f},{np.percentile(smc_bpm,75):.1f}]  "
          f"5-95% [{np.percentile(smc_bpm,5):.1f},{np.percentile(smc_bpm,95):.1f}]  sd {smc_bpm.std():.1f}")
    print(f"  fraction >70bpm: pool {np.mean(got>70):.0%}  SMC {np.mean(smc_bpm>70):.0%}")
    print("\n  per-bin counts (bpm range: pool / SMC-share%):")
    gh,_ = np.histogram(np.log(got), bins=edges)
    for i in range(a.bins):
        print(f"    {np.exp(edges[i]):6.1f}-{np.exp(edges[i+1]):6.1f}  "
              f"{gh[i]:4d}  target {tgt[i]*100:4.1f}%  got {gh[i]/max(gh.sum(),1)*100:4.1f}%")
    reps = Counter(c["name"] for c in out)
    print(f"\n  unique sources {len(reps)}, max windows from one source {max(reps.values())}")
    print("  by dataset:", Counter(c["corpus"] for c in out).most_common())
    if any(c["fold"] in a.exclude_folds for c in out):
        raise SystemExit("a held-out-fold source reached the pool -- abort")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
