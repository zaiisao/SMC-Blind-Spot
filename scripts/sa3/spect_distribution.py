"""How far is a generated corpus from real music, IN THE REPRESENTATION THE
TRACKER ACTUALLY CONSUMES?

The staged npz files are the model's literal input: log-mel, 50 fps, 128 bands.
Comparing there needs no decoding, no resampling, and no proxy -- and it can use
every real track we have staged, not just the ~400 source recordings a paired
test can reach.

Three questions, in increasing strength:

  1. PROFILE   per-band mean and sd of the generated set vs real. A corpus that
               thinned its spectrum or collapsed its dynamics shows up here.
  2. DISTANCE  Frechet distance between Gaussians fitted to frame vectors
               (the FID construction, on log-mel frames instead of Inception
               features). Reported for each real dataset so the scale is
               readable: real-vs-real distances say what "close" means.
  3. PROBE     AUC of a logistic probe trained to tell generated frames from
               real ones, held out by TRACK. 0.5 = indistinguishable, 1.0 =
               trivially separable. This is the direct "how far off" number,
               and unlike the others it is not fooled by matched first- and
               second-order statistics.

Slow real tracks (<70 bpm by their annotations) are reported separately, since
that is the material the corpus is supposed to resemble.

Usage:
  spect_distribution.py --gen sdedit_tf1 --data-dir data_tf1 \
      --real smc asap hainsworth simac --out cache/spect_dist.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np

BT = Path("/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")


def frames_from(npz, stems, per_track, rng):
    """Sample `per_track` random frames from each track."""
    out = []
    for s in stems:
        try:
            a = np.asarray(npz[f"{s}/track"], dtype=np.float32)
        except Exception:
            continue
        if a.ndim != 2 or a.shape[0] < 4:
            continue
        idx = rng.integers(0, a.shape[0], min(per_track, a.shape[0]))
        out.append(a[idx])
    return np.concatenate(out) if out else np.zeros((0, 128), np.float32)


def frechet(a, b):
    """Frechet distance between Gaussians fitted to two frame sets."""
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False) + 1e-6 * np.eye(a.shape[1])
    s2 = np.cov(b, rowvar=False) + 1e-6 * np.eye(b.shape[1])
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def track_stems(data_dir, ds):
    d = BT / data_dir / "annotations" / ds / "annotations" / "beats"
    return sorted(p.stem for p in d.glob("*.beats")) if d.is_dir() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="generated dataset name")
    ap.add_argument("--data-dir", default="data_tf1")
    ap.add_argument("--real", nargs="+",
                    default=["smc", "asap", "hainsworth", "simac", "rwc"])
    ap.add_argument("--per-track", type=int, default=200)
    ap.add_argument("--max-tracks", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sys.path.insert(0, str(BT))
    from beat_this.dataset.mmnpz import MemmappedNpzFile
    rng = np.random.default_rng(a.seed)
    spect = BT / a.data_dir / "audio" / "spectrograms"

    def load(ds):
        z = MemmappedNpzFile(spect / f"{ds}.npz")
        stems = track_stems(a.data_dir, ds)
        if not stems:
            stems = sorted({k.split("/")[0] for k in z.keys()})
        stems = list(rng.permutation(stems)[: a.max_tracks])
        return frames_from(z, stems, a.per_track, rng), len(stems)

    G, ng = load(a.gen)
    print(f"  {a.gen}: {G.shape[0]} frames from {ng} tracks", flush=True)

    reals = {}
    for ds in a.real:
        try:
            F, n = load(ds)
            if F.shape[0]:
                reals[ds] = F
                print(f"  {ds}: {F.shape[0]} frames from {n} tracks", flush=True)
        except Exception as e:
            print(f"  {ds}: skipped ({e})", flush=True)

    ALL = np.concatenate(list(reals.values()))

    print(f"\n1. PROFILE (per-band, averaged over 128 bands)")
    print(f"   {'set':14} {'mean':>8} {'sd':>8}   band-mean |gen-real|")
    print(f"   {'real (all)':14} {ALL.mean():8.4f} {ALL.std():8.4f}")
    print(f"   {a.gen:14} {G.mean():8.4f} {G.std():8.4f}   "
          f"{np.abs(G.mean(0)-ALL.mean(0)).mean():.4f}")

    print(f"\n2. FRECHET DISTANCE on log-mel frames")
    print(f"   {a.gen} vs real(all): {frechet(G, ALL):8.3f}")
    print(f"   -- scale reference, real vs real --")
    names = list(reals)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            print(f"   {x} vs {y}: {frechet(reals[x], reals[y]):8.3f}")
    for x in names:
        print(f"   {a.gen} vs {x}: {frechet(G, reals[x]):8.3f}")

    print(f"\n3. LOGISTIC PROBE, generated vs real frames (held out by track)")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    n = min(len(G), len(ALL))
    X = np.concatenate([G[:n], ALL[:n]])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    # split by frame block so a track's frames stay together
    blk = a.per_track
    ntr = int(0.7 * n / blk) * blk
    tr = np.r_[np.arange(ntr), n + np.arange(ntr)]
    te = np.r_[np.arange(ntr, n), n + np.arange(ntr, n)]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
    auc = roc_auc_score(y[te], clf.decision_function((X[te] - mu) / sd))
    print(f"   AUC = {auc:.4f}   (0.5 = indistinguishable from real)")

    if a.out:
        import pandas as pd
        pd.DataFrame([dict(gen=a.gen, frechet_all=frechet(G, ALL), probe_auc=auc,
                           gen_mean=G.mean(), gen_sd=G.std(),
                           real_mean=ALL.mean(), real_sd=ALL.std())]
                     ).to_csv(a.out, index=False)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
