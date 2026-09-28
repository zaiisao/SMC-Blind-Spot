"""Deterministic spectral repair of the SDEdit spectrograms toward real music.

The measured deficits (generated vs real slow, log-mel domain):
    transient crest   x0.48   dynamics x0.52   spectral flatness x1.46
    hf/lf brightness  x2.08

Three deterministic corrections, fitted GLOBALLY on a real slow-music reference
(never per-clip against the clip's own source -- that would leak information a
deployed pipeline would not have):

  1. BIN MATCHING     per-mel-bin mean/std affine map to the reference marginals
                      (fixes brightness + much of flatness)
  2. DYNAMICS         expand each frame's energy deviation from the clip mean by
                      the reference/generated energy_std ratio
  3. TRANSIENTS       sharpen frames in proportion to their positive flux rank:
                      out = x + alpha * w(t) * (x - median_time(x)), where w(t)
                      is the clip's flux percentile at t. Boosts what onset
                      structure exists; cannot create what is absent -- which is
                      exactly what the z-check below decides.

GATE: after repair, recompute (a) the domain d-prime table and (b) the
onset-evidence z at the labelled beats. Only if z moves meaningfully toward the
real corpus (6.0 vs the current 1.85) is a retraining arm justified.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

BT = "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"
FPS = 50


def flux_curve(sp):
    return np.concatenate([[0.0], np.maximum(np.diff(sp, axis=0), 0).sum(1)])


def fit_reference(datasets=("asap", "hainsworth", "simac", "rwc", "smc"),
                  per_ds=80, max_bpm=70):
    sys.path.insert(0, BT)
    from beat_this.dataset.mmnpz import MemmappedNpzFile
    bins = []
    estd = []
    for ds in datasets:
        z = MemmappedNpzFile(Path(BT) / f"data/audio/spectrograms/{ds}.npz")
        ad = Path(BT) / f"data/annotations/{ds}/annotations/beats"
        for s in sorted({k.split("/")[0] for k in z.files})[:per_ds]:
            fa = ad / f"{s}.beats"
            if not fa.exists():
                continue
            b = np.loadtxt(fa, ndmin=2)
            t = b[:, 0] if b.ndim == 2 else b
            if len(t) < 4 or 60 / np.median(np.diff(t)) >= max_bpm:
                continue
            sp = np.asarray(z[f"{s}/track"], dtype=np.float32)[:1000]
            if sp.shape[0] < 500:
                continue
            bins.append((sp.mean(0), sp.std(0)))
            e = sp.mean(1)
            estd.append(e.std() / (e.mean() + 1e-9))
    mu = np.mean([b[0] for b in bins], 0)
    sd = np.mean([b[1] for b in bins], 0)
    return {"bin_mu": mu, "bin_sd": sd, "energy_std": float(np.mean(estd))}


def repair(sp, ref, alpha=1.0):
    """All three corrections, in order. sp: (T,128) float32 log-mel."""
    out = sp.astype(np.float64)

    # 1. per-bin affine to reference marginals
    mu_g, sd_g = out.mean(0), out.std(0) + 1e-6
    out = (out - mu_g) / sd_g * ref["bin_sd"] + ref["bin_mu"]

    # 2. dynamics expansion toward reference energy_std
    e = out.mean(1)
    cur = e.std() / (e.mean() + 1e-9)
    gain = min(ref["energy_std"] / max(cur, 1e-6), 3.0)
    out = out + (e - e.mean())[:, None] * (gain - 1.0)

    # 3. transient sharpening: contrast boost weighted by flux percentile
    fl = flux_curve(out.astype(np.float32))
    ranks = np.argsort(np.argsort(fl)) / max(len(fl) - 1, 1)      # 0..1
    w = np.clip(ranks - 0.5, 0, None) * 2.0                       # top half only
    smooth = median_filter(out, size=(31, 1))
    out = out + alpha * w[:, None] * (out - smooth)

    return np.clip(out, 0, None).astype(np.float32)


def evidence_z(sp, beats, rng, tol=3, n_null=200):
    f = flux_curve(sp)
    f = (f - np.median(f)) / (f.std() + 1e-9)
    T = len(f)
    idx = np.round(np.asarray(beats) * FPS).astype(int)
    idx = idx[(idx >= tol) & (idx < T - tol)]
    if len(idx) < 3:
        return np.nan
    w = np.arange(-tol, tol + 1)

    def score(off):
        pos = (off[:, None] + w[None, :]).clip(0, T - 1)
        return f[pos].max(1).mean()

    obs = score(idx)
    span = T - 2 * tol
    null = np.array([score(((idx - tol + rng.integers(1, span)) % span) + tol)
                     for _ in range(n_null)])
    return float((obs - null.mean()) / (null.std() + 1e-9))


def dprime_feats(sp):
    f = {}
    fl = flux_curve(sp)
    f["flux_crest"] = fl.max() / (fl.mean() + 1e-9)
    e = sp.mean(1)
    f["energy_std"] = float(e.std() / (e.mean() + 1e-9))
    f["hf_lf_ratio"] = float(sp[:, 96:].mean() / (sp[:, :32].mean() + 1e-9))
    f["spec_flat"] = float(np.exp(np.log(sp + 1e-6).mean()) / (sp.mean() + 1e-9))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="clips to evaluate the gate on")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    sys.path.insert(0, BT)
    from beat_this.dataset.mmnpz import MemmappedNpzFile

    print("  fitting reference on real slow music...", flush=True)
    ref = fit_reference()
    print(f"    reference energy_std {ref['energy_std']:.3f}")

    gen = MemmappedNpzFile(Path(BT) / "data/audio/spectrograms/sdedit.npz")
    ann = Path(BT) / "data/annotations/sdedit/annotations/beats"
    stems = sorted({k.split("/")[0] for k in gen.files})[: a.n]
    rng = np.random.default_rng(0)

    rows = []
    for s in stems:
        sp = np.asarray(gen[f"{s}/track"], dtype=np.float32)
        b = np.loadtxt(ann / f"{s}.beats", ndmin=2)
        beats = b[:, 0] if b.ndim == 2 else b
        rp = repair(sp, ref, alpha=a.alpha)
        rows.append((evidence_z(sp, beats, rng), evidence_z(rp, beats, rng),
                     dprime_feats(sp), dprime_feats(rp)))

    z0 = np.array([r[0] for r in rows])
    z1 = np.array([r[1] for r in rows])
    print(f"\n  === GATE: onset-evidence z at the labelled beats (n={len(rows)}) ===")
    print(f"    before repair   median {np.nanmedian(z0):6.2f}   mean {np.nanmean(z0):6.2f}")
    print(f"    after  repair   median {np.nanmedian(z1):6.2f}   mean {np.nanmean(z1):6.2f}")
    print(f"    real slow music reference:      ~6.0")
    print(f"\n  === feature movement (target = real values) ===")
    tgt = {"flux_crest": 10.80, "energy_std": 0.325, "hf_lf_ratio": 0.248, "spec_flat": 0.581}
    for k in tgt:
        b0 = np.mean([r[2][k] for r in rows])
        b1 = np.mean([r[3][k] for r in rows])
        print(f"    {k:<13} {b0:7.3f} -> {b1:7.3f}   (real {tgt[k]:.3f})")


if __name__ == "__main__":
    main()
