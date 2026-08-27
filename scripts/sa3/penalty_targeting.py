"""Where does the proposed penalty actually LAND? A zero-GPU targeting screen.

The phase-weighted loss up-weights the negative term near subdivision phase. The
question that decides whether it can work is not "does F improve" but "does the
penalty fall on the peaks we want removed, and spare the ones we do not?" That is
answerable statically from cached activations -- no training, no GPU.

Uses cache/ts_tta_cache/<stem>__r1.0.npz, the fold-honest released activations
whose r=1.0 identity reproduces F 0.6270 exactly.

For every track we peak-pick, split the predicted peaks into MATCHED (within 70 ms
of an annotated beat -- these must NOT be penalised) and SPURIOUS, then measure how
much penalty mass each variant places on each group.

VARIANTS
  ungated    c = 1 + g*bump(phi)                     what ph13/ph30 actually ran
  gated      c = 1 + g*a_left*a_right*bump(phi)      Jaehoon's flanking condition:
                                                     only suppress when the model
                                                     already hits BOTH bounding
                                                     beats, so a track with no
                                                     grid is left alone
  *_norm     same, rescaled so the mean negative weight is unchanged. Without this
             gamma raises TOTAL negative pressure as well as redistributing it,
             and extra negative pressure alone produces halving -- a confound in
             every phase-loss number measured so far.

Reported per metrical category (GOOD / PHASE / DOUBLED / HALVED) because the whole
point of the gate is that it should stay SHUT on PHASE tracks, which have no
coherent grid to protect.
"""
import argparse, glob, os
import numpy as np

CACHE = "/home/sogang/jaehoon/Analyze-SMC/cache/ts_tta_cache"
ANN = "/disk1/jaehoon/dataset_store/beat_this_annotations/smc"
FPS = 50
TOL = 0.070


def peak_pick(logit, width=7, thr=0.0):
    n = len(logit)
    pad = width // 2
    x = np.pad(logit, pad, mode="constant", constant_values=-1e9)
    mx = np.max(np.lib.stride_tricks.sliding_window_view(x, width), axis=-1)
    return np.nonzero((logit == mx) & (logit > thr))[0]


def phase_of(frames, beats_f):
    """Phase in [0,1) of each frame within its enclosing annotated interval."""
    ph = np.full(len(frames), np.nan)
    lo = np.full(len(frames), -1)
    if len(beats_f) < 2:
        return ph, lo
    j = np.searchsorted(beats_f, frames, side="right") - 1
    ok = (j >= 0) & (j < len(beats_f) - 1)
    jj = j[ok]
    ibi = beats_f[jj + 1] - beats_f[jj]
    good = ibi > 0
    idx = np.where(ok)[0][good]
    ph[idx] = (frames[idx] - beats_f[jj[good]]) / ibi[good]
    lo[idx] = jj[good]
    return ph, lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=1.3)
    ap.add_argument("--sigma", type=float, default=0.06)
    ap.add_argument("--tolerance", type=int, default=3)
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(f"{CACHE}/*__r1.0.npz")):
        stem = os.path.basename(f).split("__")[0]
        bt = f"{ANN}/annotations/beats/{stem}.beats"
        if not os.path.exists(bt):
            continue
        t = np.loadtxt(bt, ndmin=2)
        if t.size == 0 or len(t) < 8:
            continue
        beats = t[:, 0]
        logit = np.load(f)["logits"]
        n = len(logit)
        bf = np.clip(np.rint(beats * FPS).astype(int), 0, n - 1)
        p = 1.0 / (1.0 + np.exp(-logit))

        pk = peak_pick(logit)
        if len(pk) < 4:
            continue
        est = pk / FPS
        # MATCHED = within tolerance of an annotated beat; must not be penalised
        matched = np.array([np.min(np.abs(beats - e)) <= TOL for e in est])

        ph, lo = phase_of(pk.astype(float), bf.astype(float))
        d = np.abs(ph - 0.5)
        d = np.minimum(d, 1 - d)
        bump = np.where(np.isnan(d), 0.0, np.exp(-np.nan_to_num(d) ** 2
                                                 / (2 * a.sigma ** 2)))
        # gate: model's own confidence AT the two bounding annotated beats
        tolw = a.tolerance
        conf = np.array([p[max(0, b - tolw): b + tolw + 1].max() for b in bf])
        gate = np.where(lo >= 0, conf[np.clip(lo, 0, len(conf) - 2)]
                        * conf[np.clip(lo + 1, 1, len(conf) - 1)], 0.0)

        pen_un = a.gamma * bump
        pen_ga = a.gamma * gate * bump
        ratio = len(est) / len(beats)
        cat = ("GOOD" if 0.7 < ratio <= 1.5 else "DOUBLED") if ratio > 0.7 else "HALVED"
        rows.append(dict(stem=stem, cat=cat, ratio=ratio, n_est=len(est),
                         matched=matched, pen_un=pen_un, pen_ga=pen_ga,
                         gate=gate, bump=bump))

    # F<0.5 split for GOOD vs PHASE needs F; approximate with matched fraction
    for r in rows:
        if r["cat"] == "GOOD" and r["matched"].mean() < 0.55:
            r["cat"] = "PHASE"

    print(f"\n  penalty targeting, gamma={a.gamma}, sigma={a.sigma}, "
          f"n={len(rows)} SMC tracks (fold-honest cached activations)\n")
    print(f"  {'variant':<10}{'on SPURIOUS':>13}{'on MATCHED':>13}"
          f"{'precision':>11}   (precision = share of penalty hitting spurious peaks)")
    for name, key in (("ungated", "pen_un"), ("gated", "pen_ga")):
        sp = sum(r[key][~r["matched"]].sum() for r in rows)
        mt = sum(r[key][r["matched"]].sum() for r in rows)
        print(f"  {name:<10}{sp:13.1f}{mt:13.1f}{sp / max(sp + mt, 1e-9):11.3f}")

    print(f"\n  per category -- the gate should stay SHUT on PHASE (no grid to protect)")
    print(f"  {'category':<10}{'n':>4}{'mean gate':>11}{'ungated pen':>13}{'gated pen':>11}"
          f"{'gated/ungated':>15}")
    for c in ("GOOD", "PHASE", "DOUBLED", "HALVED"):
        rs = [r for r in rows if r["cat"] == c]
        if not rs:
            continue
        g = np.mean([r["gate"][r["bump"] > 0.1].mean() if (r["bump"] > 0.1).any() else 0
                     for r in rs])
        u = np.mean([r["pen_un"].sum() for r in rs])
        gg = np.mean([r["pen_ga"].sum() for r in rs])
        print(f"  {c:<10}{len(rs):4d}{g:11.3f}{u:13.2f}{gg:11.2f}{gg / max(u, 1e-9):15.3f}")

    print(f"\n  NORMALISATION check -- does gamma raise TOTAL negative weight?")
    for name, key in (("ungated", "pen_un"), ("gated", "pen_ga")):
        infl = np.mean([1.0 + r[key].mean() for r in rows])
        print(f"    {name:<10} mean weight on penalised frames = {infl:.4f}x "
              f"(1.0 = pure redistribution)")


if __name__ == "__main__":
    main()
