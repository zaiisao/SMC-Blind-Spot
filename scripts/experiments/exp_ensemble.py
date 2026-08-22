"""
EXPERIMENT: activation-level ensembling of Beat This + Beat Transformer.

The paper computes its 'activation ceiling' (~F=0.67) on Beat This activations
ALONE. This tests whether fusing two independently-trained models' activation
functions moves confident-but-wrong peaks toward correct positions, breaking
that single-model ceiling -- a pure inference-time fix requiring no retraining.

All rows use the identical eval: mir_eval 5 s trim + 0.07 s tolerance.
"""
import sys, numpy as np, scipy.special
sys.path.insert(0, "scripts")
from _harness import (track_ids, load_gt, bthis_beat_prob, btrans_beat_prob,
                      resample_to, peakpick, FPS, BT_FPS)
import mir_eval

TRIM = mir_eval.beat.trim_beats

def fastF(ref, est):
    if len(est) == 0 or len(ref) == 0:
        return 0.0
    r = TRIM(ref); e = TRIM(est)
    if len(r) == 0 or len(e) == 0:
        return 0.0
    return mir_eval.beat.f_measure(r, e)

def main():
    ids = track_ids()
    refs = {t: load_gt(t) for t in ids}
    A, B = {}, {}
    for t in ids:
        A[t] = bthis_beat_prob(t)
        B[t] = resample_to(btrans_beat_prob(t), BT_FPS, FPS, len(A[t]))

    def nm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-9)

    combos = {
        "beatthis_only":  lambda a, b: a,
        "beattrans_only": lambda a, b: b,
        "mean":           lambda a, b: 0.5 * (a + b),
        "max":            lambda a, b: np.maximum(a, b),
        "geomean":        lambda a, b: np.sqrt(np.clip(a, 1e-9, 1) * np.clip(b, 1e-9, 1)),
        "norm_mean":      lambda a, b: 0.5 * (nm(a) + nm(b)),
        "norm_geomean":   lambda a, b: np.sqrt(nm(a) * nm(b)),
        "w0.7this":       lambda a, b: 0.7 * a + 0.3 * b,
        "w0.6this":       lambda a, b: 0.6 * a + 0.4 * b,
        "norm_w0.6this":  lambda a, b: 0.6 * nm(a) + 0.4 * nm(b),
        "norm_w0.7this":  lambda a, b: 0.7 * nm(a) + 0.3 * nm(b),
    }

    th_grid = np.round(np.arange(0.05, 0.901, 0.025), 4)
    fine = scipy.special.expit(np.arange(-4.0, 4.5, 0.25))

    print(f"{'combo':15s} {'gThr':>6s} {'F_glob':>7s} {'F_oracle':>8s}")
    print("-" * 42)
    for name, fn in combos.items():
        comb = {t: fn(A[t], B[t]) for t in ids}
        # cache peaks per (track, threshold) is heavy; sweep directly with fastF
        best = (0.0, -1.0)
        for th in th_grid:
            m = np.mean([fastF(refs[t], peakpick(comb[t], th, FPS)) for t in ids])
            if m > best[1]:
                best = (th, m)
        # oracle per-track threshold
        orac = []
        for t in ids:
            ref = refs[t]; p = comb[t]
            orac.append(max(fastF(ref, peakpick(p, th2, FPS)) for th2 in fine))
        print(f"{name:15s} {best[0]:>6.3f} {best[1]:>7.4f} {np.mean(orac):>8.4f}", flush=True)

if __name__ == "__main__":
    main()
