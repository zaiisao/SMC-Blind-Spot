"""Did the wide (slow-biased) tempo augmentation help where it should?

wide_ph13 draws its stretch uniformly from -60..+20 (mean -20%), narrow ph13 from
-20..+20 (mean 0). Both cold start, same corpus, seed 0, ungated phase loss
gamma=1.3. Wide therefore trains on genuinely slower audio -- corpus median moves
92.7 -> ~74 BPM, closer to SMC's 71.4 -- yet loses ~0.017 F overall.

If the slow coverage worked as intended, wide should WIN on the slowest SMC
tracks and lose elsewhere. If it loses everywhere, the extra slow data is not
buying tempo coverage at all and the cost is something else (dilution across 21
variants instead of 11, or artifacts from stretching audio 2.5x).

Scores both checkpoints per track and splits SMC fold 0 by annotated tempo.
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smc_eval import score_fold, fold_songs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--narrow", required=True)
    ap.add_argument("--wide", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"

    bpm = {stem: 60.0 / np.median(np.diff(truth))
           for stem, _, truth in fold_songs(a.fold) if len(truth) > 2}
    n = score_fold(a.narrow, a.fold, dev)
    w = score_fold(a.wide, a.fold, dev)

    stems = [s for s in sorted(set(n) & set(w)) if s in bpm]
    b = np.array([bpm[s] for s in stems])
    dn = np.array([n[s]["F"] for s in stems])
    dw = np.array([w[s]["F"] for s in stems])
    d = dw - dn

    print(f"\n  wide - narrow, per track, SMC fold {a.fold}  (n={len(stems)})")
    print(f"  narrow mean F {dn.mean():.4f}   wide mean F {dw.mean():.4f}   "
          f"diff {d.mean():+.4f}\n")

    q = np.quantile(b, [1 / 3, 2 / 3])
    groups = [("slowest", b <= q[0]), ("middle", (b > q[0]) & (b <= q[1])),
              ("fastest", b > q[1])]
    print(f"  {'tempo group':<12}{'n':>3}{'BPM range':>14}{'narrow':>9}{'wide':>9}"
          f"{'diff':>9}{'wide wins':>11}")
    for name, m in groups:
        if not m.any():
            continue
        print(f"  {name:<12}{m.sum():3d}{f'{b[m].min():.0f}-{b[m].max():.0f}':>14}"
              f"{dn[m].mean():9.4f}{dw[m].mean():9.4f}{d[m].mean():+9.4f}"
              f"{f'{(d[m] > 0).sum()}/{m.sum()}':>11}")

    r = np.corrcoef(b, d)[0, 1]
    print(f"\n  correlation(tempo, wide-narrow gain) = {r:+.3f}")
    print("  a NEGATIVE correlation means wide helps most on the SLOWEST tracks,")
    print("  which is what the slow-biased augmentation was built to do.")

    print(f"\n  biggest wide gains:")
    for i in np.argsort(-d)[:5]:
        print(f"    {stems[i]:<12} {b[i]:6.1f} BPM  {dn[i]:.3f} -> {dw[i]:.3f}  {d[i]:+.3f}")
    print(f"  biggest wide losses:")
    for i in np.argsort(d)[:5]:
        print(f"    {stems[i]:<12} {b[i]:6.1f} BPM  {dn[i]:.3f} -> {dw[i]:.3f}  {d[i]:+.3f}")


if __name__ == "__main__":
    main()
