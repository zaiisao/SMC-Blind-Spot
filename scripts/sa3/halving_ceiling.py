"""Perfect-learning ceiling for the HALVING term, on the 217-track cache.

The phase penalty can only remove beats, so it cannot touch the 28 HALVED tracks
(mean F 0.4797). The proposed mirror term boosts a TRUE beat that the model missed
while confidently hitting the beats on either side:

    h_i = a_{i-1} * a_{i+1} * (1 - a_i)        detached

This measures the best that term could ever do. For each annotated beat the decoder
missed, if its two neighbours WERE found (both within 70 ms), insert it and
re-score. That assumes perfect learning and zero collateral damage, so it is a
strict upper bound -- the same role the phi=0.5 deletion oracle played for
doubling (+0.020 at w=0.05, +0.055 at w=0.20).

Two variants, because the honest question is what evidence the term can rely on:
  ORACLE      insert every missed beat whose neighbours were hit
  GATED       insert only where the model's own confidence pattern flags it,
              i.e. h_i above a threshold -- what the term could plausibly learn
Also reports the doubling oracle and BOTH together, since the pair is the actual
proposal and the two act on disjoint frames.
"""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts/sa3")
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM/rungs_lab")
CACHE = "/home/sogang/jaehoon/Analyze-SMC/cache/ts_tta_cache"
ANN = "/disk1/jaehoon/dataset_store/beat_this_annotations/smc/annotations/beats"
FPS, TOL = 50, 0.070


def peak_pick(lg, thr=0.0, width=7):
    pad = width // 2
    x = np.pad(lg, pad, mode="constant", constant_values=-1e9)
    mx = np.max(np.lib.stride_tricks.sliding_window_view(x, width), axis=-1)
    return np.nonzero((lg == mx) & (lg > thr))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hthr", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--wdel", type=float, default=0.05)
    a = ap.parse_args()
    from training.evaluation import beat_f_measure

    T = []
    for f in sorted(glob.glob(f"{CACHE}/*__r1.0.npz")):
        stem = os.path.basename(f).split("__")[0]
        bp = f"{ANN}/{stem}.beats"
        if not os.path.exists(bp):
            continue
        truth = np.loadtxt(bp, ndmin=2)[:, 0]
        if len(truth) < 8:
            continue
        lg = np.load(f)["logits"]
        n = len(lg)
        p = 1 / (1 + np.exp(-lg))
        bf = np.clip(np.rint(truth * FPS).astype(int), 0, n - 1)
        est = peak_pick(lg) / FPS
        if len(est) < 4:
            continue
        hit = np.array([np.min(np.abs(est - b)) <= TOL for b in truth])   # per TRUE beat
        aconf = np.array([p[max(0, b - 3):b + 4].max() for b in bf])
        lo, hi = truth.min(), truth.max()
        r = ((est >= lo) & (est <= hi)).sum() / len(truth)
        T.append(dict(stem=stem, truth=truth, est=est, hit=hit, a=aconf, r=r,
                      lg=lg, bf=bf, F=float(beat_f_measure(truth, est))))

    base = float(np.mean([t["F"] for t in T]))
    r = np.array([t["r"] for t in T])
    hal, dbl = r < 0.7, r > 1.5
    print(f"\n  {len(T)} tracks   baseline F {base:.4f}")
    print(f"  HALVED {int(hal.sum())} (F {np.mean([t['F'] for t,m in zip(T,hal) if m]):.4f})"
          f"   DOUBLED {int(dbl.sum())}\n")

    # how many missed beats are actually "flanked by hits"?
    tot_missed = sum(int((~t["hit"]).sum()) for t in T)
    flanked = 0
    for t in T:
        h = t["hit"]
        for i in range(1, len(h) - 1):
            if not h[i] and h[i - 1] and h[i + 1]:
                flanked += 1
    print(f"  missed true beats: {tot_missed}   of which flanked by two hits: "
          f"{flanked} ({flanked/max(tot_missed,1):.1%})\n")

    def rescore(insert_fn, delete_w=None):
        Fs, Fh = [], []
        for t in T:
            est = t["est"]
            add = insert_fn(t)
            if len(add):
                est = np.sort(np.concatenate([est, add]))
            if delete_w is not None:
                keep = np.ones(len(est), bool)
                bfl = t["bf"].astype(float)
                fr = est * FPS
                j = np.searchsorted(bfl, fr, side="right") - 1
                ok = (j >= 0) & (j < len(bfl) - 1)
                jj = np.clip(j, 0, len(bfl) - 2)
                ibi = bfl[jj + 1] - bfl[jj]
                ph = np.where(ok & (ibi > 0), (fr - bfl[jj]) / np.maximum(ibi, 1), np.nan)
                d = np.abs(ph - 0.5); d = np.minimum(d, 1 - d)
                matched = np.array([np.min(np.abs(t["truth"] - e)) <= TOL for e in est])
                keep = ~((d < delete_w) & ~matched)
                keep |= np.isnan(d)
                est = est[keep]
            f = float(beat_f_measure(t["truth"], est))
            Fs.append(f)
            if t["r"] < 0.7:
                Fh.append(f)
        return float(np.mean(Fs)), float(np.mean(Fh))

    def flanked_missed(t, thr):
        h, aa, out = t["hit"], t["a"], []
        for i in range(1, len(h) - 1):
            if h[i] or not (h[i - 1] and h[i + 1]):
                continue
            hv = aa[i - 1] * aa[i + 1] * (1 - aa[i])
            if hv >= thr:
                out.append(t["truth"][i])
        return np.array(out)

    print(f"  {'variant':<34}{'F all':>9}{'gain':>9}{'F halved':>11}{'gain':>9}")
    bh = float(np.mean([t["F"] for t in T if t["r"] < 0.7]))
    for thr in a.hthr:
        F, Fh = rescore(lambda t, s=thr: flanked_missed(t, s))
        lab = "insert flanked-missed" + (f", h>={thr}" if thr else " (pure oracle)")
        print(f"  {lab:<34}{F:9.4f}{F-base:+9.4f}{Fh:11.4f}{Fh-bh:+9.4f}")

    Fd, Fdh = rescore(lambda t: np.array([]), delete_w=a.wdel)
    print(f"  {'delete phi=0.5 (doubling only)':<34}{Fd:9.4f}{Fd-base:+9.4f}"
          f"{Fdh:11.4f}{Fdh-bh:+9.4f}")
    Fb, Fbh = rescore(lambda t: flanked_missed(t, 0.0), delete_w=a.wdel)
    print(f"  {'BOTH terms together':<34}{Fb:9.4f}{Fb-base:+9.4f}{Fbh:11.4f}{Fbh-bh:+9.4f}")


if __name__ == "__main__":
    main()
