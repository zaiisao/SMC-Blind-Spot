"""
EXPERIMENT 4: stack both inference-time levers.
  TTA(Beat This) alone  vs  TTA(Beat This) + Beat Transformer fusion.
Reads cached TTA logits from data/tta_activations/ (produced by exp_tta.py).
All eval identical: mir_eval 5 s trim, +/-70 ms.
"""
import sys, numpy as np, scipy.special
sys.path.insert(0, "scripts")
from _harness import (track_ids, load_gt, bthis_beat_prob, btrans_beat_prob,
                      resample_to, peakpick, ev, mean_scores, FPS, BT_FPS)
from scipy.special import expit
from pathlib import Path
import mir_eval
TRIM = mir_eval.beat.trim_beats
def fastF(ref, est):
    if len(est) == 0 or len(ref) == 0: return 0.0
    r = TRIM(ref); e = TRIM(est)
    if len(r) == 0 or len(e) == 0: return 0.0
    return mir_eval.beat.f_measure(r, e)

TTA = Path("/home/sogang/jaehoon/Analyze-SMC/data/tta_activations")

def sweep(prob, refs, ids):
    best = (0.5, -1)
    for th in np.round(np.arange(0.1, 0.9, 0.025), 4):
        m = np.mean([fastF(refs[t], peakpick(prob[t], th, FPS)) for t in ids])
        if m > best[1]: best = (th, m)
    return best

def main():
    ids = track_ids(); refs = {t: load_gt(t) for t in ids}
    fine = expit(np.arange(-4.0, 4.5, 0.25))
    def nm(x):
        lo, hi = x.min(), x.max(); return (x - lo) / (hi - lo + 1e-9)

    tta = {t: expit(np.load(TTA / f"{t}.npz")["avg_logits"]) for t in ids}
    base = {t: bthis_beat_prob(t) for t in ids}
    btr = {t: resample_to(btrans_beat_prob(t), BT_FPS, FPS, len(base[t])) for t in ids}
    # match lengths for tta vs base (tta may be slightly shorter due to min-len crop)
    for t in ids:
        L = min(len(tta[t]), len(base[t]), len(btr[t]))
        tta[t], base[t], btr[t] = tta[t][:L], base[t][:L], btr[t][:L]

    configs = {
        "BeatThis base":          base,
        "TTA(BeatThis)":          tta,
        "BeatThis+BTrans mean":   {t: 0.5*(base[t]+btr[t]) for t in ids},
        "TTA+BTrans mean":        {t: 0.5*(tta[t]+btr[t]) for t in ids},
        "TTA+BTrans norm w.6":    {t: 0.6*nm(tta[t])+0.4*nm(btr[t]) for t in ids},
    }
    print(f"{'config':24s} {'gThr':>5s} {'F_glob':>7s} {'F@0.5':>7s} {'F_oracle':>8s}")
    print("-"*56)
    for name, P in configs.items():
        th, fg = sweep(P, refs, ids)
        f05 = np.mean([fastF(refs[t], peakpick(P[t], 0.5, FPS)) for t in ids])
        orc = np.mean([max(fastF(refs[t], peakpick(P[t], x, FPS)) for x in fine) for t in ids])
        print(f"{name:24s} {th:>5.2f} {fg:>7.4f} {f05:>7.4f} {orc:>8.4f}", flush=True)

if __name__ == "__main__":
    main()
