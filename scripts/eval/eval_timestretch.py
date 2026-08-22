"""
Evaluate time-stretch TTA from the cached un-warped activations.
  Phase A: single-rate F + per-tempo-band breakdown (validates rate=1.0 == 0.627)
  Phase B: rate-combination TTA
  Phase C: stack with pitch-shift TTA + Beat Transformer fusion
"""
import sys, numpy as np
sys.path.insert(0, "scripts")
from _harness import (track_ids, load_gt, gt_bpm, bthis_beat_prob, btrans_beat_prob,
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

ROOT = Path("/home/sogang/jaehoon/Analyze-SMC")
CACHE = ROOT / "data" / "ts_tta_cache"
TTA_PITCH = ROOT / "data" / "tta_activations"
RATES = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]

def load_rate(tid, r):
    return np.load(CACHE / f"{tid}__r{r}.npz")["logits"]

def sweep_global(prob, refs, ids):
    best = (0.5, -1)
    for th in np.round(np.arange(0.1, 0.9, 0.025), 4):
        m = np.mean([fastF(refs[t], peakpick(prob[t], th, FPS)) for t in ids])
        if m > best[1]: best = (th, m)
    return best

def main():
    ids = track_ids()
    refs = {t: load_gt(t) for t in ids}
    bpm = {t: gt_bpm(refs[t]) for t in ids}
    band = {t: ("<55" if bpm[t] < 55 else ("55-90" if bpm[t] < 90 else ">90")) for t in ids}
    counts = {b: sum(1 for t in ids if band[t] == b) for b in ["<55", "55-90", ">90"]}

    # ---------- Phase A: single-rate ----------
    print("== Phase A: single stretch rate (un-warped), peak-pick @p=0.5 ==")
    print(f"   tempo bands: <55={counts['<55']}  55-90={counts['55-90']}  >90={counts['>90']}")
    print(f"   {'rate':>5s} {'F_all':>7s} {'F<55':>7s} {'F55-90':>7s} {'F>90':>7s}")
    rate_prob = {}
    for r in RATES:
        P = {t: expit(load_rate(t, r)) for t in ids}
        rate_prob[r] = P
        per = {t: fastF(refs[t], peakpick(P[t], 0.5, FPS)) for t in ids}
        def bmean(bnd):
            v = [per[t] for t in ids if band[t] == bnd]
            return np.mean(v) if v else float("nan")
        tag = "  <- sanity (==0.627)" if r == 1.0 else ""
        print(f"   {r:>5.2f} {np.mean(list(per.values())):>7.4f} {bmean('<55'):>7.4f} "
              f"{bmean('55-90'):>7.4f} {bmean('>90'):>7.4f}{tag}", flush=True)

    # sanity assert
    base = {t: bthis_beat_prob(t) for t in ids}
    f_base = np.mean([fastF(refs[t], peakpick(base[t], 0.5, FPS)) for t in ids])
    f_r1 = np.mean([fastF(refs[t], peakpick(rate_prob[1.0][t], 0.5, FPS)) for t in ids])
    print(f"   [check] cached-baseline F={f_base:.4f}  rate1.0 F={f_r1:.4f}  |diff|={abs(f_base-f_r1):.4f}")

    # ---------- Phase B: rate combinations (average logits across rates) ----------
    print("\n== Phase B: multi-rate TTA (mean of un-warped logits) ==")
    def combine(rate_list):
        out = {}
        for t in ids:
            L = min(len(load_rate(t, r)) for r in rate_list)
            out[t] = expit(np.mean([load_rate(t, r)[:L] for r in rate_list], axis=0))
        return out
    combos = {
        "all 7 rates":            RATES,
        "sym {0.7,0.85,1,1.2,1.5}": [0.7, 0.85, 1.0, 1.2, 1.5],
        "speedup {1,1.2,1.5,2}":  [1.0, 1.2, 1.5, 2.0],
        "mild {0.85,1,1.2}":      [0.85, 1.0, 1.2],
    }
    print(f"   {'combo':28s} {'gThr':>5s} {'F_glob':>7s} {'F@0.5':>7s} {'F_oracle':>8s} {'CMLt':>6s} {'AMLt':>6s}")
    fine = expit(np.arange(-4.0, 4.5, 0.25))
    ts_best = None
    for name, rl in combos.items():
        P = combine(rl)
        th, fg = sweep_global(P, refs, ids)
        sc = [ev(refs[t], peakpick(P[t], 0.5, FPS)) for t in ids]
        m05 = mean_scores(sc)
        orc = np.mean([max(fastF(refs[t], peakpick(P[t], x, FPS)) for x in fine) for t in ids])
        print(f"   {name:28s} {th:>5.2f} {fg:>7.4f} {m05['F']:>7.4f} {orc:>8.4f} {m05['CMLt']:>6.4f} {m05['AMLt']:>6.4f}", flush=True)
        if name == "sym {0.7,0.85,1,1.2,1.5}":
            ts_best = P

    # ---------- Phase C: stack with pitch-shift TTA + Beat Transformer ----------
    print("\n== Phase C: stacking (peak-pick) ==")
    def nm(x):
        lo, hi = x.min(), x.max(); return (x - lo) / (hi - lo + 1e-9)
    pitch = {t: expit(np.load(TTA_PITCH / f"{t}.npz")["avg_logits"]) for t in ids} if TTA_PITCH.exists() else None
    btr = {t: resample_to(btrans_beat_prob(t), BT_FPS, FPS, len(base[t])) for t in ids}

    def report(name, P):
        # align lengths
        th, fg = sweep_global(P, refs, ids)
        sc = [ev(refs[t], peakpick(P[t], 0.5, FPS)) for t in ids]
        m = mean_scores(sc)
        orc = np.mean([max(fastF(refs[t], peakpick(P[t], x, FPS)) for x in fine) for t in ids])
        print(f"   {name:34s} gThr={th:.2f} F_glob={fg:.4f} F@.5={m['F']:.4f} "
              f"CMLt={m['CMLt']:.4f} AMLt={m['AMLt']:.4f} oracle={orc:.4f}", flush=True)

    report("Beat This base", base)
    if pitch is not None:
        report("pitch-shift TTA", pitch)
    report("time-stretch TTA (sym5)", ts_best)
    if pitch is not None:
        both = {t: 0.5 * (np.log(ts_best[t] / (1 - ts_best[t]) + 1e-12)
                          + np.log(pitch[t] / (1 - pitch[t]) + 1e-12)) for t in ids}
        both = {t: expit(both[t]) for t in ids}
        report("pitch + time-stretch TTA", both)
        L = {t: min(len(both[t]), len(btr[t])) for t in ids}
        fus = {t: 0.6 * nm(both[t][:L[t]]) + 0.4 * nm(btr[t][:L[t]]) for t in ids}
        report("pitch+TS TTA + BTrans fusion", fus)

if __name__ == "__main__":
    main()
