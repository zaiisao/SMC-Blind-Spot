"""
Fully DEPLOYABLE evaluation of TTA activations -- NO ground-truth info anywhere.
No GT tempo, no per-track oracle threshold/lambda. Only:
  - native peak-picking (p=0.5, and one global threshold tuned on the set)
  - standard wide DBN [30,215], library-default lambda=100, and one global-best lambda
  - Beat-Transformer-PREDICTED-tempo DBN (its own tempo head, 59% acc, no GT)
    at +/-20%, lambda=100 and one global-best lambda

Run one activation set per process:  eval_tta_realistic.py --set ts
sets: raw | pitch | ts | both
"""
import argparse, sys, numpy as np
sys.path.insert(0, "scripts")
from _harness import (track_ids, load_gt, bthis_beat_prob, bthis_downbeat_prob,
                      btrans_tempo, peakpick, make_dbn, dbn_beats, ev, mean_scores, FPS)
from scipy.special import expit
from pathlib import Path
import mir_eval
TRIM = mir_eval.beat.trim_beats
def fastF(ref, est):
    if len(est) == 0 or len(ref) == 0: return 0.0
    r = TRIM(ref); e = TRIM(est)
    if len(r) == 0 or len(e) == 0: return 0.0
    return mir_eval.beat.f_measure(r, e)

FULL = Path("/home/sogang/jaehoon/Analyze-SMC/cache/tta_full")
LAMBDAS = [1, 2, 5, 10, 20, 30, 50, 100]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); a = ap.parse_args()
    ids = track_ids(); refs = {t: load_gt(t) for t in ids}

    def act(t):
        if a.set == "raw":
            return bthis_beat_prob(t), bthis_downbeat_prob(t)
        f = np.load(FULL / f"{t}.npz")
        if a.set == "pitch":
            return expit(f["pitch_beat"]), expit(f["pitch_db"])
        if a.set == "ts":
            return expit(f["ts_beat"]), expit(f["ts_db"])
        L = min(len(f["pitch_beat"]), len(f["ts_beat"]))
        return expit(0.5 * (f["pitch_beat"][:L] + f["ts_beat"][:L])), \
               expit(0.5 * (f["pitch_db"][:L] + f["ts_db"][:L]))
    A = {t: act(t) for t in ids}
    btempo = {t: btrans_tempo(t)[0] for t in ids}

    out = []  # (decoder_label, F, CMLt, AMLt)

    # peak @0.5
    s = mean_scores([ev(refs[t], peakpick(A[t][0], 0.5, FPS)) for t in ids])
    out.append(("peak-pick @0.5 (untuned)", s["F"], s["CMLt"], s["AMLt"]))
    # peak best global threshold
    bestth = (0.5, -1)
    for th in np.round(np.arange(0.1, 0.9, 0.025), 4):
        m = np.mean([fastF(refs[t], peakpick(A[t][0], th, FPS)) for t in ids])
        if m > bestth[1]: bestth = (th, m)
    s = mean_scores([ev(refs[t], peakpick(A[t][0], bestth[0], FPS)) for t in ids])
    out.append((f"peak-pick @{bestth[0]:.3f} (global)", s["F"], s["CMLt"], s["AMLt"]))

    # wide DBN [30,215]: lambda=100 and global-best lambda
    perlam = {}
    for lam in LAMBDAS:
        d = make_dbn(30, 215, FPS, lam)
        perlam[lam] = [ev(refs[t], dbn_beats(A[t][0], A[t][1], d)) for t in ids]
    m100 = mean_scores(perlam[100])
    out.append(("wide DBN, lambda=100 (default)", m100["F"], m100["CMLt"], m100["AMLt"]))
    bl = max(LAMBDAS, key=lambda L: mean_scores(perlam[L])["F"])
    mb = mean_scores(perlam[bl])
    out.append((f"wide DBN, global-best lambda={bl}", mb["F"], mb["CMLt"], mb["AMLt"]))

    # BT-predicted-tempo DBN +/-20%: lambda=100 and global-best lambda
    perlam_bt = {}
    for lam in LAMBDAS:
        sc = []
        for t in ids:
            d = make_dbn(btempo[t] * 0.8, btempo[t] * 1.2, FPS, lam)
            sc.append(ev(refs[t], dbn_beats(A[t][0], A[t][1], d)))
        perlam_bt[lam] = sc
    mb100 = mean_scores(perlam_bt[100])
    out.append(("BT-pred-tempo DBN, lambda=100", mb100["F"], mb100["CMLt"], mb100["AMLt"]))
    blb = max(LAMBDAS, key=lambda L: mean_scores(perlam_bt[L])["F"])
    mbb = mean_scores(perlam_bt[blb])
    out.append((f"BT-pred-tempo DBN, global-best lambda={blb}", mbb["F"], mbb["CMLt"], mbb["AMLt"]))

    print(f"### activation set = {a.set}")
    for lbl, F, C, Aa in out:
        print(f"{a.set:6s} | {lbl:38s} F={F:.4f} CMLt={C:.4f} AMLt={Aa:.4f}", flush=True)

if __name__ == "__main__":
    main()
