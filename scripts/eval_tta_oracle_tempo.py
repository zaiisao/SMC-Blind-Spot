"""
TTA activations x oracle (GT) tempo.

For each activation set {raw BeatThis, pitch-TTA, time-stretch-TTA, pitch+TS},
decode with:
  - peak-pick @0.5                         (reference; isolates activation/F)
  - GT-tempo DBN (+/-20%, floor 30), lambda=100
  - GT-tempo DBN, per-track optimal lambda (sweep 13)
and report F / CMLt / AMLt.

Sanity: raw BeatThis + GT-tempo + optimal-lambda must reproduce the paper's
0.667 F / 0.735 CMLt (matches data/gt_tempo_lambda_sweep_results.csv).
"""
import sys, numpy as np
sys.path.insert(0, "scripts")
from _harness import (track_ids, load_gt, gt_bpm, bthis_beat_prob, bthis_downbeat_prob,
                      peakpick, make_dbn, dbn_beats, ev, mean_scores, FPS)
from scipy.special import expit
from pathlib import Path

FULL = Path("/home/sogang/jaehoon/Analyze-SMC/data/tta_full")
LAMBDAS = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]

def main():
    ids = track_ids()
    refs = {t: load_gt(t) for t in ids}
    bpm = {t: gt_bpm(refs[t]) for t in ids}

    # activation sets: (beat_prob, downbeat_prob) per track
    def load_full(t):
        return np.load(FULL / f"{t}.npz")
    raw = {t: (bthis_beat_prob(t), bthis_downbeat_prob(t)) for t in ids}
    pitch = {t: (expit(load_full(t)["pitch_beat"]), expit(load_full(t)["pitch_db"])) for t in ids}
    ts = {t: (expit(load_full(t)["ts_beat"]), expit(load_full(t)["ts_db"])) for t in ids}
    def comb(t):
        f = load_full(t)
        L = min(len(f["pitch_beat"]), len(f["ts_beat"]))
        b = expit(0.5 * (f["pitch_beat"][:L] + f["ts_beat"][:L]))
        d = expit(0.5 * (f["pitch_db"][:L] + f["ts_db"][:L]))
        return b, d
    both = {t: comb(t) for t in ids}
    SETS = {"raw BeatThis": raw, "pitch TTA": pitch, "time-stretch TTA": ts, "pitch+TS TTA": both}

    # precompute GT-tempo DBNs per (track, lambda) lazily
    def gt_dbn(t, lam):
        mn, mx = bpm[t] * 0.8, bpm[t] * 1.2
        return make_dbn(mn, mx, FPS, lam)

    print(f"{'activation set':18s} {'decoder':26s} {'F':>7s} {'CMLt':>7s} {'AMLt':>7s}")
    print("-" * 70)
    for sname, S in SETS.items():
        # peak-pick @0.5
        pk = [ev(refs[t], peakpick(S[t][0], 0.5, FPS)) for t in ids]
        m = mean_scores(pk)
        print(f"{sname:18s} {'peak-pick @0.5':26s} {m['F']:>7.4f} {m['CMLt']:>7.4f} {m['AMLt']:>7.4f}", flush=True)
        # GT-tempo DBN lambda=100
        d100 = [ev(refs[t], dbn_beats(S[t][0], S[t][1], gt_dbn(t, 100))) for t in ids]
        m = mean_scores(d100)
        print(f"{sname:18s} {'GT-tempo DBN l=100':26s} {m['F']:>7.4f} {m['CMLt']:>7.4f} {m['AMLt']:>7.4f}", flush=True)
        # GT-tempo DBN per-track optimal lambda (by F)
        optF, optC, optA = [], [], []
        for t in ids:
            best = None
            for lam in LAMBDAS:
                e = ev(refs[t], dbn_beats(S[t][0], S[t][1], gt_dbn(t, lam)))
                if best is None or e["F"] > best["F"]:
                    best = e
            optF.append(best["F"]); optC.append(best["CMLt"]); optA.append(best["AMLt"])
        print(f"{sname:18s} {'GT-tempo DBN opt-lambda':26s} {np.mean(optF):>7.4f} "
              f"{np.mean(optC):>7.4f} {np.mean(optA):>7.4f}", flush=True)
        print("-" * 70)

if __name__ == "__main__":
    main()
