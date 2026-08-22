"""Is the 0.69 decoder envelope REAL per-track structure, or the winner's curse?

THE PREMISE UNDER TEST. "The SMC Blind Spot" motivates decoder work with a per-track oracle
hyperparameter ceiling of F 0.6908 against 0.6141 for a fixed decoder. That number is a MAXIMUM over
~47 options per track (peak-pick over a threshold grid, DBN over 13 transition_lambda values),
selected with ground truth. Taking a max over K noisy scores inflates the expectation EVEN IF every
option is equally good and the differences are pure noise -- the winner's curse. So the envelope
decomposes into:

    real structure   tracks that genuinely need a different hyperparameter -> LEARNABLE in principle
    noise            options that happened to get lucky on that track     -> LEARNABLE BY NOTHING

Only the first part is a target. Nobody can predict which lambda got lucky, from any feature set of
any dimensionality. This matters far beyond one number: it is the motivation for per-track decoder
adaptation in this project AND for the audio-conditioned per-frame lambda rung of VBPM.

THE TEST. Split each track in half. Choose the oracle hyperparameter on the FIRST half; evaluate it
on the SECOND half. Compare against (a) a globally fixed hyperparameter evaluated on the second half
and (b) the second-half oracle, which is the same inflated quantity the envelope reports.

    if  transferred ~ second-half oracle   -> the gain is real per-track structure, go build a predictor
    if  transferred ~ global fixed         -> the envelope is winner's curse, the target does not exist

This is the honest version of the ceiling: it asks what an ORACLE THAT MUST COMMIT IN ADVANCE gets,
which is the most any learned predictor could ever achieve, since a predictor also cannot see the
half it is scored on.

Halves are scored independently with mir_eval's own trim, so each half is a valid excerpt rather
than a truncation artifact; tracks too short to give both halves >= 8 beats are dropped.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob, peakpick, make_dbn, dbn_beats
from _harness import bthis_downbeat_prob

LAMBDAS = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
THRESH = list(np.round(1 / (1 + np.exp(-np.arange(-4.0, 4.5, 0.5))), 4))
METRICS = ("F", "CMLt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/overnight/oracle_splithalf.json")
    args = ap.parse_args()
    dbns = {L: make_dbn(30.0, 215.0, FPS, L) for L in LAMBDAS}
    print(f"{len(THRESH)} thresholds x {len(LAMBDAS)} lambdas = "
          f"{len(THRESH)+len(LAMBDAS)} options per track", flush=True)

    rows = {}
    for n, tid in enumerate(track_ids()):
        bp = np.asarray(bthis_beat_prob(tid), np.float64)
        dp = np.asarray(bthis_downbeat_prob(tid), np.float64)
        gt = np.asarray(load_gt(tid), float)
        T = len(bp)
        mid_f, mid_t = T // 2, (T // 2) / FPS
        halves = []
        for lo_f, hi_f, lo_t, hi_t in ((0, mid_f, 0.0, mid_t), (mid_f, T, mid_t, 1e9)):
            g = gt[(gt >= lo_t) & (gt < hi_t)] - lo_t
            halves.append((bp[lo_f:hi_f], dp[lo_f:hi_f], g))
        if min(len(h[2]) for h in halves) < 8:
            continue

        # score every option on each half independently
        sc = []
        for bph, dph, g in halves:
            d = {}
            for th in THRESH:
                d[("pp", th)] = ev(g, peakpick(bph, th, FPS))
            for L in LAMBDAS:
                try:
                    d[("dbn", L)] = ev(g, dbn_beats(bph, dph, dbns[L]))
                except Exception:
                    pass
            sc.append(d)
        keys = sorted(set(sc[0]) & set(sc[1]), key=str)
        if not keys:
            continue
        r = {"_h2": {str(o): {k: sc[1][o][k] for k in METRICS} for o in keys}}
        for k in METRICS:
            fixed = ("dbn", 100)
            best_h1 = max(keys, key=lambda o: sc[0][o][k])
            r[k] = {
                "dbn100": sc[1][fixed][k] if fixed in sc[1] else float("nan"),
                "transferred": sc[1][best_h1][k],          # chosen on h1, scored on h2
                "oracle_h2": max(sc[1][o][k] for o in keys),  # the inflated envelope quantity
            }
        rows[tid] = r
        if (n + 1) % 40 == 0:
            print(f"  {n+1}/217", flush=True)

    ids = sorted(rows)
    # THE BASELINE THAT MATTERS: the best SINGLE option applied to every track -- what a
    # practitioner actually ships. Comparing against dbn(lambda=100) alone understates the
    # baseline, because peak-picking outscores the DBN on SMC, so the "gain" would partly be
    # "we stopped using the worse of the two global options".
    allkeys = sorted(set.intersection(*[set(rows[t]["_h2"]) for t in ids]))
    best_global = {k: max(allkeys, key=lambda o: np.mean([rows[t]["_h2"][o][k] for t in ids]))
                   for k in METRICS}
    print(f"\nn={len(ids)} tracks with two scorable halves")
    for k in METRICS:
        print(f"  best GLOBAL option for {k}: {best_global[k]}")
    print(f"\n{'metric':<7}{'dbn100':>9}{'bestGLOBAL':>12}{'TRANSFERRED':>13}{'oracle(h2)':>12}"
          f"{'gain vs best':>14}{'p':>9}{'% of oracle':>13}")
    out = {}
    for k in METRICS:
        gf = np.array([rows[t][k]["dbn100"] for t in ids])
        bg = np.array([rows[t]["_h2"][best_global[k]][k] for t in ids])
        tr = np.array([rows[t][k]["transferred"] for t in ids])
        oc = np.array([rows[t][k]["oracle_h2"] for t in ids])
        d = tr - bg
        p = wilcoxon(d).pvalue if np.any(d) else 1.0
        frac = (tr.mean() - bg.mean()) / max(oc.mean() - bg.mean(), 1e-9)
        print(f"{k:<7}{gf.mean():>9.4f}{bg.mean():>12.4f}{tr.mean():>13.4f}{oc.mean():>12.4f}"
              f"{d.mean():>+14.4f}{p:>9.2g}{100*frac:>12.1f}%")
        out[k] = {"dbn100": float(gf.mean()), "best_global": float(bg.mean()),
                  "best_global_opt": str(best_global[k]),
                  "transferred": float(tr.mean()), "oracle_h2": float(oc.mean()),
                  "gain_vs_best_global": float(d.mean()), "p": float(p),
                  "frac_of_oracle": float(frac)}
    print("\n'% of oracle' is the share of the apparent per-track ceiling that survives having to")
    print("commit in advance -- an upper bound on what ANY learned predictor could recover.")
    json.dump({"n": len(ids), "summary": out, "per_track": rows}, open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
