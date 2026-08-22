"""AUDIT of §4.10: sweep TTA rate subsets on BOTH decoders (peak-pick and beat-only DBN).

Addresses (b) which rate subset, and (d) TTA on both decoders in ONE script so the
"shrinkage on the correct decoder" claim is supported by matched rows.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, peakpick
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events

CACHE = ROOT / "data" / "ts_tta_cache"
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"

SUBSETS = {
    "null r1.0":       [1.0],
    "mild .85/1/1.2":  [0.85, 1.0, 1.2],
    "sym5":            [0.7, 0.85, 1.0, 1.2, 1.5],
    "all7":            [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0],
    "slow .5/.7/.85/1": [0.5, 0.7, 0.85, 1.0],
    "fast 1/1.2/1.5/2": [1.0, 1.2, 1.5, 2.0],
    "wide .7/1/1.5":   [0.7, 1.0, 1.5],
    "widest .5/1/2":   [0.5, 1.0, 2.0],
    "no1  .85/1.2":    [0.85, 1.2],
}


def logits(tid, r):
    return np.asarray(np.load(CACHE / f"{tid}__r{r}.npz")["logits"], dtype=np.float64)


def main():
    m = make_beatonly_dbn(30.0, 16, 100, DEV)
    space = m.state_spaces[0]
    ids = [t for t in track_ids() if (CACHE / f"{t}__r1.0.npz").exists()]
    gts = {t: load_gt(t) for t in ids}
    print(f"n={len(ids)}", flush=True)

    res_dbn, res_pp = {}, {}
    for name, rl in SUBSETS.items():
        P = {t: expit(np.mean([logits(t, r) for r in rl], 0)) for t in ids}
        res_pp[name] = {t: ev(gts[t], peakpick(P[t], 0.5, FPS)) for t in ids}
        d = {}
        for t in ids:
            got = emissions_beatonly(m, P[t])
            if got is None:
                d[t] = {k: 0.0 for k in METRICS}
                continue
            lcd, dec, ff = got
            path, _ = m.dynamic_programs[0].viterbi(
                m.log_initial_distributions[0], m.log_tempo_transition, lcd,
                state_to_class=m.state_to_classes[0], return_log_score=True)
            d[t] = ev(gts[t], state_path_to_events(
                path.cpu().numpy(), space, FPS, snap_to_activations=dec, first_frame=ff)["beats"])
        res_dbn[name] = d
        print(f"  done {name}", flush=True)

    for tag, res in (("PEAK-PICK @0.5", res_pp), ("BEAT-ONLY DBN2016 min30 obs16 lam100", res_dbn)):
        print(f"\n== {tag} ==")
        print(f"{'subset':<20}" + "".join(f"{k:>9}" for k in METRICS) + "   dF(p)          dCMLt(p)")
        base = res["null r1.0"]
        for name in SUBSETS:
            mu = {k: np.mean([res[name][t][k] for t in ids]) for k in METRICS}
            ex = ""
            if name != "null r1.0":
                for k in ("F", "CMLt"):
                    x = np.array([res[name][t][k] - base[t][k] for t in ids])
                    p = wilcoxon(x).pvalue if np.any(x) else 1.0
                    ex += f"  {x.mean():+.4f}(p={p:.2g})"
            print(f"{name:<20}" + "".join(f"{mu[k]:>9.4f}" for k in METRICS) + ex, flush=True)

    json.dump({"n": len(ids), "pp": res_pp, "dbn": res_dbn},
              open("/tmp/overnight/audit_tta_ratesweep.json", "w"))
    print("DONE")


if __name__ == "__main__":
    main()
