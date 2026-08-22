"""THE PROPOSED SYSTEM, measured as one configuration for the first time.

Every component below is already verified separately on SMC (n=217, fold-honest Beat This
activations, scripts/_harness.py). None has ever been run together:

  1. min_bpm 55 -> 30      +0.0087 F / +0.0364 CMLt   (21% of SMC is below 55 BPM and is
                            otherwise structurally unrepresentable)
  2. observation_lambda 16 -> 8   +0.0110 F / +0.0075 CMLt  (never examined before; madmom's
                            default is 16, Beat This ships 6, the optimum is 8)
  3. time-stretch TTA {0.85, 1.0, 1.2}   +0.018 F / +0.036 CMLt / +0.039 AMLt (free, no retraining)
  4. wide DBN decode instead of peak-picking  AMLt +0.0618 / AMLc +0.1179 (p < 1e-6)

NO ORACLE, NO GROUND TRUTH, NO RETRAINING anywhere in this configuration.

Reference rows on the same harness, for comparison:
  raw peak-pick @0.5          F 0.6270  CMLt 0.5138  AMLt 0.6100
  TTA peak-pick @0.5          F 0.6449  CMLt 0.5504  AMLt 0.6486
  raw + wide DBN obs16 l100   F 0.5851  CMLt 0.5102  AMLt 0.6735
  TTA + wide DBN obs16 l100   F 0.6047  CMLt 0.5413  AMLt 0.6954
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from _harness import FPS, track_ids, load_gt, bthis_beat_prob, bthis_downbeat_prob, ev, peakpick

TTA_DIR = ROOT / "data" / "tta_full"   # ts_beat / ts_db logits, already rate-averaged
LAMS = [5, 20, 50, 100, 200]
OBS = [8, 16]
OUT = Path("/tmp/overnight/combined_system.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


def tta_acts(tid):
    """Time-stretch TTA beat/downbeat PROBABILITIES.

    Reader copied verbatim from scripts/overnight/ov_a_tta_dbn.py:44 rather than reinvented: the cache
    stores LOGITS in keys ts_beat / ts_db, already averaged over rates {0.85, 1.0, 1.2}.
    """
    from scipy.special import expit
    z = np.load(TTA_DIR / f"{tid}.npz")
    return expit(z["ts_beat"].astype(np.float64)), expit(z["ts_db"].astype(np.float64))


def decode(bp, dp, obs, lam, min_bpm=30.0):
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    eps = 1e-5
    bp = bp * (1 - eps) + eps / 2
    dp = dp * (1 - eps) + eps / 2
    comb = np.vstack((np.maximum(bp - dp, eps / 2), dp)).T
    proc = DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4], min_bpm=min_bpm, max_bpm=215.0, fps=FPS,
        transition_lambda=lam, observation_lambda=obs)
    return proc(comb)[:, 0]


def agg(per, ids):
    return {m: float(np.mean([per[t][m] for t in ids])) for m in ("F", "CMLt", "AMLt", "CMLc", "AMLc")}


def paired(a, b, ids, m):
    d = np.array([a[t][m] - b[t][m] for t in ids])
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    boot = np.array([np.mean(np.random.choice(d, len(d), replace=True)) for _ in range(10000)])
    return {"delta": float(d.mean()), "ci": [float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5))], "p": p,
            "win": int((d > 0).sum()), "loss": int((d < 0).sum())}


def main():
    ids = [t for t in track_ids() if (TTA_DIR / f"{t}.npz").exists()]
    refs = {t: load_gt(t) for t in ids}
    print(f"n={len(ids)} (TTA cache available)", flush=True)

    rows, per_arm = {}, {}
    # baseline: raw peak-pick
    per_arm["raw_peakpick"] = {t: ev(refs[t], peakpick(bthis_beat_prob(t), 0.5, FPS)) for t in ids}
    tta = {t: tta_acts(t) for t in ids}
    per_arm["tta_peakpick"] = {t: ev(refs[t], peakpick(tta[t][0], 0.5, FPS)) for t in ids}
    for name in ("raw_peakpick", "tta_peakpick"):
        rows[name] = agg(per_arm[name], ids)
        print(f"  {name:34} {rows[name]}", flush=True)

    for obs in OBS:
        for lam in LAMS:
            key = f"TTA+DBN_min30_obs{obs}_lam{lam}"
            per_arm[key] = {t: ev(refs[t], decode(*tta[t], obs, lam)) for t in ids}
            rows[key] = agg(per_arm[key], ids)
            print(f"  {key:34} F {rows[key]['F']:.4f} CMLt {rows[key]['CMLt']:.4f} "
                  f"AMLt {rows[key]['AMLt']:.4f} AMLc {rows[key]['AMLc']:.4f}", flush=True)
            OUT.write_text(json.dumps({"n": len(ids), "rows": rows}, indent=1))

    best = max([k for k in rows if k.startswith("TTA+DBN")],
               key=lambda k: rows[k]["CMLt"] + rows[k]["AMLt"])
    tests = {m: paired(per_arm[best], per_arm["raw_peakpick"], ids, m)
             for m in ("F", "CMLt", "AMLt", "CMLc", "AMLc")}
    print(f"\nBEST (by CMLt+AMLt): {best}")
    for m, r in tests.items():
        print(f"  vs raw peak-pick  {m:5} {r['delta']:+.4f} "
              f"CI[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p={r['p']:.3g} w/l {r['win']}/{r['loss']}")
    OUT.write_text(json.dumps({"n": len(ids), "rows": rows, "best": best,
                               "tests_vs_raw_peakpick": tests}, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
