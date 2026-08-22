"""Two spec repairs to the 2016 DBN, measured together.

REPAIR 1 -- REMOVE THE TRUNCATION-TO-ZERO (new, never tested).
`rungs/bar_pointer/state_space.py::tempo_transition_probabilities` (mirroring madmom's
`exponential_transition`) does:

    probabilities = exp(-transition_lambda * |interval_ratio - 1|)
    probabilities[probabilities <= np.spacing(1)] = 0.0      # <-- HARD SUPPORT CUT
    probabilities /= probabilities.sum(axis=1, keepdims=True)

At lambda=100 that sets **50.1%** of all tempo transitions to EXACTLY zero at min_bpm=55, and
**59.4%** at min_bpm=30; some source tempi retain only 6 reachable targets. Any bar-to-bar tempo
change beyond |r-1| >= -ln(spacing(1))/lambda = 0.360 is not merely improbable, it is IMPOSSIBLE.
VBPM measured p90 = 1.76x and max = 3.32x bar-to-bar ratios on an expressive performance -- both
structurally unrepresentable. This is the same failure mode as min_bpm=55 (support excludes the
truth) and as a wrong hard tempo band (CMLt -> 0.0395).

REPAIR 2 -- min_bpm 55 -> 30 (already measured alone: the 45 sub-55 tracks go CMLt 0.0528 -> 0.3395).

ARMS (2x2, so the truncation effect is isolated at BOTH tempo ranges):
    A  min_bpm=55, truncation ON   <- shipped madmom
    B  min_bpm=30, truncation ON   <- repair 2 only
    C  min_bpm=55, truncation OFF  <- repair 1 only
    D  min_bpm=30, truncation OFF  <- both repairs

Everything else is madmom's shipped deployment (obs_lambda=16, transition_lambda=100,
num_tempi=60, threshold=0.05, correct=True), on the R1 chassis, which is certified event-identical
to madmom on 217/217 SMC tracks at min55 and 216/217 at min30 (docs/audits/PHASE0_RECERT.md section 2).

VALIDATION: arm B must reproduce the known wide-DBN row F 0.5851 / CMLt 0.5102 / AMLt 0.6735.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))

from _harness import FPS, track_ids, load_gt, gt_bpm, bthis_beat_prob, bthis_downbeat_prob, ev
from rungs.r1_2016_dbn import DBN2016
from rungs.bar_pointer.state_space import BarPointerStateSpace

N_TRACKS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
DEVICE = sys.argv[2] if len(sys.argv) > 2 else ("cuda:1" if torch.cuda.is_available() else "cpu")
OUT = Path("/tmp/overnight/spec_repairs.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

_ORIG = BarPointerStateSpace.tempo_transition_probabilities


def _no_truncation(self, transition_lambda, threshold=np.spacing(1)):
    """Same kernel, but nothing is zeroed: the support stays full."""
    ratio = (self.interval_frames.astype(float)
             / self.interval_frames.astype(float)[:, np.newaxis])
    p = np.exp(-float(transition_lambda) * np.abs(ratio - 1.0))
    # Guard only against genuine float64 underflow leaving an all-zero row; this floors at a
    # value ~1e250 times smaller than the truncation it replaces, so it is not a re-truncation.
    p = np.maximum(p, 1e-300)
    return p / p.sum(axis=1)[:, np.newaxis]


def build(min_bpm, truncate):
    BarPointerStateSpace.tempo_transition_probabilities = _ORIG if truncate else _no_truncation
    return DBN2016(fps=FPS, min_bpm=min_bpm, max_bpm=215.0, device=DEVICE,
                   observation_lambda=16, transition_lambda=100)


def main():
    ids = track_ids()[:N_TRACKS]
    refs = {t: load_gt(t) for t in ids}
    acts = {t: np.stack([bthis_beat_prob(t), bthis_downbeat_prob(t)], -1) for t in ids}
    bpm = {t: gt_bpm(refs[t]) for t in ids}
    slow = [t for t in ids if bpm[t] < 55]
    print(f"n={len(ids)} device={DEVICE}  sub-55 BPM tracks: {len(slow)}", flush=True)

    ARMS = [("A_min55_truncON", 55.0, True), ("B_min30_truncON", 30.0, True),
            ("C_min55_truncOFF", 55.0, False), ("D_min30_truncOFF", 30.0, False)]
    per, rows = {}, {}
    for name, mn, trunc in ARMS:
        m = build(mn, trunc)
        sp = m.state_spaces[0] if hasattr(m, "state_spaces") else None
        k = sp.tempo_transition_probabilities(100.0) if sp is not None else None
        zfrac = float((k == 0).mean()) if k is not None else float("nan")
        per[name] = {t: ev(refs[t], m.predict(acts[t])["beats"]) for t in ids}
        rows[name] = {mm: float(np.mean([per[name][t][mm] for t in ids]))
                      for mm in ("F", "CMLt", "AMLt", "CMLc", "AMLc")}
        rows[name]["zeroed_frac"] = zfrac
        if slow:
            rows[name]["slow_CMLt"] = float(np.mean([per[name][t]["CMLt"] for t in slow]))
        print(f"  {name:18} F {rows[name]['F']:.4f} CMLt {rows[name]['CMLt']:.4f} "
              f"AMLt {rows[name]['AMLt']:.4f} AMLc {rows[name]['AMLc']:.4f} "
              f"| zeroed {zfrac:.1%}"
              + (f" | sub55 CMLt {rows[name]['slow_CMLt']:.4f}" if slow else ""), flush=True)
        OUT.write_text(json.dumps({"n": len(ids), "rows": rows}, indent=1))
    BarPointerStateSpace.tempo_transition_probabilities = _ORIG

    def cmp(a, b, label):
        print(f"\n{label}")
        out = {}
        for mm in ("F", "CMLt", "AMLt", "CMLc", "AMLc"):
            d = np.array([per[a][t][mm] - per[b][t][mm] for t in ids])
            nz = d[d != 0]
            p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
            bt = np.array([np.mean(np.random.choice(d, len(d), True)) for _ in range(10000)])
            out[mm] = {"delta": float(d.mean()), "p": p,
                       "ci": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                       "win": int((d > 0).sum()), "loss": int((d < 0).sum())}
            print(f"  {mm:5} {d.mean():+.4f} CI[{out[mm]['ci'][0]:+.4f},{out[mm]['ci'][1]:+.4f}] "
                  f"p={p:.3g} w/l {out[mm]['win']}/{out[mm]['loss']}")
        return out

    tests = {"truncOFF_effect_at_min30": cmp("D_min30_truncOFF", "B_min30_truncON",
                                             "REPAIR 1 alone (min30): truncation OFF vs ON"),
             "truncOFF_effect_at_min55": cmp("C_min55_truncOFF", "A_min55_truncON",
                                             "REPAIR 1 alone (min55): truncation OFF vs ON"),
             "both_vs_shipped": cmp("D_min30_truncOFF", "A_min55_truncON",
                                    "BOTH REPAIRS vs shipped madmom")}
    v = rows["B_min30_truncON"]
    print(f"\nVALIDATION arm B vs known wide-DBN row: F {v['F']:.4f} (want 0.5851) "
          f"CMLt {v['CMLt']:.4f} (want 0.5102)  [subset of {len(ids)}, exact match only at n=217]")
    OUT.write_text(json.dumps({"n": len(ids), "rows": rows, "tests": tests}, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
