"""THE MISSING GATE: does a path satisfying the numerator mask EXIST?

WHAT WENT WRONG. Gates 2 and 2b certified the numerator by decoding the constrained model and
scoring its best path against the annotations (F 0.9916 / CMLt 0.9802). That is a statement about
the ARGMAX. It says nothing about whether the constraint set is non-empty. Both gates also asserted
Z_num <= Z_den -- which is a TAUTOLOGY under a non-positive additive mask and can never trip -- and
keyed their pass/fail on the MEAN over tracks, so a minority of broken tracks could not move it.

Measured with that hole closed: 61 of 138 clean fold-0 TRAINING tracks (44.2%) have NO feasible
path. Median 3 unavoidable violations, max 29 (smc_206). On those tracks the reported MMI loss is a
penalty readout, not a model term, which is why MMImean sat at ~5 across a 100x mmi_w sweep.

HOW FEASIBILITY IS MEASURED, without needing a combinatorial search. The mask adds `penalty` per
violated frame-class. If the best numerator path must violate k constraints, then log Z_num falls
linearly in `penalty` with slope k. So evaluating log Z_num at two penalties recovers k exactly:

    k = (logZ_num(p1) - logZ_num(p2)) / (p1 - p2)

k == 0 means feasible. k is integer-valued, which is itself a check on the arithmetic.

THE EMISSION IS ZEROED. With log_emission := 0 the forward sees only the mask and the transition,
so k becomes a property of (mask x state space) ALONE -- independent of Beat This, of the checkpoint,
and of how well the model happens to fit. That is what makes the result a statement about the
DECODER rather than about a particular network.

ABLATIONS, to locate the cause rather than just measure it:
  A  shipped mask            -- forbid BEAT away from annotations, AND force a beat at each one
  B  anchoring removed       -- the BEAT-forbidding half only (mmi.py's `anchored` term dropped)
  C  B x TOL_FRAMES sweep    -- is the residual just too narrow a window?
If B and C still leave violations on a large fraction, the finding is not a bug in the mask: it is
that the bar-pointer tempo grid CANNOT realise SMC's annotated inter-beat-interval sequences on that
fraction of the corpus, even given an oracle emission. That is a mechanism statement about the
decoder, measurable on all 217 tracks with no training and no n=27 problem.

Per-track counts are reported, never a mean -- the mean is what hid this.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from mmi import annotation_frames
from rungs.bar_pointer.state_space import BEAT

NO_BEAT = 0
P1, P2 = -1e4, -1e6


def build_mask(frames, T, iv, device, dtype, tol, anchor, penalty):
    """Same construction as mmi.numerator_mask, with the anchoring half switchable."""
    fr = np.asarray(frames, np.int64)
    gap = np.diff(fr).astype(float)
    bad = (gap < float(iv[0])) | (gap > float(iv[-1]))
    allowed = np.zeros(T, bool)
    for f in fr:
        allowed[max(0, f - tol):min(T, f + tol + 1)] = True
    for i in np.nonzero(bad)[0]:
        allowed[fr[i]:min(T, fr[i + 1] + 1)] = True
    anchored = np.ones(len(fr), bool)
    for i in np.nonzero(bad)[0]:
        anchored[i] = anchored[i + 1] = False
    mask = torch.zeros((T, 3), dtype=dtype, device=device)
    mask[torch.from_numpy(~allowed).to(device), BEAT] = penalty
    if anchor and anchored.any():
        mask[torch.from_numpy(fr[anchored]).to(device), NO_BEAT] = penalty
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/tmp/overnight/gate_mmi_feasible.json")
    args = ap.parse_args()
    m = make_beatonly_dbn(30.0, 16, 100, args.device)
    dp0, s2c = m.dynamic_programs[0], m.state_to_classes[0]
    iv = m.state_spaces[0].interval_frames
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition

    ARMS = [("A shipped (tol=3, anchored)", 3, True)]
    ARMS += [("B no-anchor (tol=3)", 3, False)]
    ARMS += [(f"C no-anchor tol={t}", t, False) for t in (4, 5, 6, 8)]
    ARMS += [(f"C anchored tol={t}", t, True) for t in (4, 6, 8)]

    rows = {}
    for n, tid in enumerate(track_ids()):
        got = emissions_beatonly(m, np.asarray(bthis_beat_prob(tid), np.float64))
        if got is None:
            continue
        lcd, dec, ff = got
        T = lcd.shape[0]
        fr = annotation_frames(np.asarray(load_gt(tid), float), FPS, ff, T)
        if len(fr) < 8:
            continue
        zero = torch.zeros_like(lcd)                     # EMISSION ZEROED: decoder-only property
        r = {"n_ref": int(len(fr))}
        for lbl, tol, anch in ARMS:
            z = []
            for p in (P1, P2):
                mk = build_mask(fr, T, iv, lcd.device, lcd.dtype, tol, anch, p)
                z.append(float(dp0.forward_log_likelihood(logpi, logA, zero + mk,
                                                          state_to_class=s2c)))
            r[lbl] = int(round((z[0] - z[1]) / (P1 - P2)))
        rows[tid] = r
        if (n + 1) % 40 == 0:
            print(f"  {n+1}/217", flush=True)

    print(f"\nn={len(rows)} tracks, EMISSION ZEROED (property of mask x state space alone)")
    print(f"{'arm':<28}{'infeasible':>12}{'%':>8}{'median k':>10}{'max k':>8}")
    for lbl, _, _ in ARMS:
        k = np.array([rows[t][lbl] for t in rows])
        bad = k > 0
        print(f"{lbl:<28}{bad.sum():>12}{100*bad.mean():>7.1f}%"
              f"{(np.median(k[bad]) if bad.any() else 0):>10.0f}{k.max():>8}")
    worst = sorted(rows, key=lambda t: -rows[t]["A shipped (tol=3, anchored)"])[:5]
    print("\nworst 5 under the shipped mask: "
          + ", ".join(f"{t}({rows[t]['A shipped (tol=3, anchored)']})" for t in worst))
    json.dump({"n": len(rows), "arms": [a[0] for a in ARMS], "rows": rows},
              open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
