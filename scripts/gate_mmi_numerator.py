"""GATE 2 of the MMI build: is the numerator mask a VALID numerator?

THE CLAIM UNDER TEST. MMI needs log Z_num = log-sum over paths CONSISTENT WITH THE REFERENCE. We
propose to express that constraint as a per-frame additive mask on the 3-class emission -- forbid
the BEAT class at frames far from any annotation -- so that Z_num and Z_den are the SAME certified
forward_log_likelihood call differing by one added tensor. That is worth a lot structurally (the
CRF died partly of a hand-built numerator drifting from its denominator), but it is only legitimate
if the mask actually pins paths to the annotations.

THE ARGUMENT, AND WHY IT IS NOT SELF-EVIDENT. The bar pointer must advance +1 per frame and wrap,
so it passes through the beat region periodically and cannot simply decline to emit beats. Forbidding
BEAT away from annotations should therefore force its wraps into the annotation windows. But the
model is free to choose TEMPO, and a slow-tempo path can wrap only half as often -- landing on every
OTHER annotation, satisfying the mask perfectly while being reference-INconsistent. A mask that
admits such paths is not a numerator; it is a weaker constraint that would teach the network the
wrong thing, and it would do so silently. So we measure rather than argue.

DECISIVE TEST. Viterbi-decode the CONSTRAINED model and score its beats against the annotations.
A valid numerator scores essentially perfectly -- if the best reference-consistent path is not
reference-consistent, the constraint set is wrong.

TWO VARIANTS, the second answering the failure of the first:
  A  forbid BEAT at frames farther than TOL from any annotation
  B  A, plus forbid NO_BEAT at each annotation's own frame (forces a beat AT every annotation)

A is preferable if sufficient, because it does not force the path through frames where the
annotation itself may be tens of ms off. MEASURED: it is not sufficient (F 0.916, CMLt 0.800 on a
smoke set) -- exactly the predicted hole, slow-tempo paths satisfying the mask while landing on only
some annotations. Arguing from the topology would have got this wrong.

B is exact where the annotation is REPRESENTABLE: on the smoke set it returns F 1.000 / CMLt 1.000
on 6 of 8 tracks, and the 2 failures are exactly the 2 tracks whose annotated tempo dips below the
state space's 30 BPM floor (29.6 and 25.2 BPM), where no path can reach the annotation because no
state exists for it. So the residual is a property of the DECODER'S TEMPO GRID, not of the mask, and
the fix is to exclude unrepresentable tracks from MMI training rather than to widen the grid --
widening it would change the decoder we ship and destroy attributability of any gain.

A third variant (B with each window clipped at the midpoints to its neighbours) was written and
tested on the hypothesis that adjacent windows merge at fast tempi. It is a no-op: at TOL=3 the
windows are 7 frames wide, so merging needs annotations under 140 ms apart, i.e. above 428 BPM.
Removed rather than kept, but recorded here so it is not re-invented.

B also leaves a systematic -22 ms median offset, because the beat region is
interval/observation_lambda frames wide (1 frame at 214 BPM but 7 frames = 140 ms at 30 BPM) and B
pins the path INSIDE the region while the region ENTRY -- what the read-out calls a beat -- precedes
it. At 97.4% of entries within +-70 ms this does not cost F, but it is a known bias of the numerator
and is the reason the gate reports the raw path read-out, not only the snapped one.

READ-OUT. The shipped readout snaps each beat to the strongest activation inside its region
(madmom's correct=True) and, with beats_per_bar=1, the correct=False branch emits NOTHING (all
state positions lie in [0,1) so the integer beat counter never changes). For the gate we want the
PATH, so beats are read as region ENTRIES; the snapped read-out is reported alongside so we know the
numerator is compatible with the decoder we actually evaluate.

ALSO GATED: Z_num <= Z_den on every track (a mathematical necessity -- the numerator sums a subset
of the same paths under a mask that only ever penalises -- so a violation means the mask or the
recursion is wrong), both partition functions finite, and the gradient of the MMI loss w.r.t. the
emission finite. A finite penalty is used rather than -inf precisely because -inf can produce an
all--inf logsumexp row whose backward is NaN.
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
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events
from rungs.bar_pointer.state_space import BEAT

PENALTY = -1e4          # finite, not -inf: -inf risks an all--inf logsumexp row -> NaN backward


def region_entries(path, space, fps, first_frame):
    """Beats as ENTRIES into a beat region -- the path's own beats, independent of the deployment
    snapping heuristic (and of the beats_per_bar=1 degeneracy in the correct=False read-out)."""
    inreg = space.position_classes[np.asarray(path)] >= BEAT
    ent = np.nonzero(np.diff(inreg.astype(int)) == 1)[0] + 1
    if inreg[0]:
        ent = np.r_[0, ent]
    return (ent + first_frame) / fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol-frames", type=int, default=3, help="3 frames @50fps = 60 ms <= the 70 ms metric tolerance")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", default="/tmp/overnight/gate_mmi_numerator.json")
    args = ap.parse_args()

    m = make_beatonly_dbn(30.0, 16, 100, args.device)
    space, dp0 = m.state_spaces[0], m.dynamic_programs[0]
    iv = space.interval_frames
    s2c = m.state_to_classes[0]
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition

    ids = track_ids()[:args.limit] if args.limit else track_ids()
    rows, viol, nonfinite = {}, [], []
    for tid in ids:
        act = np.asarray(bthis_beat_prob(tid), np.float64)
        got = emissions_beatonly(m, act)
        if got is None:
            continue
        lcd, dec, ff = got
        T = lcd.shape[0]
        gt = np.asarray(load_gt(tid), float)
        fr = np.round(gt * FPS).astype(int) - ff
        fr = fr[(fr >= 0) & (fr < T)]
        if len(fr) < 8:
            continue

        tolv = args.tol_frames
        near = np.zeros(T, bool)
        for f in fr:
            near[max(0, f - tolv):min(T, f + tolv + 1)] = True
        maskA = torch.zeros_like(lcd)
        maskA[torch.from_numpy(~near).to(lcd.device), BEAT] = PENALTY
        maskB = maskA.clone()
        maskB[torch.from_numpy(fr).to(lcd.device), 0] = PENALTY      # forbid NO_BEAT at annotations

        # Is the annotation representable at all? A beat gap outside the state space's tempo grid
        # has no state to occupy, so no path can reach that annotation and the numerator is
        # infeasible there through no fault of the mask.
        ibi = np.diff(fr).astype(float)
        r = {"n_ref": int(len(fr)),
             "frac_ibi_out_of_grid": float(((ibi < iv[0]) | (ibi > iv[-1])).mean())}
        Zden = dp0.forward_log_likelihood(logpi, logA, lcd, state_to_class=s2c)
        r["Zden"] = float(Zden)
        for nm, mk in (("A", maskA), ("B", maskB)):
            Znum = dp0.forward_log_likelihood(logpi, logA, lcd + mk, state_to_class=s2c)
            r[f"Znum_{nm}"] = float(Znum)
            if not (np.isfinite(r[f"Znum_{nm}"]) and np.isfinite(r["Zden"])):
                nonfinite.append((tid, nm))
            elif r[f"Znum_{nm}"] > r["Zden"] + 1e-6:
                viol.append((tid, nm, r[f"Znum_{nm}"] - r["Zden"]))

            path, _ = dp0.viterbi(logpi, logA, lcd + mk, state_to_class=s2c,
                                  return_log_score=True)
            p = path.cpu().numpy()
            r[f"path_{nm}"] = ev(gt, region_entries(p, space, FPS, ff))
            r[f"snap_{nm}"] = ev(gt, state_path_to_events(
                p, space, FPS, snap_to_activations=dec, first_frame=ff)["beats"])

        # gradient of the MMI loss wrt the emission must be finite
        em = lcd.clone().requires_grad_(True)
        loss = -(dp0.forward_log_likelihood(logpi, logA, em + maskA, state_to_class=s2c)
                 - dp0.forward_log_likelihood(logpi, logA, em, state_to_class=s2c))
        g, = torch.autograd.grad(loss, em)
        r["grad_finite"] = bool(torch.isfinite(g).all())
        r["grad_absmax"] = float(g.abs().max())
        if not r["grad_finite"]:
            nonfinite.append((tid, "grad"))
        rows[tid] = r
        if len(rows) % 25 == 0:
            print(f"  {len(rows)}/{len(ids)}", flush=True)

    allids = sorted(rows)
    feas = [t for t in allids if rows[t]["frac_ibi_out_of_grid"] == 0.0]
    print(f"\nrepresentable tracks (every beat gap inside the {60*FPS/iv[-1]:.0f}-{60*FPS/iv[0]:.0f} "
          f"BPM grid): {len(feas)} / {len(allids)}")
    for lbl, ids in (("ALL", allids), ("REPRESENTABLE", feas)):
        A = lambda k, f: float(np.mean([rows[t][k][f] for t in ids]))
        print(f"\n{lbl} n={len(ids)}   constrained-Viterbi score against the annotations")
        print(f"{'variant':<28}{'F':>9}{'CMLt':>9}{'AMLt':>9}")
        for nm, lb in (("A", "A  BEAT masked away"), ("B", "B  A + NO_BEAT at annot")):
            for rd in ("path", "snap"):
                print(f"{lb + '  [' + rd + ']':<28}"
                      + "".join(f"{A(f'{rd}_{nm}', k):>9.4f}" for k in ("F", "CMLt", "AMLt")))
            bad = [t for t in ids if rows[t][f"path_{nm}"]["CMLt"] < 0.9]
            print(f"    -> {len(bad)} / {len(ids)} tracks with numerator CMLt < 0.9")
    ids = feas
    A = lambda k, f: float(np.mean([rows[t][k][f] for t in ids]))
    gap = np.array([rows[t]["Zden"] - rows[t]["Znum_A"] for t in ids])
    print(f"\nlog Z_den - log Z_num (A): mean {gap.mean():.2f} nats, min {gap.min():.2f}")
    print(f"Z_num > Z_den violations   : {len(viol)}")
    print(f"non-finite Z or gradient   : {len(nonfinite)}")
    print(f"max |dL/d emission|        : {max(rows[t]['grad_absmax'] for t in ids):.4g}")
    ok = (not viol and not nonfinite and A("path_B", "F") > 0.98
          and A("path_B", "CMLt") > 0.98)
    print(f"\nGATE 2: {'PASS' if ok else 'FAIL'}   (variant B on the {len(feas)} representable "
          f"tracks: F {A('path_B', 'F'):.4f}, CMLt {A('path_B', 'CMLt'):.4f})")
    json.dump({"n": len(ids), "tol_frames": args.tol_frames, "rows": rows},
              open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
