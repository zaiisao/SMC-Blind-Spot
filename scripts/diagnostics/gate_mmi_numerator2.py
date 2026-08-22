"""GATE 2b: does the SHARED numerator mask (scripts/mmi.py) stay valid once unrepresentable spans
are freed rather than whole tracks dropped?

Gate 2 established that variant B is an exact numerator where the annotation is representable
(F 0.9916 / CMLt 0.9802 on 183 tracks) and infeasible where it is not -- 34 tracks contain a beat
gap outside the 30-214 BPM grid. mmi.numerator_mask() keeps those tracks by leaving the offending
SPAN unconstrained and refusing to anchor the two annotations bounding it.

That trade must be paid for honestly, so this gate reports it as a trade:
  * the numerator must remain near-exact on the 183 tracks that were already fine (no regression),
  * it must be usable on the 34 recovered tracks (scored separately, and expected to be WORSE than
    0.98 there -- the freed span is genuinely unconstrained, so paths may wander inside it),
  * and the fraction of annotations left unanchored is reported, because that is exactly how much
    supervision the concession costs.

If the recovered tracks' numerator is bad enough to teach the wrong thing, the right decision is to
drop them after all -- but that is a decision to make on measured numbers, not on the assumption
that recovering them is free.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from gate_mmi_numerator import region_entries
from mmi import annotation_frames, numerator_mask, mmi_loss

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"


def main():
    m = make_beatonly_dbn(30.0, 16, 100, DEV)
    space, dp0 = m.state_spaces[0], m.dynamic_programs[0]
    s2c = m.state_to_classes[0]
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition
    iv = space.interval_frames

    rows, viol, nonfinite = {}, [], []
    for tid in track_ids():
        act = np.asarray(bthis_beat_prob(tid), np.float64)
        got = emissions_beatonly(m, act)
        if got is None:
            continue
        lcd, dec, ff = got
        T = lcd.shape[0]
        gt = np.asarray(load_gt(tid), float)
        fr = annotation_frames(gt, FPS, ff, T)
        if len(fr) < 8:
            continue
        mask, n_anch, n_bad = numerator_mask(fr, T, iv, lcd.device, lcd.dtype)

        loss, znum, zden = mmi_loss(dp0, logpi, logA, lcd, s2c, mask)
        r = {"n_ref": int(len(fr)), "n_anchored": n_anch, "n_bad_spans": n_bad,
             "loss": float(loss), "Znum": float(znum), "Zden": float(zden)}
        if not (np.isfinite(r["Znum"]) and np.isfinite(r["Zden"])):
            nonfinite.append(tid)
        elif r["Znum"] > r["Zden"] + 1e-6:
            viol.append(tid)

        path, _ = dp0.viterbi(logpi, logA, lcd + mask, state_to_class=s2c, return_log_score=True)
        r["num"] = ev(gt, region_entries(path.cpu().numpy(), space, FPS, ff))

        em = lcd.clone().requires_grad_(True)
        g, = torch.autograd.grad(mmi_loss(dp0, logpi, logA, em, s2c, mask)[0], em)
        r["grad_finite"] = bool(torch.isfinite(g).all())
        r["grad_absmax"] = float(g.abs().max())
        if not r["grad_finite"]:
            nonfinite.append(tid + ":grad")
        rows[tid] = r
        if len(rows) % 40 == 0:
            print(f"  {len(rows)}/217", flush=True)

    ids = sorted(rows)
    clean = [t for t in ids if rows[t]["n_bad_spans"] == 0]
    recov = [t for t in ids if rows[t]["n_bad_spans"] > 0]
    A = lambda ts, k: float(np.mean([rows[t]["num"][k] for t in ts]))
    print(f"\n{'group':<34}{'n':>5}{'F':>9}{'CMLt':>9}{'AMLt':>9}{'anchored':>10}")
    for lbl, ts in (("already representable", clean), ("recovered (freed spans)", recov),
                    ("ALL", ids)):
        anch = np.mean([rows[t]["n_anchored"] / rows[t]["n_ref"] for t in ts])
        print(f"{lbl:<34}{len(ts):>5}" + "".join(f"{A(ts, k):>9.4f}" for k in ("F", "CMLt", "AMLt"))
              + f"{anch:>10.1%}")
    print(f"\n  gate-2 reference on the 183 representable tracks was F 0.9916 / CMLt 0.9802")
    bad = [t for t in ids if rows[t]["num"]["CMLt"] < 0.9]
    print(f"  tracks with numerator CMLt < 0.9 : {len(bad)} / {len(ids)}")
    print(f"  Znum > Zden violations           : {len(viol)}")
    print(f"  non-finite Z or gradient         : {len(nonfinite)}")
    print(f"  max |dL/d emission|              : {max(rows[t]['grad_absmax'] for t in ids):.4g}")
    ok = (not viol and not nonfinite and A(clean, "CMLt") > 0.97
          and max(rows[t]["grad_absmax"] for t in ids) <= 1.0 + 1e-9)
    print(f"\nGATE 2b: {'PASS' if ok else 'FAIL'}")
    json.dump({"n": len(ids), "rows": rows}, open("/tmp/overnight/gate_mmi_num2.json", "w"),
              indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
