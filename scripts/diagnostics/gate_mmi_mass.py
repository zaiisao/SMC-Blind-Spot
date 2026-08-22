"""ATTACK ON GATE 2/2b: is the numerator's probability MASS reference-consistent, or only its argmax?

Gate 2 decodes the constrained model (max) and scores the path. MMI trains on log Z_num (sum).
This probe measures, per track, using STATE-level occupancy gamma_num[t,k] = d logZ_num/d eps[t,k]:
  * E[#beats]   = sum_t sum_{k in first_states} gamma_num[t,k]   (expected region entries)
                  vs n_ref -> half/double-tempo mass shows up here directly.
  * on_target   = fraction of that expected-beat mass within +-tol frames of an annotation.
  * hit_ann     = expected beat mass attributable to each annotation (coverage).
  * viol_mass   = expected # of frames whose CLASS lands on a -1e4 masked entry (feasibility of
                  the MASS, not of the best path).
  * gap = logZ_num - viterbi_num  (nats of "spread": how much more mass than the single path).
  * penalty sweep logZ_num at -1e2/-1e4/-1e6/-1e8: a feasible numerator converges; an infeasible
    one falls linearly with the penalty.
  * region-entry offset of the mass (the -22 ms bias claim), mass-weighted.
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
from mmi import annotation_frames, numerator_mask, TOL_FRAMES

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def main():
    m = make_beatonly_dbn(30.0, 16, 100, DEV)
    space, dp0 = m.state_spaces[0], m.dynamic_programs[0]
    s2c = m.state_to_classes[0]
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition
    iv = space.interval_frames
    K = space.num_states
    first_states = torch.from_numpy(np.asarray(space.first_states).reshape(-1)).to(DEV)
    pc = torch.from_numpy(space.position_classes).to(DEV)

    ids = track_ids()[:LIMIT] if LIMIT else track_ids()
    rows = {}
    for tid in ids:
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

        # state-level emission with a differentiable zero perturbation -> exact occupancies
        base = (lcd + mask)[:, s2c]                              # [T, K]
        eps = torch.zeros_like(base, requires_grad=True)
        znum = dp0.forward_log_likelihood(logpi, logA, base + eps)
        gnum, = torch.autograd.grad(znum, eps)                   # [T,K] = P(state_t=k | num)
        base_d = lcd[:, s2c]
        eps2 = torch.zeros_like(base_d, requires_grad=True)
        zden = dp0.forward_log_likelihood(logpi, logA, base_d + eps2)
        gden, = torch.autograd.grad(zden, eps2)

        def stats(g):
            pbeat = g[:, first_states].sum(1)                    # [T] P(region entry at t)
            cls = torch.zeros(T, 3, dtype=g.dtype, device=g.device).index_add_(
                1, pc, g)                                        # class occupancy [T,3]
            return pbeat.cpu().numpy(), cls.cpu().numpy()

        pb, cls = stats(gnum)
        pb_d, cls_d = stats(gden)

        near = np.zeros(T, bool)
        for f in fr:
            near[max(0, f - TOL_FRAMES):min(T, f + TOL_FRAMES + 1)] = True
        Eb = float(pb.sum())
        on = float(pb[near].sum())
        # violation mass: class occupancy on masked entries
        mk = (mask.cpu().numpy() < 0)
        viol = float((cls * mk).sum())
        viol_d = float((cls_d * mk).sum())

        # mass-weighted offset of expected entries from nearest annotation (ms)
        t_idx = np.nonzero(pb > 1e-6)[0]
        if len(t_idx):
            d = (t_idx[:, None] - fr[None, :])
            j = np.abs(d).argmin(1)
            off = d[np.arange(len(t_idx)), j] / FPS * 1000.0
            w = pb[t_idx]
            off_mean = float((off * w).sum() / w.sum())
            srt = np.argsort(off)
            cw = np.cumsum(w[srt]) / w.sum()
            off_med = float(off[srt][np.searchsorted(cw, 0.5)])
        else:
            off_mean = off_med = float("nan")

        path, vsc = dp0.viterbi(logpi, logA, lcd + mask, state_to_class=s2c, return_log_score=True)
        num_ev = ev(gt, region_entries(path.cpu().numpy(), space, FPS, ff))

        pen = {}
        for p in (-1e2, -1e3, -1e4, -1e6, -1e8):
            mk2, _, _ = numerator_mask(fr, T, iv, lcd.device, lcd.dtype, penalty=p)
            pen[str(p)] = float(dp0.forward_log_likelihood(logpi, logA, lcd + mk2,
                                                           state_to_class=s2c))
        rows[tid] = {
            "n_ref": int(len(fr)), "n_anchored": n_anch, "n_bad_spans": n_bad, "T": T,
            "Znum": float(znum), "Zden": float(zden), "vit_num": float(vsc),
            "spread": float(znum) - float(vsc),
            "E_beats": Eb, "E_beats_den": float(pb_d.sum()),
            "on_target": on / Eb if Eb else float("nan"),
            "on_target_den": float(pb_d[near].sum()) / float(pb_d.sum()),
            "viol_mass": viol, "viol_mass_den": viol_d,
            "off_mean_ms": off_mean, "off_med_ms": off_med,
            "vit_F": num_ev["F"], "vit_CMLt": num_ev["CMLt"], "pen": pen,
        }
        if len(rows) % 20 == 0:
            print(f"  {len(rows)}", flush=True)

    json.dump(rows, open("/tmp/overnight/gate_mmi_mass.json", "w"), indent=1)
    ids = sorted(rows)
    clean = [t for t in ids if rows[t]["n_bad_spans"] == 0]
    A = lambda ts, k: float(np.median([rows[t][k] for t in ts]))
    print(f"\n{'group':<16}{'n':>4}{'E_b/nref':>10}{'onTarget':>10}{'violMass':>10}"
          f"{'spread':>10}{'vitF':>8}{'offMed':>9}")
    for lbl, ts in (("clean", clean), ("recovered", [t for t in ids if t not in clean]),
                    ("ALL", ids)):
        if not ts:
            continue
        r = float(np.median([rows[t]["E_beats"] / rows[t]["n_ref"] for t in ts]))
        print(f"{lbl:<16}{len(ts):>4}{r:>10.4f}{A(ts,'on_target'):>10.4f}"
              f"{A(ts,'viol_mass'):>10.4f}{A(ts,'spread'):>10.3f}{A(ts,'vit_F'):>8.4f}"
              f"{A(ts,'off_med_ms'):>9.1f}")
    print("\nDENOMINATOR reference: median on_target "
          f"{A(ids,'on_target_den'):.4f}, E_beats/nref "
          f"{float(np.median([rows[t]['E_beats_den']/rows[t]['n_ref'] for t in ids])):.4f}")
    for k in ("on_target", "E_beats", "viol_mass"):
        v = np.array([rows[t][k] if k != "E_beats" else rows[t]["E_beats"] / rows[t]["n_ref"]
                      for t in ids])
        print(f"  {k:<12} p5 {np.percentile(v,5):.4f}  p25 {np.percentile(v,25):.4f} "
              f" med {np.median(v):.4f}  p75 {np.percentile(v,75):.4f}  p95 {np.percentile(v,95):.4f}")
    print("\npenalty sweep, median logZ_num:  " + "  ".join(
        f"{p}:{np.median([rows[t]['pen'][p] for t in ids]):.2f}"
        for p in ("-100.0", "-1000.0", "-10000.0", "-1000000.0", "-100000000.0")))
    d = np.array([rows[t]["pen"]["-100000000.0"] - rows[t]["pen"]["-10000.0"] for t in ids])
    print(f"  tracks where logZ_num(-1e8) < logZ_num(-1e4) - 1 nat: {(d < -1).sum()}/{len(ids)}"
          f"   median drop {np.median(d):.2f}  min {d.min():.4g}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
