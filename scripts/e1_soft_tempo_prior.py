"""E1: soft, calibrated tempo prior by exact marginalisation (CPU only).

Replaces R1's uniform-over-[30,215] tempo-state initial distribution with
    pi(tau) ~ u(tau) * [ octave-smeared Beat Transformer tempo posterior ](1/T)
where u(tau) is madmom's implied tempo marginal (a tempo of i frames-per-beat owns i
states, so u(tau) ~ i). The FULL wide grid [30,215] is always decoded; no state is
ever excluded. T -> inf recovers the wide DBN exactly (prior term vanishes).
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp/e1_lab")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
import _harness as H
from rungs.r1_2016_dbn import DBN2016

FPS = 50.0
MIN_BPM, MAX_BPM = 30.0, 215.0
SMEAR = {1/3: 0.075, 1/2: 0.175, 1.0: 0.50, 2.0: 0.175, 3.0: 0.075}
FLOOR = 1e-3   # uniform mixing floor: guarantees no tempo has zero prior mass


def smeared_tempo_posterior(logits):
    """logits: [300], index == BPM. Returns q over integer BPM 0..299 (sums to 1)."""
    p = np.exp(logits - logits.max()); p /= p.sum()
    grid = np.arange(len(p), dtype=float)
    q = np.zeros_like(p)
    for r, w in SMEAR.items():
        # mass that sat at bpm0 in p moves to bpm0*r  ->  q(b) += w * p(b/r)
        q += w * np.interp(grid / r, grid, p, left=0.0, right=0.0) / r
    q /= q.sum()
    q = (1.0 - FLOOR) * q + FLOOR / len(q)
    return q


def make_model():
    return DBN2016(fps=FPS, min_bpm=MIN_BPM, max_bpm=MAX_BPM, device="cpu",
                   bounding="squeeze")


def set_prior(model, q, inv_T, mode="beat", base_tempo_transition=None):
    """inv_T = 1/T. inv_T == 0 -> untouched wide DBN.

    mode='init': tilt only the initial state distribution.
    mode='beat': also tilt every tempo transition (row-renormalised in log space),
                 i.e. the prior is re-applied at every beat boundary, which is where
                 the DBN re-chooses its tempo. This is the soft analogue of the hard
                 band, which restricts the tempo at every beat.
    """
    grid = np.arange(len(q), dtype=float)
    total = sum(s.num_states for s in model.state_spaces)
    base = -float(np.log(total))
    for i, space in enumerate(model.state_spaces):
        li = torch.full((space.num_states,), base, dtype=model.dtype)
        if inv_T != 0.0:
            bpm = 60.0 * FPS / space.state_interval_frames.astype(float)
            li = li + torch.from_numpy(inv_T * np.log(np.interp(bpm, grid, q))).to(model.dtype)
        model.log_initial_distributions[i] = li

    if mode == "beat":
        lt = base_tempo_transition.clone()
        if inv_T != 0.0:
            iv = model.state_spaces[0].interval_frames.astype(float)
            logq = np.log(np.interp(60.0 * FPS / iv, grid, q))
            lt = lt + torch.from_numpy(inv_T * logq).to(model.dtype)[None, :]
            lt = lt - torch.logsumexp(lt, dim=1, keepdim=True)   # keep a valid HMM
        model.log_tempo_transition = lt


def run(inv_T_list, out_json, mode="beat"):
    tids = H.track_ids()
    model = make_model()
    base_tt = model.log_tempo_transition.clone()
    acts, qs = {}, {}
    for tid in tids:
        b = H.bthis_beat_prob(tid); d = H.bthis_downbeat_prob(tid)
        acts[tid] = np.vstack([b, d]).T
        _, lg = H.btrans_tempo(tid)
        qs[tid] = smeared_tempo_posterior(lg)
    res = {}
    for inv_T in inv_T_list:
        per = {}
        for tid in tids:
            set_prior(model, qs[tid], inv_T, mode=mode, base_tempo_transition=base_tt)
            ev = model.predict(acts[tid])
            per[tid] = H.ev(H.load_gt(tid), np.asarray(ev["beats"]))
        res[str(inv_T)] = per
        m = H.mean_scores([per[t] for t in tids])
        print(f"[{mode}] invT={inv_T:<10g} " + " ".join(f"{k}={m[k]:.4f}" for k in m), flush=True)
        json.dump(res, open(out_json, "w"))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--invT", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--mode", type=str, default="beat")
    a = ap.parse_args()
    run([float(x) for x in a.invT.split(",")], a.out, mode=a.mode)
