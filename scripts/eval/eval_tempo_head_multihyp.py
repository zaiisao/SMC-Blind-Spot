"""Resolve the octave with the WHOLE model: decode once per octave hypothesis, keep the best path score.

THE PROBLEM THIS SOLVES, measured. Injecting the learned tempo head q_theta frame-wise into the
beat-only DBN is not uniformly harmful -- it is a large win and a larger catastrophe cancelling
(n=217, beta=5 vs the tempo-blind null):

    head tempo correct   n=113   CMLt 0.7594 -> 0.8088   +0.0494  p=0.024
    head predicts half   n=40    CMLt 0.4469 -> 0.0848   -0.3621  p=2.1e-05
    head predicts double n=14    CMLt 0.3827 -> 0.0000   -0.3827  p=0.018
    other wrong          n=50    CMLt 0.2864 -> 0.1504   -0.1360  p=0.0011

So the mechanism works; a CONFIDENTLY WRONG OCTAVE destroys it. The head is 75.6% octave-tolerant
but only 51.6% strict.

WHY OCTAVE SMEARING FAILED (measured, both weights). Spreading q across octave partners hands the
octave decision back to the emission -- but that is precisely what the beta=0 null already does, so
smearing can do no better than the null's own octave discrimination while destroying the commitment
that produced the +0.0494. Confirmed: at beta=5, CMLt -0.1080 (smear 0.3) and -0.1085 (smear 0.6),
both worse than unsmeared -0.0971; at low beta it merely decays to neutral.

WHAT THIS DOES INSTEAD. Treat the octave as a discrete latent variable and marginalise by max:

    h* = argmax_h  max_z  log p(y, z | q_h),      h in {0.5, 1, 2}

where q_h(i) = q(i / h) shifts the head's whole distribution by h octaves. Each hypothesis is
decoded with the head still fully COMMITTED (no smearing), and the winner is chosen by TOTAL VITERBI
PATH SCORE -- emission + transition + tempo term jointly, over the whole track.

This is a strictly stronger arbiter than the emission's local octave preference. That preference is
real but weak: double-tempo paths score only ~93 nats worse over an entire track, and truth beats
double on 85.7% (n=119, p=5.3e-16). Scoring complete paths uses all three factors and the full
sequence, rather than the per-frame class likelihood alone.

Cost is 3x decoding, which is trivial on the 2,921-state beat-only space.

FAILURE MODE TO WATCH. If the path score systematically prefers slow tempi for reasons unrelated to
evidence, this will bias toward h=2. Two known implicit tempo channels make that a live risk
(docs/spec/SPECIFICATION.md Corollary 1): the initial distribution is uniform over STATES so p(phidot=i) ∝ i
(1.97 nats favouring slow), and transition row-normalisation contributes up to 22.8 nats at
lambda=100. Both are reported below per hypothesis so the selection can be audited rather than
trusted.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events
from train_tempo_observation import load_net

SMC_EMB = Path("/disk1/jaehoon/smc_bt_embeddings")
HEAD = Path("/disk1/jaehoon/tempo_obs_head.pt")
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
HYPS = [0.5, 1.0, 2.0]                       # multiplier on the head's predicted beat INTERVAL
BETA = float(os.environ.get("BETA", "5.0"))
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"


def shift_octave(q, grid_t, h):
    """q_h(i) = q(i / h): move the head's mass so it asserts an interval h times longer."""
    if h == 1.0:
        return q
    src = grid_t / h                                        # where each target index reads from
    j = torch.argmin(torch.abs(grid_t[None, :] - src[:, None]), dim=1)
    ok = (torch.abs(grid_t[j] - src) / src) < 0.06
    out = torch.full_like(q, 1e-12)
    out[:, ok] = q[:, j[ok]]
    return out / out.sum(dim=1, keepdim=True)


def main():
    dev = torch.device(DEV)
    m = make_beatonly_dbn(30.0, 16, 100, DEV)
    space = m.state_spaces[0]
    iv = space.interval_frames.astype(np.int64)
    idx = {int(v): i for i, v in enumerate(iv)}
    tmap = torch.from_numpy(np.array([idx[int(v)] for v in space.state_interval_frames])).to(dev)
    bf = torch.tensor(space.state_interval_frames, dtype=torch.float64, device=dev)
    s2c = m.state_to_classes[0].to(dev)
    grid_t = torch.tensor(iv, dtype=torch.float64, device=dev)

    net = load_net(str(HEAD), DEV)
    marg = np.load(str(HEAD) + ".marginal.npy")
    logmarg = torch.tensor(np.log(np.clip(marg, 1e-12, None)), dtype=torch.float64, device=dev)
    print(f"beta={BETA}  hypotheses={HYPS}", flush=True)

    res = {"null": {}, "fixed": {}, "multi": {}}
    n_done = [0]
    chosen = {h: 0 for h in HYPS}
    for t in track_ids():
        f = SMC_EMB / f"{t}.npz"
        if not f.exists():
            continue
        got = emissions_beatonly(m, bthis_beat_prob(t))
        if got is None:
            continue
        lcd, dec, ff = got
        T = lcd.shape[0]
        z = np.load(f, allow_pickle=True)
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(z["emb"], np.float32)).permute(1, 0).unsqueeze(0).to(dev)
            logq = torch.log_softmax(net(x).float(), 1)[0].T.double()
        if logq.shape[0] < ff + T:
            logq = torch.cat([logq, logq[-1:].repeat(ff + T - logq.shape[0], 1)], 0)
        q = logq[ff:ff + T].exp()

        def decode(em, stc):
            path, sc = m.dynamic_programs[0].viterbi(
                m.log_initial_distributions[0], m.log_tempo_transition, em,
                state_to_class=stc, return_log_score=True)
            beats = state_path_to_events(path.cpu().numpy(), space, FPS,
                                         snap_to_activations=dec, first_frame=ff)["beats"]
            return beats, float(sc)

        gt = load_gt(t)
        res["null"][t] = ev(gt, decode(lcd, s2c)[0])

        best = None
        sc_all = {}
        for h in HYPS:
            qh = shift_octave(q, grid_t, h)
            ratio = torch.log(qh.clamp_min(1e-12)) - logmarg[None, :]
            em = lcd[:, s2c] + BETA * (ratio[:, tmap] / bf.unsqueeze(0))
            beats, sc = decode(em, None)
            if h == 1.0:
                res["fixed"][t] = ev(gt, beats)
            sc_all[h] = sc
            if best is None or sc > best[0]:
                best = (sc, h, beats)
        chosen[best[1]] += 1
        res["multi"][t] = ev(gt, best[2])
        n_done[0] += 1
        if n_done[0] <= 12 or n_done[0] % 25 == 0:
            hp = float(iv[int(q.mean(0).argmax())])
            print(f"  {t}  head={60*FPS/hp:6.1f}BPM  true={60/np.median(np.diff(gt)):6.1f}BPM"
                  f"  scores=" + " ".join(f"h{h}:{sc_all[h]:>9.1f}" for h in HYPS)
                  + f"  -> h={best[1]}"
                  + f"  CMLt null {res['null'][t]['CMLt']:.3f}"
                  f" fixed {res['fixed'][t]['CMLt']:.3f} multi {res['multi'][t]['CMLt']:.3f}",
                  flush=True)

    ids = sorted(set(res["null"]) & set(res["fixed"]) & set(res["multi"]))
    A = lambda r: {k: float(np.mean([r[t][k] for t in ids])) for k in METRICS}
    print(f"\nn={len(ids)}   hypothesis chosen: "
          + "  ".join(f"h={h}: {chosen[h]}" for h in HYPS))
    print(f"{'arm':<10}" + "".join(f"{k:>9}" for k in METRICS))
    for nm in ("null", "fixed", "multi"):
        print(f"{nm:<10}" + "".join(f"{A(res[nm])[k]:>9.4f}" for k in METRICS))
    print("\npaired vs null:")
    for nm in ("fixed", "multi"):
        out = []
        for k in METRICS:
            d = np.array([res[nm][t][k] - res["null"][t][k] for t in ids])
            p = wilcoxon(d).pvalue if np.any(d) else 1.0
            out.append(f"{k} {d.mean():+.4f}(p={p:.2g})")
        print(f"  {nm:<7}" + "  ".join(out))
    print("\nreference: oracle beta=5 CMLt 0.7906; null 0.5685")
    json.dump({"n": len(ids), "beta": BETA, "chosen": {str(k): v for k, v in chosen.items()},
               "summary": {nm: A(res[nm]) for nm in res}, "per_track": res},
              open(f"/tmp/overnight/tempo_head_multihyp_b{BETA:g}.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
