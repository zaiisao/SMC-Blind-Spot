"""TTA and the learned tempo observation, TOGETHER, on the beat-only decoder. All rows same setup.

WHY THIS RUN EXISTS. The two candidate gains have never been measured on the same decoder:
  * TTA's verified figure (+0.0180 F / +0.0366 CMLt) was PEAK-PICKED, no DBN.
  * The one TTA+DBN run used the JOINT beat+downbeat DBN, which we have since shown is the wrong
    decoder for SMC (TTA+joint CMLt 0.5413 is BELOW beat-only alone at 0.5685).
  * The tempo head was measured on beat-only, without TTA.
Adding numbers across those setups would be exactly the cross-setup arithmetic this project has
already been burned by, so every row below is recomputed here on the beat-only class-corrected
DBN2016 (min_bpm=30, obs_λ=16, λ=100), n=217, paired.

THE REDUNDANCY QUESTION, which is the point. TTA averages un-warped activations over time-stretched
copies. Part of why it helps on SMC is that stretching drags slow material toward the model's trained
tempo range -- the SAME disease the tempo head treats (equivariance holds inside ±20% and collapses
outside, §3/P1). So the two may be REDUNDANT rather than additive. This measures that directly:

    null            raw activations, no tempo term
    tta             TTA activations, no tempo term
    tempo           raw activations + q_theta
    tta+tempo       both

If (tta+tempo − null) < (tta − null) + (tempo − null), they overlap. That is a result either way,
and it is the number needed before claiming a combined gain in a paper.

beta defaults to 0.2, the measured optimum for the learned head (§4.7): at beta=5 the tempo term is
~27x stronger than the emission's octave evidence and a wrong octave is catastrophic.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events
from train_tempo_observation import load_net

CACHE = ROOT / "data" / "ts_tta_cache"
SMC_EMB = Path("/disk1/jaehoon/smc_bt_embeddings")
HEAD = Path("/disk1/jaehoon/tempo_obs_head.pt")
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
BETA = float(os.environ.get("BETA", "0.2"))
RATES = [float(r) for r in os.environ.get("RATES", "0.85,1.0,1.2").split(",")]
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"


def tta_prob(tid):
    """Mean of un-warped logits across rates, then sigmoid (the cache stores original-time-base logits)."""
    ls = []
    for r in RATES:
        # cache files are r0.5 r0.7 r0.85 r1.0 r1.2 r1.5 r2.0 -- ":g" renders 1.0 as "r1"
        # and matched nothing, silently skipping every track (n=0).
        f = CACHE / f"{tid}__r{r}.npz"
        if not f.exists():
            return None
        ls.append(np.asarray(np.load(f)["logits"], dtype=np.float64))
    n = min(len(x) for x in ls)
    return expit(np.stack([x[:n] for x in ls], 0).mean(0))


def raw_prob(tid):
    f = CACHE / f"{tid}__r1.0.npz"
    if not f.exists():
        return None
    return expit(np.asarray(np.load(f)["logits"], dtype=np.float64))


def main():
    dev = torch.device(DEV)
    m = make_beatonly_dbn(30.0, 16, 100, DEV)
    space = m.state_spaces[0]
    iv = space.interval_frames.astype(np.int64)
    idx = {int(v): i for i, v in enumerate(iv)}
    tmap = torch.from_numpy(np.array([idx[int(v)] for v in space.state_interval_frames])).to(dev)
    bf = torch.tensor(space.state_interval_frames, dtype=torch.float64, device=dev)
    s2c = m.state_to_classes[0].to(dev)
    net = load_net(str(HEAD), DEV)
    marg = np.load(str(HEAD) + ".marginal.npy")
    logmarg = torch.tensor(np.log(np.clip(marg, 1e-12, None)), dtype=torch.float64, device=dev)
    print(f"rates {RATES}  beta {BETA}  beat-only decoder", flush=True)

    ARMS = ("null", "tta", "tempo", "tta+tempo")
    per = {a: {} for a in ARMS}
    for t in track_ids():
        pr, pt = raw_prob(t), tta_prob(t)
        if pr is None or pt is None or not (SMC_EMB / f"{t}.npz").exists():
            continue
        z = np.load(SMC_EMB / f"{t}.npz", allow_pickle=True)
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(z["emb"], np.float32)).permute(1, 0).unsqueeze(0).to(dev)
            logq_full = torch.log_softmax(net(x).float(), 1)[0].T.double()
        gt = load_gt(t)
        for arm in ARMS:
            got = emissions_beatonly(m, pt if "tta" in arm else pr)
            if got is None:
                break
            lcd, dec, ff = got
            T = lcd.shape[0]
            if "tempo" in arm:
                lq = logq_full
                if lq.shape[0] < ff + T:
                    lq = torch.cat([lq, lq[-1:].repeat(ff + T - lq.shape[0], 1)], 0)
                ratio = lq[ff:ff + T] - logmarg[None, :]
                em, stc = lcd[:, s2c] + BETA * (ratio[:, tmap] / bf.unsqueeze(0)), None
            else:
                em, stc = lcd, s2c
            path, _ = m.dynamic_programs[0].viterbi(
                m.log_initial_distributions[0], m.log_tempo_transition, em,
                state_to_class=stc, return_log_score=True)
            per[arm][t] = ev(gt, state_path_to_events(
                path.cpu().numpy(), space, FPS, snap_to_activations=dec, first_frame=ff)["beats"])

    ids = sorted(set.intersection(*[set(per[a]) for a in ARMS]))
    A = lambda r: {k: float(np.mean([r[t][k] for t in ids])) for k in METRICS}
    print(f"\nn={len(ids)}")
    print(f"{'arm':<12}" + "".join(f"{k:>9}" for k in METRICS))
    for a in ARMS:
        print(f"{a:<12}" + "".join(f"{A(per[a])[k]:>9.4f}" for k in METRICS))
    print("\npaired vs null:")
    d = {}
    for a in ARMS[1:]:
        out = []
        for k in METRICS:
            x = np.array([per[a][t][k] - per["null"][t][k] for t in ids])
            p = wilcoxon(x).pvalue if np.any(x) else 1.0
            out.append(f"{k} {x.mean():+.4f}(p={p:.2g})")
            if k == "CMLt":
                d[a] = x.mean()
        print(f"  {a:<10}" + "  ".join(out))
    add = d["tta"] + d["tempo"]
    print(f"\nADDITIVITY on CMLt:  tta {d['tta']:+.4f}  +  tempo {d['tempo']:+.4f}  = {add:+.4f} "
          f"if independent")
    print(f"                     actual combined            = {d['tta+tempo']:+.4f}   "
          f"({'ADDITIVE' if d['tta+tempo'] >= add - 0.005 else 'REDUNDANT: overlap ' + format(add - d['tta+tempo'], '+.4f')})")
    json.dump({"n": len(ids), "beta": BETA, "rates": RATES,
               "summary": {a: A(per[a]) for a in ARMS}, "per_track": per},
              open("/tmp/overnight/tta_plus_tempo.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
