"""Decisive test the ACF-blend negative never ran.

eval_tempo_head_dbn.py injects the head's PER-FRAME blended distribution. But the 20.7%
octave-error figure is a property of a DIFFERENT estimator: median-over-frames of the per-frame
argmax (a single number per track). The oracle ceiling (+0.2221 CMLt) was injected as a
track-CONSTANT log-normal bump of width SHARP=0.15 against a UNIFORM prior at beta=5.

So the question "does a tempo estimator with 20.7% octave error help decoding?" has never been
asked with the oracle's own injection. This asks it: identical code path to the oracle, with the
oracle tempo replaced by the blended point estimate.
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events
from train_tempo_observation import load_net
SMC_EMB = Path("/disk1/jaehoon/smc_bt_embeddings"); HEAD = Path("/disk1/jaehoon/tempo_obs_head.pt")
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
SHARP = 0.15
BETAS = [float(x) for x in os.environ.get("BETAS", "0.0,0.5,1.0,2.0,5.0").split(",")]
ACF_W = float(os.environ.get("ACF_W", "0.8")); ACF_TAU = float(os.environ.get("ACF_TAU", "10.0"))
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"

def main():
    dev = torch.device(DEV)
    m = make_beatonly_dbn(30.0, 16, 100, DEV); space = m.state_spaces[0]
    iv = space.interval_frames.astype(np.int64)
    idx = {int(v): i for i, v in enumerate(iv)}
    tmap = torch.from_numpy(np.array([idx[int(v)] for v in space.state_interval_frames])).to(dev)
    bf = torch.tensor(space.state_interval_frames, dtype=torch.float64, device=dev)
    s2c = m.state_to_classes[0].to(dev)
    net = load_net(str(HEAD), DEV)
    ck = torch.load(str(HEAD), map_location="cpu", weights_only=False)
    grid = np.asarray(ck["grid"], dtype=np.int64)
    assert (grid == iv).all()
    per = {b: {} for b in BETAS}; oct_err = []; n_used = 0
    for t in track_ids():
        f = SMC_EMB / f"{t}.npz"
        if not f.exists(): continue
        got = emissions_beatonly(m, bthis_beat_prob(t))
        if got is None: continue
        lcd, dec, ff = got; T = lcd.shape[0]
        gt = np.asarray(load_gt(t), float)
        z = np.load(f, allow_pickle=True)
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(z["emb"], np.float32)).permute(1, 0).unsqueeze(0).to(dev)
            logq = torch.log_softmax(net(x).float(), 1)[0].T.double()
        if logq.shape[0] < ff + T:
            logq = torch.cat([logq, logq[-1:].repeat(ff + T - logq.shape[0], 1)], 0)
        lq = logq[ff:ff + T]
        if ACF_W > 0:
            a = torch.as_tensor(np.asarray(bthis_beat_prob(t), np.float64), device=dev)
            a = a - a.mean(); Ta = len(a)
            r = torch.stack([(a[:Ta - int(l)] * a[int(l):]).sum() / float(Ta - int(l))
                             if Ta - int(l) >= 8 else torch.tensor(-1e9, dtype=torch.float64, device=dev)
                             for l in iv])
            r = r / ((a * a).sum() / Ta + 1e-12)
            lq = (1 - ACF_W) * lq + ACF_W * torch.log_softmax(ACF_TAU * r, 0)[None, :]
        est = float(np.median(grid[lq.argmax(1).cpu().numpy()]))   # deployed aggregation
        if len(gt) >= 4:
            oct_err.append(abs(abs(np.log2(est / (np.median(np.diff(gt)) * FPS))) - 1) < 0.15)
        d = (np.log(grid.astype(np.float64)) - np.log(est)) / SHARP
        lp = torch.tensor(-0.5 * d ** 2, dtype=torch.float64, device=dev)
        lp = lp - torch.logsumexp(lp, 0)
        ratio = (lp - np.log(1.0 / len(grid)))[None, :].repeat(T, 1)
        for beta in BETAS:
            if beta == 0.0: em, stc = lcd, s2c
            else:
                em = lcd[:, s2c] + beta * (ratio[:, tmap] / bf.unsqueeze(0)); stc = None
            path, _ = m.dynamic_programs[0].viterbi(m.log_initial_distributions[0],
                m.log_tempo_transition, em, state_to_class=stc, return_log_score=True)
            per[beta][t] = ev(load_gt(t), state_path_to_events(
                path.cpu().numpy(), space, FPS, snap_to_activations=dec, first_frame=ff)["beats"])
        n_used += 1
    ids = sorted(set.intersection(*[set(per[b]) for b in BETAS]))
    A = lambda r: {k: float(np.mean([r[t][k] for t in ids])) for k in METRICS}
    print(f"n={len(ids)}  ACF_W={ACF_W} tau={ACF_TAU}  (2x/0.5x rate {np.mean(oct_err):.3f})")
    print(f"{'beta':>6}" + "".join(f"{k:>9}" for k in METRICS))
    for b in BETAS: print(f"{b:>6.2f}" + "".join(f"{A(per[b])[k]:>9.4f}" for k in METRICS))
    base = per[0.0]
    for b in BETAS[1:]:
        o = []
        for k in METRICS:
            dd = np.array([per[b][t][k] - base[t][k] for t in ids])
            o.append(f"{k} {dd.mean():+.4f}(p={wilcoxon(dd).pvalue if np.any(dd) else 1.0:.2g})")
        print(f"  beta={b:.2f}  " + "  ".join(o))
    json.dump({"n": len(ids), "summary": {str(b): A(per[b]) for b in BETAS},
               "per_track": {str(b): per[b] for b in BETAS}},
              open(f"/tmp/overnight/acf_point_w{ACF_W:g}.json", "w"), indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
