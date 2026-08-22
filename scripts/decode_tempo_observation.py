"""Inject q_theta into the beat-only DBN, with beta LEARNED (G3) and SMC held out (G4).

Emission (SPECIFICATION.md sec.4):
    log p(y_t|z_t) = log p(y_t|class(phi_t)) + beta * [ log q_theta(phidot_t|h_t) - log p(phidot_t) ]
with log p(phidot) = the HEAD'S OWN MARGINAL over held-out corpus frames (R1: NOT the label prior).
Per-frame term is divided by state_interval_frames, so each beat contributes exactly one beat's
worth of tempo evidence at every tempo -- identical to the oracle injection in
exp_oracle_tempo_beatonly.py, so the two are directly comparable.

beta is LEARNED, not swept (R3/G3): a conditional (CRF) maximum-likelihood fit of the tempo chain
on held-out CORPUS tracks. The chain has the DBN's own log_tempo_transition as its transition and
beta * (per-beat mean log-ratio) as its emission; the true tempo path comes from the annotations.
Nothing in the beta fit touches SMC.

Decoder for every reported number: beat-only PyTorch DBN2016 (class-corrected), beats_per_bar=1,
min_bpm=30, max_bpm=215, obs_lambda=16, transition_lambda=100, fps=50, Beat This activations,
mir_eval 5 s trim, +/-70 ms.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev
import _harness as H
from rungs.deployment import threshold_crop
from rungs.bar_pointer.readout import state_path_to_events
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
import train_tempo_observation as TT

METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
CACHE = Path("/disk1/jaehoon/tempo_obs_cache")


def q_for_embeddings(net, emb, dev):
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(emb, np.float32)).permute(1, 0).unsqueeze(0).to(dev)
        return torch.log_softmax(net(x).float(), 1)[0].T.cpu().numpy()   # [T,60] log q


# ------------------------------------------------------------------ beta (G3)
def fit_beta(net, marg, grid, log_T, dev, n_tracks=250):
    """Conditional ML fit of beta on held-out CORPUS tracks (SMC untouched)."""
    meta = json.load(open(CACHE / "meta.json"))["meta"]
    E = np.load(CACHE / "emb.npy", mmap_mode="r")
    isval = lambda m: int(hashlib.md5(m["tid"].encode()).hexdigest(), 16) % 10 == 0
    pool = [m for m in meta if isval(m) and m["kind"] == "plain"][:n_tracks]
    logm = np.log(marg)
    data = []
    for m in pool:
        bt = TT.load_beats(m["ds"], m["tid"])
        if bt is None or len(bt) < 5:
            continue
        lq = q_for_embeddings(net, E[m["off"]:m["off"] + m["T"]], dev)
        ratio = lq - logm[None, :]
        fr = bt * FPS - 0
        # this file's frames are already offset-aligned in the cache builder for plain files
        # via start_frame; recompute here the same way:
        z = np.load(TT.EMB / m["ds"] / f"{m['tid']}.npz", allow_pickle=True)
        fr = bt * FPS - int(z["start_frame"])
        emis, path = [], []
        for k in range(len(fr) - 1):
            a, b = max(0, int(np.ceil(fr[k]))), min(m["T"], int(np.ceil(fr[k + 1])))
            if b <= a:
                continue
            ibi = fr[k + 1] - fr[k]
            if ibi < grid.min() * 0.97 or ibi > grid.max() * 1.03:
                continue
            emis.append(ratio[a:b].mean(0))
            path.append(int(np.argmin(np.abs(grid - ibi))))
        if len(path) >= 5:
            data.append((torch.tensor(np.array(emis), dtype=torch.float64, device=dev),
                         torch.tensor(path, device=dev)))
    # SPEC sec.5.2(e): at lambda=100 the transition support is truncated, so tracks with real
    # tempo change score -inf on their own true path. Those tracks carry no usable gradient
    # (the objective is -inf regardless of beta), so they are dropped and counted, not silently
    # skipped.
    lT0 = log_T.to(dev).double()
    keep, dropped = [], 0
    for emis, path in data:
        if torch.isfinite(lT0[path[:-1], path[1:]]).all():
            keep.append((emis, path))
        else:
            dropped += 1
    data = keep
    print(f"  beta fit on {len(data)} held-out corpus tracks "
          f"({dropped} dropped: true tempo path is -inf under lambda=100)", flush=True)
    lb = torch.tensor(np.log(1.0), dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.Adam([lb], lr=0.1)
    lT = log_T.to(dev).double()
    for it in range(120):
        tot = 0.0
        opt.zero_grad()
        for emis, path in data:
            e = torch.exp(lb) * emis
            # forward algorithm over the tempo chain
            a = e[0] - np.log(len(grid))
            for t in range(1, e.shape[0]):
                a = torch.logsumexp(a.unsqueeze(1) + lT, 0) + e[t]
            logZ = torch.logsumexp(a, 0)
            sc = e[0, path[0]] - np.log(len(grid))
            sc = sc + lT[path[:-1], path[1:]].sum() + e[torch.arange(1, e.shape[0]), path[1:]].sum()
            tot = tot + (logZ - sc) / e.shape[0]
        loss = tot / len(data)
        loss.backward()
        opt.step()
        if it % 20 == 0 or it == 119:
            print(f"   it {it:3d}  beta {float(torch.exp(lb)):.4f}  NLL {float(loss):.4f}",
                  flush=True)
    return float(torch.exp(lb).detach())


# ------------------------------------------------------------------ decode (G4)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="/disk1/jaehoon/tempo_obs_head.pt")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--beta", type=float, default=None, help="override; default = LEARNED")
    ap.add_argument("--extra-betas", default="")
    ap.add_argument("--out", default="/tmp/overnight/tempo_obs_decode.json")
    a = ap.parse_args()

    dev = torch.device(a.device)
    TT.GRID = TT.tempo_grid(); TT.GRID_BPM = np.sort(60.0 * FPS / TT.GRID)
    net = TT.load_net(a.head, dev)
    marg = np.load(a.head + ".marginal.npy")

    m = make_beatonly_dbn(30.0, 16, 100, a.device)
    space = m.state_spaces[0]
    grid = space.interval_frames.astype(np.float64)
    assert np.allclose(grid, TT.GRID)
    idx = {iv: i for i, iv in enumerate(space.interval_frames)}
    tmap = torch.from_numpy(np.array([idx[v] for v in space.state_interval_frames])).to(dev)
    bf = torch.tensor(space.state_interval_frames, dtype=torch.float64, device=dev)

    beta = a.beta if a.beta is not None else fit_beta(net, marg, grid, m.log_tempo_transition, dev)
    print(f"beta = {beta:.4f}  ({'OVERRIDE' if a.beta is not None else 'LEARNED on corpus'})",
          flush=True)
    betas = [0.0, beta] + [float(x) for x in a.extra_betas.split(",") if x]

    logm = torch.tensor(np.log(marg), dtype=torch.float64, device=dev)
    per = {b: {} for b in betas}
    for t in track_ids():
        bpr = H.bthis_beat_prob(t)
        got = emissions_beatonly(m, bpr)
        if got is None:
            continue
        lcd, dec, ff = got
        emb = np.load(f"/disk1/jaehoon/smc_bt_embeddings/{t}.npz")["emb"]
        lq = torch.tensor(q_for_embeddings(net, emb, dev), dtype=torch.float64, device=dev)
        lq = lq[ff:ff + lcd.shape[0]]
        ratio = lq - logm[None, :]
        for b in betas:
            if b == 0.0:
                em, stc = lcd, m.state_to_classes[0]
            else:
                em = lcd[:, m.state_to_classes[0]] + b * (ratio[:, tmap] / bf.unsqueeze(0))
                stc = None
            path, _ = m.dynamic_programs[0].viterbi(
                m.log_initial_distributions[0], m.log_tempo_transition, em,
                state_to_class=stc, return_log_score=True)
            per[b][t] = ev(load_gt(t), state_path_to_events(
                path.cpu().numpy(), space, m.fps, snap_to_activations=dec, first_frame=ff)["beats"])
    ids = sorted(set.intersection(*[set(per[b]) for b in betas]))
    A = lambda r: {k: float(np.mean([r[t][k] for t in ids])) for k in METRICS}
    print(f"\nn={len(ids)}  beat-only PyTorch DBN2016, min_bpm=30, obs_lambda=16, lambda=100")
    print(f"{'beta':>8}" + "".join(f"{k:>9}" for k in METRICS))
    for b in betas:
        print(f"{b:>8.3f}" + "".join(f"{A(per[b])[k]:>9.4f}" for k in METRICS))
    print("\nvs beta=0 (tempo-blind), paired Wilcoxon:")
    stats = {}
    for b in betas[1:]:
        out = []
        st = {}
        for k in METRICS:
            d = np.array([per[b][t][k] - per[0.0][t][k] for t in ids])
            p = wilcoxon(d).pvalue if np.any(d) else 1.0
            out.append(f"{k} {d.mean():+.4f}(p={p:.1e})")
            st[k] = [float(d.mean()), float(p)]
        stats[b] = st
        print(f"  beta={b:.3f}  " + "  ".join(out))
    json.dump({"beta_learned": beta, "n": len(ids),
               "summary": {str(b): A(per[b]) for b in betas},
               "delta_vs_beta0": {str(k): v for k, v in stats.items()},
               "per_track": {str(b): per[b] for b in betas}},
              open(a.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
