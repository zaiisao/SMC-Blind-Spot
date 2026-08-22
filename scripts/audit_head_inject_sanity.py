"""AUDIT: push a SYNTHETIC ORACLE q through the EXACT injection path of eval_tempo_head_dbn.py.

If this does not reproduce ~+0.22 CMLt at beta=5, the bug is in eval_tempo_head_dbn.py.
Also reports diagnostics: head/oracle entropy, ratio range, frame alignment cross-check.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events
from train_tempo_observation import load_net

SMC_EMB = Path("/disk1/jaehoon/smc_bt_embeddings")
HEAD = Path("/disk1/jaehoon/tempo_obs_head.pt")
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
SHARP = 0.15
# arms: (name, source, denom, beta)
BETAS = [0.2, 1.0, 5.0]


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
    loguni = torch.full_like(logmarg, float(np.log(1.0 / len(iv))))
    g = np.log(iv.astype(np.float64))
    print("grid frames asc?", bool((np.diff(iv) > 0).all()), iv[:3], iv[-3:], flush=True)

    arms = [("null", None, None, 0.0)]
    for b in BETAS:
        arms.append((f"oracle_uni_b{b}", "oracle", "uni", b))
        arms.append((f"oracle_marg_b{b}", "oracle", "marg", b))
        arms.append((f"head_marg_b{b}", "head", "marg", b))
        arms.append((f"head_uni_b{b}", "head", "uni", b))
    per = {a[0]: {} for a in arms}
    diag = {"H_head": [], "H_oracle": [], "rng_head": [], "rng_oracle": [], "argmax_agree": []}

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
        lq_head = logq[ff:ff + T]

        # oracle, EXACTLY as exp_oracle_tempo_beatonly builds it (window-relative frames)
        bt = np.asarray(load_gt(t), float)
        fr = bt * FPS - ff
        ibi = np.diff(bt) * FPS
        true = np.full(T, np.nan)
        for k in range(len(fr) - 1):
            lo, hi = max(0, int(np.ceil(fr[k]))), min(T, int(np.ceil(fr[k + 1])))
            if hi > lo:
                true[lo:hi] = ibi[k]
        if (~np.isnan(true)).sum() < 10:
            continue
        true = np.where(~np.isnan(true), true, np.nanmedian(true))
        d = (g[None, :] - np.log(true)[:, None]) / SHARP
        lq_or = torch.tensor(-0.5 * d ** 2, dtype=torch.float64, device=dev)
        lq_or = lq_or - torch.logsumexp(lq_or, 1, keepdim=True)

        diag["H_head"].append(float(-(lq_head.exp() * lq_head).sum(1).mean()))
        diag["H_oracle"].append(float(-(lq_or.exp() * lq_or).sum(1).mean()))
        diag["rng_head"].append(float((lq_head - logmarg[None]).max() - (lq_head - logmarg[None]).min()))
        diag["rng_oracle"].append(float((lq_or - loguni[None]).max() - (lq_or - loguni[None]).min()))
        diag["argmax_agree"].append(float((lq_head.argmax(1) == lq_or.argmax(1)).double().mean()))

        for name, src, den, beta in arms:
            if beta == 0.0:
                em, stc = lcd, s2c
            else:
                lq = lq_head if src == "head" else lq_or
                dn = logmarg if den == "marg" else loguni
                em = lcd[:, s2c] + beta * ((lq - dn[None, :])[:, tmap] / bf.unsqueeze(0))
                stc = None
            path, _ = m.dynamic_programs[0].viterbi(
                m.log_initial_distributions[0], m.log_tempo_transition, em,
                state_to_class=stc, return_log_score=True)
            per[name][t] = ev(load_gt(t), state_path_to_events(
                path.cpu().numpy(), space, FPS, snap_to_activations=dec, first_frame=ff)["beats"])

    ids = sorted(set.intersection(*[set(per[a[0]]) for a in arms]))
    A = lambda r: {k: float(np.mean([r[t][k] for t in ids])) for k in METRICS}
    print(f"\nn={len(ids)}")
    for k, v in diag.items():
        print(f"  {k}: mean {np.mean(v):.4f}")
    print(f"{'arm':>18}" + "".join(f"{k:>9}" for k in METRICS) + "   dCMLt(p)")
    base = per["null"]
    for name, *_ in arms:
        a = A(per[name])
        dd = np.array([per[name][t]["CMLt"] - base[t]["CMLt"] for t in ids])
        p = wilcoxon(dd).pvalue if np.any(dd) else 1.0
        print(f"{name:>18}" + "".join(f"{a[k]:>9.4f}" for k in METRICS)
              + f"   {dd.mean():+.4f}(p={p:.2g})", flush=True)
    json.dump({"n": len(ids), "summary": {a[0]: A(per[a[0]]) for a in arms},
               "diag": {k: float(np.mean(v)) for k, v in diag.items()}},
              open("/tmp/overnight/audit_head_inject_sanity.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
