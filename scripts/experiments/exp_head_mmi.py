"""Train a small head on FROZEN features with MMI + BCE, under an honest val/test protocol.

WHY THIS COMBINATION, AFTER EVERYTHING THAT FAILED.
MMI fine-tuning of Beat This failed for reasons that were about the SETTING, not the objective:
  * 20M parameters fine-tuned on 138 clips destroyed the model (-0.137 val CMLt with pure BCE, no
    MMI involved at all -- the control degraded exactly as much as the treatment);
  * the checkpoint had already MEMORISED its training folds (CMLt 0.74-0.81 seen vs 0.63 held out),
    so there was no headroom for any loss to exploit;
  * the numerator was INFEASIBLE on 44% of training tracks, so the reported loss was a penalty
    tally rather than a log-probability.
A small head on frozen features fixes the first two structurally: it starts from random (headroom
exists), it has ~1.5M parameters (little to overfit), and it costs seconds per epoch (so a real
validation fold is affordable). This script fixes the third explicitly.

THE NUMERATOR REPAIR. mmi.numerator_mask() forces a beat at each annotation's EXACT frame, which is
often unsatisfiable: the tempo grid is 60 log-spaced points over 14..100 frames and omits 27 of the
87 integers in that range, so "be in a beat region at frame f AND at frame f+45" can have no
solution. Measured with the emission zeroed (a property of mask x state space alone, all 217 tracks):
    shipped mask (tol=3, anchored)  47.0% of tracks infeasible
    anchoring removed (tol=3)       30.4%
    anchoring removed, tol=8         9.2%
So this script (a) drops the anchoring term and (b) EXCLUDES from training any track whose numerator
is still empty, computed exactly via the two-penalty slope k = (logZ(p1)-logZ(p2))/(p1-p2). Training
on a track whose target set is empty optimises a constant.

PROTOCOL, and the mistake it exists to avoid. Yesterday every MMI number was selected on a validation
fold the checkpoint had trained on, and an hour ago I chose an epoch count by looking at TEST fold
scores. Here: test = --fold, val = a DIFFERENT held-out fold used only for epoch selection, train =
the remaining six. Test is scored once, at the val-selected epoch.

ARMS: --mmi-w 0 is the pure-BCE control, identical in every other respect. Without it nothing here
is attributable -- that was the single most costly omission of the previous round.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from exp_mert_plus_bt import Head, resample_feats
from gate_mmi_feasible import build_mask, P1, P2
from mmi import annotation_frames, mmi_loss_batched
from rungs.bar_pointer.readout import state_path_to_events

BT_EMB = Path("/disk1/jaehoon/smc_bt_embeddings")
MERT = Path("/disk1/jaehoon/vbpm_mert_layers/smc")
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0, help="TEST fold, scored once at the end")
    ap.add_argument("--val-fold", type=int, default=7, help="epoch selection only, never scored")
    ap.add_argument("--feat", default="bt", choices=["bt", "mert", "bt+mert"])
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--mmi-w", type=float, default=1.0, help="0 = pure-BCE CONTROL")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tol", type=int, default=5, help="numerator window, frames (5 = 100 ms)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--crop", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="headmmi")
    args = ap.parse_args()
    dev = torch.device(args.device)

    import beat_this.model.loss as btloss
    bce = btloss.ShiftTolerantBCELoss(pos_weight=1, tolerance=3).to(dev)
    m = make_beatonly_dbn(30.0, 16, 100, args.device)
    space, dp0 = m.state_spaces[0], m.dynamic_programs[0]
    s2c, iv = m.state_to_classes[0], space.interval_frames
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition
    edtype = logpi.dtype

    def decode(p):
        got = emissions_beatonly(m, np.clip(p, 1e-6, 1 - 1e-6))
        if got is None:
            return None
        lcd, dec, ff = got
        path, _ = dp0.viterbi(logpi, logA, lcd, state_to_class=s2c, return_log_score=True)
        return state_path_to_events(path.cpu().numpy(), space, FPS,
                                    snap_to_activations=dec, first_frame=ff)["beats"]

    print("loading frozen features...", flush=True)
    X, Y, FR, FOLD = {}, {}, {}, {}
    for tid in track_ids():
        pe, pm = BT_EMB / f"{tid}.npz", MERT / f"{tid}.pt"
        if not pe.exists():
            continue
        z = np.load(pe, allow_pickle=True)
        e = np.asarray(z["emb"], np.float32)
        T = e.shape[0]
        if args.feat != "bt":
            if not pm.exists():
                continue
            mm = resample_feats(np.asarray(torch.load(pm, map_location="cpu")["layers"][args.layer],
                                           np.float32), T).astype(np.float32)
            e = mm if args.feat == "mert" else np.concatenate([e, mm], 1)
        X[tid] = e
        v = np.zeros(T, np.float32)
        fr = np.round(np.asarray(load_gt(tid), float) * FPS).astype(int)
        fr = fr[(fr >= 0) & (fr < T)]
        v[fr] = 1.0
        Y[tid], FR[tid], FOLD[tid] = v, fr, int(z["fold"])
    ids = sorted(X)

    # ---- feasibility filter: exclude tracks whose numerator set is EMPTY ----
    feas = {}
    for tid in ids:
        got = emissions_beatonly(m, np.asarray(bthis_beat_prob(tid), np.float64))
        if got is None:
            continue
        lcd, _, ff = got
        T = lcd.shape[0]
        f2 = annotation_frames(np.asarray(load_gt(tid), float), FPS, ff, T)
        if len(f2) < 8:
            continue
        z2 = [float(dp0.forward_log_likelihood(
            logpi, logA, torch.zeros_like(lcd) + build_mask(f2, T, iv, lcd.device, lcd.dtype,
                                                            args.tol, False, p),
            state_to_class=s2c)) for p in (P1, P2)]
        feas[tid] = int(round((z2[0] - z2[1]) / (P1 - P2))) == 0

    test = [t for t in ids if FOLD[t] == args.fold]
    val = [t for t in ids if FOLD[t] == args.val_fold]
    train = [t for t in ids if FOLD[t] not in (args.fold, args.val_fold)]
    train_f = [t for t in train if feas.get(t, False)]
    print(f"feat={args.feat} d_in={X[ids[0]].shape[1]}  mmi_w={args.mmi_w} alpha={args.alpha} "
          f"tol={args.tol} (no anchoring)")
    print(f"test fold {args.fold} n={len(test)} | val fold {args.val_fold} n={len(val)} | "
          f"train {len(train)} -> {len(train_f)} after feasibility filter "
          f"({100*(1-len(train_f)/max(len(train),1)):.0f}% dropped)", flush=True)
    if args.mmi_w == 0:
        train_f = train                      # control uses the same data it would without MMI
        print(f"  CONTROL (mmi_w=0): filter not applied, train={len(train_f)}", flush=True)

    torch.manual_seed(0)
    net = Head(X[ids[0]].shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(0)

    def score(subset):
        net.eval()
        out = {}
        for t in subset:
            with torch.no_grad():
                lg = net(torch.from_numpy(X[t]).permute(1, 0).unsqueeze(0).to(dev))[0]
            e = decode(expit(lg.double().cpu().numpy()))
            if e is not None:
                out[t] = ev(np.asarray(load_gt(t), float), e)
        return out

    A = lambda r: {k: float(np.mean([v[k] for v in r.values()])) for k in METRICS}
    best_val, best_state, best_ep = -1.0, None, -1
    for ep in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(len(train_f))
        tm = tb = nb = 0
        for b0 in range(0, len(order), args.batch):
            xs, ys, frs = [], [], []
            for i in order[b0:b0 + args.batch]:
                t = train_f[i]
                if X[t].shape[0] < args.crop:
                    continue
                o = int(rng.integers(0, X[t].shape[0] - args.crop + 1))
                f2 = FR[t][(FR[t] >= o) & (FR[t] < o + args.crop)] - o
                if len(f2) < 8:
                    continue
                xs.append(X[t][o:o + args.crop]); ys.append(Y[t][o:o + args.crop]); frs.append(f2)
            if not xs:
                continue
            xb = torch.from_numpy(np.stack(xs)).permute(0, 2, 1).to(dev)
            yb = torch.from_numpy(np.stack(ys)).to(dev)
            lg = net(xb)
            loss = args.alpha * bce(lg, yb, torch.ones_like(yb, dtype=torch.bool))
            tb += float(loss)
            if args.mmi_w:
                pr = torch.sigmoid(lg).to(edtype)
                em = torch.stack([m.log_class_densities(
                    torch.stack([pr[j], torch.zeros_like(pr[j])], 1)) for j in range(len(xs))], 0)
                mk = torch.stack([build_mask(f2, args.crop, iv, em.device, em.dtype,
                                             args.tol, False, -1e4) for f2 in frs], 0)
                lm = mmi_loss_batched(dp0, logpi, logA, em, s2c, mk)[0].mean()
                loss = loss + args.mmi_w * (lm / args.crop)
                tm += float(lm) / args.crop
            nb += 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        v = A(score(val))
        print(f"ep{ep:2d}  MMI/frame {tm/max(nb,1):7.4f}  BCE {tb/max(nb,1):.4f}  "
              f"| VAL CMLt {v['CMLt']:.4f} F {v['F']:.4f}", flush=True)
        if v["CMLt"] > best_val:
            best_val, best_ep = v["CMLt"], ep
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}

    net.load_state_dict(best_state)
    print(f"\nselected epoch {best_ep} by VAL CMLt {best_val:.4f} (test never consulted)")
    res = score(test)
    base = {t: ev(np.asarray(load_gt(t), float), decode(np.asarray(bthis_beat_prob(t), np.float64)))
            for t in test}
    ids2 = sorted(set(res) & set(base))
    print(f"\nTEST fold {args.fold}, n={len(ids2)}")
    print(f"{'':<12}" + "".join(f"{k:>9}" for k in METRICS))
    print(f"{'Beat This':<12}" + "".join(f"{A({t: base[t] for t in ids2})[k]:>9.4f}" for k in METRICS))
    print(f"{'head':<12}" + "".join(f"{A({t: res[t] for t in ids2})[k]:>9.4f}" for k in METRICS))
    for k in METRICS:
        d = np.array([res[t][k] - base[t][k] for t in ids2])
        print(f"  d{k:<5} {d.mean():+.4f} (p={wilcoxon(d).pvalue:.3g})")
    json.dump({"fold": args.fold, "feat": args.feat, "mmi_w": args.mmi_w, "best_ep": best_ep,
               "best_val_cmlt": best_val, "test": res, "base": base},
              open(f"/tmp/overnight/{args.tag}.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
