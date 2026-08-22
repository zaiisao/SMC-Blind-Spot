"""Generation-1 Kalman-VAE, re-run on FOLD-HONEST Beat This features, so it lands in our table.

WHY RE-RUN IT. KVAE_GENERATIONS.md records F = 0.654 on SMC (CMLt 0.587 / AMLt 0.616) for the
2026-07 Kalman-VAE, which would sit ABOVE our deployed peak-pick (0.6270) and beat-only DBN (0.6141).
But that run used the `final0` Beat This checkpoint, which is trained on everything except GTZAN and
therefore SAW the SMC tracks it was scored on. Our numbers use, per track, the fold checkpoint that
HELD THAT TRACK OUT. Beat This is trained on SMC either way -- the difference is only which
checkpoint -- so the fix is not "don't compare" but "re-run it fold-honestly".

WHAT THE MODEL IS (KVAE_GENERATIONS.md 'What the latents actually mean'): a generic 8-dim
linear-Gaussian state with free learned A/C/Q/R and a 5-way LSTM-gated mixture. NO tempo latent, NO
beat-phase latent, NO bar pointer, NO tempo transition kernel. It is the off-diagonal cell of a
two-axis framing: exact marginalisation (a real Kalman filter/smoother, not a smoothness penalty)
WITHOUT a musical prior. That is what makes the comparison interesting rather than redundant -- the
bar-pointer DBN is the opposite cell.

FEATURES. /disk1/jaehoon/bt_emb/{ds}/*.npz for training (fold-honest 512-d penultimate embeddings,
carrying `beat`/`downbeat` logits and `start_frame`), /disk1/jaehoon/smc_bt_embeddings for eval.
SMC is never trained on.

READ-OUT. head(filtered z) -> sigmoid -> peak-pick, exactly as the original ('LEARNED-HEAD' path in
the docstring of model/kalman_vae_bar_pointer.py), so the comparison is to our peak-pick row.

THE NUMBER THIS PRODUCES IS THE POINT. If a generic-latent KVAE on fold-honest features lands near
0.62-0.63, the 0.654 was checkpoint contamination and the two-axis story stands as written. If it
lands materially higher, exact marginalisation is worth more than we credited and that changes what
the paper should claim.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/tmp/kvae_smoke")
sys.path.insert(0, "/tmp/kvae_smoke/third_party/kalman-vae")
from _harness import FPS, track_ids, load_gt, ev, peakpick
from model.kalman_vae_bar_pointer import KalmanVAEBarPointer, kvae_elbo
from kvae.sample_control import SampleControl

EMB = Path("/disk1/jaehoon/bt_emb")
SMC_EMB = Path("/disk1/jaehoon/smc_bt_embeddings")
STORE = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ballroom,hainsworth,beatles,gtzan,asap")
    ap.add_argument("--per-ds", type=int, default=60)
    ap.add_argument("--crop", type=int, default=400)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--elbo-w", type=float, default=1e-3,
                    help="ELBO is summed over 512 feature dims and dwarfs the head BCE; this scales "
                         "it back so the head actually trains")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", default="/tmp/overnight/kvae_foldhonest.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rng = np.random.default_rng(0)

    # ---------- training pool: corpus, fold-honest embeddings ----------
    pool = []
    for ds in args.datasets.split(","):
        d = EMB / ds
        if not d.exists():
            continue
        for f in sorted(d.glob("*.npz"))[:args.per_ds]:
            ann = STORE / ds / "annotations" / "beats" / f"{f.stem}.beats"
            if not ann.exists():
                continue
            z = np.load(f, allow_pickle=True)
            emb = np.asarray(z["emb"], np.float32)
            T = emb.shape[0]
            sf = int(z["start_frame"])
            rows = [l.split() for l in open(ann) if l.strip()]
            bt = np.array([float(r[0]) for r in rows]) * FPS - sf
            bn = np.array([int(float(r[1])) if len(r) > 1 else 0 for r in rows])
            yb = np.zeros(T, np.float32); yd = np.zeros(T, np.float32)
            for fr, isd in zip(np.rint(bt).astype(int), bn):
                if 0 <= fr < T:
                    yb[fr] = 1.0
                    if isd == 1:
                        yd[fr] = 1.0
            if yb.sum() < 8:
                continue
            pool.append((emb, yb, yd))
    print(f"KVAE training pool: {len(pool)} corpus tracks (fold-honest embeddings)", flush=True)

    model = KalmanVAEBarPointer(feature_dim=512, a_dim=8, z_dim=8, K=5).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sc = SampleControl(encoder="sample", decoder="mean",
                       state_transition="sample", observation="sample")
    # pos_weight: beats are ~2% of frames; without it the head learns to predict "no beat" always
    pw = torch.tensor([20.0, 90.0], dtype=torch.float32, device=dev)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    for step in range(args.steps):
        xs, ys = [], []
        for _ in range(args.bs):
            emb, yb, yd = pool[rng.integers(len(pool))]
            if emb.shape[0] <= args.crop:
                continue
            st = int(rng.integers(0, emb.shape[0] - args.crop))
            xs.append(emb[st:st + args.crop])
            ys.append(np.stack([yb[st:st + args.crop], yd[st:st + args.crop]], -1))
        if len(xs) < 2:
            continue
        x = torch.from_numpy(np.stack(xs, 1)).to(dev)             # [T, B, 512] time-major
        y = torch.from_numpy(np.stack(ys, 1)).to(dev)             # [T, B, 2]
        elbo, z_s, _ = kvae_elbo(model, x, sc)
        logits = model.head(z_s)
        loss = bce(logits, y) - args.elbo_w * elbo
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (step + 1) % 250 == 0:
            print(f"  step {step+1}  bce {float(bce(logits, y)):.4f}  elbo {float(elbo):.1f}",
                  flush=True)

    # ---------- evaluate on SMC, fold-honest embeddings, never trained on ----------
    model.eval()
    per = {}
    with torch.no_grad():
        for t in track_ids():
            f = SMC_EMB / f"{t}.npz"
            if not f.exists():
                continue
            emb = torch.from_numpy(np.asarray(np.load(f, allow_pickle=True)["emb"], np.float32))
            x = emb.unsqueeze(1).to(dev)
            _, z_s, _ = kvae_elbo(model, x, sc)
            p = torch.sigmoid(model.head(z_s)[:, 0, 0]).cpu().numpy()
            per[t] = ev(load_gt(t), peakpick(p, 0.5, FPS))
    ids = sorted(per)
    A = {k: float(np.mean([per[t][k] for t in ids])) for k in METRICS}
    print(f"\nKVAE gen-1, FOLD-HONEST features, SMC n={len(ids)}")
    print("  " + "  ".join(f"{k} {A[k]:.4f}" for k in METRICS))
    print("\nreference rows, same n, same corpus:")
    print("  Beat This peak-pick @0.5     F 0.6270  CMLt 0.5138  AMLt 0.6100")
    print("  beat-only DBN (lambda=100)   F 0.6141  CMLt 0.5685  AMLt 0.6988")
    print("  KVAE as recorded (final0, SAW SMC)  F 0.6540  CMLt 0.5870  AMLt 0.6160")
    json.dump({"n": len(ids), "kvae_foldhonest": A,
               "per_track": per}, open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
