"""Learned emission head for the 2016 DBN: p(class | h) -> scaled likelihood, ASR-hybrid style.

MOTIVATION (derived, not hopeful). Within the Boeck-2016 geometric class assignment we PROVED
that a fixed-millisecond beat region cannot be tempo-neutral: the accumulated emission for a path
at tempo i is E(i) = n_beat*log(b') + n_nonbeat*[log(1-b'-d) - log D] with n_beat = T*width/i, so
only a width proportional to i (the fractional rule) makes E(i) constant. Measured spreads across
tempi: fractional 0.0 nats, fixed-W with D=15 1671 nats, fixed-W with D=(i-W)/W 5134 nats. A
learned head is not bound by that identity because it does not assign classes by position at all.

RECIPE. Train p(class | h_t) on the 512-d penultimate Beat This embeddings, then feed the DBN
    log p(y_t | class) = log p(class | h_t) - log p(class)
(the hybrid HMM-DNN scaled likelihood). The DBN itself is UNCHANGED.

HONESTY. Embeddings are fold-honest (each track encoded by the checkpoint that held it out; SMC
alignment verified bit-exact, 0.000e+00 on 217/217). The head trains on 2559 corpus tracks
(gtzan/ballroom/hainsworth/beatles/asap) and is evaluated on the 217 SMC tracks it never saw.

GATES, both enforced before any score is reported:
  T1  TEMPO NEUTRALITY of the resulting emission. The same E(i) spread diagnostic that killed the
      fixed-ms geometric region. A head that buys constant tolerance by reintroducing a tempo bias
      will look good at frame level and decode badly -- exactly how the last two attempts failed.
  T2  The baseline is the TUNED hand-set configuration, not madmom's default.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
EMB = Path("/disk1/jaehoon/bt_emb")
NO_BEAT, BEAT, DOWNBEAT = 0, 1, 2


def load_split(emb_dir, lab_dir, datasets, max_tracks=0):
    X, Y, G = [], [], []
    for ds in datasets:
        de, dl = emb_dir / ds, lab_dir / ds
        if not de.exists() or not dl.exists():
            continue
        fs = sorted(de.glob("*.npz"))
        if max_tracks:
            fs = fs[:max_tracks]
        for f in fs:
            lp = dl / f.name
            if not lp.exists():
                continue
            z, l = np.load(f), np.load(lp)
            e, lab = z["emb"], l["label"]
            n = min(len(e), len(lab))
            e, lab = e[:n], lab[:n]
            keep = lab >= 0            # -1 marks frames with no valid target (e.g. tempo off-grid)
            if not keep.any():
                continue
            X.append(e[keep]); Y.append(lab[keep])
            G.append(np.full(int(keep.sum()), int(z["fold"]), np.int8))
    return (np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.int64),
            np.concatenate(G))


class Head(nn.Module):
    def __init__(self, dim=512, hidden=0, ctx=0, n_classes=3):
        super().__init__()
        self.ctx = ctx
        self.n_classes = n_classes
        d = dim * (2 * ctx + 1)
        self.net = nn.Linear(d, n_classes) if not hidden else nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def stack_ctx(X, ctx):
    if not ctx:
        return X
    pads = [np.roll(X, s, axis=0) for s in range(-ctx, ctx + 1)]
    return np.concatenate(pads, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/disk1/jaehoon/bt_labels_ms35")
    ap.add_argument("--datasets", default="gtzan,ballroom,hainsworth,beatles,asap")
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--max-tracks", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dsets = [d.strip() for d in args.datasets.split(",")]

    X, Y, G = load_split(EMB, Path(args.labels), dsets, args.max_tracks)
    NC = int(Y.max()) + 1
    print(f"train frames {len(X):,}  dim {X.shape[1]}  "
          f"class balance {np.bincount(Y, minlength=NC) / len(Y)}", flush=True)
    if args.ctx:
        X = stack_ctx(X, args.ctx)
    prior = np.bincount(Y, minlength=NC) / len(Y)

    dev = torch.device(args.device)
    head = Head(512, args.hidden, args.ctx, NC).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    # class-balanced loss: the NO_BEAT class dominates ~90% of frames
    w = torch.tensor((1.0 / np.clip(prior, 1e-8, None)) ** 0.5, dtype=torch.float32, device=dev)
    w = w / w.mean()
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y)
    n = len(Xt)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            xb, yb = Xt[idx].to(dev, non_blocking=True), Yt[idx].to(dev, non_blocking=True)
            opt.zero_grad(); out = head(xb); loss = lossf(out, yb)
            loss.backward(); opt.step(); tot += float(loss) * len(idx)
        with torch.no_grad():
            sub = perm[:200000]
            pred = head(Xt[sub].to(dev)).argmax(1).cpu().numpy()
            acc = float((pred == Y[sub.numpy()]).mean())
            yb2 = Y[sub.numpy()]
            on = (yb2 % (NC // 2 if NC > 3 else NC)) == 0 if NC > 3 else (yb2 != NO_BEAT)
            beat_rec = float((pred[on] == yb2[on]).mean()) if on.any() else float("nan")
        print(f"  epoch {ep}: loss {tot/n:.4f}  acc {acc:.4f}  beat-recall {beat_rec:.4f}", flush=True)

    torch.save({"state": head.state_dict(), "prior": prior, "hidden": args.hidden,
                "ctx": args.ctx, "n_classes": NC, "labels": args.labels, "datasets": dsets}, args.out)
    print(f"prior p(class) = {prior}")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
