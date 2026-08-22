"""Does MERT add information Beat This's own representation does not already contain?

THE HYPOTHESIS. ~83% of SMC's remaining F-gap is in the ACTIVATIONS (per-track oracle decoder
hyperparameters cap out at 0.6908 vs 0.6270 peak-picked). If the bottleneck is the richness of the
musical representation, then concatenating a self-supervised foundation model's features to Beat
This's own embedding should help -- MERT-v1-95M saw ~160k hours, Beat This ~4k tracks.

WHY THIS IS NOT THE EXPERIMENT THAT ALREADY FAILED. A MERT-only probe trained on corpus and tested
zero-shot on SMC scored F 0.3707 vs Beat This's 0.6141 -- but that measured DOMAIN TRANSFER, not
representation quality, since the probe trained on ballroom/beatles/hainsworth/hjdb and was tested on
rubato classical. BeatFM (arXiv 2508.09790) reports a MERT baseline of 60.1 F on SMC *with SMC in
its 8-fold training*, and 61.3 for its full method. This script uses the same 8-fold-on-SMC protocol
so the comparison is finally apples-to-apples -- and, more importantly, adds the CONTROL that
settles the actual question.

THE CONTROL IS THE POINT. Three arms share one architecture, one schedule, one seed, one fold split:
    bt        head on Beat This embedding alone            [512]
    mert      head on MERT layer L alone                   [768]   (~BeatFM's baseline setting)
    bt+mert   head on the concatenation                    [1280]
`bt+mert` minus `bt` isolates MERT's marginal contribution. Without the `bt` arm we would be
comparing our head against Beat This's head -- different capacity, different training budget -- and
could not tell a MERT effect from a head effect. That confusion is exactly what cost us yesterday.

FOLD HONESTY. smc_bt_embeddings/*.npz records the `fold` whose Beat This checkpoint produced it, so
every embedding already comes from a model that never saw that track. Heads are trained on 7 folds
and score the held-out one; concatenating the 8 held-out folds gives n=217 with no track ever scored
by a head that trained on it. MERT is self-supervised and never saw beat labels at all.

Frame rates: Beat This embeddings are 50 fps, MERT is 86.1 fps; MERT is resampled onto the 50 fps
grid so the head, the targets and the decoder all share one axis. Decoding is the beat-only DBN we
ship, unchanged, so any difference between arms is the representation.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events

BT_EMB = Path("/disk1/jaehoon/smc_bt_embeddings")
MERT = Path("/disk1/jaehoon/vbpm_mert_layers/smc")
MERT_FPS = 86.1328125
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")


class Head(nn.Module):
    def __init__(self, d_in, d=256):
        super().__init__()
        self.inp = nn.Conv1d(d_in, d, 5, padding=2)
        self.blocks = nn.ModuleList(
            [nn.Conv1d(d, d, 5, padding=2 * r, dilation=r) for r in (1, 2, 4, 8, 16, 32)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, d) for _ in range(6)])
        self.out = nn.Conv1d(d, 1, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.inp(x))
        for c, n in zip(self.blocks, self.norm):
            h = h + self.drop(torch.nn.functional.gelu(n(c(h))))
        return self.out(h)[:, 0]


def resample_feats(x, n_out):
    """[T, D] at MERT_FPS -> [n_out, D] at 50 fps, linear on the true time axis."""
    src = np.arange(x.shape[0]) / MERT_FPS
    dst = np.arange(n_out) / FPS
    return np.stack([np.interp(dst, src, x[:, d]) for d in range(x.shape[1])], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--crop", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--arms", default="bt,mert,bt+mert")
    ap.add_argument("--folds", default="",
                    help="comma-separated folds to run; empty = all 8. RUN ONE FIRST: the decisive "
                         "comparison is bt+mert minus bt, paired on the same tracks with identical "
                         "architecture, schedule and seed, so a fold-0 null kills the idea for 1/8 "
                         "of the compute. Scale to 8 folds only once one fold shows something.")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", default="/tmp/overnight/mert_plus_bt.json")
    args = ap.parse_args()
    dev = torch.device(args.device)

    import beat_this.model.loss as btloss
    bce = btloss.ShiftTolerantBCELoss(pos_weight=1, tolerance=3).to(dev)
    m = make_beatonly_dbn(30.0, 16, 100, args.device)
    space, dp0 = m.state_spaces[0], m.dynamic_programs[0]
    s2c = m.state_to_classes[0]
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition

    def decode(p):
        got = emissions_beatonly(m, np.clip(p, 1e-6, 1 - 1e-6))
        if got is None:
            return None
        lcd, dec, ff = got
        path, _ = dp0.viterbi(logpi, logA, lcd, state_to_class=s2c, return_log_score=True)
        return state_path_to_events(path.cpu().numpy(), space, FPS,
                                    snap_to_activations=dec, first_frame=ff)["beats"]

    print("loading features...", flush=True)
    F_bt, F_me, Y, FOLD = {}, {}, {}, {}
    for tid in track_ids():
        pe, pm = BT_EMB / f"{tid}.npz", MERT / f"{tid}.pt"
        if not (pe.exists() and pm.exists()):
            continue
        z = np.load(pe, allow_pickle=True)
        e = np.asarray(z["emb"], np.float32)
        T = e.shape[0]
        d = torch.load(pm, map_location="cpu")
        F_bt[tid] = e
        F_me[tid] = resample_feats(np.asarray(d["layers"][args.layer], np.float32), T).astype(np.float32)
        v = np.zeros(T, np.float32)
        fr = np.round(np.asarray(load_gt(tid), float) * FPS).astype(int)
        v[fr[(fr >= 0) & (fr < T)]] = 1.0
        Y[tid] = v
        FOLD[tid] = int(z["fold"])
    ids = sorted(F_bt)
    folds = sorted(set(FOLD.values()))
    if args.folds:
        folds = [int(x) for x in args.folds.split(",")]
    print(f"{len(ids)} tracks, folds {folds}, MERT layer {args.layer}", flush=True)

    def feats(arm, tid):
        if arm == "bt":
            return F_bt[tid]
        if arm == "mert":
            return F_me[tid]
        return np.concatenate([F_bt[tid], F_me[tid]], 1)

    res = {a: {} for a in args.arms.split(",")}
    for arm in res:
        d_in = {"bt": 512, "mert": 768, "bt+mert": 1280}[arm]
        for f in folds:
            tr = [t for t in ids if FOLD[t] != f]
            te = [t for t in ids if FOLD[t] == f]
            torch.manual_seed(0)
            net = Head(d_in).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
            rng = np.random.default_rng(0)
            for ep in range(args.epochs):
                net.train()
                order = rng.permutation(len(tr))
                for b0 in range(0, len(order), args.batch):
                    xs, ys = [], []
                    for i in order[b0:b0 + args.batch]:
                        t = tr[i]
                        x = feats(arm, t)
                        if x.shape[0] < args.crop:
                            continue
                        o = int(rng.integers(0, x.shape[0] - args.crop + 1))
                        xs.append(x[o:o + args.crop])
                        ys.append(Y[t][o:o + args.crop])
                    if not xs:
                        continue
                    xb = torch.from_numpy(np.stack(xs)).permute(0, 2, 1).to(dev)
                    yb = torch.from_numpy(np.stack(ys)).to(dev)
                    loss = bce(net(xb), yb, torch.ones_like(yb, dtype=torch.bool))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
            net.eval()
            for t in te:
                with torch.no_grad():
                    lg = net(torch.from_numpy(feats(arm, t)).permute(1, 0).unsqueeze(0).to(dev))[0]
                e = decode(expit(lg.double().cpu().numpy()))
                if e is not None:
                    res[arm][t] = ev(np.asarray(load_gt(t), float), e)
            print(f"  {arm:<8} fold {f} done ({len(te)} test)", flush=True)

    base = {}
    for t in ids:
        e = decode(np.asarray(bthis_beat_prob(t), np.float64))
        if e is not None:
            base[t] = ev(np.asarray(load_gt(t), float), e)
    common = sorted(set(base).intersection(*[set(res[a]) for a in res if res[a]]))
    if args.folds:
        print(f"\n*** FOLDS {folds} ONLY -- go/no-go signal, not a measurement ***")
    A = lambda r: {k: float(np.mean([r[t][k] for t in common])) for k in METRICS}
    print(f"\nn={len(common)}  8-fold CV on SMC, fold-honest, beat-only DBN")
    print(f"{'arm':<12}" + "".join(f"{k:>9}" for k in METRICS))
    print(f"{'Beat This':<12}" + "".join(f"{A(base)[k]:>9.4f}" for k in METRICS) + "   (shipped activation)")
    for a in res:
        print(f"{a:<12}" + "".join(f"{A(res[a])[k]:>9.4f}" for k in METRICS))
    if "bt" in res and "bt+mert" in res:
        print(f"\nMERT's MARGINAL contribution (bt+mert minus bt, same architecture/data/seed):")
        for k in METRICS:
            d = np.array([res["bt+mert"][t][k] - res["bt"][t][k] for t in common])
            print(f"  {k:<5} {d.mean():+.4f}  (p={wilcoxon(d).pvalue:.3g})")
    json.dump({"n": len(common), "layer": args.layer,
               "summary": {**{a: A(res[a]) for a in res}, "bthis_shipped": A(base)},
               "per_track": res}, open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
