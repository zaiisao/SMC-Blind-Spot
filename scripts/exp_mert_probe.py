"""Does a FROZEN music foundation model's representation carry beat evidence that Beat This's does not?

THE QUESTION THIS ANSWERS, AND WHY IT IS THE RIGHT ONE. Per-track ORACLE decoder hyperparameters
reach only F 0.6908 on SMC, against 0.6270 for peak-picked Beat This. So ~83% of the remaining gap
is in the ACTIVATIONS, not the decoder, and every intervention this project has made -- tempo term,
weight rebalance, meter sets, TTA, onset aggregation, MMI -- has lived inside that 0.69 envelope.
This is the first experiment that changes the representation the answer is computed from.

WHY MERT RATHER THAN ANOTHER TRACKER. Fusing Beat This with Beat Transformer is ensembling two
competitors that do the same thing slightly differently; at best a diagnostic. MERT-v1-95M is
self-supervised on ~160k hours and is NOT a beat tracker -- it is a representation. Asking whether a
small head on frozen MERT can localise beats on SMC tests whether the evidence is in the audio and
merely unextracted by a 20M-parameter model trained on ~4k tracks with a per-frame BCE loss.

THE SPLIT IS CLEAN, unlike everything else we have run. MERT never saw beat labels, and the probe
trains ONLY on the 300 corpus tracks (ballroom / beatles / hainsworth / hjdb). SMC is never trained
on, so all 217 tracks are test data: n=217, no folds, no memorisation, no early-stopping games.

CONFOUND TO STATE, NOT HIDE. The corpus half is percussive (ballroom, hjdb) or pop (beatles), with
no solo piano and no ASAP. SMC is largely classical and rubato. So a LOW score is ambiguous -- it
could mean MERT lacks the evidence, or merely that the probe never saw this kind of music. A HIGH
score is unambiguous and is what would matter.

LAYER SWEEP. Beat information in self-supervised audio models tends to concentrate in middle layers,
and which layer is not predictable a priori, so several are measured rather than one guessed.

Frame rates: MERT features here are 86.1 fps; the head runs at that rate and its 1-D output is
resampled to the decoder's 50 fps, which is the same single 1-D interpolation we already use for
Beat Transformer. Decoding is the beat-only DBN we ship, unchanged, so any difference is the
representation.
"""
import argparse
import glob
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
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob, resample_to
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events

MERT_FPS = 86.1328125
CORPUS = "/disk1/jaehoon/vbpm_mert_layers/corpus"
SMC = "/disk1/jaehoon/vbpm_mert_layers/smc"
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")


class Head(nn.Module):
    """Small dilated-conv head; receptive field ~4 s at 86 fps, enough to see a few beats."""

    def __init__(self, d_in=768, d=256):
        super().__init__()
        self.inp = nn.Conv1d(d_in, d, 5, padding=2)
        self.blocks = nn.ModuleList(
            [nn.Conv1d(d, d, 5, padding=2 * r, dilation=r) for r in (1, 2, 4, 8, 16, 32)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, d) for _ in range(6)])
        self.out = nn.Conv1d(d, 1, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):                      # x [B, d_in, T]
        h = torch.nn.functional.gelu(self.inp(x))
        for c, n in zip(self.blocks, self.norm):
            h = h + self.drop(torch.nn.functional.gelu(n(c(h))))
        return self.out(h)[:, 0]               # [B, T] logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="4,6,8", help="MERT layers to sweep")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--crop", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", default="/tmp/overnight/mert_probe.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)

    import beat_this.model.loss as btloss
    # tolerance 5 frames at 86 fps = 58 ms, matching Beat This's 3 frames at 50 fps = 60 ms
    bce = btloss.ShiftTolerantBCELoss(pos_weight=1, tolerance=5).to(dev)

    m = make_beatonly_dbn(30.0, 16, 100, args.device)
    space, dp0 = m.state_spaces[0], m.dynamic_programs[0]
    s2c = m.state_to_classes[0]
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition

    def decode(prob50):
        got = emissions_beatonly(m, np.clip(prob50, 1e-6, 1 - 1e-6))
        if got is None:
            return None
        lcd, dec, ff = got
        path, _ = dp0.viterbi(logpi, logA, lcd, state_to_class=s2c, return_log_score=True)
        return state_path_to_events(path.cpu().numpy(), space, FPS,
                                    snap_to_activations=dec, first_frame=ff)["beats"]

    cfiles = sorted(glob.glob(f"{CORPUS}/*.pt"))
    smc_files = {Path(p).stem: p for p in sorted(glob.glob(f"{SMC}/*.pt"))}
    print(f"corpus {len(cfiles)} tracks (train)   SMC {len(smc_files)} tracks (test, never trained on)",
          flush=True)

    results = {}
    for L in [int(x) for x in args.layers.split(",")]:
        print(f"\n=== MERT layer {L} ===", flush=True)
        X, Y = [], []
        for p in cfiles:
            d = torch.load(p, map_location="cpu")
            X.append(d["layers"][L].float())            # [T, 768]
            Y.append(d["beat_targets"].float())
        net = Head().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
        rng = np.random.default_rng(0)
        for ep in range(1, args.epochs + 1):
            net.train()
            order = rng.permutation(len(X))
            tot = nb = 0
            for b0 in range(0, len(order), args.batch):
                xs, ys = [], []
                for i in order[b0:b0 + args.batch]:
                    T = X[i].shape[0]
                    if T < args.crop:
                        continue
                    o = int(rng.integers(0, T - args.crop + 1))
                    xs.append(X[i][o:o + args.crop])
                    ys.append(Y[i][o:o + args.crop])
                if not xs:
                    continue
                x = torch.stack(xs).permute(0, 2, 1).to(dev)
                y = torch.stack(ys).to(dev)
                lg = net(x)
                loss = bce(lg, y, torch.ones_like(y, dtype=torch.bool))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += float(loss); nb += 1
            print(f"  ep{ep:2d} train BCE {tot/max(nb,1):.4f}", flush=True)

        net.eval()
        res, base = {}, {}
        for tid in track_ids():
            if tid not in smc_files:
                continue
            d = torch.load(smc_files[tid], map_location="cpu")
            with torch.no_grad():
                lg = net(d["layers"][L].float().permute(1, 0).unsqueeze(0).to(dev))[0]
            p86 = expit(lg.double().cpu().numpy())
            n50 = int(round(len(p86) * FPS / MERT_FPS))
            p50 = resample_to(p86, MERT_FPS, FPS, n50)
            gt = np.asarray(load_gt(tid), float)
            e = decode(p50)
            if e is not None:
                res[tid] = ev(gt, e)
            eb = decode(np.asarray(bthis_beat_prob(tid), np.float64))
            if eb is not None:
                base[tid] = ev(gt, eb)
        ids = sorted(set(res) & set(base))
        A = lambda r: {k: float(np.mean([r[t][k] for t in ids])) for k in METRICS}
        print(f"  layer {L}: n={len(ids)}")
        print(f"    MERT probe   " + "  ".join(f"{k} {A(res)[k]:.4f}" for k in METRICS))
        print(f"    Beat This    " + "  ".join(f"{k} {A(base)[k]:.4f}" for k in METRICS))
        for k in ("F", "CMLt"):
            dd = np.array([res[t][k] - base[t][k] for t in ids])
            print(f"    delta {k} {dd.mean():+.4f} (p={wilcoxon(dd).pvalue:.2g})")
        results[L] = {"probe": A(res), "bthis": A(base), "per_track": res}
    json.dump({"results": {str(k): {kk: v[kk] for kk in ("probe", "bthis")}
                           for k, v in results.items()},
               "per_track": {str(k): v["per_track"] for k, v in results.items()}},
              open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
