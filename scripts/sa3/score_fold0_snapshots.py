"""Score training snapshots on the held-out fold 0, broken out by subgroup.

Reuses the model's own Postprocessor and Metrics so the numbers line up with
what validation_step would report, rather than a reimplemented peak-picker.

Reports, per checkpoint:
  ALL          the whole fold-0 validation set
  SMC          smc tracks only
  <70 bpm      every corpus, tracks whose annotated median tempo is under 70
  SMC <70      the intersection
Tempo comes from the original (unquantised) annotation carried in the batch as
truth_orig_beat, so it matches what the metrics are computed against.

Usage:  score_fold0_snapshots.py --gpu 0 [--pattern '*_ep0*'] [--out results.csv]
"""

import argparse
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

BT = "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"
SNAP = "/disk1/jaehoon/ckpt_snapshots"


def build_loader(num_workers):
    from beat_this.dataset import BeatDataModule

    dm = BeatDataModule(
        Path("data"),
        batch_size=1,
        train_length=1500,
        spect_fps=50,
        num_workers=num_workers,
        test_dataset="gtzan",
        length_based_oversampling_factor=0.65,
        augmentations={},
        hung_data=False,
        no_val=False,          # keep fold 0 OUT of training, i.e. a real val set
        fold=0,
    )
    dm.setup(stage="fit")
    return dm.val_dataloader()


def load_model(ckpt_path, device):
    from beat_this.model.pl_module import PLBeatThis

    m = PLBeatThis(
        spect_dim=128, fps=50, transformer_dim=512, ff_mult=4, n_layers=6,
        stem_dim=32, dropout={"frontend": 0.1, "transformer": 0.2}, lr=8e-4,
        weight_decay=0.01, pos_weights={"beat": 20, "downbeat": 86},
        head_dim=32, loss_type="shift_tolerant_weighted_bce",
        warmup_steps=1000, max_epochs=100,
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"    warning: {len(missing)} missing keys, e.g. {missing[:2]}")
    return m.eval().to(device)


@torch.no_grad()
def score(model, loader, device):
    rows = []
    for batch in loader:
        spect = batch["spect"].to(device)
        pad = batch["padding_mask"].to(device)
        # training runs under precision="16-mixed"; match it so the weights and
        # the fp16 spectrograms agree
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.model(spect)
        pred = {k: v.float() for k, v in pred.items()}
        pb, pdb = model.postprocessor(pred["beat"], pred["downbeat"], pad)
        if not isinstance(pb, tuple):
            pb = (pb,)
        for i, est in enumerate(pb):
            truth = np.frombuffer(batch["truth_orig_beat"][i])
            if len(truth) < 3:
                continue
            met = model.metrics(truth, est, step="test")
            path = batch["spect_path"][i]
            corpus = str(path).split("/", 1)[0]
            ibis = np.diff(truth)
            rows.append(dict(
                path=str(path), corpus=corpus,
                bpm=float(60.0 / np.median(ibis)),
                F=met["F-measure"], CMLt=met["CMLt"], AMLt=met["AMLt"],
            ))
    return rows


def summarize(rows):
    groups = {
        "ALL": lambda r: True,
        "SMC": lambda r: r["corpus"] == "smc",
        "<70 bpm": lambda r: r["bpm"] < 70,
        "SMC <70": lambda r: r["corpus"] == "smc" and r["bpm"] < 70,
        ">=70 bpm": lambda r: r["bpm"] >= 70,
    }
    out = {}
    for name, fn in groups.items():
        sel = [r for r in rows if fn(r)]
        if sel:
            out[name] = (len(sel), float(np.mean([r["F"] for r in sel])),
                         float(np.mean([r["CMLt"] for r in sel])),
                         float(np.mean([r["AMLt"] for r in sel])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--pattern", default="*.ckpt")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="/home/sogang/jaehoon/Analyze-SMC/cache/fold0_scores.csv")
    args = ap.parse_args()

    os.chdir(BT)
    device = f"cuda:{args.gpu}"
    loader = build_loader(args.num_workers)

    ckpts = sorted(glob.glob(os.path.join(SNAP, args.pattern)))
    print(f"  scoring {len(ckpts)} checkpoints on fold 0\n")

    write_header = not os.path.exists(args.out)
    fh = open(args.out, "a")
    if write_header:
        fh.write("arm,epoch,group,n,F,CMLt,AMLt\n")

    for c in ckpts:
        stem = Path(c).stem
        m = re.match(r"(.+)_ep(\d+)$", stem)
        arm, ep = (m.group(1), int(m.group(2))) if m else (stem, -1)
        model = load_model(c, device)
        rows = score(model, loader, device)
        summ = summarize(rows)
        print(f"  {arm}  epoch {ep}")
        for g, (n, f, cml, aml) in summ.items():
            print(f"      {g:<10} n={n:>3}   F {f:.4f}   CMLt {cml:.4f}   AMLt {aml:.4f}")
            fh.write(f"{arm},{ep},{g},{n},{f:.6f},{cml:.6f},{aml:.6f}\n")
        fh.flush()
        print()
        del model
        torch.cuda.empty_cache()
    fh.close()
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
