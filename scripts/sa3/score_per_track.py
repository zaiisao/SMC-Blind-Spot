"""Dump PER-TRACK fold-0 scores for given checkpoints, so any grouping
(tempo bin, corpus, rubato) can be computed afterwards without re-running the
model. Reuses the model's own Postprocessor and Metrics.

Usage: score_per_track.py --gpu 2 --pattern '*_ep021.ckpt' --out per_track.csv
"""

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import torch

BT = "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"
SNAP = "/disk1/jaehoon/ckpt_snapshots"


def build_loader(num_workers):
    from beat_this.dataset import BeatDataModule
    dm = BeatDataModule(
        Path("data"), batch_size=1, train_length=1500, spect_fps=50,
        num_workers=num_workers, test_dataset="gtzan",
        length_based_oversampling_factor=0.65, augmentations={},
        hung_data=False, no_val=False, fold=0,
    )
    dm.setup(stage="fit")
    return dm.val_dataloader()


def load_model(ckpt, device):
    from beat_this.model.pl_module import PLBeatThis
    m = PLBeatThis(
        spect_dim=128, fps=50, transformer_dim=512, ff_mult=4, n_layers=6,
        stem_dim=32, dropout={"frontend": 0.1, "transformer": 0.2}, lr=8e-4,
        weight_decay=0.01, pos_weights={"beat": 20, "downbeat": 86},
        head_dim=32, loss_type="shift_tolerant_weighted_bce",
        warmup_steps=1000, max_epochs=100,
    )
    m.load_state_dict(torch.load(ckpt, map_location="cpu",
                                 weights_only=False)["state_dict"], strict=False)
    return m.eval().to(device)


@torch.no_grad()
def run(model, loader, device, arm, ep, fh):
    for batch in loader:
        spect = batch["spect"].to(device)
        pad = batch["padding_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.model(spect)
        pred = {k: v.float() for k, v in pred.items()}
        pb, _ = model.postprocessor(pred["beat"], pred["downbeat"], pad)
        if not isinstance(pb, tuple):
            pb = (pb,)
        for i, est in enumerate(pb):
            truth = np.frombuffer(batch["truth_orig_beat"][i])
            if len(truth) < 3:
                continue
            met = model.metrics(truth, est, step="test")
            path = str(batch["spect_path"][i])
            ibis = np.diff(truth)
            fh.write(f'{arm},{ep},"{path}",{path.split("/",1)[0]},'
                     f'{60.0/np.median(ibis):.4f},{np.std(ibis)/np.mean(ibis):.4f},'
                     f'{met["F-measure"]:.6f},{met["CMLt"]:.6f},{met["AMLt"]:.6f}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--pattern", default="*_ep021.ckpt")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.chdir(BT)
    device = f"cuda:{args.gpu}"
    loader = build_loader(args.num_workers)
    fh = open(args.out, "w")
    fh.write("arm,epoch,path,corpus,bpm,ibi_cv,F,CMLt,AMLt\n")
    for c in sorted(glob.glob(os.path.join(SNAP, args.pattern))):
        m = re.match(r"(.+)_ep(\d+)$", Path(c).stem)
        arm, ep = (m.group(1), int(m.group(2))) if m else (Path(c).stem, -1)
        print(f"  {arm} epoch {ep}", flush=True)
        model = load_model(c, device)
        run(model, loader, device, arm, ep, fh)
        fh.flush()
        del model
        torch.cuda.empty_cache()
    fh.close()
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
