"""Classify each fold-0 track's prediction as correct / doubled / halved /
phase-shifted / other, per checkpoint.

Method: score the estimate against the true grid and against three transformed
grids (2x, 1/2x, offbeat). Whichever reference it matches at F >= THRESH names
the failure mode. This tests the gated loss's stated mechanism -- it is supposed
to suppress doubling specifically -- rather than only its aggregate score.

Usage: classify_errors.py --gpu 2 --pattern '*_ep032.ckpt' --out cls.csv
"""

import argparse
import glob
import os
import re
from pathlib import Path

import mir_eval
import numpy as np
import torch

BT = "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"
SNAP = "/disk1/jaehoon/ckpt_snapshots"
THRESH = 0.70


def build_loader(nw):
    from beat_this.dataset import BeatDataModule
    dm = BeatDataModule(
        Path("data"), batch_size=1, train_length=1500, spect_fps=50,
        num_workers=nw, test_dataset="gtzan",
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


def f1(est, ref):
    if len(est) < 2 or len(ref) < 2:
        return 0.0
    e = mir_eval.beat.trim_beats(est)
    r = mir_eval.beat.trim_beats(ref)
    if len(e) < 1 or len(r) < 1:
        return 0.0
    return mir_eval.beat.f_measure(r, e, f_measure_threshold=0.07)


def classify(est, ref):
    dbl = np.sort(np.concatenate([ref, (ref[:-1] + ref[1:]) / 2]))
    sc = {
        "correct": f1(est, ref),
        "doubled": f1(est, dbl),
        "halved": f1(est, ref[::2]),
        "phase": f1(est, (ref[:-1] + ref[1:]) / 2),
    }
    if sc["correct"] >= THRESH:
        return "correct", sc
    best = max(sc, key=sc.get)
    return (best if sc[best] >= THRESH else "other"), sc


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
            cat, sc = classify(np.asarray(est), truth)
            path = str(batch["spect_path"][i])
            bpm = 60.0 / np.median(np.diff(truth))
            fh.write(f'{arm},{ep},"{path}",{path.split("/",1)[0]},{bpm:.4f},{cat},'
                     f'{sc["correct"]:.6f},{sc["doubled"]:.6f},{sc["halved"]:.6f},'
                     f'{sc["phase"]:.6f},{len(est)},{len(truth)}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--pattern", default="*_ep032.ckpt")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.chdir(BT)
    dev = f"cuda:{a.gpu}"
    loader = build_loader(a.num_workers)
    fh = open(a.out, "w")
    fh.write("arm,epoch,path,corpus,bpm,cat,F_correct,F_doubled,F_halved,F_phase,n_est,n_true\n")
    for c in sorted(glob.glob(os.path.join(SNAP, a.pattern))):
        m = re.match(r"(.+)_ep(\d+)$", Path(c).stem)
        arm, ep = (m.group(1), int(m.group(2))) if m else (Path(c).stem, -1)
        print(f"  {arm} epoch {ep}", flush=True)
        model = load_model(c, dev)
        run(model, loader, dev, arm, ep, fh)
        fh.flush()
        del model
        torch.cuda.empty_cache()
    fh.close()
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
