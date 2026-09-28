"""Independent cross-check of the fold-0 arm comparison.

Differs from score_fold0_snapshots.py on every axis that could be an artifact:
  - full pieces via predict_step / split_predict_aggregate, not the middle 30 s excerpt
  - fp32, no autocast
  - beat AND downbeat metrics, from the model's own predict_step
Writes one row per (arm, track).
"""
import argparse, glob, os, re, sys
from pathlib import Path
import numpy as np, torch

BT = "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"
os.chdir(BT); sys.path.insert(0, BT)

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=3)
ap.add_argument("--arms", default="vanilla_f0,vanillaS1_f0,vanillaS2_f0,vanillaS3_f0,ucorpus_s0,ucorpus_s1,ucorpus_s2,official_f0")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from beat_this.dataset import BeatDataModule
from beat_this.model.pl_module import PLBeatThis
from beat_this.dataset.dataset import BeatTrackingDataset
from torch.utils.data import DataLoader

dm = BeatDataModule(Path("data"), batch_size=1, train_length=1500, spect_fps=50,
                    num_workers=4, test_dataset="gtzan",
                    length_based_oversampling_factor=0.65, augmentations={},
                    hung_data=False, no_val=False, fold=0)
dm.setup(stage="validate")
items = sorted(dm.val_items)
if a.limit:
    items = items[: a.limit]
ds = BeatTrackingDataset(items, data_folder=Path("data"), spect_fps=50,
                         train_length=None, deterministic=True, augmentations={})
dl = DataLoader(ds, batch_size=1, num_workers=4)
print(f"full-piece val set: {len(ds)} items", flush=True)

dev = f"cuda:{a.gpu}"
fh = open(a.out, "w")
fh.write("arm,path,corpus,bpm,F,CMLt,AMLt\n")
for arm in a.arms.split(","):
    ck = f"/disk1/jaehoon/ckpt_snapshots/{arm}_ep099.ckpt"
    m = PLBeatThis(spect_dim=128, fps=50, transformer_dim=512, ff_mult=4, n_layers=6,
                   stem_dim=32, dropout={"frontend": 0.1, "transformer": 0.2}, lr=8e-4,
                   weight_decay=0.01, pos_weights={"beat": 20, "downbeat": 86}, head_dim=32,
                   loss_type="shift_tolerant_weighted_bce", warmup_steps=1000, max_epochs=100)
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    miss, unexp = m.load_state_dict(sd, strict=True), None
    m = m.eval().to(dev)
    n = 0
    with torch.no_grad():
        for b in dl:
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            b["spect"] = b["spect"].float()
            met, _, _, spath = m.predict_step(b, 0)
            truth = np.frombuffer(b["truth_orig_beat"][0])
            if len(truth) < 3:
                continue
            p = str(spath[0]); corpus = p.split("/", 1)[0]
            bpm = float(60.0 / np.median(np.diff(truth)))
            fh.write(f'"{arm}","{p}",{corpus},{bpm:.4f},{met["F-measure_beat"]:.6f},'
                     f'{met["CMLt_beat"]:.6f},{met["AMLt_beat"]:.6f}\n')
            n += 1
    fh.flush(); print(f"  {arm}: {n} tracks", flush=True)
    del m; torch.cuda.empty_cache()
fh.close()
print("done", a.out)
