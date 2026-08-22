"""Precompute log-mel spectrograms + beat targets for the training corpus, once.

FOR THE LEAVE-ONE-DATASET-OUT ABLATION: train Beat This FROM SCRATCH with and without ASAP, and
watch how SMC eval evolves. That question cannot be answered by fine-tuning -- the released
checkpoints have already absorbed ASAP, and you cannot un-learn a dataset by fine-tuning on the
others -- nor by our frozen-feature heads, whose features already encode it. It needs from-scratch
training, which needs cached spectrograms.

WHY THIS ABLATION IS THE RIGHT NEXT MOVE. Every attempt to move SMC by training has failed so far
(MMI in four settings, MERT concatenation, ASAP fine-tuning), but all of them were confounded by
fine-tuning a converged checkpoint on a narrow slice of its own training data. The ablation asks the
prior question instead: what is expressive-piano data WORTH to SMC at all?
    remove ASAP and SMC drops a lot  -> the regime matters; synthesise more of it
    remove ASAP and SMC barely moves -> the premise is wrong; stop building on it
Either answer is decision-relevant, and neither can be produced by fine-tuning.

SPECTROGRAMS ARE COMPUTED WITH BEAT THIS'S OWN signal2spect, not reimplemented -- the same function
whose output reproduces the shipped activation cache to 0.000e+00 (verified during the augmentation
work). Targets are binary impulse trains at round(t * 50), matching their dataset.py exactly.

SUBSAMPLING. Full-corpus from-scratch training is a multi-day job; the ablation only needs the
DIFFERENCE between two arms, so a balanced subsample makes it an overnight one. --per-ds caps each
dataset so that no single corpus dominates, and ASAP's share is what the two arms differ by.
GTZAN is excluded entirely: it is the field's held-out test set and Beat This never trains on it.
SMC is excluded from training too -- it is what we evaluate.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from _harness import FPS

ANN = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
AUD = Path("/disk1/jaehoon/dataset_store/audio_by_stem")
OUT = Path("/disk1/jaehoon/corpus_spect")
TRAIN_DS = ("ballroom", "beatles", "hainsworth", "hjdb", "rwc", "asap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-ds", type=int, default=150, help="cap per dataset")
    ap.add_argument("--max-secs", type=float, default=120.0,
                    help="truncate long tracks; ASAP performances run many minutes and the "
                         "model only ever sees 1500-frame (30 s) crops anyway")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from beat_this.inference import Audio2Frames
    a2f = Audio2Frames(checkpoint_path="fold0", device=args.device)   # only for signal2spect

    rng = np.random.default_rng(0)
    index = {}
    for ds in TRAIN_DS:
        adir, bdir = AUD / ds, ANN / ds / "annotations" / "beats"
        if not (adir.exists() and bdir.exists()):
            print(f"  {ds}: MISSING, skipped", flush=True)
            continue
        wavs = sorted(p for p in adir.iterdir() if p.is_file() or p.is_symlink())
        rng.shuffle(wavs)
        kept = 0
        for p in wavs:
            if kept >= args.per_ds:
                break
            stem = p.stem[len(ds) + 1:] if p.stem.startswith(ds + "_") else p.stem
            ann = bdir / f"{stem}.beats"
            if not ann.exists():
                continue
            try:
                a = np.loadtxt(str(ann))
                bt = (a if a.ndim == 1 else a[:, 0]).astype(float)
                if len(bt) < 20:
                    continue
                y, sr = sf.read(str(p), always_2d=True)
                y = y.mean(1)
                if len(y) / sr > args.max_secs:
                    y = y[:int(args.max_secs * sr)]
                    bt = bt[bt < args.max_secs]
                    if len(bt) < 20:
                        continue
                sp = a2f.signal2spect(y, sr).cpu().numpy().astype(np.float16)
            except Exception as e:
                continue
            T = sp.shape[0]
            v = np.zeros(T, np.float32)
            fr = np.round(bt * FPS).astype(int)
            v[fr[(fr >= 0) & (fr < T)]] = 1.0
            np.savez_compressed(OUT / f"{ds}__{stem}.npz", spect=sp, target=v)
            index[f"{ds}__{stem}"] = {"ds": ds, "frames": int(T),
                                      "bpm": float(60.0 / np.median(np.diff(bt)))}
            kept += 1
            if kept % 50 == 0:
                print(f"  {ds}: {kept}", flush=True)
        print(f"{ds}: cached {kept}", flush=True)

    json.dump(index, open(OUT / "index.json", "w"), indent=1)
    n = len(index)
    hrs = sum(v["frames"] for v in index.values()) / FPS / 3600
    print(f"\ncached {n} tracks, {hrs:.1f} hours")
    for ds in TRAIN_DS:
        s = [v for v in index.values() if v["ds"] == ds]
        if s:
            print(f"  {ds:<12} {len(s):>4} tracks  median {np.median([v['bpm'] for v in s]):.0f} BPM")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
