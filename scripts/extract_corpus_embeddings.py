"""Fold-honest Beat This 512-d embeddings for the training corpora, from PRECOMPUTED spectrograms.

Feeds `/disk1/jaehoon/beat_this_spect/done/{dataset}.npz` (per-track [T,128] float16 log-mel,
keyed `{track_id}/track`) straight through frontend + transformer_blocks. No audio decode, so
extraction is transformer-bound only.

FOLD HONESTY. Each dataset's `8-folds.split` maps track_id -> fold, and Beat This's `foldN`
checkpoint held fold N out of training. We therefore extract each track with the checkpoint that
did NOT train on it. GTZAN has no split file at all -- it is held out of every fold -- so any
checkpoint is honest for it; we use fold0. Loop is fold-outer so each checkpoint loads once.

WINDOW. A centred `--frames` window per track (default 1500 = 30 s) bounds cost on long-form
corpora (harmonix/asap/rwc) and avoids intros and lead-in silence. A per-frame emission head is
massively over-determined by even a few million frames, so full-length extraction (~320M frames,
~22 GPU-hours for the whole corpus) buys nothing here.

Alignment for SMC was already verified bit-exact (max deviation 0.000e+00 on 217/217) by
scripts/extract_bt_embeddings.py; this script reuses the identical chunking helper.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "beat_this"))

from beat_this.inference import load_model
from extract_bt_embeddings import embed_piece

STORE = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
SPECT = Path("/disk1/jaehoon/beat_this_spect/done")


def load_split(ds):
    p = STORE / ds / "8-folds.split"
    if not p.exists():
        return None
    out = {}
    for line in open(p):
        line = line.strip()
        if line:
            tid, f = line.split("\t")
            out[tid] = int(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="gtzan,ballroom,hainsworth,beatles,asap")
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="/disk1/jaehoon/bt_emb")
    ap.add_argument("--per-dataset-limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dsets = [d.strip() for d in args.datasets.split(",")]

    # plan: fold -> [(dataset, track_id)]
    plan, splits, missing = defaultdict(list), {}, []
    for ds in dsets:
        if not (SPECT / f"{ds}.npz").exists():
            missing.append(ds); continue
        sp = load_split(ds); splits[ds] = sp
        import zipfile
        keys = [n[:-len("/track.npy")] for n in zipfile.ZipFile(SPECT / f"{ds}.npz").namelist()
                if n.endswith("/track.npy")]
        if args.per_dataset_limit:
            keys = sorted(keys)[:args.per_dataset_limit]
        for k in keys:
            plan[0 if sp is None else sp.get(k, 0)].append((ds, k))
    if missing:
        print(f"MISSING spect archives: {missing}", flush=True)
    total = sum(len(v) for v in plan.values())
    print(f"planned {total} tracks over {len(plan)} folds; "
          f"{ {d: sum(1 for f in plan for dd,_ in plan[f] if dd==d) for d in dsets if d not in missing} }",
          flush=True)

    handles = {}
    report, done = {}, 0
    for fold in sorted(plan):
        model = load_model(f"fold{fold}", torch.device(args.device))
        model.eval()
        for ds, tid in plan[fold]:
            if ds not in handles:
                handles[ds] = np.load(SPECT / f"{ds}.npz")
            try:
                spect = np.asarray(handles[ds][f"{tid}/track"], dtype=np.float32)
            except KeyError:
                continue
            T = spect.shape[0]
            if args.frames and T > args.frames:                 # centred window
                s = (T - args.frames) // 2
                spect = spect[s:s + args.frames]
            else:
                s = 0
            st = torch.from_numpy(spect).to(args.device)
            emb, beat, down, full = embed_piece(model, st)
            d = out / ds; d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{tid}.npz", emb=emb.astype(np.float16),
                                beat=beat, downbeat=down, fold=fold, start_frame=s, orig_T=T)
            report[f"{ds}/{tid}"] = {"fold": fold, "T": int(emb.shape[0]),
                                     "start": int(s), "orig_T": int(T), "filled": bool(full)}
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{total}", flush=True)
        del model
        torch.cuda.empty_cache()

    Path(out / "_report.json").write_text(json.dumps(report, indent=1))
    nf = sum(1 for v in report.values() if not v["filled"])
    print(f"\ndone {len(report)} tracks; unfilled-frame failures: {nf}")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
