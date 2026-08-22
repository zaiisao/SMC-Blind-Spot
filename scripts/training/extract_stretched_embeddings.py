"""Beat This embeddings for corpus audio TIME-STRETCHED into the slow regime, with scaled labels.

WHY THIS DATA HAS TO BE MANUFACTURED. The tempo head reaches only 12.2% accuracy on SMC and, fed to
the DBN, actively hurts (-0.069 F at beta=0.1). The cause is covariate shift, not class imbalance:
in-domain slow tracks score 0.725, so the model CAN read slow music -- it has just barely seen any.
Reweighting a training set that lacks the region cannot fix that; the region has to be created.

Stretching is done on the WAVEFORM with a phase vocoder, then embedded. Two cheaper alternatives
are both wrong here:
  * resampling the mel time axis -- the shortcut this project has repeatedly been bitten by; a model
    can read the resampling signature instead of musical tempo, and our mel-zoom fine-tune was null
    where audio-domain stretching was not.
  * resampling the EMBEDDING time axis -- same objection one level up, and it would fabricate
    representations the encoder never actually produces.

LABELS COME FREE AND EXACT. Stretching by s maps every annotated beat time t -> t*s, so the local
inter-beat interval scales by exactly s. No new annotation is needed, and the slow-tempo labels are
as accurate as the originals.

STRETCHES target the region SMC occupies (median 70.9 BPM, 21% below 55) starting from corpora
centred near 120 BPM: s=1.7 maps 120 -> 71 BPM, s=2.4 maps 120 -> 50 BPM. s=1.0 is kept so the
head is not trained exclusively on vocoder output -- the domain-shift trap that cost -0.092 F in
the fine-tuning experiments.

FOLD HONESTY: each track is embedded with the checkpoint that held it out, matching bt_emb.
"""
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third-party" / "beat_this"))
from beat_this.inference import Audio2Frames
from beat_this.preprocessing import load_audio
from extract_bt_embeddings import embed_piece

STORE = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
AUDIO = Path("/disk1/jaehoon/dataset_store/audio_by_stem")
OUT = Path("/disk1/jaehoon/bt_emb_stretch")


def split_of(ds):
    p = STORE / ds / "8-folds.split"
    return {l.split("\t")[0]: int(l.split("\t")[1]) for l in open(p) if l.strip()} if p.exists() else {}


def annots(ds, tid):
    p = STORE / ds / "annotations" / "beats" / f"{tid}.beats"
    if not p.exists():
        return None, None
    t, b = [], []
    for line in open(p):
        f = line.split()
        if f:
            t.append(float(f[0])); b.append(int(float(f[1])) if len(f) > 1 else 0)
    return np.array(t), np.array(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ballroom,hainsworth,beatles,gtzan,asap")
    ap.add_argument("--stretches", default="1.0,1.4,1.7,2.0,2.4")
    ap.add_argument("--tracks-per-ds", type=int, default=140)
    ap.add_argument("--max-sec", type=float, default=60.0,
                    help="crop before stretching; keeps the phase vocoder (the bottleneck) bounded")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    S = [float(x) for x in args.stretches.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    a2fs = {}
    n_done = n_skip = 0
    for ds in args.datasets.split(","):
        sp = split_of(ds)
        d = AUDIO / ds
        if not d.exists():
            print(f"  {ds}: no audio dir", flush=True); continue
        od = OUT / ds; od.mkdir(parents=True, exist_ok=True)
        stems = sorted(d.glob("*.wav"))[:args.tracks_per_ds]
        for path in stems:
            tid = path.stem
            bt, bn = annots(ds, tid)
            if bt is None or len(bt) < 8:
                n_skip += 1; continue
            fold = sp.get(tid, 0)
            if fold not in a2fs:
                a2fs[fold] = Audio2Frames(checkpoint_path=f"fold{fold}", device=args.device)
            a2f = a2fs[fold]
            try:
                sig, sr = load_audio(str(path))
            except Exception:
                n_skip += 1; continue
            y = sig.mean(1) if sig.ndim > 1 else sig
            y = np.ascontiguousarray(y.astype(np.float32))
            if args.max_sec and len(y) > int(args.max_sec * sr):
                y = y[:int(args.max_sec * sr)]
                keep = bt < args.max_sec
                bt, bn = bt[keep], bn[keep]
                if len(bt) < 8:
                    n_skip += 1; continue
            for s in S:
                op = od / f"{tid}__s{s:g}.npz"
                if op.exists():
                    continue
                try:
                    ys = y if s == 1.0 else librosa.effects.time_stretch(y, rate=1.0 / s)
                    spect = a2f.signal2spect(ys, sr)
                    emb, beat, down, full = embed_piece(a2f.model, spect)
                except Exception as e:
                    print(f"    {tid} s={s}: {type(e).__name__}", flush=True); continue
                np.savez_compressed(op, emb=emb.astype(np.float16), beat=beat.astype(np.float16),
                                    beat_times=(bt * s).astype(np.float32),
                                    is_downbeat=bn.astype(np.int8),
                                    fold=int(fold), stretch=float(s), full=bool(full))
                n_done += 1
            if n_done % 200 < len(S):
                print(f"  {ds}: {n_done} written, {n_skip} skipped", flush=True)
        print(f"{ds} done ({n_done} total)", flush=True)
    print(f"DONE  written {n_done}  skipped {n_skip}", flush=True)


if __name__ == "__main__":
    main()
