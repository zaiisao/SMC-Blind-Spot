"""
Combined TTA worker that saves BOTH beat and downbeat averaged activations,
for pitch-shift and time-stretch augmentation, so they can be fed to the
GT-tempo (oracle) DBN which needs a (beat, downbeat) pair.

Usage: exp_tta_full_worker.py --folds 0,1 --device cuda:0
Saves data/tta_full/{track}.npz with keys: pitch_beat, pitch_db, ts_beat, ts_db
(all averaged LOGITS on the original 50 fps grid).
"""
import argparse, numpy as np, torch, librosa
from pathlib import Path

ROOT = Path("/home/sogang/jaehoon/Analyze-SMC")
AUDIO = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
SPLIT = ROOT / "beat_this_annotations" / "smc" / "8-folds.split"
BTHIS = ROOT / "beat_this_activations_cache"
OUT = ROOT / "data" / "tta_full"
OUT.mkdir(parents=True, exist_ok=True)
FPS = 50
PITCH = [-2, -1, 0, 1, 2]          # semitones (timing preserved)
RATES = [0.85, 1.0, 1.2]           # mild winning stretch set

def unwarp(logits_s, rate, L):
    xp = (np.arange(len(logits_s)) / FPS) * rate
    return np.interp(np.arange(L) / FPS, xp, logits_s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    folds = set(int(x) for x in args.folds.split(","))

    from beat_this.inference import Audio2Frames
    from beat_this.preprocessing import load_audio

    fold_map = {}
    for line in open(SPLIT):
        line = line.strip()
        if line:
            tid, f = line.split("\t"); fold_map[tid] = int(f)
    by_fold = {}
    for tid, f in fold_map.items():
        by_fold.setdefault(f, []).append(tid)

    for fold in sorted(folds):
        a2f = Audio2Frames(checkpoint_path=f"fold{fold}", device=args.device)
        for tid in sorted(by_fold.get(fold, [])):
            fout = OUT / f"{tid}.npz"
            if fout.exists():
                continue
            L = len(np.load(BTHIS / f"{tid}.npz")["beat"])
            sig, sr = load_audio(str(AUDIO / f"{tid.upper()}.wav"))
            sig = np.asarray(sig).astype(np.float32)
            if sig.ndim > 1:
                sig = sig.mean(axis=1)

            pb, pd_ = [], []
            for s in PITCH:
                x = sig if s == 0 else librosa.effects.pitch_shift(y=sig, sr=sr, n_steps=s)
                b, d = a2f(x, sr)
                pb.append(b.detach().cpu().numpy().astype(np.float64))
                pd_.append(d.detach().cpu().numpy().astype(np.float64))
            Lp = min(len(z) for z in pb)
            pitch_beat = np.mean([z[:Lp] for z in pb], axis=0)
            pitch_db = np.mean([z[:Lp] for z in pd_], axis=0)

            tb, td = [], []
            for r in RATES:
                x = sig if r == 1.0 else librosa.effects.time_stretch(y=sig, rate=float(r))
                b, d = a2f(x, sr)
                tb.append(unwarp(b.detach().cpu().numpy().astype(np.float64), r, L))
                td.append(unwarp(d.detach().cpu().numpy().astype(np.float64), r, L))
            ts_beat = np.mean(tb, axis=0)
            ts_db = np.mean(td, axis=0)

            np.savez(fout, pitch_beat=pitch_beat, pitch_db=pitch_db,
                     ts_beat=ts_beat, ts_db=ts_db)
        print(f"[{args.device}] fold {fold} done", flush=True)

if __name__ == "__main__":
    main()
