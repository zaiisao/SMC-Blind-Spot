"""
Time-stretch TTA worker (one GPU shard).

Usage: exp_ts_worker.py --folds 0,1 --device cuda:0

For each track in the given folds and each stretch rate r:
  - time_stretch the audio by r   (librosa: r>1 => faster/shorter)
  - run Beat This (track's held-out fold checkpoint) -> logits at 50 fps on the
    STRETCHED time axis
  - un-warp to the ORIGINAL 50 fps grid:
        a moment at stretched-time tau_s sits at original-time tau_o = tau_s * r
        (stretched audio is compressed by r, so multiply to recover original time)
    so  A_orig[k/50] = interp( k/50,  xp = (arange(L_s)/50)*r,  fp = A_s )
  - save un-warped logits to cache/ts_tta_cache/{track}__r{r}.npz

rate == 1.0 is a no-op stretch and MUST reproduce the cached activation; the
evaluator asserts this.
"""
import argparse, numpy as np, torch, librosa
from pathlib import Path

ROOT = Path("/home/sogang/jaehoon/Analyze-SMC")
AUDIO = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
SPLIT = ROOT / "third-party" / "beat_this_annotations" / "smc" / "8-folds.split"
BTHIS = ROOT / "cache" / "beat_this_activations_cache"
OUT = ROOT / "data" / "ts_tta_cache"
OUT.mkdir(parents=True, exist_ok=True)
FPS = 50
RATES = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]

def unwarp(logits_s, rate, L_orig):
    """Map stretched-axis logits onto the original 50 fps grid of length L_orig."""
    L_s = len(logits_s)
    xp = (np.arange(L_s) / FPS) * rate          # original-time of each stretched frame
    grid = np.arange(L_orig) / FPS              # original 50 fps grid
    return np.interp(grid, xp, logits_s)

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
    tracks_by_fold = {}
    for tid, f in fold_map.items():
        tracks_by_fold.setdefault(f, []).append(tid)

    for fold in sorted(folds):
        a2f = Audio2Frames(checkpoint_path=f"fold{fold}", device=args.device)
        for tid in sorted(tracks_by_fold.get(fold, [])):
            L_orig = len(np.load(BTHIS / f"{tid}.npz")["beat"])
            sig, sr = load_audio(str(AUDIO / f"{tid.upper()}.wav"))
            sig = np.asarray(sig).astype(np.float32)
            if sig.ndim > 1:
                sig = sig.mean(axis=1)
            for r in RATES:
                fout = OUT / f"{tid}__r{r}.npz"
                if fout.exists():
                    continue
                x = sig if r == 1.0 else librosa.effects.time_stretch(y=sig, rate=float(r))
                bl, _ = a2f(x, sr)
                bl = bl.detach().cpu().numpy().astype(np.float64)
                uw = unwarp(bl, r, L_orig)
                np.savez(fout, logits=uw)
        print(f"[{args.device}] fold {fold} done", flush=True)

if __name__ == "__main__":
    main()
