"""Does Beat This behave EQUIVARIANTLY under time-stretch? A no-training diagnostic.

THE QUESTION. The proposed fix for the slow-tempo failure is an equivariant contrastive objective,
    f(stretch(x, s)) = f(x) - log2 s
whose appeal is that behaviour on the well-populated 70-90 BPM region would then DETERMINE behaviour
at 40 BPM by construction, converting an extrapolation failure into interpolation. That argument is
only worth acting on if the released model actually VIOLATES the identity. If it is already
equivariant, the slow-tempo bias originates somewhere else and contrastive training cannot fix it.

THE TEST, which needs no tempo head and no retraining. Stretch audio by a known factor s, run the
fold-honest released checkpoint, peak-pick, and take the median inter-beat interval. Under perfect
equivariance IBI(s) = s * IBI(1) exactly, so the defect

    delta(s) = log2( IBI(s) / IBI(1) ) - log2 s

is 0 for every s. delta > 0 means the model reports the stretched version as SLOWER than it really
is; delta < 0 means faster. An octave error contributes +/-1.0.

WHY IBI AND NOT F. This deliberately does NOT compare against annotations. Equivariance is a
self-consistency property of the model, so the reference is the model's own unstretched output. A
model can be badly wrong about absolute tempo and still perfectly equivariant -- those are separate
defects and this separates them.

STRETCHING IS AUDIO-DOMAIN (librosa phase vocoder), not the mel time-axis zoom used by the earlier
fine-tune, because the whole concern with the contrastive idea is that a model can read the
resampling artefact instead of musical tempo. Phase vocoding preserves the spectral envelope and is
the same transformation a contrastive pipeline would use.

FOLD HONESTY. Each SMC track is run with the checkpoint that held it out.
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "beat_this"))
from beat_this.inference import Audio2Frames
from beat_this.preprocessing import load_audio
from _harness import FPS, track_ids, load_gt, peakpick

SMC_AUD = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
STORE = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
STRETCH = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]   # >1 = slower (longer), the SMC-relevant direction
N_TRACKS = 40


def med_ibi(beats):
    return float(np.median(np.diff(beats))) if len(beats) >= 4 else float("nan")


def main():
    dev = "cuda:1"
    split = {l.split("\t")[0]: int(l.split("\t")[1])
             for l in open(STORE / "smc" / "8-folds.split") if l.strip()}
    ids = [t for t in track_ids() if (SMC_AUD / f"{t.upper()}.wav").exists()][:N_TRACKS]
    by_fold = {}
    for t in ids:
        by_fold.setdefault(split.get(t, 0), []).append(t)

    gt_bpm = {}
    for t in ids:
        g = load_gt(t)
        gt_bpm[t] = 60.0 / float(np.median(np.diff(g))) if len(g) >= 4 else float("nan")

    rows = {s: {} for s in STRETCH}
    for fold, tracks in sorted(by_fold.items()):
        a2f = Audio2Frames(checkpoint_path=f"fold{fold}", device=dev)
        for t in tracks:
            sig, sr = load_audio(str(SMC_AUD / f"{t.upper()}.wav"))
            y = sig.mean(1) if sig.ndim > 1 else sig
            y = np.ascontiguousarray(y.astype(np.float32))
            for s in STRETCH:
                # librosa rate>1 speeds UP and shortens; we want s>1 to mean SLOWER, so rate = 1/s
                ys = y if s == 1.0 else librosa.effects.time_stretch(y, rate=1.0 / s)
                b, _ = a2f(ys, sr)
                bp = torch.sigmoid(b.float()).cpu().numpy()
                rows[s][t] = med_ibi(peakpick(bp, 0.5, FPS))
        print(f"  fold {fold}: {len(tracks)} tracks done", flush=True)

    print(f"\nEQUIVARIANCE DEFECT  delta(s) = log2(IBI(s)/IBI(1)) - log2(s),  0 = perfect")
    print(f"{'s':>6} {'impliedBPM':>11} {'n':>4} {'median d':>10} {'mean d':>9} {'|d|>0.5':>8}")
    out = {}
    base = rows[1.0]
    for s in STRETCH:
        d = []
        for t in ids:
            a, b = base.get(t, np.nan), rows[s].get(t, np.nan)
            if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
                d.append(np.log2(b / a) - np.log2(s))
        d = np.array(d)
        bpm = float(np.nanmedian([gt_bpm[t] for t in ids])) / s
        oct_ = int((np.abs(d) > 0.5).sum())
        out[s] = {"n": len(d), "median": float(np.median(d)), "mean": float(d.mean()),
                  "octave_errors": oct_}
        print(f"{s:>6.2f} {bpm:>11.1f} {len(d):>4} {np.median(d):>+10.3f} "
              f"{d.mean():>+9.3f} {oct_:>8d}")

    json.dump({"stretch": STRETCH, "n_tracks": len(ids), "defect": out,
               "ibi": {str(s): rows[s] for s in STRETCH}},
              open("/tmp/overnight/tempo_equivariance.json", "w"), indent=1)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
