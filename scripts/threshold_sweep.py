"""
Peak-picking threshold sweep for beat_this predictions.

The default minimal postprocessor uses logit > 0 (probability > 0.5).
This script sweeps over different thresholds to find the precision-recall
tradeoff on SMC and see if the default is optimal for this dataset.
"""
import csv
import numpy as np
import torch
import torch.nn.functional as F
import mir_eval
from collections import defaultdict
from pathlib import Path

from beat_this.inference import Audio2Frames
from beat_this.preprocessing import load_audio
from beat_this.model.postprocessor import deduplicate_peaks

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR = ROOT / "beat_this_annotations" / "smc" / "annotations" / "beats"
SPLIT_FILE = ROOT / "beat_this_annotations" / "smc" / "8-folds.split"
AUDIO_DIR = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
OUT_CSV = DATA_DIR / "threshold_sweep_results.csv"

FPS = 50


def load_gt(path):
    return np.array([float(line.strip()) for line in open(path) if line.strip()])


def pick_beats_at_threshold(beat_logits, threshold, fps=50):
    """
    Apply peak-picking at a given logit threshold.
    Replicates the minimal postprocessor logic but with a variable threshold.
    """
    # Max-pool to find local maxima within +/- 70ms (7 frames)
    logits = beat_logits.unsqueeze(0)  # (1, T)
    peaks = logits.masked_fill(
        logits != F.max_pool1d(logits, 7, 1, 3), -1000
    )
    # Apply threshold
    peak_mask = peaks.squeeze(0) > threshold
    frames = torch.nonzero(peak_mask).cpu().numpy()[:, 0]
    frames = deduplicate_peaks(frames, width=1)
    return frames / fps


def main():
    # Load fold assignments
    fold_map = {}
    with open(SPLIT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            track_id, fold_num = line.split("\t")
            fold_map[track_id] = int(fold_num)

    # Group by fold
    folds = defaultdict(list)
    for gt_path in sorted(GT_DIR.glob("*.beats")):
        track_id = gt_path.stem
        fold = fold_map.get(track_id)
        if fold is not None:
            folds[fold].append(track_id)

    # Thresholds to sweep (logit space: 0 = p=0.5, -2 = p=0.12, 2 = p=0.88)
    thresholds = np.arange(-3.0, 4.5, 0.5)

    # First pass: collect all logits per track
    print("Loading model predictions for all folds...")
    track_logits = {}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for fold_num in sorted(folds.keys()):
        tracks = folds[fold_num]
        model_name = f"fold{fold_num}"
        print(f"  Fold {fold_num}: {len(tracks)} tracks with {model_name}")
        audio2frames = Audio2Frames(checkpoint_path=model_name, device=device)

        for track_id in tracks:
            wav_path = AUDIO_DIR / f"{track_id.upper()}.wav"
            signal, sr = load_audio(str(wav_path))
            beat_logits, _ = audio2frames(signal, sr)
            track_logits[track_id] = beat_logits.cpu()

    print(f"\nLoaded {len(track_logits)} tracks. Sweeping {len(thresholds)} thresholds...\n")

    # Second pass: sweep thresholds
    results = []
    per_track_rows = []

    print(f"{'Threshold':>10} {'Prob':>6} {'MeanF':>8} {'MedF':>8} "
          f"{'CMLt':>8} {'AMLt':>8} {'AvgBeats':>9}")
    print("-" * 65)

    for thresh in thresholds:
        prob = 1.0 / (1.0 + np.exp(-thresh))
        all_f = []
        all_cmlt = []
        all_amlt = []
        all_nbeats = []

        for track_id, logits in track_logits.items():
            gt_path = GT_DIR / f"{track_id}.beats"
            ref = load_gt(gt_path)

            est = pick_beats_at_threshold(logits, thresh, FPS)
            all_nbeats.append(len(est))

            if len(ref) == 0 or len(est) == 0:
                all_f.append(0.0)
                all_cmlt.append(0.0)
                all_amlt.append(0.0)
                continue

            scores = mir_eval.beat.evaluate(ref, est)
            all_f.append(scores["F-measure"])
            all_cmlt.append(scores["Correct Metric Level Total"])
            all_amlt.append(scores["Any Metric Level Total"])

            if abs(thresh - 0.0) < 0.01:  # save per-track at default threshold for reference
                per_track_rows.append({
                    "track_id": track_id,
                    "default_F": round(scores["F-measure"], 4),
                })

        mean_f = np.mean(all_f)
        med_f = np.median(all_f)
        mean_cmlt = np.mean(all_cmlt)
        mean_amlt = np.mean(all_amlt)
        avg_beats = np.mean(all_nbeats)

        print(f"{thresh:>10.1f} {prob:>6.3f} {mean_f:>8.4f} {med_f:>8.4f} "
              f"{mean_cmlt:>8.4f} {mean_amlt:>8.4f} {avg_beats:>9.1f}")

        results.append({
            "threshold": thresh,
            "probability": round(prob, 4),
            "mean_F": round(mean_f, 4),
            "median_F": round(med_f, 4),
            "mean_CMLt": round(mean_cmlt, 4),
            "mean_AMLt": round(mean_amlt, 4),
            "avg_n_beats": round(avg_beats, 1),
        })

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Find optimal thresholds
    best_f_row = max(results, key=lambda r: r["mean_F"])
    best_cmlt_row = max(results, key=lambda r: r["mean_CMLt"])
    best_amlt_row = max(results, key=lambda r: r["mean_AMLt"])
    default_row = min(results, key=lambda r: abs(r["threshold"]))

    print(f"\n{'='*65}")
    print("OPTIMAL THRESHOLDS")
    print(f"{'='*65}")
    print(f"  Default  (t= {default_row['threshold']:>4.1f}, p={default_row['probability']:.3f}): "
          f"F={default_row['mean_F']:.4f}  CMLt={default_row['mean_CMLt']:.4f}  AMLt={default_row['mean_AMLt']:.4f}")
    print(f"  Best F   (t= {best_f_row['threshold']:>4.1f}, p={best_f_row['probability']:.3f}): "
          f"F={best_f_row['mean_F']:.4f}  CMLt={best_f_row['mean_CMLt']:.4f}  AMLt={best_f_row['mean_AMLt']:.4f}")
    print(f"  Best CMLt(t= {best_cmlt_row['threshold']:>4.1f}, p={best_cmlt_row['probability']:.3f}): "
          f"F={best_cmlt_row['mean_F']:.4f}  CMLt={best_cmlt_row['mean_CMLt']:.4f}  AMLt={best_cmlt_row['mean_AMLt']:.4f}")
    print(f"  Best AMLt(t= {best_amlt_row['threshold']:>4.1f}, p={best_amlt_row['probability']:.3f}): "
          f"F={best_amlt_row['mean_F']:.4f}  CMLt={best_amlt_row['mean_CMLt']:.4f}  AMLt={best_amlt_row['mean_AMLt']:.4f}")

    f_gain = best_f_row['mean_F'] - default_row['mean_F']
    print(f"\n  F-measure gain from threshold tuning: {f_gain:+.4f}")
    print(f"\nWrote {len(results)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
