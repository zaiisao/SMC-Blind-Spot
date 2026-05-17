"""
Octave error correction for beat tracking predictions.

Detects whether predicted beats are at double-tempo or half-tempo relative
to the dominant periodicity in the beat activation function, and corrects
by subsampling or interpolating. Evaluates the actual F-measure change.

Uses only the model's own activation function (no ground truth) for detection.
"""
import csv
import numpy as np
import torch
import mir_eval
from collections import defaultdict
from pathlib import Path

from beat_this.inference import Audio2Frames
from beat_this.preprocessing import load_audio

ROOT = Path(__file__).parent.parent
GT_DIR = ROOT / "beat_this_annotations" / "smc" / "annotations" / "beats"
PRED_DIR = ROOT / "beat_this_output"
SPLIT_FILE = ROOT / "beat_this_annotations" / "smc" / "8-folds.split"
AUDIO_DIR = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
OUT_DIR = ROOT / "beat_this_output_corrected"
OUT_CSV = ROOT / "octave_correction_results.csv"

FPS = 50


def load_gt(path):
    return np.array([float(line.strip()) for line in open(path) if line.strip()])


def load_pred(path):
    return np.array([float(line.strip().split("\t")[0]) for line in open(path) if line.strip()])


def autocorrelation_tempo(activation, fps=50, min_bpm=40, max_bpm=240):
    """
    Estimate the dominant tempo from the beat activation function
    using autocorrelation.

    Returns the dominant period in seconds.
    """
    act = activation.cpu().numpy() if isinstance(activation, torch.Tensor) else activation
    # Convert to probability
    act = 1.0 / (1.0 + np.exp(-act))  # sigmoid

    # Subtract mean
    act = act - act.mean()

    # Autocorrelation via FFT
    n = len(act)
    fft = np.fft.rfft(act, n=2 * n)
    acf = np.fft.irfft(fft * np.conj(fft))[:n]
    acf = acf / acf[0]  # normalize

    # Search range in frames
    min_lag = int(60.0 / max_bpm * fps)  # fastest tempo -> smallest lag
    max_lag = int(60.0 / min_bpm * fps)  # slowest tempo -> largest lag
    max_lag = min(max_lag, n - 1)

    if min_lag >= max_lag:
        return None

    # Find the dominant peak in the valid range
    search_range = acf[min_lag:max_lag + 1]
    peak_idx = np.argmax(search_range) + min_lag
    dominant_period = peak_idx / fps

    return dominant_period


def detect_octave_error(pred_beats, dominant_period):
    """
    Compare predicted median IBI against the autocorrelation dominant period.
    Returns:
        'correct' if ratio is ~1.0
        'double' if predicted IBI is ~0.5x the dominant period (too many beats)
        'half' if predicted IBI is ~2.0x the dominant period (too few beats)
        None if cannot determine
    """
    if len(pred_beats) < 3 or dominant_period is None:
        return None, None

    ibis = np.diff(pred_beats)
    median_ibi = np.median(ibis)

    if median_ibi <= 0 or dominant_period <= 0:
        return None, None

    ratio = median_ibi / dominant_period

    if 0.4 <= ratio < 0.7:
        return 'double', ratio  # predicted IBI is half -> too many beats
    elif 0.7 <= ratio <= 1.4:
        return 'correct', ratio
    elif 1.4 < ratio <= 2.5:
        return 'half', ratio  # predicted IBI is double -> too few beats
    else:
        return 'other', ratio


def correct_double_tempo(beats):
    """Subsample beats: keep every other beat, choosing the subset
    that better preserves the original timing structure."""
    if len(beats) < 4:
        return beats
    # Try both even and odd subsets, pick the one with more regular IBIs
    even = beats[0::2]
    odd = beats[1::2]
    even_std = np.std(np.diff(even)) if len(even) > 1 else float('inf')
    odd_std = np.std(np.diff(odd)) if len(odd) > 1 else float('inf')
    return even if even_std <= odd_std else odd


def correct_half_tempo(beats):
    """Interpolate beats: add a beat between each consecutive pair."""
    if len(beats) < 2:
        return beats
    new_beats = []
    for i in range(len(beats) - 1):
        new_beats.append(beats[i])
        mid = (beats[i] + beats[i + 1]) / 2.0
        new_beats.append(mid)
    new_beats.append(beats[-1])
    return np.array(new_beats)


def evaluate(ref, est):
    """Evaluate using mir_eval, return dict of metrics."""
    if len(ref) == 0 or len(est) == 0:
        return None
    return mir_eval.beat.evaluate(ref, est)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Load fold assignments
    fold_map = {}
    with open(SPLIT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            track_id, fold_num = line.split("\t")
            fold_map[track_id] = int(fold_num)

    # Group tracks by fold
    folds = defaultdict(list)
    for gt_path in sorted(GT_DIR.glob("*.beats")):
        track_id = gt_path.stem
        fold = fold_map.get(track_id)
        if fold is not None:
            folds[fold].append(track_id)

    rows = []
    summary = defaultdict(list)

    for fold_num in sorted(folds.keys()):
        tracks = folds[fold_num]
        model_name = f"fold{fold_num}"
        print(f"\nFold {fold_num}: loading {model_name} for {len(tracks)} tracks")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        audio2frames = Audio2Frames(checkpoint_path=model_name, device=device)

        for track_id in tracks:
            gt_path = GT_DIR / f"{track_id}.beats"
            pred_path = PRED_DIR / f"{track_id.upper()}.beats"

            if not pred_path.exists():
                continue

            ref_beats = load_gt(gt_path)
            pred_beats = load_pred(pred_path)

            # Get beat activation for this track
            wav_path = AUDIO_DIR / f"{track_id.upper()}.wav"
            signal, sr = load_audio(str(wav_path))
            beat_logits, _ = audio2frames(signal, sr)
            beat_act = beat_logits.cpu().numpy()

            # Estimate dominant period from activation
            dominant_period = autocorrelation_tempo(beat_logits, fps=FPS)

            # Detect octave error
            error_type, ratio = detect_octave_error(pred_beats, dominant_period)

            # Original scores
            orig_scores = evaluate(ref_beats, pred_beats)
            if orig_scores is None:
                continue

            # Apply correction
            if error_type == 'double':
                corrected_beats = correct_double_tempo(pred_beats)
            elif error_type == 'half':
                corrected_beats = correct_half_tempo(pred_beats)
            else:
                corrected_beats = pred_beats

            # Corrected scores
            corr_scores = evaluate(ref_beats, corrected_beats)

            # Also try the opposite corrections to see if we're wrong
            # (oracle: try all three and see which is best)
            double_beats = correct_double_tempo(pred_beats)
            half_beats = correct_half_tempo(pred_beats)
            double_scores = evaluate(ref_beats, double_beats)
            half_scores = evaluate(ref_beats, half_beats)

            # Oracle: best of original, subsampled, interpolated
            oracle_f = max(
                orig_scores["F-measure"],
                double_scores["F-measure"] if double_scores else 0,
                half_scores["F-measure"] if half_scores else 0,
            )
            # Which oracle choice?
            if oracle_f == orig_scores["F-measure"]:
                oracle_choice = "original"
            elif double_scores and oracle_f == double_scores["F-measure"]:
                oracle_choice = "subsample"
            else:
                oracle_choice = "interpolate"

            median_ibi = np.median(np.diff(pred_beats)) if len(pred_beats) > 1 else 0
            gt_ibi = np.median(np.diff(ref_beats)) if len(ref_beats) > 1 else 0

            row = {
                "track_id": track_id,
                "fold": fold_num,
                "gt_median_ibi": round(gt_ibi, 4),
                "pred_median_ibi": round(median_ibi, 4),
                "acf_period": round(dominant_period, 4) if dominant_period else "",
                "ibi_ratio": round(ratio, 4) if ratio else "",
                "detected_error": error_type or "",
                "orig_F": round(orig_scores["F-measure"], 4),
                "corrected_F": round(corr_scores["F-measure"], 4) if corr_scores else "",
                "delta_F": round(corr_scores["F-measure"] - orig_scores["F-measure"], 4) if corr_scores else "",
                "subsample_F": round(double_scores["F-measure"], 4) if double_scores else "",
                "interpolate_F": round(half_scores["F-measure"], 4) if half_scores else "",
                "oracle_F": round(oracle_f, 4),
                "oracle_choice": oracle_choice,
                "orig_CMLt": round(orig_scores["Correct Metric Level Total"], 4),
                "orig_AMLt": round(orig_scores["Any Metric Level Total"], 4),
            }
            rows.append(row)
            summary[error_type or "unknown"].append(row)

            status = ""
            if error_type in ('double', 'half') and row["delta_F"]:
                delta = row["delta_F"]
                status = f"  {'improved' if delta > 0.01 else 'hurt' if delta < -0.01 else 'same'} ({delta:+.3f})"
            print(f"  {track_id}: ratio={ratio:.2f} detected={error_type or '?'} "
                  f"F={orig_scores['F-measure']:.3f}->{corr_scores['F-measure']:.3f}{status}")

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["track_id"]))

    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print("OCTAVE ERROR DETECTION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Category':<12} {'N':>4} {'Orig F':>8} {'Corr F':>8} {'Delta':>8} {'Oracle F':>10}")
    print("-" * 58)
    all_orig_f = []
    all_corr_f = []
    all_oracle_f = []
    for cat in ['correct', 'double', 'half', 'other', 'unknown']:
        if cat not in summary:
            continue
        s = summary[cat]
        orig = np.mean([r["orig_F"] for r in s])
        corr = np.mean([r["corrected_F"] for r in s if r["corrected_F"] != ""])
        oracle = np.mean([r["oracle_F"] for r in s])
        delta = corr - orig
        print(f"{cat:<12} {len(s):>4} {orig:>8.3f} {corr:>8.3f} {delta:>+8.3f} {oracle:>10.3f}")
        all_orig_f.extend([r["orig_F"] for r in s])
        all_corr_f.extend([r["corrected_F"] for r in s if r["corrected_F"] != ""])
        all_oracle_f.extend([r["oracle_F"] for r in s])

    print("-" * 58)
    print(f"{'TOTAL':<12} {len(rows):>4} {np.mean(all_orig_f):>8.3f} "
          f"{np.mean(all_corr_f):>8.3f} {np.mean(all_corr_f)-np.mean(all_orig_f):>+8.3f} "
          f"{np.mean(all_oracle_f):>10.3f}")

    # Detection accuracy
    print(f"\n{'='*80}")
    print("DETECTION ACCURACY (vs oracle)")
    print(f"{'='*80}")
    correct_detections = 0
    total_with_error = 0
    for row in rows:
        if row["oracle_choice"] == "original":
            if row["detected_error"] == "correct":
                correct_detections += 1
        elif row["oracle_choice"] == "subsample":
            total_with_error += 1
            if row["detected_error"] == "double":
                correct_detections += 1
        elif row["oracle_choice"] == "interpolate":
            total_with_error += 1
            if row["detected_error"] == "half":
                correct_detections += 1

    oracle_changes = sum(1 for r in rows if r["oracle_choice"] != "original")
    det_changes = sum(1 for r in rows if r["detected_error"] in ("double", "half"))
    print(f"  Oracle says {oracle_changes} tracks need correction")
    print(f"  Detector flags {det_changes} tracks for correction")
    print(f"  Tracks where correction helped (delta_F > 0.01): "
          f"{sum(1 for r in rows if isinstance(r['delta_F'], float) and r['delta_F'] > 0.01)}")
    print(f"  Tracks where correction hurt (delta_F < -0.01): "
          f"{sum(1 for r in rows if isinstance(r['delta_F'], float) and r['delta_F'] < -0.01)}")

    # Oracle upper bound
    print(f"\n{'='*80}")
    print("ORACLE UPPER BOUND")
    print(f"{'='*80}")
    print(f"  Original mean F:  {np.mean(all_orig_f):.4f}")
    print(f"  Corrected mean F: {np.mean(all_corr_f):.4f} (automated detection)")
    print(f"  Oracle mean F:    {np.mean(all_oracle_f):.4f} (always pick best of 3)")
    print(f"  Oracle gain:      {np.mean(all_oracle_f) - np.mean(all_orig_f):+.4f}")


if __name__ == "__main__":
    main()
