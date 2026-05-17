"""
IBI outlier cleanup for beat tracking predictions.

Targets tracks where the model is mostly right but occasionally drops or
inserts a spurious beat, breaking continuity. Flags beats where the IBI
deviates significantly from the local median, then removes (short IBI →
extra beat) or interpolates (long IBI → missing beat).
"""
import csv
import numpy as np
import mir_eval
from pathlib import Path

ROOT = Path(__file__).parent.parent
GT_DIR = ROOT / "beat_this_annotations" / "smc" / "annotations" / "beats"
PRED_DIR = ROOT / "beat_this_output"
OUT_CSV = ROOT / "ibi_cleanup_results.csv"

FPS = 50


def load_gt(path):
    return np.array([float(line.strip()) for line in open(path) if line.strip()])


def load_pred(path):
    return np.array([float(line.strip().split("\t")[0]) for line in open(path) if line.strip()])


def local_median_ibi(beats, window=5):
    """Compute local median IBI for each beat using a sliding window."""
    if len(beats) < 3:
        return np.array([])
    ibis = np.diff(beats)
    local_medians = np.zeros(len(ibis))
    for i in range(len(ibis)):
        start = max(0, i - window)
        end = min(len(ibis), i + window + 1)
        local_medians[i] = np.median(ibis[start:end])
    return local_medians


def cleanup_beats(beats, deviation_threshold=0.5, window=5):
    """
    Remove spurious beats and interpolate missing beats.

    Args:
        beats: array of beat times
        deviation_threshold: flag IBIs that deviate by more than this
                             fraction from the local median
        window: half-window size for local median computation

    Returns:
        cleaned_beats: corrected beat array
        n_removed: number of spurious beats removed
        n_inserted: number of missing beats interpolated
    """
    if len(beats) < 4:
        return beats, 0, 0

    n_removed = 0
    n_inserted = 0

    # Pass 1: Remove spurious beats (abnormally short IBIs)
    # A spurious extra beat creates two short IBIs in a row
    cleaned = list(beats)
    changed = True
    while changed:
        changed = False
        if len(cleaned) < 4:
            break
        ibis = np.diff(cleaned)
        local_meds = local_median_ibi(np.array(cleaned), window)
        to_remove = set()
        i = 0
        while i < len(ibis):
            ratio = ibis[i] / local_meds[i] if local_meds[i] > 0 else 1.0
            if ratio < (1.0 - deviation_threshold):
                # Short IBI detected. Check if removing beat i or i+1
                # produces an IBI closer to the local median.
                # Don't remove the first or last beat.
                if i > 0 and i + 1 < len(cleaned) - 1:
                    # If next IBI is also short, this is likely a spurious beat at i+1
                    if i + 1 < len(ibis) and ibis[i + 1] / local_meds[i + 1] < (1.0 - deviation_threshold * 0.5):
                        to_remove.add(i + 1)
                        n_removed += 1
                        i += 2
                        changed = True
                        continue
                    # Otherwise remove whichever produces a closer-to-median IBI
                    ibi_without_i1 = cleaned[i + 2] - cleaned[i] if i + 2 < len(cleaned) else float('inf')
                    ibi_without_i = cleaned[i + 1] - cleaned[i - 1] if i > 0 else float('inf')
                    dev_i1 = abs(ibi_without_i1 / local_meds[i] - 1.0)
                    dev_i = abs(ibi_without_i / local_meds[i] - 1.0)
                    if dev_i1 < dev_i and i + 1 not in to_remove:
                        to_remove.add(i + 1)
                        n_removed += 1
                        changed = True
                elif i == 0 and len(cleaned) > 2:
                    # Short IBI at the start — might be a spurious first beat
                    # Only remove if the second IBI is normal
                    if len(ibis) > 1 and ibis[1] / local_meds[1] > (1.0 - deviation_threshold):
                        to_remove.add(0)
                        n_removed += 1
                        changed = True
            i += 1
        if to_remove:
            cleaned = [b for j, b in enumerate(cleaned) if j not in to_remove]

    # Pass 2: Interpolate missing beats (abnormally long IBIs)
    final = []
    ibis = np.diff(cleaned)
    local_meds = local_median_ibi(np.array(cleaned), window)
    for i in range(len(cleaned) - 1):
        final.append(cleaned[i])
        if len(local_meds) > i and local_meds[i] > 0:
            ratio = ibis[i] / local_meds[i]
            if ratio > (1.0 + deviation_threshold):
                # Long IBI — estimate how many beats are missing
                n_missing = round(ibis[i] / local_meds[i]) - 1
                if n_missing >= 1 and n_missing <= 3:  # sanity cap
                    for k in range(1, n_missing + 1):
                        interp = cleaned[i] + k * ibis[i] / (n_missing + 1)
                        final.append(interp)
                        n_inserted += 1
    final.append(cleaned[-1])

    return np.array(sorted(final)), n_removed, n_inserted


def evaluate(ref, est):
    if len(ref) == 0 or len(est) == 0:
        return None
    return mir_eval.beat.evaluate(ref, est)


def main():
    pred_files = sorted(PRED_DIR.glob("*.beats"))
    rows = []

    # Sweep over deviation thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    windows = [3, 5, 7]

    best_config = None
    best_mean_f = 0

    for thresh in thresholds:
        for win in windows:
            all_orig_f = []
            all_clean_f = []
            n_changed = 0

            for pred_path in pred_files:
                track_id = pred_path.stem.lower()
                gt_path = GT_DIR / f"{track_id}.beats"
                if not gt_path.exists():
                    continue

                ref = load_gt(gt_path)
                est = load_pred(pred_path)
                orig = evaluate(ref, est)
                if orig is None:
                    continue

                cleaned, n_rem, n_ins = cleanup_beats(est, deviation_threshold=thresh, window=win)
                clean_scores = evaluate(ref, cleaned)
                if clean_scores is None:
                    continue

                all_orig_f.append(orig["F-measure"])
                all_clean_f.append(clean_scores["F-measure"])
                if n_rem > 0 or n_ins > 0:
                    n_changed += 1

            mean_orig = np.mean(all_orig_f)
            mean_clean = np.mean(all_clean_f)
            delta = mean_clean - mean_orig

            if mean_clean > best_mean_f:
                best_mean_f = mean_clean
                best_config = (thresh, win)

            print(f"thresh={thresh:.1f} win={win}: "
                  f"F {mean_orig:.4f} -> {mean_clean:.4f} ({delta:+.4f}) "
                  f"[{n_changed} tracks changed]")

    print(f"\nBest config: thresh={best_config[0]}, window={best_config[1]}, F={best_mean_f:.4f}")

    # Run the best config and produce detailed per-track results
    thresh, win = best_config
    print(f"\n{'='*80}")
    print(f"DETAILED RESULTS WITH thresh={thresh}, window={win}")
    print(f"{'='*80}")

    all_orig_f = []
    all_clean_f = []
    all_orig_cmlt = []
    all_clean_cmlt = []

    for pred_path in pred_files:
        track_id = pred_path.stem.lower()
        gt_path = GT_DIR / f"{track_id}.beats"
        if not gt_path.exists():
            continue

        ref = load_gt(gt_path)
        est = load_pred(pred_path)
        orig = evaluate(ref, est)
        if orig is None:
            continue

        cleaned, n_rem, n_ins = cleanup_beats(est, deviation_threshold=thresh, window=win)
        clean = evaluate(ref, cleaned)
        if clean is None:
            continue

        delta_f = clean["F-measure"] - orig["F-measure"]
        delta_cmlt = clean["Correct Metric Level Total"] - orig["Correct Metric Level Total"]

        row = {
            "track_id": track_id,
            "n_beats_orig": len(est),
            "n_beats_clean": len(cleaned),
            "n_removed": n_rem,
            "n_inserted": n_ins,
            "orig_F": round(orig["F-measure"], 4),
            "clean_F": round(clean["F-measure"], 4),
            "delta_F": round(delta_f, 4),
            "orig_CMLt": round(orig["Correct Metric Level Total"], 4),
            "clean_CMLt": round(clean["Correct Metric Level Total"], 4),
            "delta_CMLt": round(delta_cmlt, 4),
            "orig_AMLt": round(orig["Any Metric Level Total"], 4),
            "clean_AMLt": round(clean["Any Metric Level Total"], 4),
        }
        rows.append(row)
        all_orig_f.append(orig["F-measure"])
        all_clean_f.append(clean["F-measure"])
        all_orig_cmlt.append(orig["Correct Metric Level Total"])
        all_clean_cmlt.append(clean["Correct Metric Level Total"])

        if n_rem > 0 or n_ins > 0:
            print(f"  {track_id}: removed={n_rem} inserted={n_ins} "
                  f"F={orig['F-measure']:.3f}->{clean['F-measure']:.3f} ({delta_f:+.3f}) "
                  f"CMLt={orig['Correct Metric Level Total']:.3f}->{clean['Correct Metric Level Total']:.3f} ({delta_cmlt:+.3f})")

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_changed = sum(1 for r in rows if r["n_removed"] > 0 or r["n_inserted"] > 0)
    n_improved = sum(1 for r in rows if r["delta_F"] > 0.01)
    n_hurt = sum(1 for r in rows if r["delta_F"] < -0.01)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  Tracks changed:  {n_changed}")
    print(f"  Tracks improved: {n_improved} (delta_F > 0.01)")
    print(f"  Tracks hurt:     {n_hurt} (delta_F < -0.01)")
    print(f"  Mean F:          {np.mean(all_orig_f):.4f} -> {np.mean(all_clean_f):.4f} ({np.mean(all_clean_f)-np.mean(all_orig_f):+.4f})")
    print(f"  Mean CMLt:       {np.mean(all_orig_cmlt):.4f} -> {np.mean(all_clean_cmlt):.4f} ({np.mean(all_clean_cmlt)-np.mean(all_orig_cmlt):+.4f})")
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
