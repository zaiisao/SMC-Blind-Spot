"""
Run beat_this inference per fold WITH DBN postprocessing,
capturing the DBN log-probability (confidence) for each track.

Uses beat_this's Audio2Frames to get frame-level logits, then
manually runs the madmom DBN to extract both beat positions and
log-probabilities — without modifying any submodule code.
"""
import csv
import itertools as it
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from beat_this.inference import Audio2Frames
from beat_this.preprocessing import load_audio
from beat_this.utils import save_beat_tsv

from madmom.features.downbeats import (
    DBNDownBeatTrackingProcessor,
    _process_dbn,
)

ROOT = Path(__file__).parent.parent
SPLIT_FILE = ROOT / "beat_this_annotations" / "smc" / "8-folds.split"
AUDIO_DIR = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
OUTPUT_DIR = ROOT / "beat_this_output_dbn"
CONFIDENCE_CSV = ROOT / "dbn_confidence.csv"

FPS = 50


def build_dbn():
    """Build the same DBN processor that beat_this uses, and return it."""
    return DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4],
        min_bpm=55.0,
        max_bpm=215.0,
        fps=FPS,
        transition_lambda=100,
    )


def run_dbn_with_confidence(dbn, beat_logits, downbeat_logits):
    """
    Replicate beat_this's _postp_dbn_item + DBNDownBeatTrackingProcessor.process(),
    but also capture the per-HMM log-probabilities.

    Returns:
        beats: np.ndarray of beat times in seconds
        downbeats: np.ndarray of downbeat times in seconds
        best_log_prob: float, log-probability of the best Viterbi path
        best_bar: int, beats_per_bar of the winning HMM (3 or 4)
        n_frames: int, number of frames
        all_log_probs: list of (beats_per_bar, log_prob) for each HMM
    """
    # Convert logits to probabilities (same as beat_this postprocessor)
    beat_prob = beat_logits.double().sigmoid()
    downbeat_prob = downbeat_logits.double().sigmoid()
    epsilon = 1e-5
    beat_prob = beat_prob * (1 - epsilon) + epsilon / 2
    downbeat_prob = downbeat_prob * (1 - epsilon) + epsilon / 2

    # Build combined activation (same as beat_this)
    combined_act = np.vstack((
        np.maximum(beat_prob.cpu().numpy() - downbeat_prob.cpu().numpy(), epsilon / 2),
        downbeat_prob.cpu().numpy(),
    )).T

    n_frames = len(combined_act)

    # Run Viterbi on each HMM (3/4 and 4/4)
    results = [hmm.viterbi(combined_act) for hmm in dbn.hmms]

    # Collect log-probs for all HMMs
    all_log_probs = []
    for i, (path, log_prob) in enumerate(results):
        all_log_probs.append((dbn.beats_per_bar[i], log_prob))

    # Pick the best HMM
    best_idx = np.argmax([r[1] for r in results])
    path, best_log_prob = results[best_idx]
    best_bar = dbn.beats_per_bar[best_idx]

    # Extract beat positions (replicating DBNDownBeatTrackingProcessor.process)
    st = dbn.hmms[best_idx].transition_model.state_space
    om = dbn.hmms[best_idx].observation_model
    positions = st.state_positions[path]
    beat_numbers = positions.astype(int) + 1

    # Correct beat positions to activation peaks
    beats = np.empty(0, dtype=int)
    beat_range = om.pointers[path] >= 1
    if beat_range.any():
        idx = np.nonzero(np.diff(beat_range.astype(int)))[0] + 1
        if beat_range[0]:
            idx = np.r_[0, idx]
        if beat_range[-1]:
            idx = np.r_[idx, beat_range.size]
        if idx.any():
            for left, right in idx.reshape((-1, 2)):
                peak = np.argmax(combined_act[left:right]) // 2 + left
                beats = np.hstack((beats, peak))

    if len(beats) == 0:
        return np.array([]), np.array([]), best_log_prob, best_bar, n_frames, all_log_probs

    beat_times = beats / float(FPS)
    downbeat_mask = beat_numbers[beats] == 1
    downbeat_times = beat_times[downbeat_mask]

    return beat_times, downbeat_times, best_log_prob, best_bar, n_frames, all_log_probs


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Parse fold assignments
    folds = defaultdict(list)
    with open(SPLIT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            track_id, fold_num = line.split("\t")
            wav_name = track_id.upper() + ".wav"
            wav_path = AUDIO_DIR / wav_name
            if wav_path.exists():
                folds[int(fold_num)].append((track_id, wav_path))
            else:
                print(f"WARNING: {wav_path} not found, skipping")

    # Build the DBN once (it's model-independent, only depends on tempo range)
    dbn = build_dbn()

    confidence_rows = []

    for fold_num in sorted(folds.keys()):
        tracks = folds[fold_num]
        model_name = f"fold{fold_num}"
        print(f"\n{'='*60}")
        print(f"Fold {fold_num}: {len(tracks)} files with checkpoint {model_name}")
        print(f"{'='*60}")

        # Load the model for this fold
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        audio2frames = Audio2Frames(checkpoint_path=model_name, device=device)

        for track_id, wav_path in tracks:
            # Get frame-level logits
            signal, sr = load_audio(str(wav_path))
            beat_logits, downbeat_logits = audio2frames(signal, sr)

            # Run DBN with confidence extraction
            beat_times, downbeat_times, log_prob, best_bar, n_frames, all_log_probs = \
                run_dbn_with_confidence(dbn, beat_logits, downbeat_logits)

            # Save .beats file (same format as beat_this CLI)
            out_path = OUTPUT_DIR / (track_id.upper() + ".beats")
            save_beat_tsv(beat_times, downbeat_times, str(out_path))

            # Compute normalized log-prob
            norm_log_prob = log_prob / n_frames if n_frames > 0 else 0.0

            # Log-prob for each HMM
            log_prob_3 = next((lp for b, lp in all_log_probs if b == 3), None)
            log_prob_4 = next((lp for b, lp in all_log_probs if b == 4), None)

            confidence_rows.append({
                "track_id": track_id,
                "fold": fold_num,
                "n_frames": n_frames,
                "n_beats": len(beat_times),
                "best_bar": best_bar,
                "log_prob": log_prob,
                "norm_log_prob": norm_log_prob,
                "log_prob_3_4": log_prob_3,
                "log_prob_4_4": log_prob_4,
            })

            print(f"  {track_id}: {len(beat_times)} beats, "
                  f"bar={best_bar}/4, "
                  f"log_p={log_prob:.1f}, "
                  f"norm={norm_log_prob:.4f}")

    # Write confidence CSV
    fieldnames = ["track_id", "fold", "n_frames", "n_beats", "best_bar",
                  "log_prob", "norm_log_prob", "log_prob_3_4", "log_prob_4_4"]
    with open(CONFIDENCE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(confidence_rows)

    print(f"\nDone! Wrote {len(confidence_rows)} rows to {CONFIDENCE_CSV}")
    print(f"Predictions saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
