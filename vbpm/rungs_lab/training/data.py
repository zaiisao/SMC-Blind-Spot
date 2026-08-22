"""Song catalog, crop sampling and target construction, shared by every training algorithm.

Knows nothing about rungs: a chassis' annotated_state_path is passed in for the eligibility filter.
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d

from data.songs import iter_songs

ROOT = Path(__file__).resolve().parent.parent
FPS = 44100 / 1024
DEMIX_ROOT = ROOT / "cache" / "bt_demix"
MAX_CROP_FRAMES = 700                                   # ~16 s
EVAL_FOLD = 0


def load_songs(annotated_state_path):
    """(train, val, skipped). Uniform eligibility filter (identical across rungs so no two arms see
    different data): demix cache exists, meter in {3, 4}, >=8 beats / >=2 downbeats, whole-song
    annotated path representable in the tempo grid. `annotated_state_path` is a callable (a DBN2016
    chassis method) used only for the representability test."""
    train, val, skipped = [], [], 0
    for song in iter_songs():
        if song.fold is None or song.audio_path is None:
            continue
        mel_path = DEMIX_ROOT / song.dataset / f"{song.stem}.npz"
        if not mel_path.exists():
            skipped += 1
            continue
        beat_times, downbeat_times = song.beats()
        if len(beat_times) < 8 or len(downbeat_times) < 2:
            skipped += 1
            continue
        beat_frames = np.round(beat_times * FPS).astype(np.int64)
        is_downbeat = np.isin(np.round(beat_times, 4), np.round(downbeat_times, 4))
        downbeat_indices = np.where(is_downbeat)[0]
        gaps = np.diff(downbeat_indices)
        if len(gaps) == 0:
            skipped += 1
            continue
        beats_per_bar = int(np.median(gaps))
        beat_in_bar = np.zeros(len(beat_frames), dtype=np.int64)
        position = -1
        for i in range(len(beat_frames)):
            position = 0 if is_downbeat[i] else (position + 1 if position >= 0 else -1)
            beat_in_bar[i] = position
        valid_from = downbeat_indices[0]                # positions defined from the first downbeat
        entry = dict(stem=song.stem, dataset=song.dataset, fold=song.fold, mel_path=mel_path,
                     beat_frames=beat_frames[valid_from:], beat_in_bar=beat_in_bar[valid_from:],
                     beats_per_bar=beats_per_bar,
                     beat_times=beat_times, downbeat_times=downbeat_times)
        if annotated_state_path(entry["beat_frames"], entry["beat_in_bar"], beats_per_bar) is None:
            skipped += 1
            continue
        (val if song.fold == EVAL_FOLD else train).append(entry)
    return train, val, skipped


def sample_crop(entry, rng):
    """Beat-aligned crop: beat indices [start, end] spanning whole beats, <= MAX_CROP_FRAMES."""
    beat_frames = entry["beat_frames"]
    start = rng.integers(0, max(1, len(beat_frames) - 4))
    end = start + 1
    while end + 1 < len(beat_frames) and beat_frames[end + 1] - beat_frames[start] <= MAX_CROP_FRAMES:
        end += 1
    return start, end


def widened_targets(beat_frames, is_downbeat_mask, num_frames):
    """Beat Transformer target construction: one-hot at beat frames, widened twice (1/.5/.25)."""
    target = np.zeros((num_frames, 2), dtype=np.float32)
    inside = (beat_frames >= 0) & (beat_frames < num_frames)
    target[beat_frames[inside], 0] = 1.0
    downbeat_frames = beat_frames[inside & is_downbeat_mask]
    target[downbeat_frames, 1] = 1.0
    for c in range(2):
        target[:, c] = np.maximum(target[:, c], maximum_filter1d(target[:, c], size=3) * 0.5)
        target[:, c] = np.maximum(target[:, c], maximum_filter1d(target[:, c], size=3) * 0.5)
    return target


def smooth_beat_frames(beat_times, tolerance=1):
    """Smoothest-in-tolerance quantization: integer frames f_i with |f_i - round(t_i*FPS)| <=
    tolerance minimizing sum |interval_i - interval_{i-1}|, removing the +-1 frame rounding jitter
    that manufactures fake tempo jumps in the target path.

    Each DP state is (offset of this beat, interval just entered); the cost is accumulated
    |interval change|."""
    rounded_frames = np.round(beat_times * FPS).astype(np.int64)
    candidate_offsets = list(range(-tolerance, tolerance + 1))
    # state -> (best cost to reach it, predecessor state). Seed: first beat, no interval yet.
    dp_states = {(offset, None): (0.0, None) for offset in candidate_offsets}
    back_pointers = []
    for beat_index in range(1, len(rounded_frames)):
        rounded_gap = int(rounded_frames[beat_index] - rounded_frames[beat_index - 1])
        next_states = {}
        for (previous_offset, previous_interval), (cost, _) in dp_states.items():
            for next_offset in candidate_offsets:
                interval = rounded_gap + next_offset - previous_offset
                if interval < 2:                        # tempo cannot exceed 1 beat / 2 frames
                    continue
                new_cost = cost + (abs(interval - previous_interval)
                                   if previous_interval is not None else 0.0)
                state = (next_offset, interval)
                if state not in next_states or new_cost < next_states[state][0]:
                    next_states[state] = (new_cost, (previous_offset, previous_interval))
        if not next_states:                             # pathological; give up on smoothing
            return rounded_frames
        dp_states = next_states
        back_pointers.append({state: predecessor for state, (_, predecessor) in dp_states.items()})
    best_final_state = min(dp_states, key=lambda state: dp_states[state][0])
    chosen_states = [best_final_state]
    for table in reversed(back_pointers):
        chosen_states.append(table[chosen_states[-1]])
    chosen_states.reverse()
    chosen_offsets = np.array([state[0] for state in chosen_states], dtype=np.int64)
    return rounded_frames + chosen_offsets


def apply_smooth_targets(entries):
    """Replace each entry's rounded beat_frames with the jitter-free path (in place)."""
    for entry in entries:
        smoothed_frames = smooth_beat_frames(entry["beat_times"])
        num_to_keep = len(entry["beat_frames"])         # load_songs cut to start at first downbeat
        if len(smoothed_frames) >= num_to_keep:
            entry["beat_frames"] = smoothed_frames[len(smoothed_frames) - num_to_keep:]


def jitter_rate(entries):
    """Fraction of beat boundaries whose interval changes from the previous one (a jitter proxy)."""
    interval_changes = np.concatenate(
        [np.abs(np.diff(np.diff(entry["beat_frames"]))) for entry in entries])
    return float((interval_changes > 0).mean())
