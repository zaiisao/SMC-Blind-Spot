"""Scoring + checkpoint I/O, shared by every algorithm. One decode->score path; one save path."""
import json

import numpy as np
import torch

from training.frontend import BT_SHIPPED_DECODE
from rungs.deployment import threshold_crop
from rungs.bar_pointer.readout import state_path_to_events


def beat_f_measure(reference_times, estimated_times, min_references=1):
    """mir_eval beat F on trimmed event lists; 0.0 when nothing was estimated.

    min_references: how many reference events the score needs to be meaningful. Passed explicitly
    by each caller so the two evaluation paths cannot silently disagree on sparse-downbeat songs."""
    import mir_eval.beat as mir_eval_beat
    reference_times = mir_eval_beat.trim_beats(reference_times)
    estimated_times = mir_eval_beat.trim_beats(estimated_times)
    if not len(estimated_times) or len(reference_times) < min_references:
        return 0.0
    return mir_eval_beat.f_measure(reference_times, estimated_times)


def evaluate(decode, entries, observations, max_songs=None):
    """Mean (beat F, downbeat F) over entries. `decode(observation) -> events` dict; `observations`
    maps stem -> the observation the rung consumes (activations or projected features)."""
    beat_f_measures, downbeat_f_measures = [], []
    for entry in entries[:max_songs]:
        events = decode(observations[entry["stem"]])
        beat_f_measures.append(beat_f_measure(entry["beat_times"], events["beats"]))
        downbeat_f_measures.append(beat_f_measure(entry["downbeat_times"], events["downbeats"]))
    return float(np.mean(beat_f_measures)), float(np.mean(downbeat_f_measures))


def evaluate_six_metric(rung, entries, features_by_stem, activations_by_stem, project, fps,
                        deployed):
    """r4_5 headline: beat/downbeat F + continuity (CMLt/AMLt), decoded from the rich features.
    deployed=True applies the shipped threshold crop + peak snap on the [T, 2] activations."""
    import mir_eval.beat as mir_eval_beat
    metric_names = ("beatF", "CMLt", "AMLt", "downbeatF", "downbeatCMLt", "downbeatAMLt")
    accumulated = {name: [] for name in metric_names}
    for entry in entries:
        features = features_by_stem[entry["stem"]]
        activations = activations_by_stem[entry["stem"]]
        if deployed:
            cropped_activations, first_frame = threshold_crop(activations,
                                                              BT_SHIPPED_DECODE["threshold"])
            features_used = features[first_frame:first_frame + cropped_activations.shape[0]]
        else:
            cropped_activations, first_frame, features_used = activations, 0, features
        if not cropped_activations.size:
            continue
        decoded = rung.decode(project(features_used))
        events = state_path_to_events(
            decoded["path"], decoded["space"], fps,
            snap_to_activations=cropped_activations if deployed else None,
            first_frame=first_frame)

        accumulated["beatF"].append(beat_f_measure(entry["beat_times"], events["beats"]))
        reference_beats = mir_eval_beat.trim_beats(entry["beat_times"])
        estimated_beats = mir_eval_beat.trim_beats(events["beats"])
        if len(estimated_beats) and len(reference_beats) > 1:
            _, continuity_required, _, continuity_allowed = mir_eval_beat.continuity(
                reference_beats, estimated_beats)
            accumulated["CMLt"].append(continuity_required)
            accumulated["AMLt"].append(continuity_allowed)
        else:
            accumulated["CMLt"].append(0.0); accumulated["AMLt"].append(0.0)

        # >=2 references, matching the continuity metrics beside it; evaluate() uses >=1.
        accumulated["downbeatF"].append(
            beat_f_measure(entry["downbeat_times"], events["downbeats"], min_references=2))
        reference_downbeats = mir_eval_beat.trim_beats(entry["downbeat_times"])
        estimated_downbeats = mir_eval_beat.trim_beats(events["downbeats"])
        if len(estimated_downbeats) and len(reference_downbeats) > 1:
            _, continuity_required, _, continuity_allowed = mir_eval_beat.continuity(
                reference_downbeats, estimated_downbeats)
            accumulated["downbeatCMLt"].append(continuity_required)
            accumulated["downbeatAMLt"].append(continuity_allowed)
        else:
            accumulated["downbeatCMLt"].append(0.0); accumulated["downbeatAMLt"].append(0.0)
    return {name: float(np.mean(values)) if values else 0.0
            for name, values in accumulated.items()}


def save_checkpoint(out_dir, arm_name, best, rung=None, model=None, projection_payload=None):
    """One checkpoint format for every rung: frontend model (e2e), rung factors, r4_5 projection."""
    checkpoint = {"arm": arm_name, "best": best}
    if model is not None:
        checkpoint["model"] = model.state_dict()
    if rung is not None:
        checkpoint["rung"] = (rung.state_dict() if isinstance(rung, torch.nn.Module)
                              else rung.training_state())
    if projection_payload is not None:
        checkpoint["projection"] = projection_payload
    torch.save(checkpoint, out_dir / f"{arm_name}_best.pt")


def write_history(out_dir, arm_name, best, history):
    (out_dir / f"{arm_name}_history.json").write_text(
        json.dumps({"best": best, "history": history}, indent=1))
