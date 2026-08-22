"""Frozen-frontend gradient algorithm: precompute the [T, 2] activations once, then per-crop
backprop through the rung's training_step. Shared by crf (a scalar lambda) and r3 (a conv net),
which differ only in trainable_parameters() / training_step()."""
import numpy as np
import torch

from training import data, frontend
from training.evaluation import evaluate, save_checkpoint, write_history


def run(rung, train_entries, val_entries, params, output_directory):
    random_generator = np.random.default_rng(params.seed)
    if params.smooth_targets:
        before = data.jitter_rate(train_entries)
        data.apply_smooth_targets(train_entries + val_entries)
        print(f"smooth targets: boundary-jump rate {before:.3f} -> "
              f"{data.jitter_rate(train_entries):.3f}", flush=True)

    frozen_model = frontend.load_frozen_model(params.frontend_checkpoint, params.device)
    print("precomputing frozen activations...", flush=True)
    train_activations = frontend.activations_for(frozen_model, train_entries, params.device)
    validation_activations = frontend.activations_for(frozen_model, val_entries, params.device)

    def evaluate_now():
        decode = lambda activations: rung.decode(activations, deploy=not params.score_bare_model)
        return evaluate(decode, val_entries, validation_activations,
                        max_songs=params.evaluation_songs)

    beat_f_measure, downbeat_f_measure = evaluate_now()             # pre-training baseline row
    print(f"baseline (rung at init): beatF {beat_f_measure:.4f} downbeatF {downbeat_f_measure:.4f}",
          flush=True)

    optimizer = torch.optim.Adam(rung.trainable_parameters(), lr=params.learning_rate)
    best = {"beat_f_measure": beat_f_measure, "downbeat_f_measure": downbeat_f_measure,
            "epoch": -1}
    history = []
    save_checkpoint(output_directory, params.arm_name, best, rung=rung)
    for epoch in range(params.epochs):
        random_generator.shuffle(train_entries)
        total_loss, num_crops = 0.0, 0
        for entry in train_entries:
            start_beat, end_beat = data.sample_crop(entry, random_generator)
            start_frame = entry["beat_frames"][start_beat]
            end_frame = entry["beat_frames"][end_beat]
            activations = train_activations[entry["stem"]]
            if start_frame < 0 or end_frame > activations.shape[0] or end_frame <= start_frame:
                continue
            built = rung.annotated_state_path(entry["beat_frames"][start_beat:end_beat + 1] - start_frame,
                                              entry["beat_in_bar"][start_beat:end_beat + 1],
                                              entry["beats_per_bar"])
            if built is None:
                continue
            state_path, meter_index = built
            crop_activations = torch.from_numpy(
                activations[start_frame:end_frame]).float().to(params.device)
            loss = rung.training_step(crop_activations, state_path, meter_index)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            if params.gradient_clip:
                torch.nn.utils.clip_grad_norm_(rung.trainable_parameters(), params.gradient_clip)
            optimizer.step()
            total_loss += float(loss)
            num_crops += 1

        line = f"epoch {epoch:02d} | loss {total_loss / max(num_crops, 1):.4f}"
        if hasattr(rung, "transition_lambda"):
            line += f" | lambda {rung.transition_lambda:.2f}"
        if (epoch + 1) % params.eval_every == 0 or epoch == params.epochs - 1:
            beat_f_measure, downbeat_f_measure = evaluate_now()
            line += f" | val beatF {beat_f_measure:.4f} downbeatF {downbeat_f_measure:.4f}"
            if beat_f_measure > best["beat_f_measure"]:
                best = {"beat_f_measure": beat_f_measure,
                        "downbeat_f_measure": downbeat_f_measure, "epoch": epoch}
                save_checkpoint(output_directory, params.arm_name, best, rung=rung)
        print(line, flush=True)
        history.append(line)
        write_history(output_directory, params.arm_name, best, history)
    print(f"BEST [{params.arm_name}]: {best}", flush=True)
    return best
