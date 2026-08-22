"""End-to-end algorithm: train the Beat Transformer frontend from scratch, optionally jointly with
a gradient rung's factors. rung=None is the vanilla arm (BCE loss only, no bar-pointer). With a
rung, the loss is HYBRID -- the rung's supervised CRF NLL plus a BCE calibration anchor: pure CRF
has no calibration pressure and the logits blow past sigmoid saturation within one epoch, where the
CRF gradient dies; BCE's gradient is maximal at saturation, so it both prevents and escapes it."""
import time

import numpy as np
import torch

from training import data, frontend
from training.evaluation import evaluate, save_checkpoint, write_history
from rungs.r1_2016_dbn import DBN2016


def run(rung, train_entries, val_entries, params, output_directory):
    from optimizer import Lookahead                    # external/beat_transformer/code
    random_generator = np.random.default_rng(params.seed)
    model = frontend.build_model(params.device, params.resume_frontend_from)
    trainable = list(model.parameters()) + (rung.trainable_parameters() if rung is not None else [])
    optimizer = Lookahead(torch.optim.RAdam(trainable, lr=params.learning_rate), k=5, alpha=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=2, threshold=1e-3, min_lr=1e-7)
    binary_cross_entropy = torch.nn.BCEWithLogitsLoss()

    def evaluate_now():
        model.eval()
        validation_activations = frontend.activations_for(
            model, val_entries[:params.evaluation_songs], params.device)
        if rung is None:
            hand_set_dbn = DBN2016(fps=data.FPS, device=params.device, dtype=torch.float32,
                                   bounding="none", transition_lambda=100.0,
                                   **frontend.BT_SHIPPED_DECODE)
            result = evaluate(hand_set_dbn.predict, val_entries[:params.evaluation_songs],
                              validation_activations)
        else:
            decode = lambda activations: rung.decode(activations, deploy=not params.score_bare_model)
            result = evaluate(decode, val_entries[:params.evaluation_songs], validation_activations)
        model.train()
        return result

    best, history = {"beat_f_measure": -1.0}, []
    for epoch in range(params.epochs):
        random_generator.shuffle(train_entries)
        epoch_loss, epoch_bce, num_crops, num_bad, num_nan = 0.0, 0.0, 0, 0, 0
        started_at = time.time()
        for entry in train_entries:
            start_beat, end_beat = data.sample_crop(entry, random_generator)
            start_frame = entry["beat_frames"][start_beat]
            end_frame = entry["beat_frames"][end_beat]
            mel_crop = np.load(entry["mel_path"])["x"][:, start_frame:end_frame]
            if mel_crop.shape[1] != end_frame - start_frame:       # annotation past audio end
                num_bad += 1
                continue
            prediction, _ = model(torch.from_numpy(mel_crop).unsqueeze(0).to(params.device))
            logits = prediction[0, :, :2]

            crop_beat_frames = entry["beat_frames"][start_beat:end_beat + 1] - start_frame
            is_downbeat = entry["beat_in_bar"][start_beat:end_beat + 1] == 0
            target = data.widened_targets(crop_beat_frames, is_downbeat, end_frame - start_frame)
            bce_loss = binary_cross_entropy(
                logits, torch.from_numpy(target).to(params.device)) * 2   # sum over 2 channels

            if rung is None:
                loss = bce_loss
            else:
                built = rung.annotated_state_path(
                    entry["beat_frames"][start_beat:end_beat + 1] - start_frame,
                    entry["beat_in_bar"][start_beat:end_beat + 1], entry["beats_per_bar"])
                if built is None:
                    num_bad += 1
                    continue
                state_path, meter_index = built
                loss = rung.training_step(torch.sigmoid(logits), state_path, meter_index) + bce_loss
            epoch_bce += float(bce_loss)

            if not torch.isfinite(loss):
                num_nan += 1
                continue
            optimizer.zero_grad()
            loss.backward()
            # clip 0 means "no clipping", but the norm is still needed for the NaN guard below.
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, params.gradient_clip if params.gradient_clip > 0 else float('inf'))
            if not torch.isfinite(gradient_norm):
                optimizer.zero_grad()
                num_nan += 1
                continue
            optimizer.step()
            epoch_loss += float(loss)
            num_crops += 1

        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            print(f"epoch {epoch:02d}: WEIGHTS NONFINITE -- aborting (guards failed)", flush=True)
            return best
        # Anneal on the BCE component for BOTH arms so a still-falling CRF term cannot keep the
        # plateau scheduler from ever firing (an unfair asymmetry, measured).
        scheduler.step(epoch_bce / max(num_crops, 1))
        line = (f"epoch {epoch:02d} | loss {epoch_loss / max(num_crops, 1):.4f} | "
                f"crops {num_crops} (bad {num_bad}, nan {num_nan}) | "
                f"{time.time() - started_at:.0f}s | lr {optimizer.param_groups[0]['lr']:.2e}")
        if rung is not None and hasattr(rung, "transition_lambda"):
            line += f" | lambda {rung.transition_lambda:.2f}"
        if (epoch + 1) % params.eval_every == 0 or epoch == params.epochs - 1:
            beat_f_measure, downbeat_f_measure = evaluate_now()
            line += f" | val beatF {beat_f_measure:.4f} downbeatF {downbeat_f_measure:.4f}"
            if beat_f_measure > best["beat_f_measure"]:
                best = {"beat_f_measure": beat_f_measure,
                        "downbeat_f_measure": downbeat_f_measure, "epoch": epoch}
                save_checkpoint(output_directory, params.arm_name, best, rung=rung, model=model)
        print(line, flush=True)
        history.append(line)
        write_history(output_directory, params.arm_name, best, history)
    print(f"BEST [{params.arm_name}]: {best}", flush=True)
    return best
