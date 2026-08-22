"""Exact Baum-Welch: frozen frontend, whole-dataset em_step, closed-form M-step, no optimizer.

Shared by r2_em (transition lambda from [T, 2] activations) and r4_5 (Gaussian emission from
[T, 256] features). The rung does the math; this module feeds it crops and scores the decode."""
import numpy as np
import torch

from training import frontend
from training.evaluation import evaluate, evaluate_six_metric, save_checkpoint, write_history


def run(rung, train_entries, val_entries, params, output_directory):
    random_generator = np.random.default_rng(params.seed)
    frozen_model = frontend.load_frozen_model(params.frontend_checkpoint, params.device)
    training_entries = train_entries[:params.num_crops]

    if rung.observes_features():
        crops, validation_observations, projection_payload, val_features, val_activations = \
            _prepare_feature_crops(rung, training_entries, val_entries, params, frozen_model,
                                   random_generator)
        decode = lambda observation: rung.decode(observation)
    else:
        crops, validation_observations = _prepare_activation_crops(
            training_entries, val_entries, params, frozen_model, random_generator)
        projection_payload = None
        decode = lambda observation: rung.decode(observation, deploy=not params.score_bare_model)

    print(f"{len(crops)} unsupervised crops ({params.crop_frames} frames) | "
          f"observes {'features' if rung.observes_features() else 'activations'}",
          flush=True)

    best, history = {"beat_f_measure": -1.0}, []
    for iteration in range(params.em_iterations):
        statistic = _em_step(rung, crops, params)
        line = _iteration_line(rung, iteration, statistic)
        if (iteration + 1) % params.eval_every == 0 or iteration == params.em_iterations - 1:
            beat_f_measure, downbeat_f_measure = evaluate(
                decode, val_entries, validation_observations, max_songs=params.evaluation_songs)
            line += f" | val beatF {beat_f_measure:.4f} downbeatF {downbeat_f_measure:.4f}"
            if beat_f_measure > best["beat_f_measure"]:
                best = {"beat_f_measure": beat_f_measure,
                        "downbeat_f_measure": downbeat_f_measure, "iteration": iteration}
                save_checkpoint(output_directory, params.arm_name, best, rung=rung,
                                projection_payload=projection_payload)
        print(line, flush=True)
        history.append(line)
        write_history(output_directory, params.arm_name, best, history)

    if rung.observes_features():
        _report_six_metric(rung, val_entries, val_features, val_activations, params,
                           projection_payload)
    print(f"BEST [{params.arm_name}]: {best}", flush=True)
    return best


def _prepare_activation_crops(training_entries, val_entries, params, frozen_model, rng):
    """[T, 2] activation crops for r2, plus the whole-song validation activations."""
    train_activations = frontend.activations_for(frozen_model, training_entries, params.device)
    validation_activations = frontend.activations_for(frozen_model, val_entries, params.device)
    crops = []
    for entry in training_entries:
        activations = train_activations[entry["stem"]]
        if activations.shape[0] > params.crop_frames + 1:
            start = int(rng.integers(0, activations.shape[0] - params.crop_frames))
            activations = activations[start:start + params.crop_frames]
        crops.append(torch.from_numpy(np.ascontiguousarray(activations)).float().to(params.device))
    return crops, validation_activations


def _prepare_feature_crops(rung, training_entries, val_entries, params, frozen_model, rng):
    """Extract features+activations, fit the projection, seed the emission, project validation."""
    train_features, train_activations = frontend.features_for(
        frozen_model, training_entries, params.device)
    val_features, val_activations = frontend.features_for(frozen_model, val_entries, params.device)

    feature_crops, activation_crops = [], []           # aligned raw feature + activation crops
    for entry in training_entries:
        features = train_features[entry["stem"]]
        activations = train_activations[entry["stem"]]
        if features.shape[0] > params.crop_frames + 1:
            start = int(rng.integers(0, features.shape[0] - params.crop_frames))
            features = features[start:start + params.crop_frames]
            activations = activations[start:start + params.crop_frames]
        feature_crops.append(features)
        activation_crops.append(
            torch.from_numpy(np.ascontiguousarray(activations)).float().to(params.device))

    if params.projection == "lda":
        # LDA needs per-frame labels; fit on the crop features they align with.
        bootstrap_labels = rung.bootstrap_labels(activation_crops)
        project, projection_payload, _ = frontend.fit_projection(
            feature_crops, "lda", rung.feature_dim, params.device, bootstrap_labels=bootstrap_labels)
    else:
        # standardize / pca fit the affine map on the FULL features of the first 50 train songs.
        bootstrap_labels = None
        full_features = [train_features[entry["stem"]] for entry in training_entries[:50]]
        project, projection_payload, _ = frontend.fit_projection(
            full_features, params.projection, rung.feature_dim, params.device)

    projected_crops = [project(features) for features in feature_crops]
    if bootstrap_labels is not None:
        rung.initialize_from_labels(projected_crops, bootstrap_labels)
    else:
        rung.self_supervised_init(projected_crops, activation_crops)

    validation_observations = {entry["stem"]: project(val_features[entry["stem"]])
                               for entry in val_entries}
    return (projected_crops, validation_observations, projection_payload,
            val_features, val_activations)


def _em_step(rung, crops, params):
    if params.arm_name == "r2_em":
        return rung.em_step(crops, learn_observation_lambda=params.learn_observation_lambda)
    return rung.em_step(crops)


def _iteration_line(rung, iteration, statistic):
    if hasattr(rung, "transition_lambda") and hasattr(rung, "observation_lambda"):
        return (f"EM iter {iteration:02d} | lambda {rung.transition_lambda:.2f} "
                f"obs_lambda {rung.observation_lambda}")
    return f"EM iter {iteration:02d} | marginal_log_likelihood {statistic:.1f}"


def _report_six_metric(rung, val_entries, val_features, val_activations, params, projection_payload):
    """r4_5 headline table (deployed + bare); rebuilds the projection from its payload."""
    device = params.device
    if "feature_std" in projection_payload:            # standardize
        feature_mean = projection_payload["feature_mean"]
        feature_std = projection_payload["feature_std"]
        project = lambda features: ((features - feature_mean) / feature_std).to(device)
    else:                                              # pca / lda
        feature_mean = projection_payload["feature_mean"]
        projection_matrix = projection_payload["projection"]
        project = lambda features: ((features - feature_mean) @ projection_matrix).to(device)
    for deployed, label in ((True, "DEPLOYED"), (False, "BARE    ")):
        metrics = evaluate_six_metric(rung, val_entries, val_features, val_activations, project,
                                      rung.chassis.fps, deployed)
        print(f"{params.arm_name} {label}: beatF {metrics['beatF']:.4f}  "
              f"CMLt {metrics['CMLt']:.4f}  AMLt {metrics['AMLt']:.4f}  "
              f"downbeatF {metrics['downbeatF']:.4f}  downbeatCMLt {metrics['downbeatCMLt']:.4f}  "
              f"downbeatAMLt {metrics['downbeatAMLt']:.4f}", flush=True)
