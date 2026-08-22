"""Config -> resolved TrainingParams, eligibility chassis, fold-split data, algorithm dispatch."""
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from training import data
from rungs.r1_2016_dbn import DBN2016

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OBSERVATION_LAMBDA = 6                          # Beat-Transformer-shipped beat-region width
DEFAULT_FRONTEND_CHECKPOINT = ROOT / "experiments" / "bt_e2e" / "vanilla_best_prelim.pt"


@dataclass
class TrainingParams:
    """Every training-loop knob, resolved (per-rung defaults <- config overrides). Rung constructor
    kwargs live in the config's `rung:` section instead."""
    arm_name: str
    device: str = "cuda:0"
    seed: int = 0
    output_directory: str = "runs"
    eval_every: int = 2
    evaluation_songs: int = 60
    score_bare_model: bool = False                     # bare (no threshold crop / peak snap) eval
    # frontend
    frontend_mode: str = "frozen"                      # "frozen" | "e2e"
    frontend_checkpoint: str = str(DEFAULT_FRONTEND_CHECKPOINT)
    resume_frontend_from: str = None                   # e2e: resume the model from here
    # gradient / e2e loops
    epochs: int = 30
    learning_rate: float = 1e-3
    gradient_clip: float = 0.0                          # 0 = off
    smooth_targets: bool = False
    # exact-EM loops
    em_iterations: int = 8
    num_crops: int = 300
    crop_frames: int = 700
    learn_observation_lambda: bool = False
    # r4_5 rich-feature projection
    projection: str = "standardize"                    # "standardize" | "pca" | "lda"


def resolve_params(arm_name, config, **rung_defaults):
    """Merge a rung's TRAIN_DEFAULTS with the config's `training:`/`frontend:` sections."""
    merged = dict(rung_defaults)
    merged.update(config.get("training", {}) or {})
    frontend = config.get("frontend", {}) or {}
    if "mode" in frontend:
        merged["frontend_mode"] = frontend["mode"]
    if "checkpoint" in frontend:
        merged["frontend_checkpoint"] = str(_resolve_path(frontend["checkpoint"]))
    if "init_from" in frontend:
        merged["resume_frontend_from"] = str(_resolve_path(frontend["init_from"]))
    known = {field.name for field in fields(TrainingParams)}
    unknown = set(merged) - known
    if unknown:
        raise ValueError(f"unknown training/frontend keys for {arm_name}: {sorted(unknown)}")
    return TrainingParams(arm_name=arm_name, **merged)


def device_of(config):
    """The device to build the rung on, before params are resolved."""
    return (config.get("training", {}) or {}).get("device", TrainingParams.device)


def rung_kwargs(config):
    """Rung constructor kwargs from the config's `rung:` section (everything but `name`)."""
    section = dict(config.get("rung", {}) or {})
    section.pop("name", None)
    return {"observation_lambda": DEFAULT_OBSERVATION_LAMBDA, **section}


def eligibility_chassis(rung, device):
    """A DBN2016 whose annotated_state_path decides song eligibility (vanilla has no rung)."""
    if rung is not None:
        return rung.chassis
    return DBN2016(fps=data.FPS, device=device, dtype=torch.float32,
                   observation_lambda=DEFAULT_OBSERVATION_LAMBDA,
                   num_tempi=None, threshold=0.0, correct=False)


def dispatch(rung_class, config):
    """Build the rung from the config and run the algorithm its TRAIN_MODE names."""
    from training import em, gradient, e2e
    arm_name = rung_class.ARM_NAME
    rung = rung_class.from_config(config)
    mode = (config.get("frontend", {}) or {}).get("mode", "frozen")

    if rung_class.TRAIN_MODE == "em":
        if mode == "e2e":
            raise ValueError(f"{arm_name} is EM-trained (closed-form M-step); the frontend cannot "
                             f"be trained end-to-end because no gradient reaches it. Use "
                             f"frontend.mode: frozen.")
        params = resolve_params(arm_name, config, **rung_class.TRAIN_DEFAULTS)
        return run(rung, params, em.run)

    if rung_class.TRAIN_MODE != "gradient":
        raise ValueError(f"{arm_name}: unknown TRAIN_MODE {rung_class.TRAIN_MODE!r}")

    if mode == "e2e":
        defaults = dict(rung_class.E2E_DEFAULTS
                        if rung_class.E2E_DEFAULTS is not None else rung_class.TRAIN_DEFAULTS)
        params = resolve_params(arm_name, config, frontend_mode="e2e", **defaults)
        return run(rung, params, e2e.run)
    params = resolve_params(arm_name, config, **rung_class.TRAIN_DEFAULTS)
    return run(rung, params, gradient.run)


def run(rung, params, algorithm):
    """Seed, prepare the output directory, load the fold-split data, and hand off to `algorithm`."""
    torch.manual_seed(params.seed)
    output_directory = Path(params.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    chassis = eligibility_chassis(rung, params.device)
    train_entries, val_entries, skipped = data.load_songs(chassis.annotated_state_path)
    print(f"[{params.arm_name}] frontend {params.frontend_mode} | train {len(train_entries)} | "
          f"val {len(val_entries)} | skipped {skipped} (no cache / meter / grid)", flush=True)
    return algorithm(rung, train_entries, val_entries, params, output_directory)


def _resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
