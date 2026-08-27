
import copy
import json

import torch

from .conditioners import BeatConditioner

BEAT_COND_ID = "beat"


def patch_diffusion_config(config: dict, cond_dim: int, cond_id: str = BEAT_COND_ID) -> dict:
    patched = copy.deepcopy(config)
    diffusion = patched["model"]["diffusion"]
    existing = diffusion.get("modular_local_cond_configs", [])
    if any(c["id"] == cond_id for c in existing):
        raise ValueError(f"modular_local_cond_configs already declares {cond_id!r}")
    diffusion["modular_local_cond_configs"] = existing + [{"id": cond_id, "dim": cond_dim}]
    return patched


def load_patched_config(config_path: str, cond_dim: int, cond_id: str = BEAT_COND_ID) -> dict:
    with open(config_path) as f:
        return patch_diffusion_config(json.load(f), cond_dim, cond_id)


def beat_projection_modules(module, cond_id: str = BEAT_COND_ID):
    return [
        (name, m)
        for name, m in module.named_modules()
        if name.endswith(f"modular_local_embeds.{cond_id}")
    ]


def beat_parameters(module, cond_id: str = BEAT_COND_ID):
    prefix = f"modular_local_embeds.{cond_id}."
    return [p for name, p in module.named_parameters() if prefix in name]


def freeze_to_beat_adapter(module, cond_id: str = BEAT_COND_ID):
    module.requires_grad_(False)
    trainable = beat_parameters(module, cond_id)
    if not trainable:
        raise ValueError(
            f"no modular_local_embeds.{cond_id} parameters found -- was the config patched?"
        )
    for p in trainable:
        p.requires_grad_(True)
    return trainable


def attach_beat_conditioner(model, include_downbeat: bool = True, cond_id: str = BEAT_COND_ID):
    conditioner = BeatConditioner(include_downbeat=include_downbeat)
    if cond_id in model.conditioner.conditioners:
        raise ValueError(f"conditioner {cond_id!r} is already attached")
    model.conditioner.conditioners[cond_id] = conditioner
    if cond_id not in model.modular_local_cond_ids:
        raise ValueError(
            f"{cond_id!r} is not in modular_local_cond_ids -- build the model from a "
            "config produced by patch_diffusion_config()"
        )
    return model


def beat_adapter_state_dict(module, cond_id: str = BEAT_COND_ID) -> dict:
    prefix = f"modular_local_embeds.{cond_id}."
    return {
        name: p.detach().cpu()
        for name, p in module.state_dict().items()
        if prefix in name
    }


def load_beat_adapter_state_dict(module, state_dict: dict, strict: bool = True):
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if strict:
        expected = set(beat_adapter_state_dict(module).keys())
        if set(state_dict.keys()) != expected:
            raise ValueError(
                f"adapter keys do not match: {sorted(set(state_dict) ^ expected)[:4]}"
            )
    return missing, unexpected


def load_dit_weights(dit_wrapper, ckpt_path: str):
    from safetensors import safe_open

    with safe_open(ckpt_path, framework="pt") as f:
        state = {
            k[len("model."):]: f.get_tensor(k)
            for k in f.keys()
            if k.startswith("model.")
        }
    missing, unexpected = dit_wrapper.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if "modular_local_embeds" not in k]
    missing = [k for k in missing if "modular_local_embeds" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"unexpected DiT weight mismatch: missing={missing[:4]} unexpected={unexpected[:4]}"
        )
    return dit_wrapper


SMALL_MUSIC_REPO = "stabilityai/stable-audio-tools-small-music"
SMALL_MUSIC_REVISION = "0fef1392cd842149a2b6d445e181c97608faac06"
SMALL_MUSIC_DIR = "/disk1/jaehoon/sa3_small_music"


def load_beat_model(
    config_path: str = None,
    ckpt_path: str = None,
    include_downbeat: bool = True,
    freeze: bool = True,
    device: str = "cpu",
    cond_id: str = BEAT_COND_ID,
):
    import json

    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict
    from safetensors.torch import load_file

    config_path = config_path or f"{SMALL_MUSIC_DIR}/model_config.json"
    ckpt_path = ckpt_path or f"{SMALL_MUSIC_DIR}/model.safetensors"

    from .conditioners import BeatConditioner

    cond_dim = BeatConditioner(include_downbeat=include_downbeat).cond_dim
    with open(config_path) as f:
        config = patch_diffusion_config(json.load(f), cond_dim, cond_id)

    model = create_model_from_config(config)
    copy_state_dict(model, load_file(ckpt_path))
    model.to(device).eval().requires_grad_(False)
    attach_beat_conditioner(model, include_downbeat=include_downbeat, cond_id=cond_id)
    model.conditioner.to(device)

    if freeze:
        freeze_to_beat_adapter(model.model, cond_id)

    return model, config


def apply_training_overrides(config: dict, overrides: dict) -> dict:
    import copy as _copy

    patched = _copy.deepcopy(config)
    training = patched.setdefault("training", {})

    def merge(dst, src):
        for k, v in src.items():
            if v is None:
                dst.pop(k, None)
            elif isinstance(v, dict) and v.get("__replace__"):
                v = {kk: vv for kk, vv in v.items() if kk != "__replace__"}
                dst[k] = _copy.deepcopy(v)
            elif isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = _copy.deepcopy(v)

    merge(training, _copy.deepcopy(overrides))
    return patched
