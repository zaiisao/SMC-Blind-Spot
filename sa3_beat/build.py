"""Attach beat conditioning to Stable Audio 3 without forking it.

Three composition seams, no modified SA3 source:

1. ``patch_diffusion_config`` declares a ``modular_local_cond_configs`` entry.
   SA3's factory then builds one zero-initialised projection per DiT block and
   derives ``modular_local_cond_ids`` from the same entry, wiring the signal all
   the way from the conditioning dict into every block's residual stream.
2. ``attach_beat_conditioner`` inserts a BeatConditioner into the live
   ``MultiConditioner.conditioners`` ModuleDict.  It cannot go in the config:
   SA3's conditioner factory ends its dispatch with ``raise ValueError`` on
   unknown types.  Injecting a conditioning tensor post-construction is the same
   thing SA3 does to itself for inpainting.
3. ``freeze_to_beat_adapter`` freezes the base model and re-enables only the new
   per-block projections.

The config is patched in memory rather than copied to a tracked file, so there
is no second source of truth to drift from the published checkpoint.
"""

import copy
import json

import torch

from .conditioners import BeatConditioner

BEAT_COND_ID = "beat"


def patch_diffusion_config(config: dict, cond_dim: int, cond_id: str = BEAT_COND_ID) -> dict:
    """Return a deep copy of ``config`` with the beat conditioning declared."""
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
    """The per-block projections created for ``cond_id``: [(name, module), ...]."""
    return [
        (name, m)
        for name, m in module.named_modules()
        if name.endswith(f"modular_local_embeds.{cond_id}")
    ]


def beat_parameters(module, cond_id: str = BEAT_COND_ID):
    prefix = f"modular_local_embeds.{cond_id}."
    return [p for name, p in module.named_parameters() if prefix in name]


def freeze_to_beat_adapter(module, cond_id: str = BEAT_COND_ID):
    """Freeze everything, then re-enable only the beat projections.

    Mirrors SA3's own LoRA recipe (``.eval().requires_grad_(False)`` on the base
    model, gradients only on the adapter).  Returns the trainable parameters.
    """
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
    """Insert a BeatConditioner into a built ConditionedDiffusionModelWrapper."""
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
    """Adapter-only state dict, so checkpoints are ~10s of MB not 2.2 GB."""
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
    """Load the published DiT weights into a bare DiTWrapper.

    Checkpoint keys are ``model.model.*`` (outer wrapper -> DiTWrapper); a bare
    DiTWrapper wants ``model.*``, so one prefix is stripped.  New beat
    projections are absent from the checkpoint and keep their zero init.
    """
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


# Published checkpoint, pinned. SA3's own ModelConfig.resolve() calls
# hf_hub_download without a revision, i.e. it silently tracks `main`; we pin so a
# rerun loads the same weights.
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
    """Build the full SA3 model with beat conditioning attached.

    Reuses the library's own factory + ``copy_state_dict``, which copies only
    keys present in the checkpoint, so the new beat projections keep their zero
    init and the model is output-identical to stock until trained.

    Set ``freeze=False`` to keep the whole model trainable (not what we want for
    the adapter recipe, but useful for diagnostics).
    """
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
    # copy_state_dict copies only keys present in the checkpoint, so the new beat
    # projections keep their zero init and the model stays output-identical.
    copy_state_dict(model, load_file(ckpt_path))
    model.to(device).eval().requires_grad_(False)
    attach_beat_conditioner(model, include_downbeat=include_downbeat, cond_id=cond_id)
    model.conditioner.to(device)

    if freeze:
        # the loader above already did requires_grad_(False); this re-enables
        # exactly the beat projections, which is what SA3's configure_optimizers
        # then picks up ([p for p in ... if p.requires_grad]).
        freeze_to_beat_adapter(model.model, cond_id)

    return model, config


def apply_training_overrides(config: dict, overrides: dict) -> dict:
    """Merge ``overrides`` into ``config["training"]``; a null value deletes a key.

    The published checkpoint's ``training`` block is Stability's ARC distillation
    recipe -- an adversarial discriminator loaded from
    ``discriminator_base_ckpt: "/path/to/base/ckpt.pt"`` (a placeholder that does
    not exist) plus a MuonAdamW optimizer.  Adapter training is a plain diffusion
    fine-tune, so that block has to be overridden rather than inherited.
    A nested override dict replaces the published subtree wholesale when it
    carries ``"__replace__": true``; otherwise it is deep-merged.
    """
    import copy as _copy

    patched = _copy.deepcopy(config)
    training = patched.setdefault("training", {})

    def merge(dst, src):
        for k, v in src.items():
            if v is None:
                dst.pop(k, None)
            elif isinstance(v, dict) and v.get("__replace__"):
                # Replace the subtree wholesale. Needed where the published
                # value is a different shape, e.g. the MuonAdamW optimizer
                # config, whose muon_* keys would survive a deep merge and be
                # passed to AdamW.
                v = {kk: vv for kk, vv in v.items() if kk != "__replace__"}
                dst[k] = _copy.deepcopy(v)
            elif isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = _copy.deepcopy(v)

    merge(training, _copy.deepcopy(overrides))
    return patched
