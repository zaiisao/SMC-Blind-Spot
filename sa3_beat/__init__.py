"""Beat-conditioned Stable Audio 3, built by composition over an unforked SA3."""

from .build import (
    BEAT_COND_ID,
    attach_beat_conditioner,
    beat_adapter_state_dict,
    beat_parameters,
    beat_projection_modules,
    freeze_to_beat_adapter,
    load_beat_model,
    load_patched_config,
    patch_diffusion_config,
)
from .conditioners import LATENT_RATE, BeatConditioner

__all__ = [
    "BEAT_COND_ID", "LATENT_RATE", "BeatConditioner",
    "attach_beat_conditioner", "beat_adapter_state_dict", "beat_parameters",
    "beat_projection_modules", "freeze_to_beat_adapter", "load_beat_model",
    "load_patched_config",
    "patch_diffusion_config",
]
