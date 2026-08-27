import json
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "third-party" / "stable-audio-tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SA3_DIR = Path(os.environ.get("SA3_DIR", "/disk1/jaehoon/sa3_small_music"))
CONFIG_PATH = SA3_DIR / "model_config.json"
CKPT_PATH = SA3_DIR / "model.safetensors"

FRAMES = 128
BATCH = 2
SEED = 0

requires_sa3 = pytest.mark.skipif(
    not (CONFIG_PATH.exists() and CKPT_PATH.exists()),
    reason=f"SA3 weights not found under {SA3_DIR}; set SA3_DIR to override",
)


@pytest.fixture(scope="session")
def base_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def beat_cond():
    from sa3_beat import BeatConditioner

    return BeatConditioner(include_downbeat=True)


@pytest.fixture(scope="session")
def beat_tensor(beat_cond):
    import numpy as np

    beats = np.arange(0, 12.0, 0.8)
    meta = [{"beats": beats, "downbeats": beats[::4], "num_frames": FRAMES}] * BATCH
    t, _ = beat_cond(meta, "cpu")
    return t.to(torch.float32)


def build_dit(config):
    from stable_audio_tools.models.diffusion import DiTWrapper

    d = config["model"]["diffusion"]
    return DiTWrapper(
        diffusion_objective=d.get("diffusion_objective", "v"),
        modular_local_cond_configs=d.get("modular_local_cond_configs", []),
        **d["config"],
    )


def fixed_inputs(config):
    g = torch.Generator(device="cpu").manual_seed(SEED)
    c = config["model"]["diffusion"]["config"]
    return dict(
        x=torch.randn(BATCH, c["io_channels"], FRAMES, generator=g),
        t=torch.rand(BATCH, generator=g),
        cross_attn_cond=torch.randn(BATCH, 16, c["cond_token_dim"], generator=g),
        global_cond=torch.randn(BATCH, c["global_cond_dim"], generator=g),
    )
