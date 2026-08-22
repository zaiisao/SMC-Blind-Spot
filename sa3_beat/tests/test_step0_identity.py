"""Prove the beat-conditioning seam on the real SA3 checkpoint.

Asserts, on the published weights:

  1. Adding the beat conditioning leaves the DiT's output BIT-IDENTICAL at
     step 0, for a non-zero beat signal (zero-init identity).
  2. Gradients still reach the new projections, so training can move them.
  3. Freezing selects exactly the new parameters and nothing else.
  4. The beat features are what we think they are (phase, not pulses).

Runs on the DiT alone -- no text encoder, so no `transformers` needed:
    /disk1/jaehoon/sa3env/bin/python sa3_beat/tests/test_step0_identity.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# sa3_beat/tests/ -> repo root. stable_audio_tools itself is installed editable from
# third-party/stable-audio-tools, so only the repo root is needed on the path.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stable_audio_tools.models.diffusion import DiTWrapper  # noqa: E402

from sa3_beat import (  # noqa: E402
    BEAT_COND_ID,
    BeatConditioner,
    beat_parameters,
    beat_projection_modules,
    freeze_to_beat_adapter,
    patch_diffusion_config,
)
from sa3_beat.build import load_dit_weights  # noqa: E402

SA3_DIR = Path("/disk1/jaehoon/sa3_small_music")
CONFIG_PATH = SA3_DIR / "model_config.json"
CKPT_PATH = SA3_DIR / "model.safetensors"

FRAMES = 128
BATCH = 2
SEED = 0


def build_dit(config: dict) -> DiTWrapper:
    """Build a bare DiTWrapper exactly as SA3's factory does."""
    diffusion_config = config["model"]["diffusion"]
    return DiTWrapper(
        diffusion_objective=diffusion_config.get("diffusion_objective", "v"),
        modular_local_cond_configs=diffusion_config.get("modular_local_cond_configs", []),
        **diffusion_config["config"],
    )


def fixed_inputs(config: dict, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(SEED)
    io_channels = config["model"]["diffusion"]["config"]["io_channels"]
    cond_dim = config["model"]["diffusion"]["config"]["cond_token_dim"]
    global_dim = config["model"]["diffusion"]["config"]["global_cond_dim"]
    return dict(
        x=torch.randn(BATCH, io_channels, FRAMES, generator=g).to(device, dtype),
        t=torch.rand(BATCH, generator=g).to(device, torch.float32),
        cross_attn_cond=torch.randn(BATCH, 16, cond_dim, generator=g).to(device, dtype),
        global_cond=torch.randn(BATCH, global_dim, generator=g).to(device, dtype),
    )


def main() -> int:
    device, dtype = "cpu", torch.float32
    with open(CONFIG_PATH) as f:
        base_config = json.load(f)
    beat_cond = BeatConditioner(include_downbeat=True)
    beat_config = patch_diffusion_config(base_config, beat_cond.cond_dim, BEAT_COND_ID)

    print(f"[cfg]  modular_local_cond_configs = "
          f"{beat_config['model']['diffusion']['modular_local_cond_configs']}")

    # --- 4. the conditioning signal itself -------------------------------
    beats = np.arange(0, 12.0, 0.8)                 # 75 BPM
    downbeats = beats[::4]
    meta = [{"beats": beats, "downbeats": downbeats, "num_frames": FRAMES}] * BATCH
    beat_tensor, _ = beat_cond(meta, device)
    beat_tensor = beat_tensor.to(dtype)

    assert beat_tensor.shape == (BATCH, beat_cond.cond_dim, FRAMES), beat_tensor.shape
    assert beat_tensor.abs().max() > 0.5, "beat signal is ~zero; identity test would be vacuous"
    # sin/cos must stay on the unit circle -> continuous phase, not pulses
    radius = (beat_tensor[0, 0] ** 2 + beat_tensor[0, 1] ** 2).sqrt()
    assert torch.allclose(radius, torch.ones_like(radius), atol=1e-4), radius.min()
    frames_per_beat = (44100 / 4096) * 0.8
    print(f"[cond] shape={tuple(beat_tensor.shape)} unit-circle OK "
          f"({frames_per_beat:.2f} frames/beat, non-integer -> sub-frame phase preserved)")

    # --- baseline: stock DiT ---------------------------------------------
    stock = build_dit(base_config).to(device, dtype).eval()
    load_dit_weights(stock, str(CKPT_PATH))

    # --- patched: same weights + beat projections ------------------------
    patched = build_dit(beat_config).to(device, dtype).eval()
    load_dit_weights(patched, str(CKPT_PATH))

    projections = beat_projection_modules(patched, BEAT_COND_ID)
    depth = beat_config["model"]["diffusion"]["config"]["depth"]
    assert len(projections) == depth, f"{len(projections)} projections for depth {depth}"
    for _, proj in projections:
        assert proj[-1].weight.abs().sum() == 0, "output layer is not zero-init"
        assert proj[-1].bias.abs().sum() == 0, "output bias is not zero-init"
    n_params = sum(p.numel() for p in beat_parameters(patched, BEAT_COND_ID))
    print(f"[init] {len(projections)} zero-init projections, {n_params/1e6:.2f}M new params")

    # --- 1. step-0 identity ----------------------------------------------
    inputs = fixed_inputs(base_config, device, dtype)
    with torch.no_grad():
        out_stock = stock(**inputs)
        out_patched = patched(**inputs, modular_local_cond={BEAT_COND_ID: beat_tensor})

    identical = torch.equal(out_stock, out_patched)
    max_delta = (out_stock - out_patched).abs().max().item()
    print(f"[step0] bit-identical={identical}  max|delta|={max_delta:.3e}")
    if not identical:
        print("FAIL: beat conditioning perturbs the model at step 0")
        return 1

    # --- 2. gradients reach the new projections --------------------------
    trainable = freeze_to_beat_adapter(patched, BEAT_COND_ID)
    patched(**inputs, modular_local_cond={BEAT_COND_ID: beat_tensor}).sum().backward()
    # At step 0 only each projection's OUTPUT layer sees gradient: the input
    # layer's gradient is scaled by the (still zero) output weight. One optimiser
    # step makes the output non-zero and gradient reaches the input layer too.
    # So the expected count here is 2 tensors (weight+bias) per block, not 4.
    out_layer = [p for n, p in patched.named_parameters()
                 if f"modular_local_embeds.{BEAT_COND_ID}.2." in n]
    live = [p for p in out_layer if p.grad is not None and p.grad.abs().sum() > 0]
    print(f"[grad] {len(live)}/{len(out_layer)} output-layer tensors have non-zero gradient "
          f"(input layers correctly still zero at step 0)")
    if len(live) != len(out_layer):
        print("FAIL: gradient does not reach every beat projection")
        return 1

    # --- 3. freezing selects exactly the adapter -------------------------
    trainable_names = {n for n, p in patched.named_parameters() if p.requires_grad}
    expected = {n for n, _ in patched.named_parameters()
                if f"modular_local_embeds.{BEAT_COND_ID}." in n}
    if trainable_names != expected:
        print(f"FAIL: freeze selected {len(trainable_names)} params, expected {len(expected)}")
        return 1
    total = sum(p.numel() for p in patched.parameters())
    print(f"[freeze] trainable {n_params/1e6:.2f}M / {total/1e6:.1f}M "
          f"({100*n_params/total:.2f}% of the DiT)")

    print("\nPASS -- zero-init identity holds, gradients flow, freeze is exact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
