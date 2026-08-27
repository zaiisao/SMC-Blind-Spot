import pytest
import torch

from conftest import (BATCH, CKPT_PATH, FRAMES, build_dit, fixed_inputs,
                      requires_sa3)

pytestmark = requires_sa3

BEAT = "beat"


@pytest.fixture(scope="module")
def beat_config(base_config, beat_cond):
    from sa3_beat import patch_diffusion_config

    return patch_diffusion_config(base_config, beat_cond.cond_dim, BEAT)


@pytest.fixture(scope="module")
def stock(base_config):
    from sa3_beat.build import load_dit_weights

    m = build_dit(base_config).eval()
    load_dit_weights(m, str(CKPT_PATH))
    return m


@pytest.fixture(scope="module")
def patched(beat_config):
    from sa3_beat.build import load_dit_weights

    m = build_dit(beat_config).eval()
    load_dit_weights(m, str(CKPT_PATH))
    return m


def test_beat_signal_is_not_vacuous(beat_tensor, beat_cond):
    assert beat_tensor.shape == (BATCH, beat_cond.cond_dim, FRAMES)
    assert beat_tensor.abs().max() > 0.5


def test_phase_stays_on_the_unit_circle(beat_tensor):
    radius = (beat_tensor[0, 0] ** 2 + beat_tensor[0, 1] ** 2).sqrt()
    assert torch.allclose(radius, torch.ones_like(radius), atol=1e-4)


def test_every_block_gets_a_zero_init_projection(patched, beat_config):
    from sa3_beat import beat_projection_modules

    projections = beat_projection_modules(patched, BEAT)
    assert len(projections) == beat_config["model"]["diffusion"]["config"]["depth"]
    for _, proj in projections:
        assert proj[-1].weight.abs().sum() == 0
        assert proj[-1].bias.abs().sum() == 0


def test_step0_output_is_bit_identical(stock, patched, base_config, beat_tensor):
    inputs = fixed_inputs(base_config)
    with torch.no_grad():
        a = stock(**inputs)
        b = patched(**inputs, modular_local_cond={BEAT: beat_tensor})
    assert torch.equal(a, b), f"max|delta|={(a - b).abs().max().item():.3e}"


def test_gradient_reaches_every_projection(patched, base_config, beat_tensor):
    from sa3_beat import freeze_to_beat_adapter

    freeze_to_beat_adapter(patched, BEAT)
    patched.zero_grad(set_to_none=True)
    patched(**fixed_inputs(base_config),
            modular_local_cond={BEAT: beat_tensor}).sum().backward()
    out = [p for n, p in patched.named_parameters()
           if f"modular_local_embeds.{BEAT}.2." in n]
    assert out
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in out)


def test_freeze_selects_exactly_the_adapter(patched):
    from sa3_beat import freeze_to_beat_adapter

    freeze_to_beat_adapter(patched, BEAT)
    trainable = {n for n, p in patched.named_parameters() if p.requires_grad}
    expected = {n for n, _ in patched.named_parameters()
                if f"modular_local_embeds.{BEAT}." in n}
    assert trainable == expected
