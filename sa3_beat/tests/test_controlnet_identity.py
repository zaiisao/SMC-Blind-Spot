import json
import math

import pytest
import torch

from conftest import CKPT_PATH, CONFIG_PATH, requires_sa3

pytestmark = requires_sa3

SEQ = 64


@pytest.fixture(scope="module")
def backbone():
    from safetensors.torch import load_file
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict

    config = json.loads(open(CONFIG_PATH).read())
    model = create_model_from_config(config)
    copy_state_dict(model, load_file(str(CKPT_PATH)))
    model = model.eval().requires_grad_(False)
    return model.model.model.transformer


@pytest.fixture(scope="module")
def cn(backbone):
    from sa3_beat.controlnet import BeatControlNet

    return BeatControlNet(backbone, include_downbeat=True)


@pytest.fixture(scope="module")
def inputs(backbone):
    torch.manual_seed(0)
    x = torch.randn(1, SEQ, backbone.project_in.in_features)
    g = torch.randn(1, backbone.dim)
    ph = 2 * math.pi * torch.arange(SEQ) / 9.0
    beat = torch.stack([ph.sin(), ph.cos(), (ph / 4).sin(), (ph / 4).cos()], -1)[None]
    return x, g, beat


def _fwd(tf, x, g):
    with torch.no_grad():
        return tf(x, global_cond=g, use_checkpointing=False)


def test_control_branch_is_smaller_than_the_backbone(cn, backbone):
    n_cn = sum(p.numel() for p in cn.parameters())
    n_bb = sum(p.numel() for p in backbone.parameters())
    assert 0 < n_cn < n_bb


def test_attached_controlnet_is_identity_before_training(cn, backbone, inputs):
    x, g, beat = inputs
    ref = _fwd(backbone, x, g)
    cn.attach().set_beat(beat)
    try:
        assert torch.equal(ref, _fwd(backbone, x, g))
    finally:
        cn.set_beat(None)


def test_identity_also_holds_with_no_beat(cn, backbone, inputs):
    x, g, _ = inputs
    cn.attach().set_beat(None)
    ref_detached = _fwd(backbone, x, g)
    assert torch.equal(ref_detached, _fwd(backbone, x, g))


def test_gradient_reaches_the_branch_and_not_the_backbone(cn, backbone, inputs):
    x, g, beat = inputs
    cn.attach().freeze_backbone().set_beat(beat)
    backbone.zero_grad(set_to_none=True)
    cn.zero_grad(set_to_none=True)
    backbone(x, global_cond=g, use_checkpointing=False).square().mean().backward()

    out = [p for n, p in cn.named_parameters()
           if p.requires_grad and n.startswith("out_projs.")]
    assert len(out) == 2 * cn.n_blocks
    assert all(p.grad is not None and p.grad.abs().max() > 0 for p in out)

    behind = [p for n, p in cn.named_parameters()
              if p.requires_grad and not n.startswith("out_projs.")]
    assert all(p.grad is None or p.grad.abs().max() == 0 for p in behind)

    assert not any(p.grad is not None and p.grad.abs().max() > 0
                   for p in backbone.parameters())
