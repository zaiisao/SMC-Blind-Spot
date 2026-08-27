"""The ControlNet is bit-identical to the checkpoint until it is trained.

Same contract as the DiT adapter and the encoder seam: zero-init output
projections mean installing the branch changes nothing, while gradient still
reaches every trainable tensor.  A non-zero beat signal is fed, since a
zero-init path maps anything to zero and passing zeros would test nothing.
"""

import json
import sys

import torch

SA3_DIR = "/disk1/jaehoon/sa3_small_music"


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC")
    from safetensors.torch import load_file
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict

    from sa3_beat.controlnet import BeatControlNet

    config = json.loads(open(f"{SA3_DIR}/model_config.json").read())
    model = create_model_from_config(config)
    copy_state_dict(model, load_file(f"{SA3_DIR}/model.safetensors"))
    model = model.to(device).eval().requires_grad_(False)

    dit = model.model.model            # DiffusionTransformer
    tf = dit.transformer
    print(f"backbone blocks: {len(tf.layers)}  dim: {tf.dim}")

    cn = BeatControlNet(tf, include_downbeat=True).to(device)
    n_cn = sum(p.numel() for p in cn.parameters())
    n_bb = sum(p.numel() for p in tf.parameters())
    print(f"control blocks : {cn.n_blocks}   params {n_cn/1e6:.1f}M "
          f"vs backbone {n_bb/1e6:.1f}M ({100*n_cn/n_bb:.0f}%)")

    b, seq = 1, 64
    in_dim = tf.project_in.in_features
    torch.manual_seed(0)
    x = torch.randn(b, seq, in_dim, device=device)
    g = torch.randn(b, tf.dim, device=device)
    print(f"input dim      : {in_dim} (project_in) -> {tf.dim}")
    ph = 2 * torch.pi * torch.arange(seq, device=device) / 9.0
    beat = torch.stack([ph.sin(), ph.cos(), (ph / 4).sin(), (ph / 4).cos()], -1)[None]

    with torch.no_grad():
        ref = tf(x, global_cond=g, use_checkpointing=False)

    cn.attach().set_beat(beat)
    with torch.no_grad():
        got = tf(x, global_cond=g, use_checkpointing=False)
    cn.set_beat(None)
    with torch.no_grad():
        off = tf(x, global_cond=g, use_checkpointing=False)

    d_on = (ref - got).abs().max().item()
    d_off = (ref - off).abs().max().item()
    print(f"max|ref - controlnet(beat)| : {d_on:.3e}")
    print(f"max|ref - controlnet(None)| : {d_off:.3e}")

    cn.freeze_backbone().set_beat(beat)
    out = tf(x, global_cond=g, use_checkpointing=False)
    out.square().mean().backward()
    named = [(n, p) for n, p in cn.named_parameters() if p.requires_grad]
    nog = [n for n, p in named if p.grad is None or p.grad.abs().max() == 0]
    gmax = max(p.grad.abs().max().item() for _, p in named if p.grad is not None)
    print(f"trainable tensors           : {len(named)}   no-gradient: {len(nog)}")
    print(f"max gradient                : {gmax:.3e}")
    bb_grad = any(p.grad is not None and p.grad.abs().max() > 0
                  for p in tf.parameters())
    print(f"backbone received gradient  : {bb_grad}  (must be False)")

    ok = d_on == 0.0 and d_off == 0.0 and gmax > 0 and not bb_grad
    print("\nPASS" if ok else "\nFAIL")
    if nog:
        print("  no-gradient tensors:", nog[:6])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
