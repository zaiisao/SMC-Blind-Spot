"""Step 1: the beat-conditioned encoder is bit-identical to the checkpoint.

The whole zero-init argument rests on one claim -- that installing the beat seam
changes nothing until it is trained.  This checks it against the published
weights rather than against a freshly constructed model, and it feeds a
*non-zero* beat signal: a zero-init projection must map any input to exactly
zero, so passing zeros would test nothing.
"""

import json
import sys

import torch

SA3_DIR = "/disk1/jaehoon/sa3_small_music"


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"
    from safetensors.torch import load_file
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict

    sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC")
    from sa3_beat.encoder import BeatConditionedEncoder

    with open(f"{SA3_DIR}/model_config.json") as f:
        config = json.load(f)
    model = create_model_from_config(config)
    copy_state_dict(model, load_file(f"{SA3_DIR}/model.safetensors"))
    ae = model.pretransform.model.to(device).eval().requires_grad_(False)

    enc = ae.encoder
    wrapped = BeatConditionedEncoder(enc, include_downbeat=True).to(device).eval()

    ds = model.pretransform.downsampling_ratio
    patch = config["model"]["pretransform"]["config"]["pretransform"]["config"]["patch_size"]
    seconds, sr = 4.0, config["sample_rate"]
    n_samples = (int(seconds * sr) // ds) * ds

    torch.manual_seed(0)
    audio = torch.randn(1, 2, n_samples, device=device)
    # The encoder consumes patched input, not raw waveform.
    x = model.pretransform.model.encoder  # noqa: F841  (kept for clarity)
    patched = audio.reshape(1, 2, n_samples // patch, patch)
    patched = patched.permute(0, 1, 3, 2).reshape(1, 2 * patch, n_samples // patch)

    # Derived from the model, not from arithmetic: the block zero-pads its input
    # up to a multiple of chunk_size, so 43 latent frames of audio become 44.
    with torch.no_grad():
        n_latent = enc(patched).shape[-1]

    # A deliberately non-trivial beat signal: sin/cos of continuous phase.
    t = torch.arange(n_latent, device=device, dtype=torch.float32) * (ds / sr)
    ph = 2 * torch.pi * t / 0.5
    beat = torch.stack([ph.sin(), ph.cos(), (ph / 4).sin(), (ph / 4).cos()])[None]

    with torch.no_grad():
        ref = enc(patched)
        got = wrapped(patched, beat=beat)
        stock = wrapped(patched, beat=None)

    print(f"latent shape          : {tuple(ref.shape)}  (n_latent={n_latent}, naive={n_samples // ds})")
    print(f"beat signal           : {tuple(beat.shape)}  nonzero={bool(beat.abs().sum() > 0)}")
    print(f"proj output all zero  : {bool(wrapped.beat_proj(beat).abs().max() == 0)}")
    d_beat = (ref - got).abs().max().item()
    d_none = (ref - stock).abs().max().item()
    print(f"max|ref - conditioned|: {d_beat:.3e}")
    print(f"max|ref - beat=None|  : {d_none:.3e}")

    n_beat = sum(p.numel() for p in wrapped.beat_parameters())
    n_enc = sum(p.numel() for p in enc.parameters())
    print(f"beat params           : {n_beat/1e6:.3f}M   encoder: {n_enc/1e6:.2f}M")

    # Gradient must still reach the zeroed layer, or the seam is dead weight.
    wrapped.freeze_backbone()
    out = wrapped(patched, beat=beat)
    out.square().mean().backward()
    grads = [p.grad for p in wrapped.beat_parameters() if p.grad is not None]
    gmax = max(g.abs().max().item() for g in grads) if grads else 0.0
    print(f"max grad on beat proj : {gmax:.3e}")
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    print(f"trainable after freeze: {trainable/1e6:.3f}M")

    ok = (d_beat == 0.0 and d_none == 0.0 and gmax > 0
          and trainable == n_beat)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
