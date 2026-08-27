"""Beat conditioning for SA3's *encoder*, through the stock ``new_tokens`` seam.

The handover (§3.4) proposes widening the encoder's first conv from 2 to 3 input
channels, or failing that a FiLM modulation.  Neither is needed here, and the
first does not even apply: SA3's encoder is a ``SAMEEncoder`` whose entry point
is a 512-dim patch mapping, not a waveform conv.

What it *does* have is a conditioning seam already built for this shape.
``SAMEEncoder`` downsamples by appending learned query tokens -- one per output
latent frame -- to each chunk of input tokens and reading the queries back out
as the latent.  ``TransformerResamplingBlock.forward`` accepts
``override_new_tokens`` and adds it to those queries:

    new_tokens = self.new_tokens.expand(...)
    if override_new_tokens is not None:
        new_tokens = new_tokens + override_new_tokens

That is per-latent-frame, time-aligned to the latent grid by construction, and
additive -- so a zero-init projection is an exact identity at step 0, the same
guarantee the DiT adapter relies on.  The seam is simply never wired up:
``SAMEEncoder.forward`` takes ``**kwargs`` and drops them instead of passing
them to its layers.  This module supplies the missing pass-through without
touching the submodule.

Deliberately NOT conditioned (§3.2): the decoder.  If the decoder could see the
beats, the encoder would be free to leave timing out of ``z`` and let the
decoder reconstruct it from the side channel.  Keeping the decoder blind is what
forces ``z`` to carry the beats -- which is required, because ``z`` is all the
DiT ever produces.
"""

import torch
from torch import nn

from stable_audio_tools.models.autoencoders import (SAMEEncoder,
                                                    TransformerResamplingBlock)

from .conditioners import BeatConditioner


def latent_frames(input_length: int, block: TransformerResamplingBlock) -> int:
    """Latent frames a resampling block emits for ``input_length`` input tokens.

    The block zero-pads its input up to a multiple of ``chunk_size`` before
    chunking, so this is not simply ``input_length // stride``: a 688-token
    input at stride 16 and chunk size 32 yields 44 frames, not 43.  Getting this
    wrong surfaces as an einops shape error deep inside the submodule, so the
    rule is reproduced here and checked against the real block.
    """
    stride = block.stride
    if block.sliding_window_latents is None:
        pad_modulo = block.chunk_size
    else:
        pad_modulo = stride
    padded = -(-input_length // pad_modulo) * pad_modulo
    return padded // stride


class BeatTokenProjection(nn.Module):
    """Beat features -> one additive token per latent frame, zero-init.

    ``Linear -> SiLU -> Linear`` with the output layer zeroed, matching the
    shape SA3 already uses for its own local conditioning.  At init the output
    is exactly zero, so the encoder is bit-identical to the published
    checkpoint while gradient still flows into both layers.
    """

    def __init__(self, beat_dim: int, token_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(beat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, beat):
        # beat: (batch, beat_dim, frames) -- the layout BeatConditioner emits.
        return self.net(beat.transpose(1, 2))  # -> (batch, frames, token_dim)


class BeatConditionedEncoder(nn.Module):
    """Wraps a ``SAMEEncoder`` and injects beat tokens into its first block.

    Only the first resampling block is conditioned: it is the one whose queries
    are still at the input time resolution, and in this checkpoint it is also
    the only one (``c_mults`` has length 1).

    ``include_downbeat`` mirrors ``BeatConditioner``: 4 channels (beat and bar
    phase) or 2 (beat only).  Tracks lacking downbeat annotations keep the bar
    channels at zero, which is off the unit circle and so distinguishable from
    every real bar phase without a mask channel.
    """

    def __init__(self, encoder: SAMEEncoder, include_downbeat: bool = True,
                 hidden_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        # Derived, not hardcoded: ``build.py`` sizes the DiT's projection the
        # same way, so ``include_downbeat`` stays a single switch instead of a
        # width that has to be kept in sync by hand.
        beat_dim = BeatConditioner(include_downbeat=include_downbeat).cond_dim
        self.beat_dim = beat_dim

        blocks = [m for m in encoder.layers
                  if isinstance(m, TransformerResamplingBlock)]
        if not blocks:
            raise ValueError("encoder has no TransformerResamplingBlock")
        self.beat_proj = BeatTokenProjection(
            beat_dim, blocks[0].new_tokens.shape[-1], hidden_dim)

    def forward(self, x, beat=None, override_stride=None):
        """``beat`` is ``(batch, beat_dim, frames)`` or None (-> stock encoder).

        ``frames`` must equal the number of latent frames the encoder will
        produce; a mismatch is a silent shape bug inside einops, so it is
        checked here instead.
        """
        if beat is not None and beat.shape[1] != self.beat_dim:
            raise ValueError(
                f"beat has {beat.shape[1]} channels, expected {self.beat_dim} "
                f"(include_downbeat mismatch?)")
        tokens = None if beat is None else self.beat_proj(beat)

        first = True
        for i, layer in enumerate(self.encoder.layers):
            if isinstance(layer, TransformerResamplingBlock):
                stride = None if override_stride is None else override_stride[i]
                if first and tokens is not None:
                    expected = latent_frames(x.shape[-1], layer)
                    if tokens.shape[1] != expected:
                        raise ValueError(
                            f"beat has {tokens.shape[1]} frames, encoder block "
                            f"produces {expected} for input length {x.shape[-1]}")
                    x = layer(x, stride=stride, override_new_tokens=tokens)
                else:
                    x = layer(x, stride=stride)
                first = False
            else:
                x = layer(x)
        return x

    def beat_parameters(self):
        return self.beat_proj.parameters()

    def freeze_backbone(self):
        """Train only the beat projection; the encoder itself stays fixed.

        Used for the first training stage, where the question is whether the
        seam can steer anything at all before the encoder is allowed to move.
        """
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.beat_proj.parameters():
            p.requires_grad = True
