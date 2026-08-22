"""Beat This frontend (Foscarin, Schlueter & Widmer, ISMIR 2024; external/beat_this submodule).

Wraps the OFFICIAL beat_this.inference.Audio2Frames -- their spectrogram, their chunked
split-predict-aggregate, their checkpoint loading -- rather than re-plumbing the model. We add only
the property surface the Tracker needs (fps, ACTIVATION_FORM) and optional resampling of the
activation grid.

Properties of what this emits:
  * native grid is EXACTLY 50 fps (22050 Hz audio, hop 441); pass target_fps to linearly
    interpolate the logits onto another grid (our activation caches historically use
    86.1328125 = 22050/256, so cached and live activations line up frame for frame).
  * activations are LOGITS (ACTIVATION_FORM="logit"): Beat This's own DBN path feeds
    sigmoid(logits) to madmom, and the Tracker applies the same conversion where a bar-pointer
    model wants probabilities.

Checkpoints (all cached locally under ~/.cache/torch/hub/checkpoints/):
  * "final0" -- trained on everything except GTZAN. Fine for deployment and demos; NOT fold-honest
    for evaluation on the training datasets.
  * "fold0".."fold7" -- the Beat This 8-fold protocol. For any number reported on our val folds,
    use the checkpoint whose held-out fold matches; final0 numbers on those songs are leakage.
"""
import sys
from pathlib import Path
from typing import Optional

import torch

from frontends import Frontend

_BEAT_THIS_ROOT = Path(__file__).resolve().parent.parent / "external" / "beat_this"


class BeatThisFrontend(Frontend):
    OUTPUT_MODES = {"activations": 2, "features": 512}

    ACTIVATION_FORM = "logit"
    BOUNDING = "squeeze"       # Beat This's published convention: sigmoid(x)*(1-eps) + eps/2.
                               # With this, Tracker(frontend, "madmom_dbn") is EVENT-IDENTICAL to
                               # the official Audio2Beats(dbn=True) (verified 8/8 GTZAN songs);
                               # with "clip" it diverged on 1/8 -- the convention is NOT always
                               # event-neutral, though mean F is identical to 4 decimals.
    NATIVE_FPS = 50.0

    def __init__(self, checkpoint: str = "final0", device: str = "cuda",
                 target_fps: Optional[float] = None, float16: bool = False,
                 output: str = "activations"):
        if output not in self.OUTPUT_MODES:
            raise KeyError(f"unknown output mode {output!r} for {self.name} "
                           f"(have: {sorted(self.OUTPUT_MODES)})")
        self.output = output
        if str(_BEAT_THIS_ROOT) not in sys.path:
            sys.path.insert(0, str(_BEAT_THIS_ROOT))
        from beat_this.inference import Audio2Frames

        self._audio2frames = Audio2Frames(checkpoint_path=checkpoint, device=device,
                                          float16=float16)
        # The features width is the loaded checkpoint's transformer_dim (the frontend stack's final
        # linear projects into it); read it off the model so small checkpoints declare honestly.
        transformer_dim = self._audio2frames.model.frontend.linear.out_features
        self.OUTPUT_MODES = {**self.OUTPUT_MODES, "features": transformer_dim}
        self.checkpoint = checkpoint
        self.device = device
        self.fps = target_fps if target_fps is not None else self.NATIVE_FPS

    @torch.no_grad()
    def get_features(self, signal, sample_rate: int) -> torch.Tensor:
        """[num_samples] mono (any sample rate; Beat This resamples internally) -> [num_frames,
        num_channels] at self.fps: (beat, downbeat) LOGITS in "activations" mode, the penultimate
        transformer_blocks features in "features" mode."""
        if self.output == "activations":
            beat_logits, downbeat_logits = self._audio2frames(signal, sample_rate)
            out = torch.stack([beat_logits, downbeat_logits], dim=-1)         # [T@50fps, 2]
        else:
            out = self._penultimate(signal, sample_rate)                      # [T@50fps, C]
        if self.fps != self.NATIVE_FPS:
            num_target = int(round(out.shape[0] * self.fps / self.NATIVE_FPS))
            out = torch.nn.functional.interpolate(
                out.t().unsqueeze(0), size=num_target, mode="linear", align_corners=False
            ).squeeze(0).t()
        return out.cpu()

    # spect2frames' chunking constants (external/beat_this/beat_this/inference.py) -- the features
    # path MUST split/aggregate identically or feature frames misalign with the logits pipeline.
    _CHUNK_SIZE = 1500
    _BORDER_SIZE = 6

    def _penultimate(self, signal, sample_rate: int) -> torch.Tensor:
        """[T, transformer_dim] transformer_blocks output -- the model with its final task heads
        cut off, non-destructively: a forward hook captures the heads' input during the normal
        forward pass (the heads still run; their logits are ignored).

        Replicates Spect2Frames.spect2frames' split-predict-aggregate exactly (same split_piece,
        chunk_size=1500, border_size=6, overlap_mode="keep_first"), applied to feature chunks
        instead of logit chunks: borders are discarded, and earlier chunks win in overlaps
        (iterate reversed, later writes overwritten by earlier ones). Verified in __main__:
        task_heads(aggregated features) == aggregated logits (exact on a single chunk, float noise
        ~1e-6 across separate CUDA passes on multi-chunk), which certifies both the hook placement
        and the aggregation, since the heads are frame-wise.
        """
        from beat_this.inference import split_piece

        model = self._audio2frames.model
        spect = self._audio2frames.signal2spect(signal, sample_rate)
        captured = []
        handle = model.transformer_blocks.register_forward_hook(
            lambda module, inputs, output: captured.append(output))
        try:
            with torch.inference_mode():
                with torch.autocast(enabled=self._audio2frames.float16,
                                    device_type=self._audio2frames.device.type):
                    chunks, starts = split_piece(spect, self._CHUNK_SIZE,
                                                 border_size=self._BORDER_SIZE,
                                                 avoid_short_end=True)
                    for chunk in chunks:
                        model(chunk.unsqueeze(0))
        finally:
            handle.remove()
        feature_chunks = [chunk[0].float() for chunk in captured]             # each [chunk_T, C]

        piece = torch.zeros((spect.shape[0], feature_chunks[0].shape[1]), device=spect.device)
        border, chunk_size = self._BORDER_SIZE, self._CHUNK_SIZE
        for start, chunk in reversed(list(zip(starts, feature_chunks))):      # keep_first
            piece[max(start + border, 0):start + chunk_size - border] = chunk[border:-border]
        return piece


if __name__ == "__main__":
    # Smoke test: a synthetic 120 BPM 4/4 click track through frontend -> both bar-pointer models.
    import numpy as np

    from tracker import Tracker

    SR, SECONDS, BPM = 22050, 12, 120.0
    signal = np.zeros(SR * SECONDS, dtype=np.float32)
    click = (0.8 * np.sin(2 * np.pi * 1000 * np.arange(0.02 * SR) / SR)
             * np.hanning(int(0.02 * SR))).astype(np.float32)
    beat_samples = (np.arange(0, SECONDS * BPM / 60) * 60 / BPM * SR).astype(int)
    for i, s in enumerate(beat_samples):
        signal[s:s + len(click)] += click * (1.5 if i % 4 == 0 else 1.0)   # accent the downbeats

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frontend = BeatThisFrontend(device=device)
    print(f"frontend: {frontend.name} ckpt={frontend.checkpoint} fps={frontend.fps} "
          f"form={frontend.ACTIVATION_FORM}")
    features = frontend.get_features(signal, SR)
    print(f"features: {tuple(features.shape)} (expect ~{SECONDS * 50} frames)")

    for bar_pointer in ("madmom_dbn", "2016_dbn"):
        kwargs = {} if bar_pointer == "madmom_dbn" else {"device": device}
        events = Tracker(frontend, bar_pointer, **kwargs).track(signal, SR)
        n_beats, n_downbeats = len(events["beats"]), len(events["downbeats"])
        ibi = float(np.diff(events["beats"]).mean()) if n_beats > 1 else float("nan")
        print(f"  {bar_pointer:16s}: {n_beats:3d} beats (expect ~{SECONDS * 2}), "
              f"{n_downbeats:2d} downbeats (expect ~{SECONDS // 2}), "
              f"mean IBI {ibi:.3f}s -> {60.0 / ibi:.1f} BPM (expect ~120)")

    # Certify the "features" mode against the logits pipeline: the task heads are frame-wise, so
    # heads(aggregated features) must reproduce the aggregated logits. Run on a long signal too,
    # so the multi-chunk split/aggregate path is exercised, not just the single-chunk one.
    features_frontend = BeatThisFrontend(device=device, output="features")
    long_signal = np.concatenate([signal] * 4)                    # ~48 s -> multiple chunks
    for label, sig in (("single-chunk", signal), ("multi-chunk", long_signal)):
        features = features_frontend.get_features(sig, SR)
        logits = frontend.get_features(sig, SR)
        with torch.inference_mode():
            heads = features_frontend._audio2frames.model.task_heads(
                features.to(device).unsqueeze(0))
        rederived = torch.stack([heads["beat"][0], heads["downbeat"][0]], dim=-1).cpu()
        error = float((rederived - logits).abs().max())
        assert features.shape == (logits.shape[0], features_frontend.num_channels)
        assert error < 1e-4, f"features/logits misaligned ({label}): max|diff|={error}"
        print(f"  features ({label:12s}): {tuple(features.shape)}, "
              f"heads(features) vs logits max|diff| = {error:.2e}  OK")
