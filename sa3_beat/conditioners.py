
import numpy as np
import torch

from stable_audio_tools.models.conditioners import Conditioner

# 44100 / 4096 -- SAME's downsampling ratio at the model's sample rate.
LATENT_RATE = 44100 / 4096


def _continuous_phase(event_times, frame_times):
    event_times = np.asarray(event_times, dtype=np.float64)
    if event_times.size < 2:
        return np.zeros_like(frame_times)

    idx = np.arange(event_times.size, dtype=np.float64)
    phase = np.interp(frame_times, event_times, idx)

    first_ibi = event_times[1] - event_times[0]
    last_ibi = event_times[-1] - event_times[-2]
    if first_ibi > 0:
        before = frame_times < event_times[0]
        phase[before] = (frame_times[before] - event_times[0]) / first_ibi
    if last_ibi > 0:
        after = frame_times > event_times[-1]
        phase[after] = idx[-1] + (frame_times[after] - event_times[-1]) / last_ibi
    return phase


class BeatConditioner(Conditioner):

    def __init__(self, include_downbeat: bool = True, latent_rate: float = LATENT_RATE,
                 cfg_dropout_prob: float = 0.0):
        dim = 4 if include_downbeat else 2
        super().__init__(dim=dim, output_dim=dim)
        self.include_downbeat = include_downbeat
        self.latent_rate = latent_rate
        self.cfg_dropout_prob = cfg_dropout_prob

    @property
    def cond_dim(self) -> int:
        return self.dim

    def encode_one(self, beats, downbeats, num_frames, frame_offset=0) -> np.ndarray:
        frame_times = (
            np.arange(num_frames, dtype=np.float64) + frame_offset
        ) / self.latent_rate

        beat_phase = _continuous_phase(beats, frame_times)
        channels = [np.sin(2 * np.pi * beat_phase), np.cos(2 * np.pi * beat_phase)]

        if self.include_downbeat:
            if downbeats is None or len(np.asarray(downbeats)) < 2:
                channels += [np.zeros(num_frames), np.zeros(num_frames)]
            else:
                bar_phase = _continuous_phase(downbeats, frame_times)
                channels += [np.sin(2 * np.pi * bar_phase), np.cos(2 * np.pi * bar_phase)]

        return np.stack(channels, axis=0)

    def forward(self, inputs, device):
        encoded = []
        for item in inputs:
            if isinstance(item, dict):
                beats = item["beats"]
                downbeats = item.get("downbeats")
                num_frames = int(item["num_frames"])
                frame_offset = int(item.get("frame_offset", 0))
            else:
                raise TypeError(
                    "BeatConditioner expects a dict with 'beats' and 'num_frames', "
                    f"got {type(item).__name__}"
                )
            encoded.append(self.encode_one(beats, downbeats, num_frames, frame_offset))

        if self.cfg_dropout_prob > 0:
            keep = np.random.rand(len(encoded)) >= self.cfg_dropout_prob
            encoded = [e if k else np.zeros_like(e) for e, k in zip(encoded, keep)]

        max_frames = max(e.shape[1] for e in encoded)
        batch = np.zeros((len(encoded), self.dim, max_frames), dtype=np.float32)
        mask = np.zeros((len(encoded), max_frames), dtype=bool)
        for i, e in enumerate(encoded):
            batch[i, :, max_frames - e.shape[1]:] = e
            mask[i, max_frames - e.shape[1]:] = True

        return (
            torch.from_numpy(batch).to(device),
            torch.from_numpy(mask).to(device),
        )
