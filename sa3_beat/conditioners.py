"""Beat conditioning signal for Stable Audio 3.

Produces the tensor that gets injected into every DiT block through the stock
``modular_local_cond`` path.  Nothing here subclasses or patches SA3 beyond
implementing its ``Conditioner`` contract: ``forward(list_of_inputs, device)``
returns ``(tensor, mask)`` and the wrapper takes element ``[0]``.

Beats are encoded as sin/cos of *continuous* beat phase rather than a binary
pulse train, so sub-frame beat position stays expressible on the 10.77 Hz latent
grid (SPECIFICATION-style rationale: at 92.9 ms per frame a pulse quantises beat
position to +/-46 ms, which is most of the +/-70 ms evaluation tolerance).
"""

import numpy as np
import torch

from stable_audio_tools.models.conditioners import Conditioner

# 44100 / 4096 -- SAME's downsampling ratio at the model's sample rate.
LATENT_RATE = 44100 / 4096


def _continuous_phase(event_times, frame_times):
    """Continuous event index at each frame time.

    Returns a float array that increases by exactly 1.0 between consecutive
    events, so ``sin/cos(2*pi*phase)`` is a smooth periodic encoding whose zero
    crossings land on the events themselves.  Outside the annotated span the
    phase is extrapolated linearly from the first/last interval instead of being
    clamped, which would otherwise create a spurious stationary region.
    """
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
    """Beat/downbeat phase -> ``(batch, channels, frames)``.

    Channel layout: ``[sin(beat), cos(beat)]`` and, when ``include_downbeat``,
    ``[sin(bar), cos(bar)]``.  Input per batch item is a dict with ``beats``
    (seconds), optional ``downbeats`` (seconds) and ``num_frames``.
    """

    def __init__(self, include_downbeat: bool = True, latent_rate: float = LATENT_RATE,
                 cfg_dropout_prob: float = 0.0):
        dim = 4 if include_downbeat else 2
        # dim == output_dim and project_out=False -> proj_out is nn.Identity, so
        # this conditioner holds no parameters. All learning lives in the DiT's
        # per-block modular projections.
        super().__init__(dim=dim, output_dim=dim)
        self.include_downbeat = include_downbeat
        self.latent_rate = latent_rate
        # SA3's own CFG dropout covers cross_attn_cond and prepend_cond only,
        # and get_conditioning_inputs(negative=True) does not carry
        # modular_local_cond -- so the beat signal is identical in both CFG
        # branches and cancels out of the guidance direction. Guiding ON beats
        # requires the model to have seen them absent, which is what this does.
        # Whole items are dropped, not individual frames: the null has to be
        # "this clip has no beat conditioning", not a ragged grid.
        self.cfg_dropout_prob = cfg_dropout_prob

    @property
    def cond_dim(self) -> int:
        return self.dim

    def encode_one(self, beats, downbeats, num_frames, frame_offset=0) -> np.ndarray:
        """``(channels, num_frames)`` features for a single item.

        ``frame_offset`` is the latent frame index this window starts at, so a
        randomly cropped clip still lines its beat phase up with the audio it was
        cropped from.  Beat times stay absolute in the source track's timebase.
        """
        frame_times = (
            np.arange(num_frames, dtype=np.float64) + frame_offset
        ) / self.latent_rate

        beat_phase = _continuous_phase(beats, frame_times)
        channels = [np.sin(2 * np.pi * beat_phase), np.cos(2 * np.pi * beat_phase)]

        if self.include_downbeat:
            if downbeats is None or len(np.asarray(downbeats)) < 2:
                # No downbeat annotation: leave the bar channels at zero rather
                # than inventing a bar phase the annotation does not support.
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

        # Deliberately NOT gated on ``self.training``: SA3's training wrapper
        # keeps the diffusion module (and therefore this conditioner) in eval
        # mode throughout training -- Lightning reports ~1347 modules in eval
        # mode -- so a training-mode check silently disables the dropout while
        # still printing that it is on. The probability is the switch: it is
        # set only by train.py and defaults to 0 everywhere else.
        if self.cfg_dropout_prob > 0:
            keep = np.random.rand(len(encoded)) >= self.cfg_dropout_prob
            encoded = [e if k else np.zeros_like(e) for e, k in zip(encoded, keep)]

        max_frames = max(e.shape[1] for e in encoded)
        batch = np.zeros((len(encoded), self.dim, max_frames), dtype=np.float32)
        mask = np.zeros((len(encoded), max_frames), dtype=bool)
        for i, e in enumerate(encoded):
            # Right-aligned to match _left_pad_to_match in SA3's transformer,
            # which left-pads / right-trims local conditioning onto x.
            batch[i, :, max_frames - e.shape[1]:] = e
            mask[i, max_frames - e.shape[1]:] = True

        return (
            torch.from_numpy(batch).to(device),
            torch.from_numpy(mask).to(device),
        )
