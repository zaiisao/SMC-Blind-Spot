"""``custom_metadata_module`` for stable-audio-tools' dataset config.

The library loads this file by path via importlib and calls the module-level
``get_custom_metadata(info, audio)``, merging the result into the item's
metadata.  Configuration therefore arrives through environment variables rather
than function arguments -- the contract has no place to pass them.

    SA3_BEAT_ANNOTATIONS   directory of .beats files, matched by track stem
    SA3_BEAT_PROMPT        text prompt for every item (default "solo piano")
    SA3_BEAT_SUFFIX        annotation extension (default ".beats")

For pre-encoded latents the library calls this *after* cropping and passes
``latent_crop_start`` / ``latent_crop_length`` in ``info``, so the beat window is
aligned to the crop rather than to the start of the source track.
"""

import os
from pathlib import Path

# Absolute imports on purpose: the library loads this file by path with
# importlib (module name "metadata_module", no package), so relative imports
# would raise ImportError. Requires the repo root on sys.path.
from sa3_beat.conditioners import LATENT_RATE
from sa3_beat.data import beat_metadata, load_beats

_ANNOTATIONS = os.environ.get("SA3_BEAT_ANNOTATIONS")
_PROMPT = os.environ.get("SA3_BEAT_PROMPT", "solo piano")
_SUFFIX = os.environ.get("SA3_BEAT_SUFFIX", ".beats")


def get_custom_metadata(info, _data):
    if not _ANNOTATIONS:
        raise RuntimeError("SA3_BEAT_ANNOTATIONS is not set; cannot resolve beat annotations")

    stem = Path(info.get("path") or info.get("latent_filename", "")).stem
    ann = Path(_ANNOTATIONS) / f"{stem}{_SUFFIX}"
    if not ann.exists():
        return {"__reject__": True}

    beats, downbeats = load_beats(ann)
    if beats.size < 2:
        return {"__reject__": True}

    num_frames = int(info.get("latent_crop_length") or 0)
    if not num_frames:
        num_frames = int(round(float(info.get("seconds_total", 0)) * LATENT_RATE))

    return beat_metadata(beats, downbeats, num_frames,
                         int(info.get("latent_crop_start", 0)), _PROMPT)
