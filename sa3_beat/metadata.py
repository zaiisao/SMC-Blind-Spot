
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
