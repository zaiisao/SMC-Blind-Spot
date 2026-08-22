"""CLI inference: python track.py song.wav [--config configs/track.yaml].

The audio path is the only positional input; the whole tracker composition -- frontend, bar-pointer
model, and every kwarg of each -- comes from the config YAML (see configs/track.yaml, the default).
Prints one event per line ("<seconds>  beat" / "<seconds>  DOWNBEAT"). This is deliberately just
file loading around tracker.py's definitions -- training and evaluation build their Trackers from
the same build_tracker_from_config.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile
import yaml

from tracker import build_tracker_from_config

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "track.yaml"


def main():
    parser = argparse.ArgumentParser(description="Track beats/downbeats: frontend x bar-pointer model.")
    parser.add_argument("audio", help="path to an audio file (anything soundfile reads)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="tracker composition YAML (default: configs/track.yaml)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    tracker = build_tracker_from_config(config)

    signal, sample_rate = soundfile.read(args.audio, dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    events = tracker.track(signal, sample_rate)

    downbeats = set(np.round(events["downbeats"], decimals=6))
    for t in events["beats"]:
        print(f"{t:10.3f}  {'DOWNBEAT' if round(float(t), 6) in downbeats else 'beat'}")


if __name__ == "__main__":
    main()
