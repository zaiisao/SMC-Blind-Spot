"""Train a ladder rung from a config: python train.py [--config configs/train.yaml].

Which rung, which frontend, and the training hyperparameters all live in the YAML (see
configs/train.yaml, mirroring configs/track.yaml); this script loads it and calls the selected
rung's fit(). --device / --rung override the config.
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "bt_e2e"))     # crf_baseline (discriminative arm)

from rungs.r2_em_dbn import R2GenerativeLambda                # noqa: E402
from rungs.r3_conditioned_dbn import R3ConditionedFactors    # noqa: E402
from rungs.r4_neural_hmm import R4NeuralHMM                   # noqa: E402

from crf_baseline import CRFLearnedFactors                    # noqa: E402


class R45GaussianAlias(R4NeuralHMM):
    """`rung: {name: r4_5}` == r4 with the gaussian (rich-feature) emission. Own ARM_NAME keeps
    its checkpoints separate from bock-emission r4 runs."""
    ARM_NAME = "r4_5"
    TRAIN_DEFAULTS = dict(em_iterations=10)

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["rung"] = {**(config.get("rung") or {}), "emission": "gaussian"}
        config["rung"].pop("name", None)
        return super().from_config(config)

DEFAULT_CONFIG = ROOT / "configs" / "train.yaml"


def train_vanilla(config):
    """The rung-less arm: train the Beat Transformer frontend with the BCE loss only."""
    from training import harness, e2e
    params = harness.resolve_params("vanilla", config, frontend_mode="e2e", epochs=30,
                                    learning_rate=1e-3, gradient_clip=0.5)
    if params.frontend_mode != "e2e":
        raise SystemExit("the vanilla arm trains the frontend; frontend.mode must be e2e")
    return harness.run(None, params, e2e.run)


TRAINABLE_RUNGS = (CRFLearnedFactors, R3ConditionedFactors, R2GenerativeLambda,
                   R4NeuralHMM, R45GaussianAlias)
TRAINERS = {rung_class.ARM_NAME: rung_class.fit for rung_class in TRAINABLE_RUNGS}
TRAINERS["vanilla"] = train_vanilla


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="training composition YAML (default: configs/train.yaml)")
    parser.add_argument("--rung", default=None, choices=tuple(TRAINERS),
                        help="override rung.name from the config")
    parser.add_argument("--device", default=None, help="override training.device from the config")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    if args.rung:
        config.setdefault("rung", {})["name"] = args.rung
    if args.device:
        config.setdefault("training", {})["device"] = args.device

    rung_name = (config.get("rung") or {}).get("name")
    if rung_name not in TRAINERS:
        raise SystemExit(f"config rung.name must be one of {sorted(TRAINERS)}, got {rung_name!r}")
    TRAINERS[rung_name](config)


if __name__ == "__main__":
    main()
