"""Train the beat adapter with stable-audio-tools' own training stack.

Everything that can come from config does: the model, the beat conditioning
declaration, the optimizer, the timestep sampler and the inpainting masks all
live in the model config -- the published SA3 checkpoint already ships a
``training`` block, and ``create_training_wrapper_from_config`` consumes it.  The
dataset, cropping and metadata hook come from the dataset config.

This file exists only for the one thing config cannot express: the library's
conditioner factory is a hard-coded dispatch, so ``BeatConditioner`` has to be
attached to the live model between construction and training.

    /disk1/jaehoon/sa3env/bin/python -m sa3_beat.train \
        --model-config sa3_beat/configs/beat_small_music.json \
        --dataset-config sa3_beat/configs/dataset_beat_example.json \
        --save-dir runs/beat
"""

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from safetensors.torch import save_file

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.training.factory import create_training_wrapper_from_config

from .build import BEAT_COND_ID, beat_adapter_state_dict, load_beat_model


class SaveBeatAdapter(pl.Callback):
    """Checkpoint only the adapter -- tens of MB, not the 2.2 GB base model."""

    def __init__(self, save_dir, every_n_steps=1000):
        self.save_dir = Path(save_dir)
        self.every_n_steps = every_n_steps
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, module, tag):
        sd = beat_adapter_state_dict(module.diffusion.model, BEAT_COND_ID)
        path = self.save_dir / f"beat_adapter_{tag}.safetensors"
        save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
        return path

    def on_train_batch_end(self, trainer, module, *_):
        if trainer.global_step and trainer.global_step % self.every_n_steps == 0:
            print(f"[save] {self._save(module, f'step{trainer.global_step}')}")

    def on_train_end(self, trainer, module):
        print(f"[save] {self._save(module, 'final')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", required=True)
    p.add_argument("--dataset-config", required=True)
    p.add_argument("--save-dir", default="runs/beat_adapter")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", default="bf16-mixed")
    args = p.parse_args(argv)

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    with open(args.model_config) as f:
        beat_cfg = json.load(f)
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_config = load_beat_model(
        config_path=beat_cfg["base_model_config"],
        ckpt_path=beat_cfg["ckpt_path"],
        include_downbeat=beat_cfg.get("include_downbeat", True),
        freeze=True,
        device=device,
    )

    # Pre-tokenize text in dataloader workers, as the library's own trainer does.
    tokenizers = {k: (c.tokenizer, c.max_length)
                  for k, c in model.conditioner.conditioners.items()
                  if hasattr(c, "tokenizer") and hasattr(c, "max_length")}

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
        tokenizers=tokenizers,
    )

    wrapper = create_training_wrapper_from_config(model_config, model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] {trainable/1e6:.2f}M trainable params on {device}")

    pl.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        max_steps=args.max_steps,
        precision=args.precision if device == "cuda" else "32-true",
        callbacks=[SaveBeatAdapter(args.save_dir, args.save_every)],
        default_root_dir=args.save_dir,
        logger=pl.loggers.CSVLogger(args.save_dir),
        enable_checkpointing=False,   # adapter-only saving is handled above
    ).fit(wrapper, train_dl)


if __name__ == "__main__":
    main()
