
import argparse
import json
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from safetensors.torch import save_file

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.training.factory import create_training_wrapper_from_config


from .build import (BEAT_COND_ID, apply_training_overrides,
                    beat_adapter_state_dict, beat_parameters, load_beat_model)
from .controlnet import BeatControlNet


def preflight_annotations(dataset_config, strict=True):
    ann_dir = os.environ.get("SA3_BEAT_ANNOTATIONS")
    if not ann_dir:
        raise RuntimeError("SA3_BEAT_ANNOTATIONS is not set")
    suffix = os.environ.get("SA3_BEAT_SUFFIX", ".beats")

    stems, missing = [], []
    for d in dataset_config.get("datasets", []):
        for npy in sorted(Path(d["path"]).glob("*.npy")):
            stems.append(npy.stem)
            if not (Path(ann_dir) / f"{npy.stem}{suffix}").exists():
                missing.append(npy.stem)

    n = len(stems)
    print(f"[preflight] {n - len(missing)}/{n} annotations resolved in {ann_dir}")
    if missing:
        msg = (f"{len(missing)} of {n} latents have no annotation, e.g. "
               f"{missing[:5]}. Training would silently use only the matched "
               f"subset (or spin forever if none match).")
        if strict:
            raise RuntimeError(msg)
        print(f"[preflight] WARNING: {msg}")
    return n - len(missing)


def attach_controlnet(model, n_blocks=None, include_downbeat=True, cond_id=BEAT_COND_ID):
    dit = model.model.model
    cn = BeatControlNet(dit.transformer, n_blocks=n_blocks,
                        include_downbeat=include_downbeat, cond_id=cond_id)
    cn.to(next(dit.parameters()).device)
    dit.beat_controlnet = cn
    cn.attach()
    for p_ in cn.parameters():
        p_.requires_grad_(True)
    return cn


class LogBeatSensitivity(pl.Callback):

    def __init__(self, every_n_steps=50):
        self.every_n_steps = every_n_steps

    def on_train_start(self, trainer, module):
        cn = getattr(module.diffusion.model.model, "beat_controlnet", None)
        if cn is not None:
            cn.probe_every = self.every_n_steps

    def on_train_batch_end(self, trainer, module, *_):
        cn = getattr(module.diffusion.model.model, "beat_controlnet", None)
        if cn is not None and cn.last_sensitivity is not None:
            module.log("train/beat_sensitivity", cn.last_sensitivity,
                       prog_bar=True)
            cn.last_sensitivity = None


class SaveBeatAdapter(pl.Callback):

    def __init__(self, save_dir, every_n_steps=1000, keep_last=5):
        self.save_dir = Path(save_dir)
        self.every_n_steps = every_n_steps
        self.keep_last = keep_last
        self.saved = []
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, module, tag):
        dit = module.diffusion.model
        sd = beat_adapter_state_dict(dit, BEAT_COND_ID)
        cn = getattr(dit.model, "beat_controlnet", None)
        if cn is not None:
            sd = {k: v for k, v in sd.items()
                  if "beat_controlnet." not in k}
            sd.update({f"beat_controlnet.{k}": v
                       for k, v in cn.state_dict().items()})
        path = self.save_dir / f"beat_adapter_{tag}.safetensors"
        save_file({k: v.contiguous().clone() for k, v in sd.items()}, str(path))
        self.saved.append(path)
        while self.keep_last and len(self.saved) > self.keep_last:
            old = self.saved.pop(0)
            if old.name.endswith("final.safetensors"):
                continue
            old.unlink(missing_ok=True)
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
    p.add_argument("--controlnet", action="store_true",
                   help="train a ControlNet branch instead of the 21M "
                        "per-frame adapter alone")
    p.add_argument("--controlnet-blocks", type=int, default=None,
                   help="backbone blocks to copy (default: half the depth)")
    p.add_argument("--train-both", action="store_true",
                   help="also train the 21M per-frame adapter alongside the "
                        "ControlNet (confounds ablations; off by default)")
    p.add_argument("--keep-last", type=int, default=5,
                   help="checkpoints to retain; each is ~1.1 GB with a ControlNet")
    p.add_argument("--probe-every", type=int, default=50,
                   help="steps between beat-sensitivity probes (0 disables)")
    p.add_argument("--cfg-dropout", type=float, default=0.0,
                   help="probability of dropping the beat signal per item; "
                        "required if you want CFG to guide ON beats")
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

    include_downbeat = beat_cfg.get("include_downbeat", True)
    if args.cfg_dropout > 0:
        model.conditioner.conditioners[BEAT_COND_ID].cfg_dropout_prob = args.cfg_dropout
        print(f"[train] beat CFG dropout p={args.cfg_dropout}")

    if args.controlnet:
        cn = attach_controlnet(model, n_blocks=args.controlnet_blocks,
                               include_downbeat=include_downbeat)
        n_cn = sum(q.numel() for q in cn.parameters())
        if not args.train_both:
            prefix = f"modular_local_embeds.{BEAT_COND_ID}."
            for name, q in model.model.named_parameters():
                if prefix in name and "beat_controlnet." not in name:
                    q.requires_grad_(False)
            live = sum(q.numel() for q in model.parameters() if q.requires_grad)
            assert live == n_cn, f"expected {n_cn} trainable, got {live}"
        print(f"[train] ControlNet: {cn.n_blocks} blocks, {n_cn/1e6:.1f}M params"
              f"{' (+ per-frame adapter, JOINT)' if args.train_both else ' only'}")

    # Pre-tokenize text in dataloader workers, as the library's own trainer does.
    tokenizers = {k: (c.tokenizer, c.max_length)
                  for k, c in model.conditioner.conditioners.items()
                  if hasattr(c, "tokenizer") and hasattr(c, "max_length")}

    preflight_annotations(dataset_config)

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
        tokenizers=tokenizers,
    )

    model_config = apply_training_overrides(
        model_config, beat_cfg.get("training_overrides", {}))
    wrapper = create_training_wrapper_from_config(model_config, model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] {trainable/1e6:.2f}M trainable params on {device}")

    pl.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        max_steps=args.max_steps,
        precision=args.precision if device == "cuda" else "32-true",
        callbacks=[SaveBeatAdapter(args.save_dir, args.save_every,
                                       args.keep_last),
                   LogBeatSensitivity(args.probe_every)],
        default_root_dir=args.save_dir,
        logger=pl.loggers.CSVLogger(args.save_dir, flush_logs_every_n_steps=10),
        log_every_n_steps=10,
        enable_checkpointing=False,   # adapter-only saving is handled above
    ).fit(wrapper, train_dl)


if __name__ == "__main__":
    main()
