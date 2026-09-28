"""Can the trained discriminator see the beat grid at all?

train/contrastive_loss_dis sat at its chance value (softplus(0) * weight) for
every step of r1, r2 and r3. Two things produce that: a discriminator that is
blind to the grid, or a bug in how the term is reduced. This tells them apart
without training anything.

Method: freeze a saved discriminator, take held-out REAL latents, and score
each one twice -- under its own beat grid and under a neighbour's grid
re-anchored to its crop (arc_data.roll_beat_metadata, the exact negative used
in training). If the conditioning pathway carries signal, the true-grid score
exceeds the wrong-grid score more often than chance. If the two are bitwise
equal, the grid never reaches the scoring head.

Reported per checkpoint:
  bitwise identical   fraction of items where score(true) == score(wrong) exactly
  win rate            fraction of score positions where true outscores wrong
                      (0.5 = blind; this is what contrastive_loss_dis measures)
  mean gap            mean(score_true - score_wrong), with a paired bootstrap CI
  softplus            the loss value implied by the gap, to compare against the
                      chance line printed alongside it

Usage:
    arc_disc_probe.py --gpu 3 --disc /disk1/jaehoon/arc_runs/r3/disc_002000.safetensors
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "third-party" / "stable-audio-tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import arc_data  # noqa: E402
import arc_beat_train as abt  # noqa: E402
from sa3_beat.build import load_beat_model  # noqa: E402

SA3_DIR = Path("/disk1/jaehoon/sa3_small_music")


def boot(d, n=10000, seed=0):
    r = np.random.default_rng(seed)
    return np.percentile(
        np.asarray(d)[r.integers(0, len(d), (n, len(d)))].mean(1), [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disc", nargs="+", required=True,
                    help="saved disc_*.safetensors (disc.* + head.* keys)")
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--t-gan", type=float, default=0.5,
                    help="fixed noise level, so every checkpoint is scored on "
                         "identical inputs instead of a fresh random draw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapter", default=abt.DEFAULT_ADAPTER)
    a = ap.parse_args()

    from stable_audio_tools.training.arc import ARCTrainingWrapper

    torch.manual_seed(a.seed)
    device = f"cuda:{a.gpu}"

    # The generator is only here because ARCTrainingWrapper needs one; it is
    # never stepped and never scores anything in this probe.
    model, config = load_beat_model(device=device)
    from sa3_beat.generate import load_adapter, _patch_conditioning_inputs
    load_adapter(model, a.adapter, device)
    _patch_conditioning_inputs(model)
    abt.wrap_conditioner_with_inpaint(model, config["model"]["io_channels"])
    for p in model.model.parameters():
        p.requires_grad_(False)

    disc, _ = abt.build_discriminator(device)

    arc_config = json.loads(
        (SA3_DIR / "model_config.json").read_text())["training"]["arc"]
    arc_config.pop("use_model_as_discriminator", None)
    arc_config.pop("discriminator_base_ckpt", None)
    dc = arc_config["discriminator"]
    dc["contrastive"] = True

    opt = {"optimizer": {"type": "AdamW", "config": {"lr": 1e-5}}}
    wrapper = ARCTrainingWrapper(
        model, arc_config, {"diffusion": opt, "discriminator": opt},
        discriminator=disc, use_ema=False, pre_encoded=True,
        sample_rate=config["sample_rate"],
        sample_size=int(arc_data.CROP * model.pretransform.downsampling_ratio),
    ).to(device)
    wrapper.eval()

    # held-out real material only: fold 0 never entered ARC training
    items, _ = arc_data.real_items()
    ds = arc_data.BeatARCDataset(items, seed=a.seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size, shuffle=False,
        collate_fn=arc_data.collate, num_workers=2, drop_last=True)

    # freeze one set of batches so all checkpoints see identical inputs
    batches = []
    for i, b in enumerate(loader):
        if i >= a.batches:
            break
        batches.append(b)
    print(f"[data] {len(batches)} batches x {a.batch_size} "
          f"= {len(batches)*a.batch_size} real crops", flush=True)

    from safetensors.torch import load_file

    for path in a.disc:
        sd = load_file(path)
        d_sd = {k[5:]: v for k, v in sd.items() if k.startswith("disc.")}
        h_sd = {k[5:]: v for k, v in sd.items() if k.startswith("head.")}
        md = wrapper.discriminator.load_state_dict(d_sd, strict=False)
        mh = wrapper.discriminator_head.load_state_dict(h_sd, strict=False)
        if md.missing_keys or mh.missing_keys:
            raise SystemExit(f"{path} is incomplete: {len(md.missing_keys)} "
                             f"disc / {len(mh.missing_keys)} head keys unset")
        wrapper.eval()

        gaps, wins, ident, n_rolled, n_items = [], [], [], 0, 0
        g = torch.Generator(device="cpu").manual_seed(a.seed)
        for reals, metadata in batches:
            reals = reals.to(device)
            rolled, unrolled = arc_data.roll_beat_metadata(metadata)
            n_rolled += len(metadata) - len(unrolled)
            n_items += len(metadata)

            noise = torch.randn(reals.shape, generator=g).to(device)
            t = torch.full((reals.shape[0],), a.t_gan, device=device)
            x_t = reals * (1 - t)[:, None, None] + noise * t[:, None, None]

            with torch.no_grad():
                c_true = wrapper.discriminator.conditioner(metadata, device)
                c_wrong = wrapper.discriminator.conditioner(rolled, device)
                s_true = wrapper._disc_scores(x_t, t, c_true)
                s_wrong = wrapper._disc_scores(x_t, t, c_wrong)

            # scores come back as a list over scales; flatten per item
            for st, sw in zip(s_true, s_wrong):
                d = (st - sw).float()
                wins.append((d > 0).float().mean(dim=tuple(range(1, d.ndim))).cpu().numpy())
                gaps.append(d.mean(dim=tuple(range(1, d.ndim))).cpu().numpy())
                ident.append(torch.tensor(
                    [bool((x == y).all()) for x, y in zip(st, sw)]).numpy())

        gaps = np.concatenate(gaps)
        wins = np.concatenate(wins)
        ident = np.concatenate(ident)
        lo, hi = boot(gaps)
        sp = float(np.log1p(np.exp(-np.mean(gaps))))
        print(f"\n=== {path}")
        print(f"  usable rolls        {n_rolled}/{n_items}")
        print(f"  bitwise identical   {ident.mean():.1%} of scored items")
        print(f"  win rate            {wins.mean():.4f}   (0.5 = blind)")
        print(f"  mean gap            {gaps.mean():+.5f}  [{lo:+.5f}, {hi:+.5f}]")
        print(f"  softplus(-gap)      {sp:.4f}   (chance = {np.log(2):.4f})")


if __name__ == "__main__":
    main()
