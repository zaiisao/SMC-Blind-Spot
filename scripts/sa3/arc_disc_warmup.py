"""Is the beat-conditional task learnable at all?

In r1/r2/r3 the discriminator's contrastive term never left its chance value.
Two readings: the task is unlearnable at the noise the discriminator works at,
or it was learnable but drowned -- in r2/r3 the contrastive term was weighted 5x
and made up ~83% of the discriminator loss while real-vs-fake collapsed to
chance, so both objectives failed together.

This isolates the question. No generator, no real-vs-fake, no adversarial
dynamics: the discriminator is trained on ONE supervised binary task, real audio
under its own beat grid versus the same audio under a neighbour's grid
(arc_data.roll_beat_metadata, the exact negative used in ARC). Loss is the same
relativistic softplus, so the numbers are directly comparable to
train/contrastive_loss_dis in the ARC logs; chance is ln2.

If this stays at chance with all of the discriminator's capacity on it and
nothing competing, the score-head representation does not encode onset
placement and no conditioning design fixes that. If it drops, the task is
learnable, the ARC failure was in the setup, and this checkpoint is a warm start.

Held-out fold 0 never enters training (arc_data excludes it); --val-every scores
a fixed set of held-out batches so a drop in train loss cannot be memorisation.

Usage:
    arc_disc_warmup.py --gpu 3 --steps 2000 --out-dir /disk1/jaehoon/arc_runs/warm
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--disc-lr", type=float, default=1e-4)
    ap.add_argument("--disc-beat-init", type=float, default=0.02)
    ap.add_argument("--t-max", type=float, default=1.0,
                    help="cap on the sampled noise level. The ARC schedule puts "
                         "the median at 0.62 and only 3%% of samples below 0.2; "
                         "if beats are only legible in clean audio the task can "
                         "look unlearnable purely because of that. 1.0 keeps the "
                         "ARC distribution untouched.")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-batches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapter", default=abt.DEFAULT_ADAPTER)
    a = ap.parse_args()

    from stable_audio_tools.training.arc import ARCTrainingWrapper

    torch.manual_seed(a.seed)
    device = f"cuda:{a.gpu}"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "run_config.json", "w"), indent=2)

    # generator is required by the wrapper's constructor and is never stepped
    model, config = load_beat_model(device=device)
    from sa3_beat.generate import load_adapter, _patch_conditioning_inputs
    load_adapter(model, a.adapter, device)
    _patch_conditioning_inputs(model)
    abt.wrap_conditioner_with_inpaint(model, config["model"]["io_channels"])
    for p in model.model.parameters():
        p.requires_grad_(False)

    disc, _ = abt.build_discriminator(device, beat_init_std=a.disc_beat_init)

    arc_config = json.loads(
        (SA3_DIR / "model_config.json").read_text())["training"]["arc"]
    arc_config.pop("use_model_as_discriminator", None)
    arc_config.pop("discriminator_base_ckpt", None)
    arc_config["discriminator"]["contrastive"] = True
    o = {"optimizer": {"type": "AdamW", "config": {"lr": a.disc_lr}}}
    wrapper = ARCTrainingWrapper(
        model, arc_config, {"diffusion": o, "discriminator": o},
        discriminator=disc, use_ema=False, pre_encoded=True,
        sample_rate=config["sample_rate"],
        sample_size=int(arc_data.CROP * model.pretransform.downsampling_ratio),
    ).to(device)

    params = ([p for p in wrapper.discriminator.parameters() if p.requires_grad]
              + [p for p in wrapper.discriminator_head.parameters()])
    optim = torch.optim.AdamW(params, lr=a.disc_lr, betas=(0.9, 0.99),
                              weight_decay=0.001)
    print(f"[disc] optimising {sum(p.numel() for p in params)/1e6:.1f}M params",
          flush=True)

    items, _ = arc_data.real_items()
    ds = arc_data.BeatARCDataset(items, seed=a.seed)
    n_val = max(a.val_batches * a.batch_size, 1)
    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(len(ds), generator=g).tolist()
    val_ds = torch.utils.data.Subset(ds, perm[:n_val])
    tr_ds = torch.utils.data.Subset(ds, perm[n_val:])
    mk = lambda d, sh: torch.utils.data.DataLoader(
        d, batch_size=a.batch_size, shuffle=sh, collate_fn=arc_data.collate,
        num_workers=2, drop_last=True, persistent_workers=True)
    tr, va = mk(tr_ds, True), mk(val_ds, False)
    print(f"[data] {len(tr_ds)} train / {len(val_ds)} val crops "
          f"(fold {arc_data.HELD_OUT_FOLD} excluded from both)", flush=True)

    tgen = torch.Generator(device="cpu").manual_seed(a.seed)

    def batch_loss(reals, metadata, train):
        reals = reals.to(device)
        rolled, unrolled = arc_data.roll_beat_metadata(metadata)
        if len(unrolled) == len(metadata):
            return None, None
        noise = torch.randn(reals.shape, generator=tgen).to(device)
        t = wrapper.sample_t_with_shift(
            wrapper.dis_noise_dist, reals.shape[0], reals.shape[2], None)
        t = t.clamp(max=a.t_max)
        x_t = reals * (1 - t)[:, None, None] + noise * t[:, None, None]
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            c_true = wrapper.discriminator.conditioner(metadata, device)
            c_wrong = wrapper.discriminator.conditioner(rolled, device)
            s_true = wrapper._disc_scores(x_t, t, c_true)
            s_wrong = wrapper._disc_scores(x_t, t, c_wrong)
            loss = wrapper.calculate_disc_loss(s_true, s_wrong)
        with torch.no_grad():
            acc = torch.stack([(a_ > b_).float().mean()
                               for a_, b_ in zip(s_true, s_wrong)]).mean()
        return loss, float(acc)

    chance = float(np.log(2))
    print(f"[task] chance loss = {chance:.4f}, chance acc = 0.5", flush=True)
    step, run_l, run_a, n = 0, 0.0, 0.0, 0
    best = None
    while step < a.steps:
        for reals, metadata in tr:
            if step >= a.steps:
                break
            wrapper.train()
            loss, acc = batch_loss(reals, metadata, True)
            if loss is None:
                continue
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            run_l += float(loss); run_a += acc; n += 1
            step += 1

            if step % a.log_every == 0:
                print(f"  step {step:5d}  train_loss {run_l/n:.4f}  "
                      f"train_acc {run_a/n:.4f}   (chance {chance:.4f}/0.5)",
                      flush=True)
                run_l = run_a = 0.0; n = 0

            if step % a.val_every == 0:
                wrapper.eval()
                vl, vac, m = 0.0, 0.0, 0
                for vr, vm in va:
                    l, ac = batch_loss(vr, vm, False)
                    if l is None:
                        continue
                    vl += float(l); vac += ac; m += 1
                if m:
                    vl, vac = vl/m, vac/m
                    flag = "  <-- below chance" if vl < chance - 0.005 else ""
                    print(f"  step {step:5d}  VAL loss {vl:.4f}  acc {vac:.4f}"
                          f"{flag}", flush=True)
                    if best is None or vl < best:
                        best = vl
                        from safetensors.torch import save_file
                        sd = {f"disc.{k}": v.detach().cpu().contiguous()
                              for k, v in wrapper.discriminator.state_dict().items()}
                        sd.update({f"head.{k}": v.detach().cpu().contiguous()
                                   for k, v in wrapper.discriminator_head.state_dict().items()})
                        save_file(sd, str(out / "disc_best.safetensors"))

    print(f"\nbest val loss {best:.4f} vs chance {chance:.4f} "
          f"-> {'LEARNABLE' if best is not None and best < chance - 0.005 else 'AT CHANCE'}")


if __name__ == "__main__":
    main()
