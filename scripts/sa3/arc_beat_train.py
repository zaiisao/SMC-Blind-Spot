"""ARC-style adversarial post-training of the SA3 generator + beat ControlNet
against a BEAT-CONDITIONAL discriminator.

What is trained (small-parameter, not the full model):
    * the beat ControlNet adapter -- the component that knows where the grid is
    * optionally the per-layer `modular_local_embeds.beat` projections of the
      generator (--train-beat-proj)
Everything else in the generator is frozen. A checkpoint therefore holds only
those tensors and is loadable by `sa3_beat.generate.load_adapter`, i.e. it can
be handed straight to scripts/sa3/arc_eval.py as `--test-ckpt`.

The discriminator is a SECOND copy of the model built from the same config
PATCHED with the beat conditioner (sa3_beat.build.patch_diffusion_config), so
the grid reaches it through per-layer modular_local_cond projections. This is
deliberate: the generator's ControlNet is attached by forward hooks, and ARC
runs the discriminator under `checkpoint()`, which recomputes layers in
backward -- a hook-held residual dict would leak the last forward's
conditioning into the recomputation. Only the discriminator's beat projections
and its scoring head start from random weights.

Negatives, and what a null looks like:
    fake      the generator's own denoised latent, same grid          (ARC's usual)
    wrong     REAL audio scored under a NEIGHBOUR's grid, re-anchored to this
              item's crop (arc_data.roll_beat_metadata). This is the term that
              forces the discriminator to judge placement rather than timbre.
    If the discriminator cannot see the grid, the wrong-grid score equals the
    real score bitwise, train/contrastive_loss_dis pins at softplus(0)=0.6931
    and nothing about beats is being learned. --blind-check asserts this is not
    the case before any optimizer step, and --broken-contrastive reproduces it
    on purpose as the deliberately-broken control.

Fold 0 of data/pools/slow_under70.json never enters training: arc_data excludes
it unconditionally on the real side and drops generated clips whose source is a
fold-0 recording.

Examples
    # 20-step smoke run, no checkpoints kept
    python scripts/sa3/arc_beat_train.py --gpu 3 --max-steps 20 \
        --out-dir /disk1/jaehoon/arc_runs/smoke --blind-check
    # real run
    python scripts/sa3/arc_beat_train.py --gpu 3 --max-steps 4000 \
        --batch-size 4 --lr 1e-5 --disc-lr 1e-4 --ckpt-every 500 \
        --out-dir /disk1/jaehoon/arc_runs/r1
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
from sa3_beat.build import (BEAT_COND_ID, attach_beat_conditioner,  # noqa: E402
                            load_beat_model, patch_diffusion_config)

SA3_DIR = Path("/disk1/jaehoon/sa3_small_music")
DEFAULT_ADAPTER = ("/disk1/jaehoon/sa3_beat_runs/cn_fold0_rw/"
                   "beat_adapter_final.safetensors")


# --------------------------------------------------------------------------
# conditioning plumbing
# --------------------------------------------------------------------------

def wrap_conditioner_with_inpaint(model, io_channels):
    """`local_add_cond_ids` is ['inpaint_mask', 'inpaint_masked_input'], so the
    DiT wrapper KeyErrors unless those tensors are present. Training here is
    never an inpainting task, so supply the all-zero (nothing masked) case, the
    same thing generation does."""
    cond = model.conditioner
    if getattr(cond, "_inpaint_wrapped", False):
        return model
    original = cond.forward

    def forward(metadata, device):
        out = original(metadata, device)
        b = len(metadata)
        n = out[BEAT_COND_ID][0].shape[-1]
        out["inpaint_mask"] = [torch.zeros(b, 1, n, device=device), None]
        out["inpaint_masked_input"] = [
            torch.zeros(b, io_channels, n, device=device), None]
        return out

    cond.forward = forward
    cond._inpaint_wrapped = True
    return model


def init_beat_projections(module, std, cond_id=BEAT_COND_ID):
    """The beat projections are ZERO-initialised on their output layer, by
    design, so that adding a new conditioning id cannot change a pretrained
    model. For the DISCRIMINATOR that is wrong: at init it is exactly blind to
    the grid (score(true) == score(wrong) bitwise), the contrastive term is
    identically 0.6931 and only the gradient through the zero weight can break
    the symmetry. There is nothing to preserve in a discriminator, so give the
    output layer a small random init instead."""
    from sa3_beat.build import beat_projection_modules
    n = 0
    for _, proj in beat_projection_modules(module, cond_id):
        last = proj[-1]
        torch.nn.init.normal_(last.weight, std=std)
        torch.nn.init.zeros_(last.bias)
        n += 1
    if n == 0:
        raise SystemExit(f"no modular_local_embeds.{cond_id} found on the "
                         "discriminator -- was its config patched?")
    return n


def build_discriminator(device, include_downbeat=True, beat_init_std=0.02):
    """Second copy of the model, built from the beat-patched config so the grid
    reaches it as modular local conditioning at every layer."""
    from safetensors.torch import load_file
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict
    from sa3_beat.conditioners import BeatConditioner

    cond_dim = BeatConditioner(include_downbeat=include_downbeat).cond_dim
    config = patch_diffusion_config(
        json.loads((SA3_DIR / "model_config.json").read_text()), cond_dim)
    disc = create_model_from_config(config)
    copy_state_dict(disc, load_file(str(SA3_DIR / "model.safetensors")))
    attach_beat_conditioner(disc, include_downbeat=include_downbeat)
    n_proj = init_beat_projections(disc.model, beat_init_std)
    print(f"[disc] re-initialised {n_proj} beat projection stack(s) "
          f"(std={beat_init_std}) -- they ship zero-init, i.e. blind")
    disc.pretransform = None      # latents only; never decodes
    wrap_conditioner_with_inpaint(disc, config["model"]["io_channels"])
    return disc.to(device), config


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------

def adapter_state(model, base_keys):
    """The FULL adapter tensor set, keyed the way `load_adapter` expects:
    ControlNet tensors under 'beat_controlnet.*', everything else relative to
    `model.model` (the DiT wrapper).

    Every key of the adapter we started from is written back, trained or not.
    Saving only the trainable tensors looks equivalent and is not: the released
    adapter carries 80 generator beat-projection tensors that this run freezes,
    and `load_adapter` reports a checkpoint without them as "0 adapter tensors,
    0 unexpected" and moves on. A candidate checkpoint would then silently run
    with PRETRAINED beat projections while the control ran with the released
    ones, so the two would differ for a reason that has nothing to do with
    training. Only the ControlNet block is missing-key checked; this half is
    not, so it has to be got right here.
    """
    sd = model.model.state_dict()
    out = {}
    for k in base_keys:
        src = ("model." + k) if k.startswith("beat_controlnet.") else k
        if src not in sd:
            raise SystemExit(f"adapter key {k!r} has no counterpart {src!r} in "
                             "the live model -- checkpoint would be incomplete")
        out[k] = sd[src].detach().float().cpu().contiguous()
    return out


def save_discriminator(wrapper, path):
    """Write the discriminator copy AND its scoring head.

    Round 1 did not do this, and only found out afterwards: its
    step_002000.safetensors holds 352 GENERATOR tensors (beat_controlnet.* and
    the generator's beat projections) and nothing of the discriminator, so the
    question "did the trained D learn to tell a right grid from a wrong one?"
    could not be asked of that run at all. The discriminator is where the
    conditional signal lives; it has to be recoverable.
    """
    from safetensors.torch import save_file
    sd = {"disc." + k: v.detach().float().cpu().contiguous()
          for k, v in wrapper.discriminator.state_dict().items()}
    sd.update({"head." + k: v.detach().float().cpu().contiguous()
               for k, v in wrapper.discriminator_head.state_dict().items()})
    save_file(sd, str(path))
    print(f"[ckpt] discriminator -> {path}", flush=True)


class PeriodicCheckpoint:
    def __init__(self, model, out_dir, every, base_keys):
        self.model, self.out_dir, self.every = model, Path(out_dir), every
        self.base_keys = base_keys
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, step):
        from safetensors.torch import save_file
        path = self.out_dir / f"step_{step:06d}.safetensors"
        save_file(adapter_state(self.model, self.base_keys), str(path))
        print(f"[ckpt] {path}", flush=True)
        return path


# --------------------------------------------------------------------------
# lightning glue
# --------------------------------------------------------------------------

def grad_report(model, disc, wrapper, tag):
    """Which parameter groups actually received gradient this step.

    Reports, per group: how many tensors have a non-None .grad, how many have a
    non-zero one, and the global grad norm. The frozen generator must show 0/N
    non-None -- if it does not, `requires_grad_(False)` did not take and we are
    silently training the whole backbone.
    """
    def stat(named):
        named = list(named)
        got = [(n, p) for n, p in named if p.grad is not None]
        zero = [n for n, p in got if float(p.grad.abs().max()) == 0]
        norm = float(torch.sqrt(sum((p.grad.float() ** 2).sum()
                                    for _, p in got))) if got else 0.0
        none = [n for n, p in named if p.grad is None]
        s = (f"{len(got)-len(zero)}nz/{len(got)}grad/{len(named)} "
             f"|g|={norm:.3e}")
        if zero:
            s += f"  zero-grad e.g. {zero[:2]}"
        if none:
            s += f"  no-grad e.g. {none[:2]}"
        return s

    train_ids = {id(p) for p in wrapper._gen_params}
    gen_train = [(n, p) for n, p in model.model.named_parameters()
                 if id(p) in train_ids]
    gen_froz = [(n, p) for n, p in model.model.named_parameters()
                if id(p) not in train_ids]
    disc_head = list(wrapper.discriminator_head.named_parameters())
    head_ids = {id(p) for _, p in disc_head}
    from sa3_beat.build import beat_projection_modules, BEAT_COND_ID
    dbeat = [(f"{k}.{n}", p)
             for k, m in beat_projection_modules(disc.model, BEAT_COND_ID)
             for n, p in m.named_parameters()]
    beat_ids = {id(p) for _, p in dbeat}
    disc_body = [(n, p) for n, p in disc.named_parameters()
                 if id(p) not in head_ids and id(p) not in beat_ids]
    print(f"[grad {tag}] gen_trainable   {stat(gen_train)}\n"
          f"[grad {tag}] gen_frozen      {stat(gen_froz)}\n"
          f"[grad {tag}] disc_beat_proj  {stat(dbeat)}\n"
          f"[grad {tag}] disc_backbone   {stat(disc_body)}\n"
          f"[grad {tag}] disc_head       {stat(disc_head)}", flush=True)


def make_callbacks(pl, model, args, wrapper, disc=None, base_keys=()):
    saver = PeriodicCheckpoint(model, args.out_dir, args.ckpt_every, base_keys)

    class Cb(pl.Callback):
        def on_train_batch_start(self, trainer, module, batch, idx):
            if trainer.global_step < args.shape_report:
                reals, md = batch
                print(f"[shape step {trainer.global_step}] reals={tuple(reals.shape)} "
                      f"dtype={reals.dtype} finite={bool(torch.isfinite(reals).all())} "
                      f"std={float(reals.float().std()):.3f}  n_md={len(md)} "
                      f"beats0={len(md[0]['beat']['beats'])} "
                      f"pad0={tuple(md[0]['padding_mask'][0].shape)}", flush=True)

        def on_train_batch_end(self, trainer, module, outputs, batch, idx):
            step = trainer.global_step
            if disc is not None and step <= args.grad_audit:
                grad_report(model, disc, wrapper, f"step {step}")
            m = {k: float(v) for k, v in trainer.callback_metrics.items()}
            if step % args.log_every == 0:
                keys = ["train/gen_loss", "train/adv_loss",
                        "train/denoiser_anchor", "train/discriminator_loss",
                        "train/dis_loss", "train/contrastive_loss_dis",
                        "train/disc_acc_real", "train/disc_acc_fake",
                        "train/disc_acc_pairwise",
                        "disc_score_gap", "disc_score_center"]
                s = "  ".join(f"{k.split('/')[-1]}={m[k]:.4f}"
                              for k in keys if k in m)
                print(f"[step {step:6d}] {s}", flush=True)
            if args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
                saver.save(step)
                if args.save_disc:
                    save_discriminator(
                        module, Path(args.out_dir) / f"disc_{step:06d}.safetensors")

        def on_train_end(self, trainer, module):
            saver.save(trainer.global_step)
            if args.save_disc:
                save_discriminator(
                    module,
                    Path(args.out_dir) / f"disc_{trainer.global_step:06d}.safetensors")

    return [Cb()]


def blind_check(wrapper, batch, device):
    """Assert the discriminator actually sees the beat grid.

    NULL (what today's stock wiring gives): the score under the true grid and
    the score under a half-IBI-shifted grid are bitwise identical, so the
    contrastive term has zero gradient. A non-zero difference only proves the
    tensor reaches the discriminator -- the beat projections start random, so
    the size of the difference means nothing.
    """
    reals, metadata = batch
    reals = reals.to(device)
    lat = reals / wrapper.diffusion.pretransform.scale
    t = torch.full((reals.shape[0],), 0.5, device=device)
    x = lat * 0.5 + torch.randn_like(lat) * 0.5

    shifted = []
    for md in metadata:
        b = np.asarray(md["beat"]["beats"], dtype=np.float64)
        ibi = float(np.median(np.diff(b))) if b.size > 1 else 0.5
        m = dict(md)
        m["beat"] = dict(md["beat"])
        m["beat"]["beats"] = b + 0.5 * ibi
        db = md["beat"]["downbeats"]
        m["beat"]["downbeats"] = None if db is None else np.asarray(db) + 0.5 * ibi
        shifted.append(m)

    with torch.no_grad():
        s_true = wrapper._disc_scores(
            x, t, wrapper.discriminator.conditioner(metadata, device))[0]
        s_shift = wrapper._disc_scores(
            x, t, wrapper.discriminator.conditioner(shifted, device))[0]
    d = float((s_true - s_shift).abs().mean())
    print(f"[blind-check] mean |score(true grid) - score(half-IBI shift)| "
          f"= {d:.3e}", flush=True)
    if d == 0.0:
        raise SystemExit(
            "[blind-check] FAILED: the discriminator is blind to the beat grid "
            "-- the contrastive negative is a no-op and training it would be "
            "measuring nothing.")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=None,
                    help="alias for --max-steps")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--disc-lr", type=float, default=1e-4)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resume", default=None,
                    help="checkpoint of trainable tensors to start from "
                         "(default: the released ControlNet adapter)")
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--train-beat-proj", action="store_true",
                    help="also train the generator's modular_local_embeds.beat "
                         "projections (they are unused by the ControlNet path "
                         "and start at their pretrained values)")
    ap.add_argument("--anchor-weight", type=float, default=1.0,
                    help="reconstruction anchor on the generator's x0 "
                         "prediction. 0 = purely adversarial, which leaves "
                         "nothing holding beat PLACEMENT in place (GUARD 1)")
    ap.add_argument("--contrastive-beta", type=float, default=1.0)
    ap.add_argument("--contrastive-weight", type=float, default=1.0,
                    help="Weight on the contrastive (wrong-grid) discriminator "
                         "term relative to the real-vs-fake term. ARC reads it "
                         "from discriminator.weights.contrastive.")
    ap.add_argument("--disc-update-interval", type=int, default=2)
    ap.add_argument("--logit-center-weight", type=float, default=0.0,
                    help="penalty on (score_real + score_fake)^2. The shipped "
                         "config omits it, so it defaults to 0 and the "
                         "relativistic loss constrains only the DIFFERENCE -- "
                         "measured drift of disc_score_center over 20 steps "
                         "was -0.6 -> +18.7 -> -11.3, which makes disc_acc_real "
                         "and disc_acc_fake read the drift rather than any "
                         "separation. 0 keeps the shipped behaviour.")
    ap.add_argument("--dilations", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16],
                    help="discriminator receptive field. The shipped [1]*5 is "
                         "~2.5 s at the 10.77 Hz latent rate, under 3 beats at "
                         "60 bpm -- too short to judge periodicity.")
    ap.add_argument("--gen-latent-dirs", nargs="*", default=[],
                    help="cached latents of generated clips to add as extra "
                         "negatives (kind='gen')")
    ap.add_argument("--gen-weight", type=float, default=0.0)
    ap.add_argument("--broken-contrastive", action="store_true",
                    help="CONTROL: the 'wrong grid' negative is the item's OWN "
                         "grid, so the contrastive term is provably empty. Any "
                         "improvement that survives this is not coming from "
                         "grid-conditional discrimination.")
    ap.add_argument("--disc-beat-init", type=float, default=0.02,
                    help="std of the discriminator's beat projection output "
                         "layer; 0 reproduces the shipped zero-init, which is "
                         "blind at step 0")
    ap.add_argument("--resume-disc", default=None,
                    help="warm-start the discriminator (and its scoring head) "
                         "from a save_discriminator()/arc_grid_probe "
                         "checkpoint: 'disc.*' keys go to the D copy, 'head.*' "
                         "to the scoring head. Round 2 showed "
                         "contrastive_loss_dis pinned at 5*ln2 for all 2000 "
                         "steps, i.e. D never learned the grid axis while also "
                         "fighting the real/fake axis; the probe showed it CAN "
                         "learn it alone. This is how that D is carried in.")
    ap.add_argument("--save-disc", action="store_true", default=True,
                    help="also checkpoint the discriminator and its head "
                         "alongside each adapter checkpoint. Round 1 saved "
                         "only the generator adapter, so its discriminator "
                         "cannot be re-probed; on by default now.")
    ap.add_argument("--no-save-disc", dest="save_disc", action="store_false")
    ap.add_argument("--dis-t-max", type=float, default=None,
                    help="cap the noise level the DISCRIMINATOR judges at "
                         "(applied after dist_shift, clamp). The stock schedule "
                         "puts the median at 0.62 noise, where beat placement is "
                         "mostly destroyed but timbre survives -- so the "
                         "real-vs-fake job stays easy while the beat check is "
                         "near-impossible, and the easy job's gradient "
                         "overwrites the beat skill (measured: warm-started "
                         "win rate 0.62 -> 0.50 within 1000 steps in r5). "
                         "Generator noise levels are untouched.")
    ap.add_argument("--blind-check", action="store_true")
    ap.add_argument("--grad-audit", type=int, default=0,
                    help="report per-group gradient status for the first N steps")
    ap.add_argument("--shape-report", type=int, default=0,
                    help="print the shapes of the first N batches")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.steps is not None:
        a.max_steps = a.steps

    import pytorch_lightning as pl
    from stable_audio_tools.training.arc import ARCTrainingWrapper

    torch.manual_seed(a.seed)
    device = f"cuda:{a.gpu}"
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- generator -------------------------------------------------------
    model, config = load_beat_model(device=device)
    from sa3_beat.generate import load_adapter, _patch_conditioning_inputs
    adapter_path = a.resume or a.adapter
    load_adapter(model, adapter_path, device)
    from safetensors.torch import load_file as _load_file
    base_keys = sorted(_load_file(adapter_path).keys())
    print(f"[gen] adapter {adapter_path} defines {len(base_keys)} tensors; "
          f"every checkpoint will carry all of them", flush=True)
    _patch_conditioning_inputs(model)
    wrap_conditioner_with_inpaint(model, config["model"]["io_channels"])

    for p in model.model.parameters():
        p.requires_grad_(False)
    cn = getattr(model.model.model, "beat_controlnet", None)
    if cn is None:
        raise SystemExit("no beat ControlNet on the generator -- nothing to train")
    for p in cn.parameters():
        p.requires_grad_(True)
    if a.train_beat_proj:
        from sa3_beat.build import beat_parameters
        for p in beat_parameters(model.model):
            p.requires_grad_(True)
    trainable = [p for p in model.model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.model.parameters())
    print(f"[gen] trainable {n_train/1e6:.2f}M / {n_total/1e6:.1f}M params "
          f"({100*n_train/n_total:.2f}%)", flush=True)
    model.model.train()

    # ---- discriminator ---------------------------------------------------
    disc, _ = build_discriminator(device, beat_init_std=a.disc_beat_init)
    print(f"[disc] separate copy, beat conditioning via modular_local_cond "
          f"at every layer", flush=True)

    # ---- data ------------------------------------------------------------
    items, missing = arc_data.real_items()
    print(f"[data] {len(items)} real cached tracks (fold "
          f"{arc_data.HELD_OUT_FOLD} excluded), {len(missing)} pool tracks "
          f"without latents", flush=True)
    weights = {}
    for d in a.gen_latent_dirs:
        gi = arc_data.gen_items(d)
        gi, dropped = arc_data.filter_gen_by_source(gi, d)
        print(f"[data] {len(gi)} generated clips from {d} "
              f"({len(dropped)} dropped: fold-0 source)", flush=True)
        items += gi
        weights["gen"] = a.gen_weight
    if any(it["fold"] == arc_data.HELD_OUT_FOLD for it in items
           if it["fold"] is not None):
        raise SystemExit("fold-0 material reached the training set -- abort")

    ds = arc_data.BeatARCDataset(items, seed=a.seed)
    sampler = (arc_data.make_sampler(items, weights, seed=a.seed)
               if weights else None)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size, sampler=sampler,
        shuffle=(sampler is None), collate_fn=arc_data.collate,
        num_workers=2, drop_last=True, persistent_workers=True)

    # ---- ARC config ------------------------------------------------------
    arc_config = json.loads(
        (SA3_DIR / "model_config.json").read_text())["training"]["arc"]
    arc_config.pop("use_model_as_discriminator", None)
    arc_config.pop("discriminator_base_ckpt", None)
    arc_config["denoiser_loss_weight"] = a.anchor_weight
    arc_config["disc_update_interval"] = a.disc_update_interval
    dc = arc_config["discriminator"]
    dc["config"]["dilations"] = list(a.dilations)
    dc["contrastive"] = True
    dc["contrastive_beta"] = a.contrastive_beta
    dc["logit_center_weight"] = a.logit_center_weight
    dc.setdefault("weights", {})["contrastive"] = a.contrastive_weight

    def opt(lr):
        return {"optimizer": {"type": "AdamW",
                              "config": {"lr": lr, "betas": [0.9, 0.99],
                                         "weight_decay": 0.001}}}

    wrapper = ARCTrainingWrapper(
        model, arc_config,
        {"diffusion": opt(a.lr), "discriminator": opt(a.disc_lr)},
        discriminator=disc, use_ema=False, pre_encoded=True,
        sample_rate=config["sample_rate"],
        sample_size=int(arc_data.CROP * model.pretransform.downsampling_ratio),
    )
    # Optimize only what we chose to train (the default sweeps in every frozen
    # parameter of the generator).
    wrapper._gen_params = trainable

    if a.dis_t_max is not None:
        # Clamp only the discriminator's t draws; sample_t_with_shift is shared
        # with the generator path, so key on which noise dist was passed in.
        _orig_sample = wrapper.sample_t_with_shift

        def _capped(fn, *args, **kw):
            t = _orig_sample(fn, *args, **kw)
            if fn is wrapper.dis_noise_dist:
                t = t.clamp(max=a.dis_t_max)
            return t

        wrapper.sample_t_with_shift = _capped
        print(f"[dis-t-max] discriminator judges at t <= {a.dis_t_max} "
              f"(generator schedule unchanged)", flush=True)

    if a.resume_disc:
        from safetensors.torch import load_file as _lf
        _sd = _lf(a.resume_disc)
        d_sd = {k[5:]: v for k, v in _sd.items() if k.startswith("disc.")}
        h_sd = {k[5:]: v for k, v in _sd.items() if k.startswith("head.")}
        if not d_sd or not h_sd:
            raise SystemExit(
                f"--resume-disc {a.resume_disc}: expected 'disc.*' and "
                f"'head.*' keys, got {len(d_sd)} and {len(h_sd)}")
        md = wrapper.discriminator.load_state_dict(d_sd, strict=False)
        mh = wrapper.discriminator_head.load_state_dict(h_sd, strict=False)
        if md.unexpected_keys or mh.unexpected_keys:
            raise SystemExit(
                f"unexpected keys in {a.resume_disc}: "
                f"{md.unexpected_keys[:5]} {mh.unexpected_keys[:5]}")
        if md.missing_keys or mh.missing_keys:
            raise SystemExit(
                f"--resume-disc {a.resume_disc} is incomplete: "
                f"{len(md.missing_keys)} disc and {len(mh.missing_keys)} head "
                f"keys left at init -- a partly-warm D is not a control")
        print(f"[resume-disc] {len(d_sd)} disc + {len(h_sd)} head tensors "
              f"from {a.resume_disc}, 0 missing, 0 unexpected", flush=True)

    # ARC accumulates metrics in a StaggeredLogger and flushes the WINDOW MEAN
    # every 10 steps; between flushes trainer.callback_metrics keeps the last
    # flushed value, so printing every step would repeat a stale average ten
    # times and hide exactly the early-step behaviour a smoke test is looking
    # for. Tie the flush interval to --log-every so every printed line is the
    # step it is labelled with.
    wrapper._staggered_logger.every_n_steps = max(1, a.log_every)

    # The wrong-grid negative. Without this hook ARC rolls only 'prompt', which
    # our beat conditioning does not travel in.
    if a.broken_contrastive:
        print("[control] BROKEN contrastive: negatives keep their own grid",
              flush=True)
        wrapper.contrastive_roll_fn = lambda md: [dict(m) for m in md]
    else:
        def roll(md):
            rolled, unrolled = arc_data.roll_beat_metadata(md)
            if unrolled:
                print(f"[roll] {len(unrolled)}/{len(md)} items had no "
                      f"distinguishable partner this batch", flush=True)
            return rolled
        wrapper.contrastive_roll_fn = roll

    if a.blind_check:
        wrapper.to(device)
        batch = arc_data.collate([ds[i] for i in range(a.batch_size)])
        blind_check(wrapper, batch, device)

    (out_dir / "run_config.json").write_text(json.dumps(
        {"args": vars(a), "arc": arc_config, "n_items": len(items),
         "trainable_params": n_train}, indent=1, default=str))

    trainer = pl.Trainer(
        devices=[a.gpu], accelerator="gpu", precision="32-true",
        max_steps=a.max_steps, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, log_every_n_steps=a.log_every,
        callbacks=make_callbacks(pl, model, a, wrapper, disc=disc,
                                 base_keys=base_keys),
    )
    trainer.fit(wrapper, loader)


if __name__ == "__main__":
    main()
