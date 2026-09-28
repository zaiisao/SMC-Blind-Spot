"""Grid-sensitivity probe for the ARC beat-conditional discriminator.

This is a GATE, not a training round. It answers one question directly:

    can the discriminator tell a CORRECT beat grid from a WRONG one?

A fixed batch of REAL latents is scored under (a) its own grid and (b) a
wrong grid, and the separation is reported. Arms:

    rolled   neighbour's grid, re-anchored (arc_data.roll_beat_metadata) --
             exactly the negative ARC's contrastive branch builds
    shift    the item's own grid moved by half an inter-beat interval
    noise    random beat times, same count
    broken   the item's OWN grid.  STATED NULL: separation must be exactly
             0.0, because the two forwards are bitwise identical.

Reported per arm, over n_items real items:
    mean |s_own - s_wrong|      raw score separation
    median (s_own - s_wrong)    signed; a useful D wants this > 0
    AUC                         P(s_own > s_wrong) across items, 0.5 = chance
    contrastive loss            softplus(-(s_own - s_wrong)); softplus(0)
                                = 0.6931 is the "learned nothing" value

--train-steps > 0 additionally trains ONLY the discriminator (its beat
projections + scoring head, or --train-all for the whole D copy) on the
contrastive objective alone, on a small FIXED pool of items, and re-probes.
This is the decisive test of whether the pathway can learn at all: if D
cannot separate own-grid from wrong-grid on a handful of items it has seen
hundreds of times, the conditional pathway is unworkable, not merely
mis-weighted.

Fold 0 never enters: arc_data.real_items() excludes it unconditionally.

Examples
    python scripts/sa3/arc_grid_probe.py --gpu 3 --out-dir OUT \
        --beat-init 0.0 0.02 0.2 --n-items 32
    python scripts/sa3/arc_grid_probe.py --gpu 3 --out-dir OUT \
        --beat-init 0.2 --train-steps 400 --pool-items 16
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

SA3_DIR = Path("/disk1/jaehoon/sa3_small_music")


# --------------------------------------------------------------------------
# wrong-grid constructions
# --------------------------------------------------------------------------

def _copy(md):
    m = dict(md)
    m["beat"] = dict(md["beat"])
    return m


def wrong_metadata(metadata, arm, rng):
    """Return the negative-conditioning metadata list for one arm."""
    if arm == "broken":
        return [_copy(m) for m in metadata]
    if arm == "rolled":
        rolled, _ = arc_data.roll_beat_metadata(metadata)
        return rolled
    out = []
    for md in metadata:
        b = np.asarray(md["beat"]["beats"], dtype=np.float64)
        m = _copy(md)
        db = md["beat"]["downbeats"]
        if arm == "shift":
            ibi = float(np.median(np.diff(b))) if b.size > 1 else 0.5
            m["beat"]["beats"] = b + 0.5 * ibi
            m["beat"]["downbeats"] = (None if db is None
                                      else np.asarray(db) + 0.5 * ibi)
        elif arm == "noise":
            span = float(b.max() - b.min()) if b.size > 1 else 20.0
            lo = float(b.min())
            m["beat"]["beats"] = np.sort(rng.uniform(lo, lo + span, size=b.size))
            m["beat"]["downbeats"] = (None if db is None else
                                      np.sort(rng.uniform(lo, lo + span,
                                                          size=len(db))))
        else:
            raise ValueError(arm)
        out.append(m)
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def item_scores(wrapper, x, t, metadata, device):
    """Mean score per item, (B,), from the discriminator's score map."""
    s = wrapper._disc_scores(x, t, wrapper.discriminator.conditioner(
        metadata, device))
    # relativistic head returns a list of (B,1,T) maps; average over scales
    return torch.stack([m.flatten(1).mean(1) for m in s]).mean(0)


def noised(wrapper, reals, t_val, device, gen):
    lat = reals.to(device) / wrapper.diffusion.pretransform.scale
    t = torch.full((lat.shape[0],), t_val, device=device)
    n = torch.randn(lat.shape, device=device, generator=gen)
    return lat * (1 - t_val) + n * t_val, t


def probe(wrapper, batches, arms, device, t_val, seed=0):
    """Score separation per arm. Uses the SAME noised x for every arm."""
    rng = np.random.default_rng(seed)
    acc = {a: {"own": [], "wrong": []} for a in arms}
    for reals, metadata in batches:
        gen = torch.Generator(device=device).manual_seed(seed)
        x, t = noised(wrapper, reals, t_val, device, gen)
        with torch.no_grad():
            s_own = item_scores(wrapper, x, t, metadata, device)
            for arm in arms:
                wmd = wrong_metadata(metadata, arm, rng)
                s_w = item_scores(wrapper, x, t, wmd, device)
                acc[arm]["own"].append(s_own.float().cpu().numpy())
                acc[arm]["wrong"].append(s_w.float().cpu().numpy())
    out = {}
    for arm in arms:
        o = np.concatenate(acc[arm]["own"])
        w = np.concatenate(acc[arm]["wrong"])
        d = o - w
        out[arm] = {
            "n": int(d.size),
            "mean_abs_sep": float(np.mean(np.abs(d))),
            "median_signed_sep": float(np.median(d)),
            "auc": float(np.mean(d > 0) + 0.5 * np.mean(d == 0)),
            "contrastive_loss": float(np.mean(np.log1p(np.exp(-np.clip(d, -30, 30))))),
            "score_own_mean": float(o.mean()),
            "score_wrong_mean": float(w.mean()),
        }
    return out


# --------------------------------------------------------------------------
# discriminator-only contrastive training
# --------------------------------------------------------------------------

def disc_train(wrapper, batches, args, device):
    from sa3_beat.build import beat_projection_modules, BEAT_COND_ID
    if args.train_all:
        params = [p for p in wrapper.discriminator.parameters()
                  if p.requires_grad]
    else:
        params = [p for _, m in beat_projection_modules(
            wrapper.discriminator.model, BEAT_COND_ID) for p in m.parameters()]
    params = params + list(wrapper.discriminator_head.parameters())
    n = sum(p.numel() for p in params)
    print(f"[dtrain] optimising {len(params)} tensors / {n/1e6:.2f}M params "
          f"({'whole D' if args.train_all else 'beat proj + head'})",
          flush=True)
    opt = torch.optim.AdamW(params, lr=args.disc_lr, betas=(0.9, 0.99),
                            weight_decay=1e-3)
    rng = np.random.default_rng(args.seed)
    hist = []
    for step in range(1, args.train_steps + 1):
        reals, metadata = batches[step % len(batches)]
        gen = torch.Generator(device=device).manual_seed(step)
        t_val = float(rng.uniform(0.2, 0.8)) if args.random_t else args.t
        x, t = noised(wrapper, reals, t_val, device, gen)
        s_own = wrapper._disc_scores(
            x, t, wrapper.discriminator.conditioner(metadata, device))
        wmd = wrong_metadata(metadata, args.train_arm, rng)
        s_w = wrapper._disc_scores(
            x, t, wrapper.discriminator.conditioner(wmd, device))
        loss = wrapper.calculate_disc_loss(s_own, s_w,
                                           beta=args.contrastive_beta)
        if args.logit_center_weight:
            c = torch.stack([(a + b).mean() for a, b in zip(s_own, s_w)]).mean()
            loss = loss + args.logit_center_weight * c.pow(2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1e9)
        opt.step()
        hist.append({"step": step, "loss": float(loss),
                     "grad_norm": float(gnorm)})
        if step % args.log_every == 0 or step == 1:
            print(f"[dtrain {step:5d}] contrastive={float(loss):.4f} "
                  f"|g|={float(gnorm):.3e}", flush=True)
    return hist


# --------------------------------------------------------------------------

def make_batches(ds, n_items, batch_size):
    idx = list(range(min(n_items, len(ds))))
    return [arc_data.collate([ds[i] for i in idx[k:k + batch_size]])
            for k in range(0, len(idx), batch_size)
            if len(idx[k:k + batch_size]) == batch_size]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-items", type=int, default=32,
                    help="items used for the static probe")
    ap.add_argument("--pool-items", type=int, default=16,
                    help="items the D-only training loop cycles over")
    ap.add_argument("--beat-init", type=float, nargs="+", default=[0.02],
                    help="std of the discriminator's beat projection output "
                         "layer; 0.0 reproduces the shipped zero-init")
    ap.add_argument("--arms", nargs="+",
                    default=["rolled", "shift", "noise", "broken"])
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--random-t", action="store_true")
    ap.add_argument("--load-disc", default=None,
                    help="safetensors written by arc_beat_train's "
                         "save_discriminator(): 'disc.*' keys go to the "
                         "discriminator copy, 'head.*' to the scoring head. "
                         "Use with --train-steps 0 to probe a TRAINED D.")
    ap.add_argument("--train-steps", type=int, default=0)
    ap.add_argument("--train-arm", default="rolled")
    ap.add_argument("--train-all", action="store_true",
                    help="optimise the whole discriminator copy, not just its "
                         "beat projections and head")
    ap.add_argument("--disc-lr", type=float, default=1e-4)
    ap.add_argument("--contrastive-beta", type=float, default=1.0)
    ap.add_argument("--logit-center-weight", type=float, default=0.0)
    ap.add_argument("--dilations", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16])
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-disc", action="store_true",
                    help="write the trained discriminator state (the round-1 "
                         "training script did NOT save it, which is why the "
                         "round-1 D cannot be re-probed)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from stable_audio_tools.training.arc import ARCTrainingWrapper
    from sa3_beat.build import load_beat_model
    from sa3_beat.generate import load_adapter, _patch_conditioning_inputs

    torch.manual_seed(a.seed)
    device = f"cuda:{a.gpu}"
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, config = load_beat_model(device=device)
    load_adapter(model, abt.DEFAULT_ADAPTER, device)
    _patch_conditioning_inputs(model)
    abt.wrap_conditioner_with_inpaint(model, config["model"]["io_channels"])
    model.model.eval()
    for p in model.model.parameters():
        p.requires_grad_(False)

    items, missing = arc_data.real_items()
    print(f"[data] {len(items)} real cached tracks (fold "
          f"{arc_data.HELD_OUT_FOLD} excluded), {len(missing)} without latents",
          flush=True)
    ds = arc_data.BeatARCDataset(items, seed=a.seed)

    arc_config = json.loads(
        (SA3_DIR / "model_config.json").read_text())["training"]["arc"]
    arc_config.pop("use_model_as_discriminator", None)
    arc_config.pop("discriminator_base_ckpt", None)
    dc = arc_config["discriminator"]
    dc["config"]["dilations"] = list(a.dilations)
    dc["contrastive"] = True

    results = {"args": vars(a), "configs": []}
    probe_batches = make_batches(ds, a.n_items, a.batch_size)
    train_batches = make_batches(ds, a.pool_items, a.batch_size)

    for init_std in a.beat_init:
        print(f"\n===== disc_beat_init = {init_std} =====", flush=True)
        torch.manual_seed(a.seed)
        disc, _ = abt.build_discriminator(device, beat_init_std=init_std)
        wrapper = ARCTrainingWrapper(
            model, arc_config,
            {"diffusion": {"optimizer": {"type": "AdamW",
                                         "config": {"lr": 1e-5}}},
             "discriminator": {"optimizer": {"type": "AdamW",
                                             "config": {"lr": a.disc_lr}}}},
            discriminator=disc, use_ema=False, pre_encoded=True,
            sample_rate=config["sample_rate"],
            sample_size=int(arc_data.CROP
                            * model.pretransform.downsampling_ratio),
        ).to(device)

        entry = {"beat_init": init_std}
        if a.load_disc:
            from safetensors.torch import load_file
            sd = load_file(a.load_disc)
            d_sd = {k[5:]: v for k, v in sd.items() if k.startswith("disc.")}
            h_sd = {k[5:]: v for k, v in sd.items() if k.startswith("head.")}
            md = wrapper.discriminator.load_state_dict(d_sd, strict=False)
            mh = wrapper.discriminator_head.load_state_dict(h_sd, strict=False)
            if md.unexpected_keys or mh.unexpected_keys:
                raise SystemExit(
                    f"unexpected keys in {a.load_disc}: "
                    f"{md.unexpected_keys[:5]} {mh.unexpected_keys[:5]}")
            print(f"[load-disc] {len(d_sd)} disc + {len(h_sd)} head tensors "
                  f"from {a.load_disc}; {len(md.missing_keys)} disc and "
                  f"{len(mh.missing_keys)} head keys left at init", flush=True)
            entry["load_disc"] = {"path": a.load_disc, "n_disc": len(d_sd),
                                  "n_head": len(h_sd),
                                  "missing_disc": len(md.missing_keys),
                                  "missing_head": len(mh.missing_keys)}
        entry["untrained"] = probe(wrapper, probe_batches, a.arms, device,
                                   a.t, seed=a.seed)
        for arm, v in entry["untrained"].items():
            print(f"[untrained {arm:7s}] |sep|={v['mean_abs_sep']:.4e} "
                  f"med={v['median_signed_sep']:+.4e} AUC={v['auc']:.3f} "
                  f"L={v['contrastive_loss']:.4f}", flush=True)

        if a.train_steps > 0:
            entry["train_hist"] = disc_train(wrapper, train_batches, a, device)
            entry["after_train_seen"] = probe(wrapper, train_batches, a.arms,
                                              device, a.t, seed=a.seed)
            entry["after_train_heldout"] = probe(wrapper, probe_batches[
                len(train_batches):] or probe_batches, a.arms, device, a.t,
                seed=a.seed)
            for tag in ("after_train_seen", "after_train_heldout"):
                for arm, v in entry[tag].items():
                    print(f"[{tag} {arm:7s}] |sep|={v['mean_abs_sep']:.4e} "
                          f"med={v['median_signed_sep']:+.4e} "
                          f"AUC={v['auc']:.3f} L={v['contrastive_loss']:.4f}",
                          flush=True)
            if a.save_disc:
                from safetensors.torch import save_file
                # 'disc.' prefix: this is the format --load-disc and the
                # trainer's --resume-disc read. Without it the file loads as
                # zero disc tensors and silently stays at init.
                sd = {"disc." + k: v.detach().float().cpu().contiguous()
                      for k, v in wrapper.discriminator.state_dict().items()}
                sd.update({"head." + k: v.detach().float().cpu().contiguous()
                           for k, v in
                           wrapper.discriminator_head.state_dict().items()})
                p = out_dir / f"disc_init{init_std}.safetensors"
                save_file(sd, str(p))
                print(f"[ckpt] discriminator -> {p}", flush=True)

        results["configs"].append(entry)
        del wrapper, disc
        torch.cuda.empty_cache()

    (out_dir / "probe.json").write_text(json.dumps(results, indent=1,
                                                   default=str))
    print(f"\n[done] {out_dir/'probe.json'}", flush=True)


if __name__ == "__main__":
    main()
