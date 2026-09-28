"""Fine-tune the beat ControlNet against a FROZEN Beat This tracker.

FIX 2 of the discriminator post-mortem. Instead of teaching a music-realism
discriminator to also check beat placement (which failed in r1/r2/r3/r5 -- the
easy real-vs-fake job overwrites the beat skill every time), use the judge we
already own: a trained Beat This checkpoint, frozen. No discriminator at all.

Per step, on real latents from the ARC pool:
  anchor    the model's own rectified-flow denoising loss (keeps it a music model)
  tracker   noise the latent to t ~ U(t_lo, t_hi) (t=0.4 is the SDEdit operating
            point), one-step denoise, decode THROUGH the VAE with gradients,
            resample to 22.05k, mel, frozen Beat This, ShiftTolerantBCELoss
            against the clip's own annotation. Gradient reaches only the
            ControlNet.

Known trap handled here: the pretransform ships with enable_grad off and a
nested no_grad in decode; go through `pre.model.decode(z * pre.scale)` after
setting enable_grad, or the tracker term silently has no gradient. The
--grad-check flag asserts a nonzero ControlNet gradient on the tracker term
alone before training starts, and the run refuses to continue without it.

Startup invariant (derive before GPU): decode REAL latents (no noise) and track
them; the mean F against their own annotations must clear --real-f-floor
(default 0.5). If clean real audio through this exact decode+resample+mel path
scores garbage, the pipeline is misaligned and training would optimize nonsense.

Honest caveat, stated up front: this optimizes audio to be legible to a Beat
This checkpoint, and the corpus would later train Beat This models. Circularity
is a real concern for the eventual paper; this run is to find out whether the
generator CAN be pulled toward beat-legible audio at all.

Usage:
    arc_tracker_ft.py --gpu 0 --steps 1500 --out-dir /disk1/jaehoon/arc_runs/tf1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
BT = REPO / "third-party" / "beat_this"
for p in (str(REPO), str(REPO / "third-party" / "stable-audio-tools"), str(BT)):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import arc_data  # noqa: E402
import arc_beat_train as abt  # noqa: E402
from sa3_beat.build import load_beat_model  # noqa: E402

BT_FPS = 50
BT_SR = 22050


def build_tracker(ckpt, device):
    from beat_this.model.pl_module import PLBeatThis
    m = PLBeatThis(
        spect_dim=128, fps=50, transformer_dim=512, ff_mult=4, n_layers=6,
        stem_dim=32, dropout={"frontend": 0.1, "transformer": 0.2}, lr=8e-4,
        weight_decay=0.01, pos_weights={"beat": 20, "downbeat": 86},
        head_dim=32, loss_type="shift_tolerant_weighted_bce",
        warmup_steps=1000, max_epochs=100,
    )
    m.load_state_dict(torch.load(ckpt, map_location="cpu",
                                 weights_only=False)["state_dict"], strict=False)
    m = m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def beat_target(meta, n_frames):
    """0/1 beat frames at 50 fps for one crop; ShiftTolerantBCELoss widens it."""
    b = meta["beat"]
    t0 = b["frame_offset"] / arc_data.LATENT_RATE
    beats = np.asarray(b["beats"], dtype=np.float64)
    idx = np.round((beats - t0) * BT_FPS).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    y = torch.zeros(n_frames)
    y[torch.as_tensor(idx, dtype=torch.long)] = 1.0
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--crop", type=int, default=107,
                    help="latent frames (~10 s); decode-with-grad memory scales with this")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--tracker-weight", type=float, default=1.0)
    ap.add_argument("--t-lo", type=float, default=0.3)
    ap.add_argument("--t-hi", type=float, default=0.5)
    ap.add_argument("--tracker-ckpt",
                    default="/disk1/jaehoon/ckpt_snapshots/vanilla_f0_ep099.ckpt")
    ap.add_argument("--adapter", default=abt.DEFAULT_ADAPTER)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--real-f-floor", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = f"cuda:{a.gpu}"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "run_config.json", "w"), indent=2)

    # ---- generator -------------------------------------------------------
    model, config = load_beat_model(device=device)
    from sa3_beat.generate import load_adapter, _patch_conditioning_inputs
    load_adapter(model, a.adapter, device)
    from safetensors.torch import load_file, save_file
    base_keys = sorted(load_file(a.adapter).keys())
    _patch_conditioning_inputs(model)
    abt.wrap_conditioner_with_inpaint(model, config["model"]["io_channels"])
    sr = config["sample_rate"]

    for p in model.model.parameters():
        p.requires_grad_(False)
    cn = getattr(model.model.model, "beat_controlnet", None)
    if cn is None:
        raise SystemExit("no beat ControlNet on the generator")
    for p in cn.parameters():
        p.requires_grad_(True)
    trainable = [p for p in model.model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.99),
                              weight_decay=0.001)
    print(f"[gen] training {sum(p.numel() for p in trainable)/1e6:.2f}M "
          f"ControlNet params", flush=True)

    # decode-with-grad path. The no_grad trap is NESTED: AutoencoderPretransform
    # gates on its own enable_grad, and the AudioAutoencoder inside it holds a
    # SECOND pretransform with a second enable_grad-gated no_grad
    # (autoencoders.py:904). Setting only the outer one leaves decode gradless
    # with no error. Walk the whole chain.
    pre = model.pretransform
    obj, depth = pre, 0
    while obj is not None and depth < 6:
        if hasattr(obj, "enable_grad"):
            obj.enable_grad = True
            print(f"[decode-grad] enable_grad=True on {type(obj).__name__}",
                  flush=True)
        obj = getattr(obj, "model", None) or getattr(obj, "pretransform", None)
        if obj is not None and not hasattr(obj, "enable_grad") \
                and hasattr(obj, "pretransform"):
            obj = obj.pretransform
        depth += 1
    inner = pre.model
    for p in inner.parameters():
        p.requires_grad_(False)

    def decode(z):
        return inner.decode(z * pre.scale)

    # ---- frozen tracker + loss -------------------------------------------
    tracker = build_tracker(a.tracker_ckpt, device)
    from beat_this.model.loss import ShiftTolerantBCELoss
    from beat_this.preprocessing import LogMelSpect
    st_loss = ShiftTolerantBCELoss(pos_weight=20)
    mel = LogMelSpect(BT_SR).to(device)
    import torchaudio

    def track_logits(z):
        """latent [B,C,N] -> beat logits [B,T] through decode/resample/mel."""
        audio = decode(z)                       # [B, ch, samples] at sr
        audio = audio.mean(1)
        if sr != BT_SR:
            audio = torchaudio.functional.resample(audio, sr, BT_SR)
        specs = torch.stack([mel(w) for w in audio])
        return tracker.model(specs)["beat"]

    def denoise_step(x_t, t, conditioning):
        """rf objective: v = noise - x0, so x0_hat = x_t - t * v_pred."""
        v = model.model(x_t, t, **model.get_conditioning_inputs(conditioning))
        return x_t - t[:, None, None] * v, v

    # ---- data ------------------------------------------------------------
    items, _ = arc_data.real_items()
    ds = arc_data.BeatARCDataset(items, crop=a.crop, seed=a.seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size, shuffle=True,
        collate_fn=arc_data.collate, num_workers=2, drop_last=True,
        persistent_workers=True)
    print(f"[data] {len(ds)} real crops, crop {a.crop} latent frames "
          f"(~{a.crop/arc_data.LATENT_RATE:.1f}s)", flush=True)

    # ---- INVARIANT: clean real audio through this exact path must track ---
    import mir_eval
    fs, checked = [], 0
    with torch.no_grad():
        for reals, metadata in loader:
            logits = track_logits(reals.to(device)).float()
            pad = torch.ones(logits.shape, dtype=torch.bool, device=device)
            picks, _ = tracker.postprocessor(logits, logits, pad)
            while isinstance(picks, tuple):
                picks = picks[0]
            if not isinstance(picks, (list, tuple)):
                picks = [picks]
            for est, m in zip(picks, metadata):
                t0 = m["beat"]["frame_offset"] / arc_data.LATENT_RATE
                ref = np.asarray(m["beat"]["beats"])
                ref = ref[(ref >= t0) & (ref < t0 + a.crop/arc_data.LATENT_RATE)] - t0
                est = np.asarray(est, dtype=float)
                if len(ref) >= 2 and len(est) >= 2:
                    fs.append(mir_eval.beat.f_measure(
                        mir_eval.beat.trim_beats(ref),
                        mir_eval.beat.trim_beats(est),
                        f_measure_threshold=0.07))
            checked += len(metadata)
            if checked >= 24:
                break
    f_real = float(np.mean(fs)) if fs else 0.0
    print(f"[invariant] frozen tracker on CLEAN real audio via decode path: "
          f"mean F {f_real:.3f} over {len(fs)} crops (floor {a.real_f_floor})",
          flush=True)
    if f_real < a.real_f_floor:
        raise SystemExit("pipeline misaligned -- training would optimize nonsense")

    # ---- GRAD CHECK: tracker term alone must reach the ControlNet ---------
    reals, metadata = next(iter(loader))
    reals = reals.to(device)
    conditioning = model.conditioner(metadata, device)
    t = torch.full((reals.shape[0],), 0.4, device=device)
    noise = torch.randn_like(reals)
    x_t = reals * (1 - t)[:, None, None] + noise * t[:, None, None]
    x0_hat, _ = denoise_step(x_t, t, conditioning)
    logits = track_logits(x0_hat)
    y = torch.stack([beat_target(m, logits.shape[1]) for m in metadata]).to(device)
    st_loss.to(device)(logits, y).backward()
    gnorm = sum(float(p.grad.norm()) for p in trainable if p.grad is not None)
    optim.zero_grad(set_to_none=True)
    print(f"[grad-check] ControlNet grad norm from tracker term alone: "
          f"{gnorm:.3e}", flush=True)
    if gnorm == 0.0:
        raise SystemExit("tracker term has NO gradient into the ControlNet -- "
                         "the no_grad trap is back; refusing to train")

    ckpt = abt.PeriodicCheckpoint(model, out, a.ckpt_every, base_keys)
    st = st_loss.to(device)
    step, ra, rt, n = 0, 0.0, 0.0, 0
    model.model.train()
    while step < a.steps:
        for reals, metadata in loader:
            if step >= a.steps:
                break
            reals = reals.to(device)
            conditioning = model.conditioner(metadata, device)

            # anchor: plain rf denoising on real latents
            t_a = torch.sigmoid(torch.randn(reals.shape[0], device=device))
            nz = torch.randn_like(reals)
            x_ta = reals * (1 - t_a)[:, None, None] + nz * t_a[:, None, None]
            _, v_pred = denoise_step(x_ta, t_a, conditioning)
            anchor = F.mse_loss(v_pred, nz - reals)

            # tracker: one-step denoise at the SDEdit operating range, then track
            t_t = (torch.rand(reals.shape[0], device=device)
                   * (a.t_hi - a.t_lo) + a.t_lo)
            nz2 = torch.randn_like(reals)
            x_tt = reals * (1 - t_t)[:, None, None] + nz2 * t_t[:, None, None]
            x0_hat, _ = denoise_step(x_tt, t_t, conditioning)
            logits = track_logits(x0_hat)
            y = torch.stack([beat_target(m, logits.shape[1])
                             for m in metadata]).to(device)
            tr = st(logits, y)

            loss = a.anchor_weight * anchor + a.tracker_weight * tr
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()

            ra += float(anchor); rt += float(tr); n += 1
            step += 1
            if step % a.log_every == 0:
                print(f"  step {step:5d}  anchor {ra/n:.4f}  tracker {rt/n:.4f}",
                      flush=True)
                ra = rt = 0.0; n = 0
            if step % a.ckpt_every == 0:
                ckpt.save(step)

    ckpt.save(step)
    print("done", flush=True)


if __name__ == "__main__":
    main()
