"""Rung 1: zero-shot beat-evidence guidance during SDEdit sampling.

At each ping-pong step the sampler computes `denoised` (the x0 estimate in
latent space) and hands it to our callback BEFORE renoising. We decode it,
compute a differentiable beat-evidence objective on the audio -- spectral-flux
mass concentrated at the conditioned beat positions -- and nudge the latents
along its gradient, in place.

Objective (maximised):
    conc = sum over beats of maxpool_{+-3 frames}(flux) / (sum of flux + eps)
"commit onset energy at the conditioned beats, not elsewhere". This is the
differentiable cousin of ac_period_vs_best, the filter that won the evidence
search; unpaired ABSOLUTE evidence was shown to be an easiness proxy, but here
the grid is given, so concentration at it is exactly what we want more of.

Everything else (model, adapter, windows, seeds, noise 0.4) matches the spec8
probe batch (8 steps, cfg 1), so output pairs 1:1 with both spec8 and the
original 60-step clips for crest / onset-z comparison.
"""
import argparse, os, sys, glob, json, random
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

POOL = "/home/sogang/jaehoon/Analyze-SMC/data/pools/slow_under70.json"


def build_pool(exclude_folds=(0,)):
    stems = []
    for r in json.load(open(POOL)):
        if r["fold"] in exclude_folds:
            continue
        if not os.path.exists(r["audio"]) or not os.path.exists(r["beats"]):
            continue
        stems.append((r["name"], r["audio"], r["beats"], r["fold"]))
    return stems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--g-scale", type=float, default=0.1,
                    help="guidance strength; 0 disables (control)")
    ap.add_argument("--adapter", default="/disk1/jaehoon/sa3_beat_runs/cn_fold0_rw/"
                                         "beat_adapter_final.safetensors")
    ap.add_argument("--gpu", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    import torch, soxr
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    from sa3_beat.build import load_beat_model
    from sa3_beat.generate import load_adapter, beat_conditioning, _patch_conditioning_inputs

    dev = f"cuda:{a.gpu}"
    model, cfg = load_beat_model(device=dev)
    load_adapter(model, a.adapter, dev)
    _patch_conditioning_inputs(model)
    sr_gen = cfg["sample_rate"]
    ds = model.pretransform.downsampling_ratio
    io_ch = cfg["model"]["io_channels"]

    # the autoencoder's own NESTED pretransform wraps its decode in no_grad
    # unless enable_grad is set; without this the guidance graph is detached
    # at the final decode stage.
    inner = getattr(model.pretransform.model, "pretransform", None)
    if inner is not None:
        inner.enable_grad = True

    HOP = 512
    fps_stft = sr_gen / HOP
    win = torch.hann_window(2048, device=dev)

    def make_callback(beat_sec, n_samples):
        idx = torch.tensor(np.round(np.asarray(beat_sec) * fps_stft).astype(int), device=dev)

        def flux_conc(z):
            # bypass pretransform.decode: its chunked crossfade path detaches
            # the autograd graph. Direct unchunked decode keeps gradients.
            pre = model.pretransform
            audio = pre.model.decode(z * pre.scale)       # (B, ch, n)
            mono = audio.mean(1)[0][:n_samples]
            S = torch.stft(mono, 2048, HOP, window=win, return_complex=True).abs()
            fl = torch.relu(S[:, 1:] - S[:, :-1]).sum(0)   # (T-1,)
            T = fl.shape[0]
            ii = idx[(idx >= 3) & (idx < T - 3)]
            if ii.numel() < 3:
                return None
            pool = torch.stack([fl[ii + k] for k in range(-3, 4)]).max(0).values
            return pool.sum() / (fl.sum() + 1e-6)

        def cb(d):
            if a.g_scale == 0:
                return
            den = d["denoised"]
            with torch.enable_grad():
                z = den.detach().float().requires_grad_(True)
                obj = flux_conc(z)
                if obj is None:
                    return
                (-obj).backward()
                g = z.grad
            g = g * (den.float().std() / (g.std() + 1e-9))
            den.add_((a.g_scale * g).to(den.dtype))
        return cb

    pool = build_pool()
    rng = random.Random(0)
    rng.shuffle(pool)
    made = 0
    for i in range(a.n):
        stem, wav, bt, fold = pool[i % len(pool)]
        out_base = os.path.join(a.out_dir, f"sd_{i:05d}")
        if os.path.exists(out_base + ".wav"):
            made += 1
            continue
        r = random.Random(i)
        try:
            y, sr = sf.read(wav, dtype="float32", always_2d=True)
        except Exception:
            continue
        y = y.mean(axis=1)
        yg = soxr.resample(y, sr, sr_gen) if sr != sr_gen else y
        beats = np.loadtxt(bt, ndmin=2)
        db_col = beats[:, 1] if beats.shape[1] > 1 else None
        beats = beats[:, 0]
        if len(beats) < 12:
            continue
        lo, hi = beats[2], max(beats[2], beats[-2] - a.seconds)
        start = float(beats[min(range(len(beats)),
                                key=lambda k: abs(beats[k] - r.uniform(lo, hi)))])
        seg = yg[int(start * sr_gen): int((start + a.seconds) * sr_gen)]
        n = (len(seg) // ds) * ds
        if n < ds * 8:
            continue
        seg = seg[:n]
        m = (beats >= start) & (beats < start + n / sr_gen)
        b = beats[m] - start
        d_ = (beats[m][db_col[m] == 1] - start) if db_col is not None else None
        if len(b) < 8:
            continue

        init = torch.from_numpy(np.stack([seg, seg])).to(dev)
        n_lat = n // ds
        item = beat_conditioning(b, d_, n_lat, "", n / sr_gen)
        cond = model.conditioner([item], dev)
        cond["inpaint_mask"] = [torch.zeros(1, 1, n_lat, device=dev), None]
        cond["inpaint_masked_input"] = [torch.zeros(1, io_ch, n_lat, device=dev), None]
        out = generate_diffusion_cond(
            model, steps=a.steps, cfg_scale=a.cfg_scale, conditioning_tensors=cond,
            sample_size=n, seed=i, device=dev,
            init_audio=(sr_gen, init), init_noise_level=a.noise,
            callback=make_callback(b, n))
        o = out.squeeze(0).float().cpu().numpy()
        o = o.mean(axis=0) if o.ndim > 1 else o
        o = o / (np.abs(o).max() + 1e-9) * 0.9
        sf.write(out_base + ".wav", o, sr_gen, format="WAV")
        with open(out_base + ".beats", "w") as fh:
            for k, t in enumerate(b):
                fh.write(f"{t:.6f}\t{1 if (d_ is not None and t in d_) else 2}\n")
        json.dump({"source": stem, "source_wav": wav, "window_start": start,
                   "noise": a.noise, "steps": a.steps, "cfg_scale": a.cfg_scale,
                   "g_scale": a.g_scale, "n_beats": int(len(b))},
                  open(out_base + ".json", "w"))
        made += 1
        print(f"  [{made}/{a.n}] {stem}", flush=True)
    print(f"DONE {made}")


if __name__ == "__main__":
    main()
