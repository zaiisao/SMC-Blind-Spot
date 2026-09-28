"""SDEdit style transfer that keeps the beat grid: easy genre in, hard genre out.

The point is verifiable labels. Generating slow drumless music directly gives
clips whose beats nobody can locate -- not the tracker, and (2026-08-24 listening)
not a human either, so the label is an unchecked assertion. But generating lofi or
dub at 70 bpm gives CMLt 1.00: we KNOW where the beats are because the tracker
agrees with the grid we asked for.

So make the hard clip by transforming a verified easy one. Encode it, add partial
noise, denoise under a new prompt with the SAME beat conditioning. Timing survives
at low noise levels and the timbre changes; at high noise levels the model forgets
the source and the label stops being true. The noise level is the whole experiment.

Verification afterwards is non-circular, because there is a known-good BEFORE:

    onsets still on grid + tracker now fails   -> genuinely hard, valid label  <- goal
    onsets still on grid + tracker still fine  -> did not get harder
    onsets gone                                -> timing destroyed, discard
"""
import argparse, os, sys, glob
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def add_ticks(y, sr, beats, downbeats=None):
    out = y.copy()
    n = len(out)
    for i, b in enumerate(beats):
        s = int(b * sr)
        if not (0 <= s < n):
            continue
        dur = int(0.030 * sr)
        t = np.arange(min(dur, n - s)) / sr
        freq = 1800.0 if (downbeats is not None and i in downbeats) else 1100.0
        out[s:s + len(t)] += 0.35 * np.exp(-t * 90.0) * np.sin(2 * np.pi * freq * t)
    return np.clip(out, -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, help="verified clips (wav + .beats)")
    ap.add_argument("--clips", nargs="+", required=True, help="stems to transform")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--adapter", default="/disk1/jaehoon/sa3_beat_runs/cn_fold0_rw/"
                                         "beat_adapter_final.safetensors")
    ap.add_argument("--prompt", default="solo piano, classical, expressive, no drums")
    ap.add_argument("--noise", type=float, nargs="+", default=[0.4, 0.6, 0.8])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg-scale", type=float, default=6.0)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    import torch, soxr
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    from sa3_beat.build import load_beat_model
    from sa3_beat.generate import load_adapter, beat_conditioning, _patch_conditioning_inputs

    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    model, cfg = load_beat_model(device=dev)
    load_adapter(model, a.adapter, dev)
    _patch_conditioning_inputs(model)
    sr_gen = cfg["sample_rate"]
    ds = model.pretransform.downsampling_ratio
    io_ch = cfg["model"]["io_channels"]

    for stem in a.clips:
        wav = os.path.join(a.src_dir, stem + ".wav")
        bt = os.path.join(a.src_dir, stem + ".beats")
        y, sr = sf.read(wav, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        yg = soxr.resample(y, sr, sr_gen) if sr != sr_gen else y
        n = (len(yg) // ds) * ds
        yg = yg[:n]
        beats = np.loadtxt(bt, ndmin=2)[:, 0]
        seconds = n / sr_gen
        beats = beats[beats < seconds]

        # BEFORE, copied verbatim so the comparison is like for like
        sf.write(os.path.join(a.out_dir, f"{stem}__BEFORE.wav"), yg, sr_gen, format="WAV")
        sf.write(os.path.join(a.out_dir, f"{stem}__BEFORE_TICKS.wav"),
                 add_ticks(yg, sr_gen, beats), sr_gen, format="WAV")
        np.savetxt(os.path.join(a.out_dir, f"{stem}__BEFORE.beats"), beats, fmt="%.6f")

        init = torch.from_numpy(np.stack([yg, yg])).to(dev)
        n_lat = n // ds
        # SAME beat grid as the source: the transfer is told to keep the timing.
        item = beat_conditioning(beats, beats[::4], n_lat, a.prompt, seconds)
        cond = model.conditioner([item], dev)
        cond["inpaint_mask"] = [torch.zeros(1, 1, n_lat, device=dev), None]
        cond["inpaint_masked_input"] = [torch.zeros(1, io_ch, n_lat, device=dev), None]

        for nl in a.noise:
            out = generate_diffusion_cond(
                model, steps=a.steps, cfg_scale=a.cfg_scale, conditioning_tensors=cond,
                sample_size=n, seed=0, device=dev,
                init_audio=(sr_gen, init), init_noise_level=nl)
            o = out.squeeze(0).float().cpu().numpy()
            o = o.mean(axis=0) if o.ndim > 1 else o
            o = o / (np.abs(o).max() + 1e-9) * 0.9
            tag = f"{stem}__AFTER_noise{nl:.2f}"
            sf.write(os.path.join(a.out_dir, tag + ".wav"), o, sr_gen, format="WAV")
            sf.write(os.path.join(a.out_dir, tag + "_TICKS.wav"),
                     add_ticks(o, sr_gen, beats), sr_gen, format="WAV")
            np.savetxt(os.path.join(a.out_dir, tag + ".beats"), beats, fmt="%.6f")
            print(f"[sdedit] {stem}  noise={nl:.2f} -> {tag}.wav", flush=True)


if __name__ == "__main__":
    main()
