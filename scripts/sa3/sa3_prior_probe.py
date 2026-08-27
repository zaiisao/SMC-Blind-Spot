"""HANDOVER §5.1 + §5.2: what does SA3's prior actually contain?

§5.2  Can the base model render quiet, expressive solo piano at all?  If not, the
      text prompt cannot supply the style and the composition argument in §6
      collapses.
§5.1  Did the DiT inherit the encoder's timing blind spot?  It was trained on
      encoder outputs, and §2.3 showed the encoder smears onsets.  If generated
      audio has systematically softer onsets than real audio, a ControlNet would
      be steering toward a region the prior never represented -- which is the
      difference between path 1 (small job) and path 2 (encoder+DiT retraining)
      in §3.5.

Measures onset sharpness (attack slope, peak prominence of the onset envelope)
and inter-onset regularity on generated audio, against real SMC audio scored the
same way.  No training. Writes .wav files so §5.2 can also be judged by ear.

    /disk1/jaehoon/sa3env/bin/python scripts/sa3/sa3_prior_probe.py --gpu 2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SA3_DIR = Path("/disk1/jaehoon/sa3_small_music")

PROMPTS = [
    ("piano_expressive", "solo piano, quiet, expressive rubato, classical, intimate recording"),
    ("piano_plain",      "solo piano, classical"),
    ("percussive",       "acoustic drum kit, steady groove, dry recording"),
]


def onset_features(audio, sr):
    """Onset sharpness + inter-onset regularity for one mono signal."""
    import librosa

    env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=3,
                                   post_avg=5, delta=0.1, wait=10)
    if len(peaks) < 3:
        return None

    # Sharpness: rise of the onset envelope into each peak, normalised by peak
    # height so it measures attack shape rather than loudness.
    slopes, prominences = [], []
    for p in peaks:
        if p < 2:
            continue
        rise = env[p] - env[max(0, p - 2)]
        slopes.append(rise / (env[p] + 1e-9))
        local = env[max(0, p - 10):p + 10]
        prominences.append((env[p] - np.median(local)) / (env[p] + 1e-9))

    iois = np.diff(peaks) * 512 / sr
    return {
        "n_onsets": int(len(peaks)),
        "attack_slope": float(np.mean(slopes)),
        "peak_prominence": float(np.mean(prominences)),
        "ioi_cv": float(np.std(iois) / (np.mean(iois) + 1e-9)),
        "onset_rate_hz": float(len(peaks) / (len(audio) / sr)),
    }


def summarise(name, feats):
    ok = [f for f in feats if f]
    if not ok:
        return f"{name:22s}  (no usable onsets)"
    def m(k):
        return np.mean([f[k] for f in ok])
    return (f"{name:22s}  n={len(ok):2d}  attack_slope={m('attack_slope'):.3f}  "
            f"prominence={m('peak_prominence'):.3f}  ioi_cv={m('ioi_cv'):.3f}  "
            f"onsets/s={m('onset_rate_hz'):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg-scale", type=float, default=6.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--n-per-prompt", type=int, default=3)
    ap.add_argument("--n-real", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "cache" / "sa3_prior_probe"))
    args = ap.parse_args()

    import soundfile as sf
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict
    from safetensors.torch import load_file

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    with open(SA3_DIR / "model_config.json") as f:
        config = json.load(f)
    model = create_model_from_config(config)
    copy_state_dict(model, load_file(str(SA3_DIR / "model.safetensors")))
    model = model.to(device).eval().requires_grad_(False)

    sr = config["sample_rate"]
    ds = model.pretransform.downsampling_ratio
    sample_size = (int(args.seconds * sr) // ds) * ds
    print(f"[setup] device={device} sr={sr} sample_size={sample_size} "
          f"({sample_size/sr:.1f}s) steps={args.steps} cfg={args.cfg_scale}\n")

    results = {}
    for tag, prompt in PROMPTS:
        feats = []
        for i in range(args.n_per_prompt):
            # This checkpoint is diffusion_cond_inpaint, so the DiT always
            # expects the inpaint local conditioning. Pure text-to-music is the
            # FULL_MASK case the trainer uses for validation: both tensors zero.
            cond_t = model.conditioner(
                [{"prompt": prompt, "seconds_total": args.seconds}], device)
            n_latent = sample_size // ds
            cond_t["inpaint_mask"] = [
                torch.zeros(1, 1, n_latent, device=device), None]
            cond_t["inpaint_masked_input"] = [
                torch.zeros(1, config["model"]["io_channels"], n_latent, device=device), None]

            audio = generate_diffusion_cond(
                model,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                conditioning_tensors=cond_t,
                sample_size=sample_size,
                seed=1000 + i,
                device=device,
            )
            wav = audio.squeeze(0).to(torch.float32).cpu().numpy()
            wav = wav / (np.abs(wav).max() + 1e-9)
            path = out / f"{tag}_{i}.wav"
            sf.write(str(path), wav.T, sr)
            feats.append(onset_features(wav.mean(axis=0), sr))
            print(f"  [gen] {path.name}")
        results[tag] = feats
        print(summarise(tag, feats) + "\n")

    # Real reference: SMC audio, the material we actually care about.
    smc = sorted((ROOT / "SMC_MIREX" / "SMC_MIREX_Audio").glob("*.wav"))[: args.n_real]
    if smc:
        import librosa
        real = []
        for p in smc:
            y, _sr = librosa.load(str(p), sr=sr, mono=True, duration=args.seconds)
            real.append(onset_features(y, sr))
        results["real_smc"] = real
        print(summarise("real_smc", real))

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[out] {out}/results.json  (+ .wav files for the §5.2 listen test)")


if __name__ == "__main__":
    main()
