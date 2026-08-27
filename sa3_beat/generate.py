
import argparse
import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_SECONDS = 20.0


class _CastableDict(dict):

    def type(self, dtype):
        return _CastableDict(
            {k: (v.type(dtype) if v is not None else v) for k, v in self.items()})


def _patch_conditioning_inputs(model):
    if getattr(model, "_beat_cast_patched", False):
        return model
    original = model.get_conditioning_inputs

    def wrapped(conditioning_tensors, negative=False):
        inputs = original(conditioning_tensors, negative=negative)
        mlc = inputs.get("modular_local_cond")
        if isinstance(mlc, dict):
            inputs["modular_local_cond"] = _CastableDict(mlc)
        return inputs

    model.get_conditioning_inputs = wrapped
    model._beat_cast_patched = True
    return model


def beat_conditioning(beats, downbeats, num_frames, prompt, seconds):
    return {
        "prompt": prompt,
        "seconds_total": seconds,
        "beat": {
            "beats": np.asarray(beats, dtype=np.float64),
            "downbeats": None if downbeats is None else np.asarray(downbeats, dtype=np.float64),
            "num_frames": int(num_frames),
        },
    }


def isochronous(bpm, seconds, beats_per_bar=4, offset=0.0):
    period = 60.0 / bpm
    beats = np.arange(offset, seconds, period)
    downbeats = beats[::beats_per_bar] if beats_per_bar else None
    return beats, downbeats


def rubato(bpm, seconds, depth=0.25, period_bars=2.0, beats_per_bar=4, offset=0.0):
    period = 60.0 / bpm
    times, t, k = [], offset, 0
    while t < seconds:
        times.append(t)
        rate = 1.0 + depth * np.sin(2 * np.pi * k / (period_bars * beats_per_bar))
        t += period / rate
        k += 1
    beats = np.array(times)
    downbeats = beats[::beats_per_bar] if beats_per_bar else None
    return beats, downbeats


def generate(model, config, prompt, beats, downbeats=None, seconds=DEFAULT_SECONDS,
             steps=100, cfg_scale=6.0, seed=0, device="cuda"):
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    _patch_conditioning_inputs(model)

    sr = config["sample_rate"]
    ds = model.pretransform.downsampling_ratio
    sample_size = (int(seconds * sr) // ds) * ds
    n_latent = sample_size // ds

    item = beat_conditioning(beats, downbeats, n_latent, prompt, seconds)
    cond = model.conditioner([item], device)

    io_ch = config["model"]["io_channels"]
    cond["inpaint_mask"] = [torch.zeros(1, 1, n_latent, device=device), None]
    cond["inpaint_masked_input"] = [
        torch.zeros(1, io_ch, n_latent, device=device), None]

    audio = generate_diffusion_cond(
        model, steps=steps, cfg_scale=cfg_scale, conditioning_tensors=cond,
        sample_size=sample_size, seed=seed, device=device)
    return audio.squeeze(0).to(torch.float32).cpu().numpy(), n_latent


def load_adapter(model, path, device):
    from safetensors.torch import load_file

    state = (load_file(path, device=str(device)) if str(path).endswith(".safetensors")
             else torch.load(path, map_location=device))

    cn_keys = [k for k in state if k.startswith("beat_controlnet.")]
    if cn_keys:
        from .controlnet import BeatControlNet
        n_blocks = 1 + max(int(k.split(".")[2]) for k in cn_keys
                           if k.startswith("beat_controlnet.out_projs."))
        dit = model.model.model
        cn = BeatControlNet(dit.transformer, n_blocks=n_blocks).to(device)
        dit.beat_controlnet = cn
        cn.attach()
        cn_state = {k[len("beat_controlnet."):]: v for k, v in state.items()
                    if k.startswith("beat_controlnet.")}
        result = cn.load_state_dict(cn_state, strict=False)
        if result.missing_keys:
            raise RuntimeError(
                f"ControlNet checkpoint is missing {len(result.missing_keys)} "
                f"tensors, e.g. {result.missing_keys[:3]}")
        print(f"[adapter] ControlNet: {n_blocks} blocks, {len(cn_state)} tensors")

    rest = {k: v for k, v in state.items() if not k.startswith("beat_controlnet.")}
    info = model.model.load_state_dict(rest, strict=False)
    unexpected = list(info.unexpected_keys)
    print(f"[adapter] loaded {path}: {len(rest)} adapter tensors, "
          f"{len(unexpected)} unexpected")
    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} checkpoint tensors matched nothing in the model, "
            f"e.g. {unexpected[:3]} -- they would have been silently ignored")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="solo piano, quiet, expressive rubato, "
                                        "classical, intimate recording")
    ap.add_argument("--bpm", type=float, default=100.0)
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--rubato", type=float, default=0.0,
                    help="peak fractional tempo deviation; 0 = metronomic")
    ap.add_argument("--no-downbeat", action="store_true")
    ap.add_argument("--beats-file", default=None,
                    help="annotation file to take the grid from instead")
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg-scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--adapter", default=None,
                    help="beat adapter checkpoint; omitted = untrained no-op")
    ap.add_argument("--check", action="store_true",
                    help="generate twice with different grids and report whether "
                         "the beat signal changed the output at all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import soundfile as sf
    from .build import load_beat_model

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model, config = load_beat_model(device=device,
                                    include_downbeat=not args.no_downbeat)
    if args.adapter:
        load_adapter(model, args.adapter, device)
    else:
        print("[adapter] none -- beat conditioning is a zero-init no-op")

    if args.beats_file:
        from .data import load_beats
        beats, downbeats = load_beats(args.beats_file)
        beats = beats[beats < args.seconds]
        downbeats = None if downbeats is None else downbeats[downbeats < args.seconds]
    elif args.rubato > 0:
        beats, downbeats = rubato(args.bpm, args.seconds, depth=args.rubato,
                                  beats_per_bar=args.beats_per_bar)
    else:
        beats, downbeats = isochronous(args.bpm, args.seconds,
                                       beats_per_bar=args.beats_per_bar)
    if args.no_downbeat:
        downbeats = None

    ibi = np.diff(beats)
    print(f"[grid] {len(beats)} beats, {0 if downbeats is None else len(downbeats)} "
          f"downbeats | IBI {ibi.min():.3f}-{ibi.max():.3f}s "
          f"(cv {ibi.std()/ibi.mean():.3f})")

    sr = config["sample_rate"]
    out = Path(args.out or f"cache/sa3_generate/seed{args.seed}")
    out.parent.mkdir(parents=True, exist_ok=True); out.mkdir(exist_ok=True)

    wav, n_latent = generate(model, config, args.prompt, beats, downbeats,
                             args.seconds, args.steps, args.cfg_scale,
                             args.seed, device)
    wav = wav / (np.abs(wav).max() + 1e-9)
    sf.write(str(out / "beats.wav"), wav.T, sr)
    print(f"[gen] {out}/beats.wav  latent_frames={n_latent}")

    if args.check:
        alt, alt_db = isochronous(args.bpm * 1.37, args.seconds,
                                  beats_per_bar=args.beats_per_bar)
        wav2, _ = generate(model, config, args.prompt, alt, alt_db,
                           args.seconds, args.steps, args.cfg_scale,
                           args.seed, device)
        wav2 = wav2 / (np.abs(wav2).max() + 1e-9)
        sf.write(str(out / "beats_alt.wav"), wav2.T, sr)
        d = float(np.abs(wav - wav2).max())
        print(f"\n[check] max|audio(grid A) - audio(grid B)| = {d:.3e}")
        print("[check] " + ("beat signal has NO effect -- adapter is untrained "
                            "(expected until training)" if d == 0.0 else
                            "beat signal changes the output"))

    json.dump({"prompt": args.prompt, "beats": list(map(float, beats)),
               "downbeats": None if downbeats is None else list(map(float, downbeats)),
               "args": vars(args)}, open(out / "grid.json", "w"), indent=1)


if __name__ == "__main__":
    main()
