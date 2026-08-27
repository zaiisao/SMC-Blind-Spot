"""Beat-conditioned generation: text prompt + beat times -> audio.

The routing already exists upstream -- ``ConditionedDiffusionModelWrapper``
copies every id in ``modular_local_cond_ids`` out of the conditioning tensors
and hands it to the DiT -- so this module only has to assemble the inputs:

* text prompt and ``seconds_total``, as any SA3 generation does;
* the beat signal, as continuous phase at the latent frame rate, so a rubato
  grid is as expressible as a metronomic one;
* zeroed ``inpaint_mask`` / ``inpaint_masked_input``.  This checkpoint is
  ``diffusion_cond_inpaint``, so the DiT always expects the inpaint pair;
  pure text-to-music is the FULL_MASK case with both tensors zero.

Note on frame counts: generation is *not* subject to the encoder's chunk
padding.  The diffusion latent length is exactly ``sample_size // downsampling
ratio``.  ``sa3_beat.encoder.latent_frames`` applies to the encode path only.

UNTRAINED, THE BEAT SIGNAL DOES NOTHING.  Both the ``modular_local_cond``
adapter and the ControlNet are zero-init, so they contribute exactly zero until
trained: generation will run and ignore the beats entirely.  ``--check`` exists
to measure that rather than assume it -- it generates twice from one seed with
two different beat grids and reports the difference.  While the adapter is a
no-op the difference is exactly 0, which makes the check a regression test that
becomes meaningful the moment training starts.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_SECONDS = 20.0


class _CastableDict(dict):
    """A dict that answers ``.type(dtype)`` by casting its values.

    ``generate_diffusion_cond`` casts every conditioning input with
    ``v.type(model_dtype)``, which assumes each value is a tensor.  But
    ``get_conditioning_inputs`` returns ``modular_local_cond`` as a *dict* of
    per-id tensors, so the stock sampler raises ``AttributeError`` the moment
    any modular conditioning is present -- beat conditioning cannot be sampled
    through the library helper as written.

    Rather than fork the submodule for one line, the dict is returned as this
    subclass, which satisfies the call and casts the tensors it holds.
    """

    def type(self, dtype):
        return _CastableDict(
            {k: (v.type(dtype) if v is not None else v) for k, v in self.items()})


def _patch_conditioning_inputs(model):
    """Wrap ``get_conditioning_inputs`` so modular conditioning survives casting."""
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
    """One batch item for ``model.conditioner``."""
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
    """A plain metronomic grid -- the baseline to compare rubato against."""
    period = 60.0 / bpm
    beats = np.arange(offset, seconds, period)
    downbeats = beats[::beats_per_bar] if beats_per_bar else None
    return beats, downbeats


def rubato(bpm, seconds, depth=0.25, period_bars=2.0, beats_per_bar=4, offset=0.0):
    """A grid whose tempo breathes sinusoidally, for testing expressive timing.

    ``depth`` is the peak fractional tempo deviation, so 0.25 means the local
    tempo swings +/-25%.  Built by integrating the tempo curve rather than by
    jittering beat times, which keeps the grid monotonic and musically coherent.
    """
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
    """Generate one clip. Returns ``(audio, n_latent)`` with audio as (ch, n)."""
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
    """Load a beat checkpoint, building the ControlNet branch if it has one.

    Two traps this exists to avoid, both of which fail *silently*:

    * checkpoints are written with ``safetensors.save_file``, so ``torch.load``
      is the wrong reader;
    * a ControlNet checkpoint carries ``beat_controlnet.*`` tensors, but the
      branch does not exist on a freshly built model.  Loading with
      ``strict=False`` would drop every one of them into ``unexpected_keys``
      and generate exactly as if untrained -- indistinguishable from a failed
      training run.

    The branch depth is recovered from the checkpoint rather than assumed, so a
    model trained with a non-default ``--controlnet-blocks`` reloads correctly.
    """
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
        # Same seed, a deliberately different grid: any difference in the output
        # is attributable to the beat signal alone.
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
