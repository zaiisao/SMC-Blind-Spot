"""Generate to a UNIFORM synthetic grid at a fixed tempo. Nothing borrowed.

The corpus pipeline builds each clip by combining three independently sampled
things: a real song's beat grid, a target tempo drawn from SMC's empirical
distribution, and a prompt from a bank. Most combinations are fine, so the
aggregate statistics look healthy, but the tail is incoherent -- a Beethoven
sonata window compressed to 193.5 bpm and prompted for bluegrass banjo. That is
where the unlabelable clips come from.

This does the simple thing instead: beats at 0, 1.5, 3.0, 4.5 ... at one chosen
tempo, one matching prompt, and generate. If the ControlNet places beats, the
ticks land on the pulse and there is nothing else to blame.
"""
import argparse, os, sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def add_ticks(y, sr, beats, downbeats_every=4):
    out = y.copy()
    n = len(out)
    for i, b in enumerate(beats):
        s = int(b * sr)
        if not (0 <= s < n):
            continue
        dur = int(0.030 * sr)
        t = np.arange(min(dur, n - s)) / sr
        freq = 1800.0 if (i % downbeats_every == 0) else 1100.0
        out[s:s + len(t)] += 0.35 * np.exp(-t * 90.0) * np.sin(2 * np.pi * freq * t)
    return np.clip(out, -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/disk1/jaehoon/sa3_beat_runs/cn_fold0_rw/"
                                         "beat_adapter_final.safetensors")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bpm", type=float, nargs="+", default=[40, 50, 60, 70])
    ap.add_argument("--prompt", default="solo piano, slow, expressive, "
                                        "clear note attacks")
    ap.add_argument("--prompts", nargs="+", default=None,
                    help="tag=prompt pairs, e.g. lofi='lofi hip hop, ...'. "
                         "Overrides --prompt; the tag goes in the filename.")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg-scale", type=float, default=6.0)
    ap.add_argument("--per-bpm", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="beat-pathway guidance. For UNIFORM grids the metronomic "
                         "twin equals the grid itself and guidance is a no-op, so "
                         "this uses the NULL-beat reference (Prof. Jung's Eq. 13): "
                         "res = res(null) + g * (res(true) - res(null)).")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    import torch
    from sa3_beat.build import load_beat_model
    from sa3_beat.generate import load_adapter, generate, isochronous

    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    model, cfg = load_beat_model(device=dev)
    load_adapter(model, a.adapter, dev)
    sr = cfg["sample_rate"]

    cn = getattr(model.model.model, "beat_controlnet", None)
    if a.guidance != 1.0:
        assert cn is not None, "guidance requested but no beat_controlnet found"
        # null reference: the conditioner emits zeros for an empty grid, exactly
        # the 'no beat guidance' state percussify relied on.
        from sa3_beat.build import BEAT_COND_ID
        from sa3_beat.generate import beat_conditioning
        ds_ratio = model.pretransform.downsampling_ratio
        n_lat = (int(a.seconds * sr) // ds_ratio)
        null_item = beat_conditioning(np.array([]), None, n_lat, "", a.seconds)
        nt = model.conditioner([null_item], dev)[BEAT_COND_ID][0]
        if nt.shape[1] < nt.shape[2]:
            nt = nt.transpose(1, 2)
        cn.guidance = float(a.guidance)
        cn.twin_beat = nt
        print(f"[guidance] beat-pathway CFG ON  g={a.guidance}  ref=null", flush=True)

    if a.prompts:
        banks = [p.split("=", 1) for p in a.prompts]
    else:
        banks = [("piano", a.prompt)]

    for tag, prompt in banks:
        for bpm in a.bpm:
            ibi = 60.0 / bpm
            beats, downbeats = isochronous(bpm, a.seconds)   # 0, 1.5, 3.0, 4.5 ...
            for k in range(a.per_bpm):
                seed = int(bpm) * 100 + k
                wav, _ = generate(model, cfg, prompt, beats, downbeats,
                                  a.seconds, a.steps, a.cfg_scale, seed=seed, device=dev)
                y = wav.mean(axis=0) if wav.ndim > 1 else wav
                y = y / (np.abs(y).max() + 1e-9) * 0.9
                base = os.path.join(a.out_dir, f"{tag}_{int(bpm):03d}bpm_{k}")
                sf.write(base + ".wav", y, sr, format="WAV")
                sf.write(base + "_TICKS.wav", add_ticks(y, sr, beats), sr, format="WAV")
                np.savetxt(base + ".beats", beats, fmt="%.6f")
                print(f"[gen] {tag:<10} {bpm:5.1f} bpm  ibi={ibi:.3f}s  "
                      f"{len(beats)} beats  -> {base}.wav", flush=True)


if __name__ == "__main__":
    main()
