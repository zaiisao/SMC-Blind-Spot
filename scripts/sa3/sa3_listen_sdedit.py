"""Generate a listening set: SDEdit on real slow/rubato clips, with and without ticks.

Uses the validated configuration:
  - SOURCE is real annotated audio (SMC / ASAP), so beat times are HUMAN
    annotations, not asserted by us. That is the whole point of SDEdit here.
  - beat-pathway guidance g=2.0. Measured (sa3_sdedit_fidelity, 24 clips):
    at noise 0.5 it holds median lag 5.8 -> 4.4 ms, cuts clips over the 20 ms
    budget 8% -> 4%, raises onset salience 1.286 -> 1.394 and tracker F
    0.4163 -> 0.5047. At noise 0.3 g makes no difference, which is the right
    sanity check -- it should only matter when the task is hard.
  - noise 0.4 and 0.5. 0.4 was the validated ceiling without guidance
    (0% of clips past 20 ms); 0.5 is the extension guidance buys.

Writes three files per clip:
    <stem>__n<noise>_g<g>.wav         the generated audio
    <stem>__n<noise>_g<g>_TICKS.wav   same, with clicks at the GROUND-TRUTH beats
    <stem>__n<noise>_g<g>.beats       the beat times, for reference
plus the untouched source clip, so the comparison is audible.

The ticks mark the INHERITED annotation. If the generated audio has drifted, the
clicks will audibly lead or lag the music -- that is the check being made by ear,
and it is the one measurement a listener can make that the metrics cannot.
"""
import argparse, glob, os, sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

CACHE = "/disk1/jaehoon/sa3_beat_cache"
AUDIO = "/disk1/jaehoon/dataset_store/audio_by_stem"


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


def ibi_cv(b):
    d = np.diff(np.asarray(b, dtype=np.float64))
    return float(np.std(d) / (np.mean(d) + 1e-12)) if len(d) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/disk1/jaehoon/sa3_beat_runs/cn_fold0_rw/"
                                         "beat_adapter_final.safetensors")
    ap.add_argument("--datasets", nargs="+", default=["smc", "asap"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.4, 0.5])
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--cfg-scale", type=float, default=9.0,
                    help="text CFG. Raised from the 6.0 default per request.")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--max-bpm", type=float, default=80.0, help="slow clips only")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--out-dir", default="/disk1/jaehoon/listen_sdedit_g2")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    import torch, soxr
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    from sa3_beat.build import load_beat_model, BEAT_COND_ID
    from sa3_beat.generate import (load_adapter, beat_conditioning,
                                   _patch_conditioning_inputs)

    dev = f"cuda:{a.gpu}"
    model, cfg = load_beat_model(device=dev)
    load_adapter(model, a.adapter, dev)
    _patch_conditioning_inputs(model)
    sr_gen = cfg["sample_rate"]
    ds = model.pretransform.downsampling_ratio
    io_ch = cfg["model"]["io_channels"]
    cn = getattr(model.model.model, "beat_controlnet", None)
    assert cn is not None, "no beat_controlnet on the DiT"

    def metronomic_twin(b):
        b = np.asarray(b, dtype=np.float64)
        if len(b) < 3:
            return b
        return b[0] + np.arange(len(b)) * float(np.median(np.diff(b)))

    # pick SLOW and/or RUBATO real clips
    cands = []
    for dsn in a.datasets:
        for bt in sorted(glob.glob(f"{CACHE}/{dsn}__*.beats")):
            stem = os.path.basename(bt)[:-6]
            name = stem.split("__", 1)[1]
            wav = None
            for e in (".wav", ".flac", ".mp3"):
                p = os.path.join(AUDIO, dsn, name + e)
                if os.path.exists(p):
                    wav = p
                    break
            if wav is None:
                continue
            t = np.loadtxt(bt, ndmin=2)
            if t.size == 0 or len(t) < 12:
                continue
            b = t[:, 0]
            bpm = 60.0 / float(np.median(np.diff(b)))
            cv = ibi_cv(b)
            if bpm <= a.max_bpm or 0.06 <= cv <= 0.35:
                cands.append((stem, wav, bt, bpm, cv))
    # most rubato first among the slow ones, so the set is genuinely hard
    cands.sort(key=lambda x: -(x[4] + max(0.0, (a.max_bpm - x[3]) / a.max_bpm)))
    cands = cands[: a.n]
    print(f"[set] {len(cands)} clips  median bpm "
          f"{np.median([c[3] for c in cands]):.1f}  median IBI cv "
          f"{np.median([c[4] for c in cands]):.3f}", flush=True)

    for stem, wav, bt, bpm, cv in cands:
        y, sr = sf.read(wav, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        yg = soxr.resample(y, sr, sr_gen) if sr != sr_gen else y
        beats = np.loadtxt(bt, ndmin=2)[:, 0]
        start = float(beats[len(beats) // 4])
        seg = yg[int(start * sr_gen): int((start + a.seconds) * sr_gen)]
        n = (len(seg) // ds) * ds
        if n < ds * 8:
            continue
        seg = seg[:n]
        b = beats[(beats >= start) & (beats < start + n / sr_gen)] - start
        if len(b) < 8:
            continue

        # the untouched source, so the ear has a reference
        base = os.path.join(a.out_dir, f"{stem}__SOURCE")
        sf.write(base + ".wav", seg / (np.abs(seg).max() + 1e-9) * 0.9, sr_gen)
        sf.write(base + "_TICKS.wav",
                 add_ticks(seg / (np.abs(seg).max() + 1e-9) * 0.9, sr_gen, b), sr_gen)
        np.savetxt(base + ".beats", b, fmt="%.6f")

        init = torch.from_numpy(np.stack([seg, seg])).to(dev)
        n_lat = n // ds
        item = beat_conditioning(b, None, n_lat, a.prompt, n / sr_gen)
        cond = model.conditioner([item], dev)
        cond["inpaint_mask"] = [torch.zeros(1, 1, n_lat, device=dev), None]
        cond["inpaint_masked_input"] = [torch.zeros(1, io_ch, n_lat, device=dev), None]
        tw = None
        if a.guidance != 1.0:
            twin_item = beat_conditioning(metronomic_twin(b), None, n_lat,
                                          a.prompt, n / sr_gen)
            tw = model.conditioner([twin_item], dev)[BEAT_COND_ID][0]
            if tw.shape[1] < tw.shape[2]:
                tw = tw.transpose(1, 2)

        for nl in a.noise:
            cn.guidance = float(a.guidance)
            cn.twin_beat = None if a.guidance == 1.0 else tw
            out = generate_diffusion_cond(
                model, steps=a.steps, cfg_scale=a.cfg_scale,
                conditioning_tensors=cond, sample_size=n, seed=0, device=dev,
                init_audio=(sr_gen, init), init_noise_level=nl)
            o = out.squeeze(0).float().cpu().numpy()
            o = o.mean(axis=0) if o.ndim > 1 else o
            o = o / (np.abs(o).max() + 1e-9) * 0.9
            tag = os.path.join(a.out_dir,
                               f"{stem}__n{nl:.1f}_g{a.guidance:.1f}")
            sf.write(tag + ".wav", o, sr_gen)
            sf.write(tag + "_TICKS.wav", add_ticks(o, sr_gen, b), sr_gen)
            np.savetxt(tag + ".beats", b, fmt="%.6f")
        cn.guidance, cn.twin_beat = 1.0, None
        print(f"  {stem[:40]:<40} bpm={bpm:5.1f} cv={cv:.3f} done", flush=True)

    print(f"\n[done] {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
