"""Test-time transform: regenerate a drumless song WITH percussion, then track that.

Idea (user's): SMC is drumless, and the percussion-bias campaign found the failure is
"a cliff at exactly zero drum energy". So move the INPUT across that cliff -- ask SA3
to re-render the song with drums, run the released tracker on the result, and report
those beats for the original.

No ground-truth beat information is used anywhere. SA3 has to infer the pulse from the
music alone; the drums it adds are its answer. That distinguishes this from the failed
v13d overdub, which rendered drums ON the annotated beats (0.635 vs 0.655 baseline).

The failure mode to watch is timing drift: if SDEdit shifts the audio, the beats read
off the regenerated version are wrong for the original. Measured here by cross-
correlation lag between source and output.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/sogang/jaehoon/Analyze-SMC")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "sa3"))
ANN = Path("/disk1/jaehoon/dataset_store/beat_this_annotations/smc/annotations/beats")
AUD = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"


def lag_ms(a, b, sr, max_ms=200):
    n = int(max_ms * sr / 1000)
    a = a[: min(len(a), len(b))]; b = b[: len(a)]
    a = a - a.mean(); b = b - b.mean()
    c = np.correlate(b[n:-n], a, mode="valid")
    return (np.argmax(c) - n) / sr * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--prompt", default="heavy percussion, strong drums, clear steady beat")
    ap.add_argument("--n", type=int, default=27)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--cfg-scale", type=float, default=6.0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="cache/percussify.json")
    a = ap.parse_args()

    import torch, soundfile as sf, soxr
    from sa3_beat.build import load_beat_model
    from sa3_beat.generate import beat_conditioning, _patch_conditioning_inputs
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    from smc_eval import fold_songs, load_net
    from beat_this.preprocessing import LogMelSpect
    from beat_this.inference import split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from training.evaluation import beat_f_measure

    dev = f"cuda:{a.gpu}"
    model, cfg = load_beat_model(device=dev)          # NO controlnet adapter loaded
    _patch_conditioning_inputs(model)
    io_ch = cfg["model"]["io_channels"]
    sr_gen = cfg["sample_rate"]; ds = 4096
    tracker = load_net("fold0", dev); tracker.eval()
    spect = LogMelSpect(device=dev); post = Postprocessor(type="minimal", fps=50)

    songs = fold_songs(0)[: a.n]
    print(f"[percussify] {len(songs)} SMC fold-0 songs, prompt='{a.prompt}'", flush=True)

    def track(y22):
        with torch.inference_mode():
            S = spect(torch.from_numpy(y22).to(dev))
            pr = split_predict_aggregate(S, 1500, 6, "keep_first", tracker)
            bt, _ = post(pr["beat"].float(), pr["downbeat"].float())
        return np.asarray(bt)

    res = {"baseline": {}, "runs": {}}
    for stem, wav, truth in songs:
        y, sr = sf.read(str(wav), dtype="float32", always_2d=True); y = y.mean(axis=1)
        y22 = soxr.resample(y, sr, 22050) if sr != 22050 else y
        res["baseline"][stem] = float(beat_f_measure(truth, track(y22)))
    print(f"[percussify] baseline (original audio) F={np.mean(list(res['baseline'].values())):.4f}", flush=True)

    for nl in a.noise:
        fs, lags = {}, []
        for stem, wav, truth in songs:
            y, sr = sf.read(str(wav), dtype="float32", always_2d=True); y = y.mean(axis=1)
            yg = soxr.resample(y, sr, sr_gen) if sr != sr_gen else y
            n = (len(yg) // ds) * ds
            yg = yg[:n]
            init = torch.from_numpy(np.stack([yg, yg])).to(dev)   # (channels, samples)
            # empty beat grid -> the conditioner emits zeros, i.e. NO beat guidance.
            # SA3 must infer the pulse from the audio itself; that is the whole point.
            n_lat = n // ds
            item = beat_conditioning(np.array([]), None, n_lat, a.prompt, n / sr_gen)
            cond = model.conditioner([item], dev)
            cond["inpaint_mask"] = [torch.zeros(1, 1, n_lat, device=dev), None]
            cond["inpaint_masked_input"] = [torch.zeros(1, io_ch, n_lat, device=dev), None]
            out = generate_diffusion_cond(
                model, steps=a.steps, cfg_scale=a.cfg_scale,
                conditioning_tensors=cond,
                sample_size=n, seed=0, device=dev,
                init_audio=(sr_gen, init), init_noise_level=nl)
            o = out.squeeze(0).float().cpu().numpy()
            o = o.mean(axis=0) if o.ndim > 1 else o
            lags.append(lag_ms(yg[: len(o)], o, sr_gen))
            o22 = soxr.resample(o, sr_gen, 22050)
            fs[stem] = float(beat_f_measure(truth, track(o22)))
        res["runs"][str(nl)] = {"F": fs, "lag_ms": lags}
        base = np.array([res["baseline"][s] for s in fs])
        new = np.array([fs[s] for s in fs])
        print(f"  noise={nl:.2f}  F={new.mean():.4f}  (baseline {base.mean():.4f}, "
              f"delta {new.mean()-base.mean():+.4f})  {int((new>base).sum())}/{len(fs)} improved  "
              f"|lag| median {np.median(np.abs(lags)):.0f} ms", flush=True)
    json.dump(res, open(a.out, "w"))
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
