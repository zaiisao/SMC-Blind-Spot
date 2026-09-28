import argparse, os, sys, glob, json, random
from pathlib import Path
import numpy as np
import soundfile as sf


CACHE = "/disk1/jaehoon/sa3_beat_cache"
AUDIO = "/disk1/jaehoon/dataset_store/audio_by_stem"
POOL = os.environ.get("SA3_POOL",
    str(Path(__file__).resolve().parents[1] / "data" / "pools" / "slow_under70.json"))


def build_pool(datasets, exclude_folds=(0,)):
    import json
    stems = []
    for r in json.load(open(POOL)):
        if datasets and r["corpus"] not in datasets:
            continue
        if r["fold"] in exclude_folds:
            continue
        if not os.path.exists(r["audio"]) or not os.path.exists(r["beats"]):
            continue
        stems.append((r["name"], r["audio"], r["beats"], r["fold"]))
    return stems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--exclude-folds", type=int, nargs="*", default=[0])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--adapter", default="/disk1/jaehoon/sa3_beat_runs/cn_fold0_rw/"
                                         "beat_adapter_final.safetensors")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--cfg-scale", type=float, default=6.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    import torch, soxr
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    from sa3_beat.build import load_beat_model
    from sa3_beat.generate import load_adapter, beat_conditioning, _patch_conditioning_inputs

    pool = build_pool(set(a.datasets), tuple(a.exclude_folds))
    import collections
    fc = collections.Counter(f for _, _, _, f in pool)
    print(f"[pool] {len(pool)} sources, folds excluded {a.exclude_folds}, "
          f"remaining folds " + " ".join(f'f{k}:{fc[k]}' for k in sorted(fc)), flush=True)
    rng = random.Random(0)
    rng.shuffle(pool)
    jobs = [(i, pool[i % len(pool)]) for i in range(a.n)]
    jobs = [j for j in jobs if j[0] % a.shards == a.shard]

    dev = f"cuda:{a.gpu}"
    model, cfg = load_beat_model(device=dev)
    load_adapter(model, a.adapter, dev)
    _patch_conditioning_inputs(model)
    sr_gen = cfg["sample_rate"]
    ds = model.pretransform.downsampling_ratio
    io_ch = cfg["model"]["io_channels"]

    made = 0
    for idx, (stem, wav, bt, fold) in jobs:
        out_base = os.path.join(a.out_dir, f"sd_{idx:05d}")
        if os.path.exists(out_base + ".beats") and os.path.getsize(out_base + ".beats") > 0 \
           and os.path.exists(out_base + ".wav") and os.path.getsize(out_base + ".wav") > 0:
            made += 1
            continue
        # draw all randomness BEFORE any skip, so resuming matches a clean run
        r = random.Random(idx)
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
                                key=lambda i: abs(beats[i] - r.uniform(lo, hi)))])
        seg = yg[int(start * sr_gen): int((start + a.seconds) * sr_gen)]
        n = (len(seg) // ds) * ds
        if n < ds * 8:
            continue
        seg = seg[:n]
        m = (beats >= start) & (beats < start + n / sr_gen)
        b = beats[m] - start
        d = (beats[m][db_col[m] == 1] - start) if db_col is not None else None
        if len(b) < 8:
            continue

        init = torch.from_numpy(np.stack([seg, seg])).to(dev)
        n_lat = n // ds
        item = beat_conditioning(b, d, n_lat, a.prompt, n / sr_gen)
        cond = model.conditioner([item], dev)
        cond["inpaint_mask"] = [torch.zeros(1, 1, n_lat, device=dev), None]
        cond["inpaint_masked_input"] = [torch.zeros(1, io_ch, n_lat, device=dev), None]
        out = generate_diffusion_cond(
            model, steps=a.steps, cfg_scale=a.cfg_scale, conditioning_tensors=cond,
            sample_size=n, seed=idx, device=dev,
            init_audio=(sr_gen, init), init_noise_level=a.noise)
        o = out.squeeze(0).float().cpu().numpy()
        o = o.mean(axis=0) if o.ndim > 1 else o
        o = o / (np.abs(o).max() + 1e-9) * 0.9

        tmp = out_base + ".wav.tmp"
        sf.write(tmp, o, sr_gen, format="WAV")     # explicit format: .tmp hides it
        os.replace(tmp, out_base + ".wav")
        with open(out_base + ".beats.tmp", "w") as fh:
            for i, t in enumerate(b):
                fh.write(f"{t:.6f}\t{1 if (d is not None and t in d) else 2}\n")
        os.replace(out_base + ".beats.tmp", out_base + ".beats")
        json.dump({"source": stem, "source_wav": wav, "window_start": start,
                   "noise": a.noise, "n_beats": int(len(b))},
                  open(out_base + ".json", "w"))
        made += 1
        if made % 25 == 0:
            print(f"[bulk] shard {a.shard}: {made} clips", flush=True)
    print(f"[bulk] shard {a.shard} DONE {made} clips", flush=True)


if __name__ == "__main__":
    main()
