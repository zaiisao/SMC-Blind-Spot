"""Decompose the generation pipeline's transient loss: autoencoder floor vs
diffusion/SDEdit.

For N clips we already generated at noise 0.4, take the SAME source window and
measure beat-relevant statistics at three points:
    A. original source audio            (upper bound)
    B. VAE encode->decode round trip    (the autoencoder floor: no diffusion)
    C. the SDEdit 0.4 output on disk    (full pipeline)

Statistics per clip, computed on the beat_this log-mel at 22050/50fps:
    flux crest (transient peakiness), onset-evidence z at the labelled beats.

If B ~= A, the VAE is innocent and the transient loss happens in diffusion; a
percussion-control project should target the sampler/conditioning.
If B ~= C, the loss is already in the autoencoder and no amount of diffusion
control can fix it; the project must start at the VAE.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = "/home/sogang/jaehoon/Analyze-SMC"
BT = f"{REPO}/third-party/beat_this"
SD = "/disk1/jaehoon/sdedit_slow"
FPS = 50
SR_BT = 22050


def flux(sp):
    return np.concatenate([[0.0], np.maximum(np.diff(sp, axis=0), 0).sum(1)])


def crest(sp):
    f = flux(sp)
    return float(f.max() / (f.mean() + 1e-9))


def zscore(sp, beats, rng, tol=3, n=200):
    f = flux(sp)
    f = (f - np.median(f)) / (f.std() + 1e-9)
    T = len(f)
    idx = np.round(np.asarray(beats) * FPS).astype(int)
    idx = idx[(idx >= tol) & (idx < T - tol)]
    if len(idx) < 3:
        return np.nan
    w = np.arange(-tol, tol + 1)

    def sc(o):
        return f[(o[:, None] + w[None, :]).clip(0, T - 1)].max(1).mean()

    obs = sc(idx)
    span = T - 2 * tol
    null = np.array([sc(((idx - tol + rng.integers(1, span)) % span) + tol)
                     for _ in range(n)])
    return float((obs - null.mean()) / (null.std() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, REPO)
    sys.path.insert(0, BT)
    sys.path.insert(0, f"{REPO}/third-party/stable-audio-tools")
    import soxr
    import torch
    from beat_this.preprocessing import LogMelSpect
    from sa3_beat.build import load_beat_model

    dev = f"cuda:{a.gpu}"
    mel = LogMelSpect(SR_BT)
    model, cfg = load_beat_model(device=dev)
    sr_gen = cfg["sample_rate"]
    pre = model.pretransform
    rng = np.random.default_rng(0)

    jfs = sorted(Path(SD).glob("*.json"))[: a.n]
    fh = open(a.out, "w")
    fh.write("stem,bpm,crest_orig,crest_vae,crest_gen,z_orig,z_vae,z_gen\n")
    done = 0
    for jf in jfs:
        j = json.load(open(jf))
        stem = jf.stem
        beats = np.loadtxt(Path(SD) / f"{stem}.beats", ndmin=2)[:, 0]
        gen_wav, gsr = sf.read(Path(SD) / f"{stem}.wav", dtype="float32", always_2d=True)
        gen_mono = gen_wav.mean(1)
        dur = len(gen_mono) / gsr

        src, ssr = sf.read(j["source_wav"], dtype="float32", always_2d=True)
        src = src.mean(1)
        s0 = int(round(j["window_start"] * ssr))
        seg = src[s0: s0 + int(round(dur * ssr))]
        if len(seg) < ssr:
            continue

        # A. original, at beat_this rate
        seg_bt = soxr.resample(seg, in_rate=ssr, out_rate=SR_BT)
        # B. VAE round trip at the generator's rate, then to beat_this rate
        seg_gen_sr = soxr.resample(seg, in_rate=ssr, out_rate=sr_gen)
        x = torch.tensor(seg_gen_sr, device=dev)[None, None, :].repeat(1, cfg["audio_channels"], 1)
        with torch.no_grad():
            lat = pre.encode(x)
            rec = pre.decode(lat)[0].mean(0).float().cpu().numpy()
        rec_bt = soxr.resample(rec, in_rate=sr_gen, out_rate=SR_BT)
        # C. the SDEdit output
        gen_bt = soxr.resample(gen_mono, in_rate=gsr, out_rate=SR_BT)

        import torch as _t
        sps = {}
        for k, w in [("orig", seg_bt), ("vae", rec_bt), ("gen", gen_bt)]:
            with torch.no_grad():
                sps[k] = mel(_t.as_tensor(np.ascontiguousarray(w), dtype=_t.float32)).numpy()
        fh.write(f"{stem},{60/np.median(np.diff(beats)):.2f},"
                 f"{crest(sps['orig']):.3f},{crest(sps['vae']):.3f},{crest(sps['gen']):.3f},"
                 f"{zscore(sps['orig'],beats,rng):.3f},{zscore(sps['vae'],beats,rng):.3f},"
                 f"{zscore(sps['gen'],beats,rng):.3f}\n")
        done += 1
        if done % 10 == 0:
            fh.flush()
            print(f"  {done}/{len(jfs)}", flush=True)
    fh.close()
    print(f"  wrote {a.out} ({done} clips)")


if __name__ == "__main__":
    main()
