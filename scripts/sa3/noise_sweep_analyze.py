"""Compare SDEdit noise levels on the two axes that matter.

VALIDITY  -- does the audio still support its inherited labels?
            ac_period_vs_best (the workflow's winning filter): autocorrelation of
            spectral flux at the labelled period minus the max over the surrounding
            lag band. <= 0 always; 0 means the labelled period IS the dominant
            periodicity. No beat model involved.
            Plus paired retention / sharp_ratio against the clip's own source.

NOVELTY   -- did the generation actually change anything? At noise -> 0 the output
            is the source, which is a duplicate and worthless as augmentation.
            Measured as correlation between the generated and source flux curves,
            and as log-mel spectral distance. High correlation = low novelty.

Both axes are needed: validity alone would just recommend noise 0.

Clips at index i are the SAME source and window at every noise level (the bulk
script maps job i -> pool[i % len(pool)] and draws the window from Random(i)), so
everything here is paired. Works on partial output -- scores whatever exists.

CPU only by default; --gpu N additionally runs a beat model for F_gen / F_src.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

BT = "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"
FPS = 50
SR = 22050


def flux(spect):
    d = np.diff(spect, axis=0)
    return np.concatenate([[0.0], np.maximum(d, 0).sum(axis=1)])


def evidence(spect, beats):
    beats = np.asarray(beats, float)
    if len(beats) < 4:
        return np.nan
    f = flux(spect)
    f = f - f.mean()
    s = f.std()
    if s < 1e-9:
        return np.nan
    f = f / s
    T = len(f)
    n = int(2 ** np.ceil(np.log2(2 * T)))
    F = np.fft.rfft(f, n)
    ac = np.fft.irfft(F * np.conj(F), n)[:T]
    ac = ac / (np.arange(T, 0, -1) * (ac[0] / T + 1e-12))
    P = np.median(np.diff(beats)) * FPS
    lo, hi = int(np.floor(0.35 * P)), min(int(np.ceil(4.0 * P)), T - 2)
    if P >= T - 2 or hi <= lo + 2:
        return np.nan
    i0 = int(np.floor(P))
    fr = P - i0
    return float(ac[i0] * (1 - fr) + ac[i0 + 1] * fr - ac[lo:hi + 1].max())


def zflux(spect):
    f = flux(spect)
    return (f - np.median(f)) / (f.std() + 1e-9)


def paired_stats(gen_sp, src_sp, beats, tol=3):
    """retention, sharp_ratio, and novelty (flux correlation) against the source."""
    zg, zs = zflux(gen_sp), zflux(src_sp)
    T = min(len(zg), len(zs))
    zg, zs = zg[:T], zs[:T]
    idx = np.round(np.asarray(beats) * FPS).astype(int)
    idx = idx[(idx >= tol) & (idx < T - tol)]
    if len(idx) < 3:
        return np.nan, np.nan, np.nan
    w = np.arange(-tol, tol + 1)
    pos = (idx[:, None] + w[None, :]).clip(0, T - 1)
    pg, ps = zg[pos].max(1), zs[pos].max(1)
    retention = float(np.mean(pg >= 0.5 * ps))
    sharp = float(np.mean(pg) / (np.mean(ps) + 1e-9))
    novelty_corr = float(np.corrcoef(zg, zs)[0, 1])
    return retention, sharp, novelty_corr


def load_mel(wav_path, mel, torch, soxr, start=None, dur=None):
    y, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    y = y.mean(1)
    if start is not None:
        s0 = int(round(start * sr))
        s1 = s0 + int(round(dur * sr))
        y = y[s0:s1]
    if len(y) < sr // 2:
        return None
    if sr != SR:
        y = soxr.resample(y, in_rate=sr, out_rate=SR)
    with torch.no_grad():
        return mel(torch.as_tensor(y, dtype=torch.float32)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="noise=DIR pairs, e.g. 0.2=/disk1/jaehoon/sdedit_n02")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.chdir(BT)
    sys.path.insert(0, BT)
    import soxr
    import torch
    from beat_this.preprocessing import LogMelSpect
    mel = LogMelSpect(SR)

    src_cache = {}
    fh = open(a.out, "w")
    fh.write("noise,idx,stem,source,bpm,ev_gen,ev_src,retention,sharp_ratio,novelty_corr\n")
    for spec in a.dirs:
        noise, d = spec.split("=", 1)
        clips = sorted(Path(d).glob("*.wav"))
        print(f"  noise {noise}: {len(clips)} clips in {d}", flush=True)
        for w in clips:
            stem = w.stem
            jf = Path(d) / f"{stem}.json"
            bf = Path(d) / f"{stem}.beats"
            if not (jf.exists() and bf.exists()):
                continue
            j = json.load(open(jf))
            b = np.loadtxt(bf, ndmin=2)
            beats = b[:, 0] if b.ndim == 2 else b
            if len(beats) < 4:
                continue
            g = load_mel(str(w), mel, torch, soxr)
            if g is None:
                continue
            dur = g.shape[0] / FPS
            key = (j["source_wav"], round(j["window_start"], 3), round(dur, 2))
            if key not in src_cache:
                src_cache[key] = load_mel(j["source_wav"], mel, torch, soxr,
                                          start=j["window_start"], dur=dur)
            s = src_cache[key]
            if s is None:
                continue
            ret, sh, nov = paired_stats(g, s, beats)
            fh.write(f"{noise},{int(stem.split('_')[1])},{stem},{j['source']},"
                     f"{60.0/np.median(np.diff(beats)):.3f},"
                     f"{evidence(g,beats):.6f},{evidence(s,beats):.6f},"
                     f"{ret:.4f},{sh:.4f},{nov:.4f}\n")
        fh.flush()
    fh.close()
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
