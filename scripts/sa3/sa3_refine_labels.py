"""Correct generated-clip labels to describe the audio that was actually produced.

Listening check (2026-08-24) found the failure is NOT missing beats: the clips have a
clear pulse, but the conditioning grid sits at the wrong phase -- one clip was audibly
on the offbeat. The released tracker scores our corpus 0.45-0.49 against its own labels
while scoring real time-warped audio 0.84, which is consistent with mislabelling rather
than unbeatable audio.

The conditioning grid is a REQUEST. If SA3 played it half a beat off, the true label is
where it played. Two corrections, both local and both grid-preserving:

  1. GLOBAL PHASE: compare onset energy on the grid against the half-beat-shifted grid.
     If the offbeat carries more energy, the whole clip is anti-phase; shift by IBI/2.
  2. LOCAL SNAP: move each beat to the strongest onset within +/-max_snap of it
     (default 0.20 of an IBI), keeping order and never crossing a neighbour.

This uses onset detection only -- never a beat tracker -- so it cannot become
self-training on tracker output (a route already refuted in this project).
"""
import argparse, json, os, shutil
from pathlib import Path
import numpy as np


def onset_env(y, sr, hop=128):
    import librosa
    e = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    e = np.maximum(e - np.median(e), 0.0)
    return e, np.arange(len(e)) * hop / sr


def energy_at(env, t_env, times, tol):
    k = max(1, int(round(tol / (t_env[1] - t_env[0]))))
    idx = np.clip(np.searchsorted(t_env, times), 0, len(env) - 1)
    return float(np.mean([env[max(0, i - k): i + k + 1].max() for i in idx]))


def refine(beats, y, sr, max_snap=0.20):
    env, te = onset_env(y, sr)
    if env.max() <= 0 or len(beats) < 4:
        return beats, False, 0
    ibi = float(np.median(np.diff(beats)))
    tol = 0.25 * ibi
    flipped = False
    half = beats + ibi / 2.0
    half = half[half < te[-1]]
    if len(half) >= 4 and energy_at(env, te, half, tol) > energy_at(env, te, beats, tol) * 1.15:
        beats = half
        flipped = True
    snap = max_snap * ibi
    out, moved = [], 0
    for j, b in enumerate(beats):
        lo, hi = b - snap, b + snap
        m = (te >= lo) & (te <= hi)
        if m.any() and env[m].max() > 0:
            cand = te[m][int(np.argmax(env[m]))]
            if (not out or cand > out[-1] + 0.4 * ibi):
                if abs(cand - b) > 1e-6:
                    moved += 1
                out.append(cand); continue
        out.append(b)
    return np.array(out), flipped, moved


_CFG = {}


def _one(bp):
    import soundfile as sf
    from pathlib import Path as _P
    bp = _P(bp); dst = _P(_CFG["dst"])
    arr = np.loadtxt(bp, ndmin=2)
    if arr.size == 0 or len(arr) < 8:
        return None
    b = arr[:, 0]
    y, sr = sf.read(str(bp.with_suffix(".wav")), dtype="float32", always_2d=True)
    nb, flip, moved = refine(b, y.mean(axis=1), sr, _CFG["max_snap"])
    np.savetxt(dst / bp.name, np.c_[nb, np.full(len(nb), 2)], fmt="%.6f\t%d")
    for ext in (".wav", ".json"):
        t = dst / (bp.stem + ext)
        if not t.exists():
            try: os.symlink(bp.with_suffix(ext), t)
            except FileExistsError: pass
    return flip, moved, len(nb)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-snap", type=float, default=0.20)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()

    import multiprocessing as mp
    src, dst = Path(a.gen_dir), Path(a.out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    _CFG["dst"] = str(dst); _CFG["max_snap"] = a.max_snap

    files = [str(p) for p in sorted(src.glob("*.beats")) if p.stat().st_size > 0]
    print(f"[refine] {len(files)} clips -> {dst}", flush=True)
    with mp.Pool(a.workers) as pool:
        res = [r for r in pool.map(_one, files, chunksize=16) if r]
    fl = sum(r[0] for r in res); mv = sum(r[1] for r in res); tot = sum(r[2] for r in res)
    print(f"[refine] phase-flipped {fl}/{len(res)} clips ({100*fl/len(res):.0f}%)")
    print(f"[refine] snapped {mv}/{tot} beats ({100*mv/tot:.0f}%)")


if __name__ == "__main__":
    main()
