"""Keep only generated clips whose own label is the best explanation of the audio.

Measured on the current corpus: on slow clips (<55 bpm) the labelled grid wins
against the 1.5x/2x alternatives only 51% of the time, and mean adherence is
+0.123 against +0.279 for fast clips. So about half the slow training data does
not clearly demonstrate the thing it was generated to teach. Quantity was never
the problem -- 3,977 clips bought nothing.

A clip is kept only if ALL hold:
  * the labelled grid scores higher than the 0.5x, 1.5x and 2x grids
    (margin test -- the label must be the BEST story, not merely a good one)
  * labelled phase_align >= --min-align (absolute floor)
  * the half-beat-shifted grid scores negative (energy sits ON beats, not between)

Only phase_align is computed -- no rotation null -- which is ~200x faster than
sa3_score_corpus.py and lets the whole corpus be filtered in minutes.
"""
import argparse, json, os
from pathlib import Path
import numpy as np

SR_TARGET = None   # use each file's native rate; passing the wrong sr silently
                   # halves every number (this bug invalidated an earlier pass)


def phase_align(env, times, beats):
    beats = np.asarray(beats, float)
    if len(beats) < 4:
        return 0.0
    idx = np.searchsorted(beats, times) - 1
    ok = (idx >= 0) & (idx < len(beats) - 1)
    if not ok.any():
        return 0.0
    i = idx[ok]
    span = np.maximum(beats[i + 1] - beats[i], 1e-9)
    ph = 2 * np.pi * (times[ok] - beats[i]) / span
    w = env[ok] ** 2
    if w.sum() <= 0:
        return 0.0
    return float(np.real(np.sum(w * np.exp(1j * ph)) / w.sum()))


def score_clip(path):
    import soundfile as sf, librosa
    b = np.loadtxt(str(path), ndmin=2)
    if b.size == 0 or len(b) < 8:
        return None
    b = b[:, 0]
    wav = str(path)[:-6] + ".wav"
    y, sr = sf.read(wav, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env = np.maximum(env - np.median(env), 0.0)
    t = np.arange(len(env)) * hop / sr
    ibi = float(np.median(np.diff(b)))
    grids = {
        "x1": b,
        "x0.5": b[::2],
        "x1.5": np.arange(b[0], b[-1] + 1e-9, ibi / 1.5),
        "x2": np.arange(b[0], b[-1] + 1e-9, ibi / 2.0),
        "half": b + ibi / 2.0,
    }
    s = {k: phase_align(env, t, g) for k, g in grids.items()}
    return dict(stem=Path(path).stem, bpm=60.0 / ibi, **s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", default="/disk1/jaehoon/sa3_gen_corpus_fold0")
    ap.add_argument("--min-align", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default="cache/corpus_filter.json")
    a = ap.parse_args()

    import multiprocessing as mp
    files = [p for p in sorted(Path(a.gen_dir).glob("*.beats")) if p.stat().st_size > 0]
    print(f"[filter] scoring {len(files)} clips on {a.workers} workers", flush=True)
    with mp.Pool(a.workers) as pool:
        rows = [r for r in pool.imap_unordered(score_clip, files, chunksize=16) if r]

    keep = []
    for r in rows:
        best = r["x1"] > max(r["x0.5"], r["x1.5"], r["x2"])
        if best and r["x1"] >= a.min_align and r["half"] < 0:
            keep.append(r["stem"])
    kept = set(keep)
    bpm = np.array([r["bpm"] for r in rows])
    al = np.array([r["x1"] for r in rows])
    k = np.array([r["stem"] in kept for r in rows])
    print(f"\n[filter] kept {k.sum()}/{len(rows)} ({100*k.mean():.0f}%)")
    print(f"         adherence {al.mean():+.4f} -> {al[k].mean():+.4f}")
    print(f"\n{'tempo band':16s} {'n':>5s} {'kept':>6s} {'%':>5s} {'align before':>13s} {'after':>8s}")
    for lo, hi in [(0, 55), (55, 70), (70, 90), (90, 999)]:
        m = (bpm >= lo) & (bpm < hi)
        if m.sum():
            print(f"  {lo:3.0f}-{hi:3.0f} bpm    {m.sum():5d} {int((m&k).sum()):6d} "
                  f"{100*(m&k).sum()/m.sum():5.0f} {al[m].mean():13.4f} "
                  f"{al[m&k].mean() if (m&k).any() else float('nan'):8.4f}")
    json.dump({"kept": sorted(kept), "min_align": a.min_align,
               "per_clip": rows}, open(a.out, "w"))
    Path(a.out).with_suffix(".txt").write_text("\n".join(sorted(kept)) + "\n")
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
