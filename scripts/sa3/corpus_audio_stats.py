"""How far is a generated corpus from real music, at the signal level?

The tracker-critic fine-tune (arc_tracker_ft.py) optimises generated audio to be
LEGIBLE TO A BEAT TRACKER. That is exactly the setup where a model can win the
stated objective by degenerate means -- superimposing click-like transients on
the beat grid, thinning the spectrum, collapsing to near-silence between beats --
and the paired beat-tracking test would happily report success. Beat legibility
is not audio realism, and nothing measured so far separates them.

So: measure each generated clip against THE SAME WINDOW of its own source
recording, and report generated-minus-source per corpus. A corpus that is close
to real music sits near zero on every row; a corpus that gamed the critic shows
up as inflated crest, raised flatness, collapsed RMS, or a large log-mel
distance while its onset evidence looks fine.

  rms          mean |amplitude|            has it collapsed to near-silence?
  crest        flux.max() / flux.mean()    click-like spikes on the grid?
  flatness     spectral flatness           has it become noise?
  centroid     spectral centroid (Hz)      brightness shift
  onset_z      onset evidence at the annotated beats vs a circular-shift null
               (beat_this log-mel, 50 fps, tol +-3 frames, 200 shifts) --
               the same estimator as vae_floor_probe.py
  mel_l1       mean |log-mel(gen) - log-mel(src)|, the direct fidelity distance

Usage:
  corpus_audio_stats.py --source-json /disk1/jaehoon/sdedit_r3 \
      --corpus tf1=/disk1/jaehoon/sdedit_tf1_full r3=/disk1/jaehoon/sdedit_r3 \
      --limit 300 --out cache/corpus_audio_stats.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SR = 22050
FPS = 50
HOP = SR // FPS


def logmel(y, n_mels=128):
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=HOP,
                                       n_mels=n_mels, fmin=30, fmax=11000,
                                       power=1.0)
    return np.log1p(1000 * S)


def flux(mel):
    d = np.diff(mel, axis=1)
    return np.maximum(d, 0).sum(0)


def onset_z(f, beats, t0, n_null=200, tol=3, seed=0):
    """Evidence at the annotated beats against a circular-shift null."""
    if len(beats) < 3 or f.size < 10:
        return np.nan
    idx = np.round((np.asarray(beats) - t0) * FPS).astype(int)
    idx = idx[(idx >= tol) & (idx < f.size - tol)]
    if idx.size < 3:
        return np.nan
    fz = (f - f.mean()) / (f.std() + 1e-9)

    def score(shift):
        j = (idx + shift) % f.size
        w = np.stack([np.take(fz, (j + k) % f.size) for k in range(-tol, tol + 1)])
        return w.max(0).mean()

    obs = score(0)
    r = np.random.default_rng(seed)
    null = np.array([score(int(s)) for s in
                     r.integers(tol + 1, f.size - tol - 1, n_null)])
    return float((obs - null.mean()) / (null.std() + 1e-9))


def stats(y, beats, t0):
    m = logmel(y)
    f = flux(m)
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) + 1e-10
    return dict(
        rms=float(np.abs(y).mean()),
        crest=float(f.max() / (f.mean() + 1e-9)),
        flatness=float(np.exp(np.log(spec).mean()) / spec.mean()),
        centroid=float((np.arange(spec.size) * spec).sum() / spec.sum()
                       * (SR / 2) / spec.size),
        onset_z=onset_z(f, beats, t0),
    ), m


def load(p):
    y, sr = sf.read(str(p), dtype="float32", always_2d=True)
    y = y.mean(1)
    return soxr.resample(y, sr, SR) if sr != SR else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True,
                    help="name=/path/to/clip_dir, repeatable")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    corpora = dict(c.split("=", 1) for c in a.corpus)
    ref = list(corpora.values())[0]
    stems = sorted(p.stem for p in Path(ref).glob("*.json"))[: a.limit]
    print(f"  {len(stems)} stems x {len(corpora)} corpora", flush=True)

    rows = []
    for i, stem in enumerate(stems):
        j = json.load(open(Path(ref) / f"{stem}.json"))
        try:
            src = load(j["source_wav"])
        except Exception:
            continue
        beats = np.loadtxt(Path(ref) / f"{stem}.beats", ndmin=2)[:, 0]
        t0 = float(j["window_start"])

        # the source window, matched in length to the generated clip
        any_gen = load(Path(ref) / f"{stem}.wav")
        s0 = int(round(t0 * SR))
        seg = src[s0: s0 + len(any_gen)]
        if len(seg) < SR:
            continue
        s_stats, s_mel = stats(seg, beats, t0)
        rows.append(dict(stem=stem, corpus="source", **s_stats, mel_l1=0.0))

        for name, d in corpora.items():
            w = Path(d) / f"{stem}.wav"
            if not w.exists():
                continue
            y = load(w)
            n = min(len(y), len(seg))
            g_stats, g_mel = stats(y[:n], beats, t0)
            k = min(g_mel.shape[1], s_mel.shape[1])
            g_stats["mel_l1"] = float(np.abs(g_mel[:, :k] - s_mel[:, :k]).mean())
            rows.append(dict(stem=stem, corpus=name, **g_stats))
        if (i + 1) % 50 == 0:
            print(f"    {i+1} stems", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    def boot(d, n=10000, seed=0):
        r = np.random.default_rng(seed)
        d = np.asarray(d)[~np.isnan(d)]
        if d.size == 0:
            return (np.nan, np.nan)
        return tuple(np.percentile(
            d[r.integers(0, d.size, (n, d.size))].mean(1), [2.5, 97.5]))

    src = df[df.corpus == "source"].set_index("stem")
    print(f"\n{'metric':10} {'source':>10} " +
          " ".join(f"{c:>10}" for c in corpora) + "   paired gen-src (95% CI)")
    for met in ["rms", "crest", "flatness", "centroid", "onset_z", "mel_l1"]:
        line = f"{met:10} {src[met].mean():10.4f} "
        tail = []
        for c in corpora:
            g = df[df.corpus == c].set_index("stem")
            common = src.index.intersection(g.index)
            line += f"{g.loc[common, met].mean():10.4f} "
            d = (g.loc[common, met] - src.loc[common, met])
            lo, hi = boot(d)
            tail.append(f"{c}: {np.nanmean(d):+.3f} [{lo:+.3f},{hi:+.3f}]")
        print(line + "   " + "   ".join(tail))
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
