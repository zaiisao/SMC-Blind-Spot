"""Build a listening set with ticks on the labelled beats, stratified by bucket.

Motivation (2026-08-24): every corpus-quality number so far is scored with Beat
This, which is the very model that fails on hard music. A clip we call
"unfollowable" (AMLt<0.2) is therefore ambiguous: it may be degenerate audio with
no rhythm, or audio with real beats the tracker cannot find -- and the second kind
is the training data we actually want. Optimising the corpus against the tracker
pushes toward material the tracker already handles, which is the redundancy trap
the percussion work already paid for.

Only ears resolve that. This writes clips with a tick on every labelled beat,
sampled across buckets, plus a signal-only pulse-clarity score that uses no
tracker: the autocorrelation peak of the onset envelope at the labelled period.

  pulse_clarity HIGH + AMLt LOW  -> hard but real: the valuable case
  pulse_clarity LOW  + AMLt LOW  -> degenerate audio: drop it
"""
import argparse, csv, os, glob
import numpy as np
import soundfile as sf


def pulse_clarity(y, sr, period_s):
    """Autocorrelation of the onset envelope at the labelled beat period.

    Tracker-free: it asks whether energy actually recurs at that period, not
    whether any model agrees it is the beat.
    """
    import librosa
    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env = env - env.mean()
    if env.std() < 1e-9:
        return 0.0
    env = env / env.std()
    lag = int(round(period_s * sr / hop))
    if lag < 2 or lag >= len(env):
        return 0.0
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = max(1, int(lag * 0.9)), min(len(ac) - 1, int(lag * 1.1) + 1)
    return float(ac[lo:hi].max()) if hi > lo else 0.0


def add_ticks(y, sr, beats, downbeats=None):
    out = y.copy()
    n = len(out)
    for i, b in enumerate(beats):
        s = int(b * sr)
        if not (0 <= s < n):
            continue
        dur = int(0.030 * sr)
        t = np.arange(min(dur, n - s)) / sr
        freq = 1800.0 if (downbeats is not None and i in downbeats) else 1100.0
        env = np.exp(-t * 90.0)
        out[s:s + len(t)] += 0.35 * env * np.sin(2 * np.pi * freq * t)
    return np.clip(out, -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--perclip-csv", required=True, help="from compare_corpora --dump-dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-bucket", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    rows = {r["clip"]: r for r in csv.DictReader(open(a.perclip_csv))}
    buckets = {"correct": [], "wrong_level": [], "unfollowable": [], "middling": []}
    for clip, r in rows.items():
        C, A = float(r["CMLt"]), float(r["AMLt"])
        if C > 0.5:
            buckets["correct"].append(clip)
        elif A > 0.5 and C < 0.2:
            buckets["wrong_level"].append(clip)
        elif A < 0.2:
            buckets["unfollowable"].append(clip)
        else:
            buckets["middling"].append(clip)

    rng = np.random.RandomState(0)
    manifest = []
    for b, clips in buckets.items():
        if not clips:
            continue
        sel = [clips[i] for i in rng.choice(len(clips), min(a.per_bucket, len(clips)),
                                            replace=False)]
        for clip in sel:
            wav = os.path.join(a.gen_dir, clip + ".wav")
            bt = os.path.join(a.gen_dir, clip + ".beats")
            if not (os.path.exists(wav) and os.path.exists(bt)):
                continue
            y, sr = sf.read(wav, dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            t = np.loadtxt(bt, ndmin=2)
            beats = t[:, 0]
            db = set(np.where(t[:, 1] == 1)[0]) if t.shape[1] > 1 else None
            period = float(np.median(np.diff(beats)))
            pc = pulse_clarity(y, sr, period)
            out = os.path.join(a.out_dir, f"{b}__pc{pc:.2f}__{clip}.wav")
            sf.write(out, add_ticks(y, sr, beats, db), sr, format="WAV")
            manifest.append((b, clip, round(60.0 / period, 1), pc,
                             float(rows[clip]["CMLt"]), float(rows[clip]["AMLt"])))

    manifest.sort()
    print(f"{'bucket':<14} {'bpm':>6} {'pulse_clarity':>14} {'CMLt':>7} {'AMLt':>7}")
    for b, clip, bpm, pc, C, A in manifest:
        print(f"{b:<14} {bpm:6.1f} {pc:14.3f} {C:7.3f} {A:7.3f}")
    print(f"\n[listen] {len(manifest)} clips -> {a.out_dir}")


if __name__ == "__main__":
    main()
