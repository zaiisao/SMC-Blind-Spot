"""Time-stretch a generated corpus to SMC tempos, moving labels with the audio.

The 2x2 on 2026-08-24 showed tempo, not material, is what defeats the generator:
percussive prompts score CMLt 0.5054 at 100-130 bpm and 0.2147 at SMC tempos, and
asking for drums at 50-70 bpm is the worst cell we measured (33.3% unfollowable).
SA3 wants to play at its own tempo and fights a slow grid.

So generate where the model works, then slow the result down. Stretching is
label-preserving by construction -- beat times scale by exactly 1/rate -- and we
already know stretched audio stays trackable, since warp_SMC (real SMC slowed the
same way) scores F 0.6184.

What this cannot fix is metrical ambiguity: AMLt-CMLt is 0.17-0.27 on every
generated corpus against 0.035 on real audio. Stretching moves the tempo, not the
model's indecision about which level is the beat.
"""
import argparse, os, glob
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf


def stretch_one(args):
    beats_path, out_dir, lo, hi, seed = args
    wav_path = beats_path[:-6] + ".wav"
    if not os.path.exists(wav_path):
        return None
    t = np.loadtxt(beats_path, ndmin=2)
    if t.size == 0 or len(t) < 8:
        return None
    beats = t[:, 0]
    src_bpm = 60.0 / float(np.median(np.diff(beats)))

    rng = np.random.RandomState(seed)
    want = rng.uniform(lo, hi)
    rate = want / src_bpm                      # <1 slows the audio down
    if not (0.2 < rate < 2.0):
        return None

    y, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)

    # sox's `tempo` (WSOLA) rather than a phase vocoder. Measured 2026-08-24:
    # librosa's phase vocoder at rate 0.639 LOST 50% of onset crest factor across
    # the corpus and audibly turned clips into a reverberant wash ("recorded in a
    # shower"). sox preserves attacks and actually raises crest ~10%, since slowing
    # separates transients further without smearing them.
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        a, b = os.path.join(td, "i.wav"), os.path.join(td, "o.wav")
        sf.write(a, y, sr, format="WAV")
        r = subprocess.run(["sox", a, b, "tempo", f"{rate:.6f}"],
                           capture_output=True)
        if r.returncode != 0 or not os.path.exists(b):
            return None
        ys, sr = sf.read(b, dtype="float32", always_2d=True)
        ys = ys.mean(axis=1)

    # time_stretch by `rate` maps t -> t / rate, so labels scale the same way.
    nb = beats / rate
    dur = len(ys) / sr
    keep = nb < dur
    nb = nb[keep]
    if len(nb) < 8:
        return None

    base = os.path.basename(beats_path)[:-6]
    sf.write(os.path.join(out_dir, base + ".wav"), ys, sr, format="WAV")
    cols = t[keep] if t.shape[1] > 1 else None
    with open(os.path.join(out_dir, base + ".beats"), "w") as fh:
        for i, b in enumerate(nb):
            if cols is not None and cols.shape[1] > 1:
                fh.write(f"{b:.6f}\t{int(cols[i, 1])}\n")
            else:
                fh.write(f"{b:.6f}\n")
    return (src_bpm, want)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bpm-range", type=float, nargs=2, default=[44.0, 114.0],
                    help="SMC's empirical p10..p90")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(a.gen_dir, "*.beats")))
    jobs = [(f, a.out_dir, a.bpm_range[0], a.bpm_range[1], i) for i, f in enumerate(fs)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = [r for r in ex.map(stretch_one, jobs) if r is not None]
    src = np.array([r[0] for r in res]); dst = np.array([r[1] for r in res])
    print(f"[stretch] {len(res)}/{len(fs)} clips -> {a.out_dir}")
    print(f"[stretch] bpm {np.median(src):.1f} -> {np.median(dst):.1f} "
          f"(median rate {np.median(dst/src):.3f})")


if __name__ == "__main__":
    main()
