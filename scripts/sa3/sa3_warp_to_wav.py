"""Re-materialize the warp-augment corpus as AUDIO for Beat This fine-tuning.

sa3_warp_augment.py warped real annotated audio through a piecewise-linear
time map but only saved the SA3 VAE *latents* (its consumer was the latent
ControlNet). The Beat This fine-tune arm needs the warped waveforms themselves.

This script replays the exact same corpus generation -- same source pool
(train_excl_fold0.txt), same 2 variants, same rate_curve(depth=0.20,
phase=2*pi*v/2, global_rate=1.0), same max 60 s -- which is fully
deterministic (verified: recomputed warped beat times match the stored
``*__w{v}.beats`` to 0.000000), and writes the warped waveform as mono
16-bit wav at the source 44.1 kHz instead of encoding it. The existing
``.beats`` files are copied verbatim: they are exact by construction.

CPU only; the phase vocoder needs no GPU.
"""
import argparse, math, shutil, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "sa3"))

from sa3_beat.data import load_beats  # noqa: E402
from sa3_warp_augment import rate_curve, warp_audio  # noqa: E402

AUDIO = Path("/disk1/jaehoon/dataset_store/audio_by_stem")


def _produce(job):
    import librosa, soundfile as sf
    uid, v, cache, out_dir, depth, variants, max_s, n_fft, hop = job
    try:
        C, O = Path(cache), Path(out_dir)
        out_uid = f"{uid}__w{v}"
        wav_out = O / f"{out_uid}.wav"
        src_beats = C / f"{out_uid}.beats"
        if not src_beats.exists():
            return ("no-beats", out_uid)
        if wav_out.exists() and (O / f"{out_uid}.beats").exists():
            return ("cached", out_uid)
        beats, _ = load_beats(C / f"{uid}.beats")
        corpus, stem = uid.split("__", 1)
        y, sr = librosa.load(str(AUDIO / corpus / f"{stem}.wav"),
                             sr=44100, mono=True, duration=max_s)
        dur = len(y) / sr
        beats = beats[beats < dur]
        if len(beats) < 8:
            return ("short", out_uid)
        edges, rates = rate_curve(dur, depth, phase=2 * math.pi * v / variants,
                                  global_rate=1.0)
        yw = warp_audio(y[None, :], sr, edges, rates, n_fft, hop)[0]
        peak = float(np.max(np.abs(yw))) if yw.size else 0.0
        if peak > 0.999:                       # phase vocoder can overshoot
            yw = yw * (0.999 / peak)
        tmp = wav_out.with_suffix(".wav.part")
        sf.write(str(tmp), yw.astype(np.float32), sr, subtype="PCM_16",
                 format="WAV")
        tmp.rename(wav_out)
        shutil.copy(src_beats, O / f"{out_uid}.beats")
        return ("ok", out_uid)
    except Exception as e:                                       # noqa: BLE001
        return ("fail", f"{uid} v{v}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="/disk1/jaehoon/sa3_beat_cache")
    ap.add_argument("--out", default="/disk1/jaehoon/warp_wav_corpus")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--depth", type=float, default=0.20)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--workers", type=int, default=56)
    args = ap.parse_args()

    import multiprocessing as mp
    C = Path(args.cache); O = Path(args.out); O.mkdir(parents=True, exist_ok=True)
    train = [l.strip()[:-4] for l in open(C / f"train_excl_fold{args.fold}.txt")
             if l.strip()]
    jobs = [(uid, v, str(C), str(O), args.depth, args.variants,
             args.max_seconds, args.n_fft, args.hop)
            for uid in train for v in range(args.variants)]
    print(f"[plan] {len(train)} tracks x {args.variants} variants = {len(jobs)} wavs "
          f"-> {O}", flush=True)

    from collections import Counter
    stats, done = Counter(), 0
    with mp.Pool(processes=args.workers) as pool:
        for status, msg in pool.imap_unordered(_produce, jobs, chunksize=4):
            stats[status] += 1; done += 1
            if status == "fail":
                print(f"[warp-fail] {msg}", flush=True)
            if done % 200 == 0:
                print(f"  [{done}/{len(jobs)}] {dict(stats)}", flush=True)
    print(f"[done] {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
