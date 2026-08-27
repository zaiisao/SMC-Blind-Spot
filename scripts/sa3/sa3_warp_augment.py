"""Manufacture rubato training data by label-preserving non-uniform time warping.

The ControlNet learned "beats are periodic" because ~50% of the corpus is
rigidly metronomic (ballroom/hjdb/beatles median IBI cv ~0.02 against ASAP's
0.167). It cannot learn timing *deviation* from data that contains none.

A varying tempo curve fixes that at the source: warping audio and its
annotations through the SAME piecewise-linear time map yields genuine rubato
with labels that are exact by construction -- no model compliance required,
which is the §4 editing argument applied to the generator's own training set.

Audio is warped with a variable-rate phase vocoder rather than by stretching
segments separately and concatenating, so there are no boundary discontinuities
where the rate changes.

    --verify   warp a sample, score it against its WARPED labels with the
               non-circular adherence verifier, and compare to the unwarped
               original. If warping smears onsets, the labels stop describing
               the audio and we would be manufacturing an artifact distribution
               -- the Spleeter failure. Run this before generating a corpus.
"""

import argparse, json, math, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "sa3"))
SA3_DIR = Path("/disk1/jaehoon/sa3_small_music")

from sa3_beat.data import load_beats, warp_times_curve   # noqa: E402


def rate_curve(duration, depth, n_seg=24, period_seg=8.0, phase=0.0, global_rate=1.0):
    """Breathing tempo curve, optionally around a global rate != 1.

    ``global_rate < 1`` slows the whole track down. That is a separate axis from
    ``depth``: depth teaches timing DEVIATION, global_rate moves the track into a
    different tempo REGION. Both matter here -- the corpus median is 117 BPM
    while SMC's is 71.6, and only 14.4% of training tracks fall below 70 BPM
    against 45.8% of SMC.
    """
    edges = np.linspace(0.0, duration, n_seg + 1)
    k = np.arange(n_seg)
    rates = global_rate * (1.0 + depth * np.sin(2 * np.pi * k / period_seg + phase))
    return edges, rates


def variable_phase_vocoder(D, time_steps, hop_length):
    """Phase vocoder over an arbitrary (monotone) source-frame trajectory.

    Same algorithm as librosa's fixed-rate ``phase_vocoder``; the only change is
    that ``time_steps`` may be non-uniform, which is what makes the rate vary.
    """
    n_bins = D.shape[0]
    phase_advance = np.linspace(0, np.pi * hop_length, n_bins)
    out = np.zeros((n_bins, len(time_steps)), dtype=np.complex64)
    phase_acc = np.angle(D[:, 0])
    Dp = np.pad(D, [(0, 0), (0, 2)], mode="constant")
    for t, step in enumerate(time_steps):
        i = int(step); alpha = step - i
        col0, col1 = Dp[:, i], Dp[:, i + 1]
        mag = (1 - alpha) * np.abs(col0) + alpha * np.abs(col1)
        out[:, t] = mag * np.exp(1j * phase_acc)
        dphase = np.angle(col1) - np.angle(col0) - phase_advance
        dphase = dphase - 2 * np.pi * np.round(dphase / (2 * np.pi))
        phase_acc = phase_acc + phase_advance + dphase
    return out


def warp_audio(y, sr, edges, rates, n_fft=1024, hop=256):
    """Warp ``y`` (channels, n) through the same time map as warp_times_curve."""
    import librosa
    dst_edges = np.concatenate([[0.0], np.cumsum(np.diff(edges) / rates)])
    dur_dst = dst_edges[-1]
    n_out_frames = int(dur_dst * sr / hop)
    dst_t = np.arange(n_out_frames) * hop / sr
    # invert the monotone map: dst time -> src time -> src frame index
    src_t = np.interp(dst_t, dst_edges, edges)
    src_frames = np.clip(src_t * sr / hop, 0, None)
    chans = []
    for ch in y:
        D = librosa.stft(ch, n_fft=n_fft, hop_length=hop)
        src = np.clip(src_frames, 0, D.shape[1] - 1.001)
        chans.append(librosa.istft(variable_phase_vocoder(D, src, hop),
                                   hop_length=hop, n_fft=n_fft))
    n = min(len(c) for c in chans)
    return np.stack([c[:n] for c in chans])


def _produce(job):
    """Module-level so multiprocessing can pickle it; all state comes in `job`."""
    import librosa
    uid, v, cache, depth, variants, max_s, n_fft, hop, audio_root, grate = job
    try:
        C = Path(cache)
        beats, db = load_beats(C / f"{uid}.beats")
        corpus, stem = uid.split("__", 1)
        y, sr = librosa.load(str(Path(audio_root) / corpus / f"{stem}.wav"),
                             sr=44100, mono=False, duration=max_s)
        if y.ndim == 1:
            y = np.stack([y, y])
        dur = y.shape[1] / sr
        beats = beats[beats < dur]
        if len(beats) < 8:
            return None
        edges, rates = rate_curve(dur, depth, phase=2 * math.pi * v / variants,
                                  global_rate=grate)
        yw = warp_audio(y, sr, edges, rates, n_fft, hop)
        bw = warp_times_curve(beats, edges, rates)
        dbw = None if db is None else warp_times_curve(db[db < dur], edges, rates)
        return uid, v, yw.astype(np.float32), bw, dbw
    except Exception as e:                                       # noqa: BLE001
        print(f"[warp-fail] {uid} v{v}: {type(e).__name__}: {e}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="/disk1/jaehoon/sa3_beat_cache")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--depth", type=float, default=0.20)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--global-rate", type=float, default=1.0,
                    help="<1 slows the track; covers SMC's tempo region")
    ap.add_argument("--global-rate-range", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="sample the rate per variant instead of fixing it. A "
                         "fixed rate only TRANSLATES the tempo distribution; "
                         "SMC spans q10=45 to q90=115 BPM, so coverage needs a "
                         "spread, not a shifted point mass.")
    ap.add_argument("--tag", default="w",
                    help="variant tag. Must not collide with corpus "
                         "prefixes: tag 's' makes *__s* also match "
                         "smc__s*, silently duplicating SMC tracks "
                         "into the training filelist.")
    ap.add_argument("--max-seconds", type=float, default=120.0)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--verify", type=int, default=0,
                    help="verify N tracks and exit without writing a corpus")
    args = ap.parse_args()

    import librosa, torch
    from sa3_beat_adherence import beat_adherence
    C = Path(args.cache)
    train = [l.strip()[:-4] for l in open(C / f"train_excl_fold{args.fold}.txt") if l.strip()]

    AUDIO = Path("/disk1/jaehoon/dataset_store/audio_by_stem")
    def audio_path(uid):
        corpus, stem = uid.split("__", 1)
        return AUDIO / corpus / f"{stem}.wav"

    if args.verify:
        rng = np.random.RandomState(0)
        sel = [train[i] for i in rng.choice(len(train), args.verify, replace=False)]
        print(f"verifying {len(sel)} tracks, depth={args.depth}\n")
        rows = []
        for uid in sel:
            beats, _ = load_beats(C / f"{uid}.beats")
            y, sr = librosa.load(str(audio_path(uid)), sr=44100, mono=False,
                                 duration=args.max_seconds)
            if y.ndim == 1: y = np.stack([y, y])
            dur = y.shape[1] / sr
            beats = beats[beats < dur]
            if len(beats) < 8: continue
            edges, rates = rate_curve(dur, args.depth, global_rate=args.global_rate)
            yw = warp_audio(y, sr, edges, rates, args.n_fft, args.hop)
            bw = warp_times_curve(beats, edges, rates)
            bw = bw[bw < yw.shape[1] / sr]
            a = beat_adherence(y.mean(axis=0), beats)["phase_align"]
            b = beat_adherence(yw.mean(axis=0), bw)["phase_align"]
            bpm0 = 60.0/np.median(np.diff(beats)); bpm1 = 60.0/np.median(np.diff(bw))
            cv0 = np.std(np.diff(beats)) / np.mean(np.diff(beats))
            cv1 = np.std(np.diff(bw)) / np.mean(np.diff(bw))
            rows.append((uid, a, b, cv0, cv1, bpm0, bpm1))
            print(f"  {uid[:30]:30s} adh {a:+.4f}->{b:+.4f}  cv {cv0:.3f}->{cv1:.3f}  bpm {bpm0:5.1f}->{bpm1:5.1f}")
        A = np.array([r[1] for r in rows]); B = np.array([r[2] for r in rows])
        C0 = np.array([r[3] for r in rows]); C1 = np.array([r[4] for r in rows])
        print(f"\n  mean phase_align  original={A.mean():+.4f}  warped={B.mean():+.4f}  "
              f"delta={B.mean()-A.mean():+.4f}")
        print(f"  mean IBI cv       original={C0.mean():.4f}   warped={C1.mean():.4f}")
        B0=np.array([r[5] for r in rows]); B1=np.array([r[6] for r in rows])
        print(f"  median BPM        original={np.median(B0):.1f}      warped={np.median(B1):.1f}")
        print("\n  labels survive warping" if B.mean() > A.mean() - 0.05 else
              "\n  WARNING: warping degraded adherence -- labels no longer describe the audio")
        return

    # ---- corpus generation -------------------------------------------------
    import multiprocessing as mp, shutil, torch
    from safetensors.torch import load_file
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict

    cfg = json.loads((SA3_DIR / "model_config.json").read_text())
    model = create_model_from_config(cfg)
    copy_state_dict(model, load_file(str(SA3_DIR / "model.safetensors")))
    dev = f"cuda:{args.gpu}"
    pt = model.pretransform.to(dev).eval()
    sr, ds = cfg["sample_rate"], model.pretransform.downsampling_ratio

    AUD = str(AUDIO)
    jobs = [(uid, v, str(C), args.depth, args.variants, args.max_seconds,
             args.n_fft, args.hop, AUD,
             (args.global_rate if not args.global_rate_range else
              float(np.random.RandomState(hash((uid, v)) % 2**31)
                    .uniform(*args.global_rate_range))))
            for uid in train for v in range(args.variants)]
    print(f"[plan] {len(train)} tracks x {args.variants} variants = {len(jobs)} warps "
          f"(depth={args.depth}, max {args.max_seconds}s)", flush=True)

    written = 0
    with mp.Pool(processes=12) as pool:
        for res in pool.imap_unordered(_produce, jobs, chunksize=4):
            if res is None: continue
            uid, v, yw, bw, dbw = res
            out_uid = f"{uid}__{args.tag}{v}"
            n = max(ds, (yw.shape[1] // ds) * ds)
            if yw.shape[1] < n: yw = np.pad(yw, ((0,0),(0, n - yw.shape[1])))
            x = torch.tensor(yw[:, :n], dtype=torch.float32, device=dev)[None]
            with torch.no_grad():
                z = pt.encode(x).squeeze(0).float().cpu().numpy()
            np.save(C / f"{out_uid}.npy", z)
            nf = z.shape[-1]
            json.dump({"path": f"{out_uid}.wav", "seconds_total": round(n/sr, 3),
                       "padding_mask": [1]*nf, "warped_from": uid, "variant": v},
                      open(C / f"{out_uid}.json", "w"))
            lines = [f"{t:.6f}\t{1 if (dbw is not None and np.min(np.abs(dbw-t))<1e-6) else 2}"
                     for t in bw]
            (C / f"{out_uid}.beats").write_text("\n".join(lines) + "\n")
            written += 1
            if written % 200 == 0: print(f"  [{written}/{len(jobs)}]", flush=True)

    # Match the variant suffix explicitly (…__<tag><digits>) rather than a
    # loose glob, which would also match corpus prefixes like "smc__s…".
    import re as _re
    pat = _re.compile(r"__(w|tempo)\d+$")
    warped = sorted(p.stem for p in C.glob("*.npy") if pat.search(p.stem))
    base = [l.strip()[:-4] for l in open(C / f"train_excl_fold{args.fold}.txt") if l.strip()]
    (C / f"train_excl_fold{args.fold}_warped.txt").write_text(
        "\n".join(f"{u}.npy" for u in sorted(base + warped)) + "\n")
    print(f"[done] wrote {written} warped tracks; filelist = {len(base)+len(warped)} entries")


if __name__ == "__main__":
    main()
