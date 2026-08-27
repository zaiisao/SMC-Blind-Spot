"""Run Beat This's own audio preprocessing, parallelised across processes.

The vanilla launch_scripts/preprocess_audio.py is correct but leaves the machine
half idle: its ThreadPoolExecutor() defaults to min(32, cpu+4) workers, which is
32 here against 64 cores, and pedalboard's pitch-shift/time-stretch dominate the
runtime. This driver imports that module unchanged and only changes how the work
is spread: --shards processes over disjoint clip subsets, each with --threads
threads. Stems are disjoint per shard and every output path is per-stem, so the
shards never touch the same file.

Stage 3 (npz bundling) must run once, after all shards finish.
"""
import argparse, os, sys, time
from pathlib import Path

BT = Path("/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")


def load_pp(root, threads):
    sys.path.insert(0, str(BT)); sys.path.insert(0, str(BT / "launch_scripts"))
    import concurrent.futures as cf
    _orig = cf.ThreadPoolExecutor

    class Pool(_orig):                      # widen the default worker count
        def __init__(self, max_workers=None, *a, **k):
            super().__init__(max_workers=max_workers or threads, *a, **k)
    cf.ThreadPoolExecutor = Pool
    import preprocess_audio as pp
    pp.concurrent.futures.ThreadPoolExecutor = Pool
    pp.BASEPATH = Path(root)
    return pp


MEL = dict(n_fft=1024, hop_length=441, f_min=30, f_max=11000, n_mels=128,
           mel_scale="slaney", normalized="frame_length", power=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/disk1/jaehoon/bt_sa3")
    ap.add_argument("--stage", choices=["audio", "spect", "npz"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.root)
    pp = load_pp(root, args.threads)
    ps, ts = (-5, 6), (20, 4)
    t0 = time.time()

    if args.stage == "audio":
        # shard by giving this process its own audio_paths.csv over a subset
        ds_name, src = open(root / "audio_paths.csv").read().strip().split(",")
        src = Path(src)
        wavs = sorted(src.glob("*.wav"))
        mine = [w for i, w in enumerate(wavs) if i % args.shards == args.shard]
        sub = root / f".shard{args.shard}"
        sub.mkdir(exist_ok=True)
        for w in mine:
            d = sub / w.name
            if not d.exists():
                os.symlink(w, d)
        csv = root / f".paths{args.shard}.csv"
        csv.write_text(f"{ds_name},{sub}\n")
        dp = pp.AudioPreprocessing(orig_audio_paths=str(csv), out_sr=22050,
                                   aug_sr=44100, pitch_shift=ps, time_stretch=ts,
                                   verbose=False)
        dp.preprocess_audio()
        print(f"[shard {args.shard}] audio: {len(mine)} clips in {time.time()-t0:.0f}s",
              flush=True)

    elif args.stage == "spect":
        sc = pp.SpectCreation(pitch_shift=ps, time_stretch=ts, audio_sr=22050,
                              mel_args=MEL, verbose=False)
        # shard the piece list the same way
        pieces = sorted(p for d in sc.mono_tracks_dir.iterdir() if d.is_dir()
                        for p in d.iterdir() if p.is_dir())
        mine = [p for i, p in enumerate(pieces) if i % args.shards == args.shard]
        # signature is (preprocessed_audio_folder, beat_path, dataset_name), where
        # beat_path is relative to annotations_dir -- mirror the vanilla call exactly.
        sc.spectrograms_dir.mkdir(parents=True, exist_ok=True)
        import concurrent.futures as cf

        def one(piece_dir):
            ds = piece_dir.parent.name
            return sc.create_spect_piece(
                piece_dir,
                Path(ds) / "annotations" / "beats" / f"{piece_dir.name}.beats",
                ds)

        with cf.ThreadPoolExecutor(args.threads) as ex:
            done = sum(1 for r in ex.map(one, mine) if r)
        print(f"[shard {args.shard}] created {done}/{len(mine)} spectrograms", flush=True)
        print(f"[shard {args.shard}] spect: {len(mine)} pieces in {time.time()-t0:.0f}s",
              flush=True)

    else:  # npz -- once, after all shards
        sc = pp.SpectCreation(pitch_shift=ps, time_stretch=ts, audio_sr=22050,
                              mel_args=MEL, verbose=False)
        for d in [c for c in sc.spectrograms_dir.iterdir() if c.is_dir()]:
            pp.create_npz(d, d.with_suffix(".npz"), sc.augmentations, False)
            print(f"[npz] {d.name} -> {d.with_suffix('.npz')} "
                  f"({d.with_suffix('.npz').stat().st_size/2**30:.1f} GB)", flush=True)
        print(f"[npz] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
