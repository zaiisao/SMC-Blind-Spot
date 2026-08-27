"""Render ASYMMETRIC wide-range tempo augmentation, -60%..+20% instead of +/-20%.

Phase 2 of the tempo-coverage plan. The measured cause (2026-08-25):

    fraction of augmented training draws below 70 bpm    0.100
    fraction of SMC below 70 bpm                         0.475

and Gate 1a established causality -- slowing easy music (F 0.9645) into the
45-70 bpm band costs 0.266 F, while a resample-there-and-back control returns
0.9645 EXACTLY (dC = 0.0000, n=245). So display tempo alone produces an
SMC-sized deficit, and the training pool barely covers that region.

ASYMMETRY IS THE POINT, not an optimisation. Beat This builds its range as
range(-ts, ts+1, stride), so widening it to +/-60 would add +24..+60 speed-ups
alongside the slow-downs and push the below-70 fraction back down. We want only
the slow side.

TWO UPSTREAM BUGS this works around, both in
third-party/beat_this/launch_scripts/preprocess_audio.py:
  1. AudioPreprocessing.process_audio_file (line ~253) builds its resume check
     from `self.time_stretch[0]` and `[1]` AFTER __init__ converted it to a
     range, so for (20,4) it looks for {ts-12, ts4, ts20} -- 3 names instead of
     10. Those never all exist, so the "skip if done" branch never fires and
     every run re-renders everything. Generation itself is correct.
  2. The same indexing makes an asymmetric range impossible to express.

Both are handled by setting explicit stretch lists and rebuilding the filename
lists directly.
"""
import argparse, os, sys
from pathlib import Path

# overridable so the second machine (titan), which has no shared filesystem,
# can point at its own copy of the beat_this checkout
BT = Path(os.environ.get("BT_ROOT",
                         "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"))


def load_pp(root, threads):
    sys.path.insert(0, str(BT))
    sys.path.insert(0, str(BT / "launch_scripts"))
    import preprocess_audio as pp
    pp.BASEPATH = Path(root)
    return pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/disk1/jaehoon/bt_wide")
    ap.add_argument("--stage", choices=["audio", "spect"], required=True)
    ap.add_argument("--lo", type=int, default=-60, help="most negative stretch %%")
    ap.add_argument("--hi", type=int, default=20, help="most positive stretch %%")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard-list", default=None,
                    help="comma-separated shard ids for THIS process, e.g. '0,6'. "
                         "Decouples the split (which must stay fixed so the two "
                         "machines do not render the same track twice) from the "
                         "process count (which we tune for throughput -- pedalboard "
                         "runs its own thread pool, so more processes is not more "
                         "work done).")
    ap.add_argument("--track-list", default=None,
                    help="file of track stems this machine owns; overrides --shard-list")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pitch", type=int, nargs=2, default=[-5, 6])
    a = ap.parse_args()

    root = Path(a.root)
    pp = load_pp(root, a.threads)
    from beat_this.dataset.augment import precomputed_augmentation_filenames

    stretches = [p for p in range(a.lo, a.hi + 1, a.stride)]
    aug = {"pitch": {"min": a.pitch[0], "max": a.pitch[1]},
           "tempo": {"min": a.lo, "max": a.hi, "stride": a.stride}}
    names = precomputed_augmentation_filenames(aug, "wav")
    print(f"[wide] stretches {a.lo}..{a.hi} step {a.stride}  "
          f"({len(stretches)} values, {len(names)} files/track)", flush=True)

    MEL = dict(n_fft=1024, hop_length=441, f_min=30, f_max=11000, n_mels=128,
               mel_scale="slaney", normalized="frame_length", power=1)

    if a.stage == "audio":
        ds_name, src = open(root / "audio_paths.csv").read().strip().split(",")
        src = Path(src)
        wavs = sorted([p for p in src.iterdir() if p.suffix.lower()
                       in (".wav", ".flac", ".mp3")])
        if a.track_list:
            # explicit assignment. Once both machines have partial output, "who
            # owes what" stops being a function of shard id -- it depends on what
            # is already complete on each box, so the balancer computes the split
            # and hands each side a literal list.
            want = set(open(a.track_list).read().split())
            mine = [w for w in wavs if w.stem in want]
            mine = [w for i, w in enumerate(mine) if i % a.shards == a.shard]
            tag = f"L{a.shard}"
        else:
            ids = ([int(x) for x in a.shard_list.split(",")] if a.shard_list
                   else [a.shard])
            mine = [w for i, w in enumerate(wavs) if i % a.shards in ids]
            tag = "_".join(str(i) for i in ids)
        print(f"[wide] {len(mine)} tracks assigned to this worker", flush=True)
        sub = root / f".wshard{tag}"
        sub.mkdir(parents=True, exist_ok=True)
        for w in mine:
            d = sub / w.name
            if not d.exists():
                os.symlink(w, d)
        csv = root / f".wpaths{tag}.csv"
        csv.write_text(f"{ds_name},{sub}\n")
        dp = pp.AudioPreprocessing(orig_audio_paths=str(csv), out_sr=22050,
                                   aug_sr=44100, pitch_shift=tuple(a.pitch),
                                   time_stretch=(abs(a.lo), a.stride), verbose=False)
        dp.time_stretch = stretches          # explicit, asymmetric
        # make the resume check look for the CORRECT filenames, so re-runs skip
        # work instead of re-rendering everything
        orig = pp.AudioPreprocessing.process_audio_file
        def patched(self, dataset_name, audio_path, _names=tuple(names)):
            folder = Path(self.audio_dir, "mono_tracks", dataset_name, audio_path.stem)
            if folder.exists() and all((folder / n).exists() for n in _names):
                return True
            return orig(self, dataset_name, audio_path)
        pp.AudioPreprocessing.process_audio_file = patched
        dp.preprocess_audio()
        print(f"[wide] shards {tag} audio done", flush=True)

    else:
        sc = pp.SpectCreation(pitch_shift=tuple(a.pitch),
                              time_stretch=(abs(a.lo), a.stride),
                              audio_sr=22050, mel_args=MEL, verbose=False)
        sc.augmentations = aug
        sc.filenames = names                 # explicit, asymmetric
        pieces = sorted(p for d in sc.mono_tracks_dir.iterdir() if d.is_dir()
                        for p in d.iterdir() if p.is_dir())
        # only fully-rendered tracks: a partial dir would yield a spectrogram
        # set with missing augmentations, which the dataloader would then index
        # into and fail on -- or worse, silently train on the wrong variant.
        pieces = [p for p in pieces if len(list(p.iterdir())) >= len(names)]
        if a.track_list:
            # ~100 tracks got rendered on BOTH machines during rebalancing, so
            # spectrograms must be assigned explicitly or they'd be computed
            # twice and merged into duplicate npz.
            want = set(open(a.track_list).read().split())
            pieces = [p for p in pieces if p.name in want]
        mine = [p for i, p in enumerate(pieces) if i % a.shards == a.shard]
        print(f"[wide] shard {a.shard}: {len(mine)} pieces", flush=True)
        # create_spect_piece writes one .npy per augmentation into
        # spectrograms_dir/<ds>/<track>/ and skips files that already exist, so
        # it is resumable on its own -- no wrapper npz, and no re-doing work
        # after a restart.
        done = fail = 0
        for p in mine:
            ds = p.parent.name
            try:
                sc.create_spect_piece(
                    p, Path(ds) / "annotations" / "beats" / f"{p.name}.beats", ds)
                done += 1
            except Exception as e:
                fail += 1
                print(f"  [FAIL] {p.name}: {type(e).__name__} {e}", flush=True)
            if (done + fail) % 50 == 0:
                print(f"  [wide] shard {a.shard}: {done + fail}/{len(mine)} "
                      f"({fail} failed)", flush=True)
        print(f"[wide] shard {a.shard} spect done: {done} ok, {fail} failed",
              flush=True)


if __name__ == "__main__":
    main()
