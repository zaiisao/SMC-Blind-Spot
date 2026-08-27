"""Stage the tempo-selected tracks as one pseudo-dataset for wide-range rendering.

Selection rule (the paper's criterion): a track is included if its annotated tempo
can reach the 45-70 bpm band within a stretch factor of 1.75. That band holds 47.5%
of SMC but only 11.5% of the shipped +/-20% augmented training pool; rendering
-60..-24% for the selected tracks moves that to 34.4%, with the sub-55 fraction
landing at 0.211 against SMC's 0.207.

Everything is flattened into a single "wide" dataset with <corpus>__<stem> names so
one audio_paths.csv line covers all 13 corpora, mirroring how the generated corpora
were staged.
"""
import argparse, glob, os, shutil
from pathlib import Path

import numpy as np

ANN = "/disk1/jaehoon/dataset_store/beat_this_annotations"
AUD = "/disk1/jaehoon/dataset_store/audio_by_stem"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/disk3/jaehoon/bt_wide")
    ap.add_argument("--lo", type=float, default=45.0)
    ap.add_argument("--hi", type=float, default=70.0)
    ap.add_argument("--max-stretch", type=float, default=1.75)
    ap.add_argument("--min-per-corpus", type=int, default=30,
                    help="corpora contributing fewer selected tracks than this "
                         "cannot shift the augmented tempo marginal; excluded")
    a = ap.parse_args()

    root = Path(a.root)
    adir = root / "audio_src"
    bdir = root / "data" / "annotations" / "wide" / "annotations" / "beats"
    for d in (adir, bdir):
        d.mkdir(parents=True, exist_ok=True)

    def stretch(b):
        tgt = min(max(b, a.lo), a.hi) if b < a.hi else a.hi
        return b / tgt

    corpora = sorted(d for d in os.listdir(AUD) if not d.startswith("."))
    picked = {}
    for c in corpora:
        got = []
        for bt in glob.glob(f"{ANN}/{c}/annotations/beats/*.beats"):
            stem = os.path.basename(bt)[:-6]
            src = None
            for e in (".wav", ".flac", ".mp3"):
                p = os.path.join(AUD, c, stem + e)
                if os.path.exists(p):
                    src = p
                    break
            if src is None:
                continue
            t = np.loadtxt(bt, ndmin=2)
            if len(t) < 8:
                continue
            ibi = float(np.median(np.diff(t[:, 0])))
            if ibi <= 0:
                continue
            if stretch(60.0 / ibi) <= a.max_stretch:
                got.append((stem, src, bt))
        picked[c] = got

    n_tot = 0
    for c, got in sorted(picked.items(), key=lambda kv: -len(kv[1])):
        if len(got) < a.min_per_corpus:
            print(f"  {c:<14}{len(got):5d}  EXCLUDED (below --min-per-corpus)")
            continue
        for stem, src, bt in got:
            name = f"{c}__{stem}"
            dst = adir / (name + os.path.splitext(src)[1])
            if not dst.exists():
                os.symlink(src, dst)
            shutil.copy(bt, bdir / f"{name}.beats")
        print(f"  {c:<14}{len(got):5d}  staged")
        n_tot += len(got)

    (root / "audio_paths.csv").write_text(f"wide,{adir}\n")
    # has_downbeats is a PER-DATASET flag upstream (smc/simac false, ballroom/
    # gtzan/rwc true). Flattening 13 corpora into one pseudo-dataset forces a
    # single value, and writing `true` makes the dataloader silently DROP every
    # beats-only track -- 519 simac + 200 smc + 5 gtzan = 724 of 2678, and they
    # are disproportionately the slow ones this corpus exists to add.
    # Write the honest flag: false means downbeat supervision is skipped for the
    # merged set, but no track is discarded. To keep downbeat supervision, stage
    # two pseudo-datasets (wide_db / wide_nodb) instead of one.
    n_nodb = sum(1 for f in bdir.glob("*.beats")
                 if len((f.read_text().strip().split("\n") or [""])[0].split()) < 2)
    info = root / "data" / "annotations" / "wide" / "info.json"
    has_db = n_nodb == 0
    info.write_text(f'{{"has_downbeats": {str(has_db).lower()}}}\n')
    if n_nodb:
        print(f"  WARNING: {n_nodb} of {n_tot} staged tracks are beats-only.")
        print(f"  Wrote has_downbeats=false so NONE are dropped (downbeat head "
              f"gets no supervision from this set).")
        print(f"  For downbeat supervision, split into wide_db / wide_nodb.")

    # FOLD DISCIPLINE. SMC contributes 200 tracks here and IS the evaluation set,
    # so a stretched SMC track must inherit its ORIGINAL fold -- otherwise a
    # fold-k eval track re-enters training wearing a new name, which is the
    # leak this project already paid for once (an SMC eval run on a checkpoint
    # that had trained on 7/8 of SMC, voiding those absolutes).
    orig = {}
    for c in picked:
        sp = Path(ANN) / c / "8-folds.split"
        if sp.exists():
            for line in sp.read_text().splitlines():
                p = line.split()
                if len(p) >= 2:
                    orig[f"{c}__{p[0]}"] = int(p[1])
    # GTZAN has NO fold file because upstream never trains on it -- it is the
    # held-out test set. Round-robining it (the old `i % 8`) puts ~600 GTZAN
    # tracks into training at 7/8, so any later GTZAN number from a model trained
    # here is contaminated. Pin it to a single fold so one `--fold` value excludes
    # all of it, and refuse to silently round-robin anything else.
    TEST_ONLY = {"gtzan"}
    lines, inherited, assigned, pinned = [], 0, 0, 0
    for i, p in enumerate(sorted(bdir.glob("*.beats"))):
        corpus = p.stem.split("__", 1)[0]
        if p.stem in orig:
            lines.append(f"{p.stem}\t{orig[p.stem]}\n")
            inherited += 1
        elif corpus in TEST_ONLY:
            lines.append(f"{p.stem}\t0\n")      # all in fold 0 => excluded together
            pinned += 1
        else:
            lines.append(f"{p.stem}\t{i % 8}\n")
            assigned += 1
    if pinned:
        print(f"  {pinned} test-only tracks ({'/'.join(sorted(TEST_ONLY))}) pinned to "
              f"fold 0 -- train with --fold 0 to exclude them entirely")
    (root / "data" / "annotations" / "wide" / "8-folds.split").write_text("".join(lines))
    print(f"  folds: {inherited} inherited from source corpora, {assigned} assigned")
    print(f"\nstaged {n_tot} tracks -> {root}")
    print(f"audio_paths.csv -> wide,{adir}")


if __name__ == "__main__":
    main()
