"""Register the generated corpus as a Beat This dataset.

Beat This reads annotations from ``<annotation_dir>/<dataset>/annotations/beats``
and decides train/val membership from ``<dataset>/8-folds.split``:

    val_items   = split.piece[split.part == fold]
    train_items = split.piece[split.part != fold]

so every synthetic item is assigned a part != the evaluation fold. They are
training material only and can never leak into validation for this pilot.

Audio is registered in ``audio_paths.csv`` (dataset_name, audio_path) for
``launch_scripts/preprocess_audio.py`` to turn into spectrograms.
"""
import argparse, csv, shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", default="/disk1/jaehoon/sa3_gen_corpus_fold0")
    ap.add_argument("--name", default="sa3gen")
    ap.add_argument("--annotation-dir", default="/disk1/jaehoon/dataset_store/beat_this_annotations")
    ap.add_argument("--eval-fold", type=int, default=0)
    ap.add_argument("--audio-paths-csv", default=None)
    args = ap.parse_args()

    gen = Path(args.gen_dir)
    wavs = sorted(gen.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no wavs in {gen}")

    ann_root = Path(args.annotation_dir) / args.name
    beats_dir = ann_root / "annotations" / "beats"
    beats_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, w in enumerate(wavs):
        src = w.with_suffix(".beats")
        if not src.exists():
            continue
        shutil.copy(src, beats_dir / f"{w.stem}.beats")
        # spread across the folds that are NOT the evaluation fold, so the split
        # file stays valid for any fold while these stay training-only here.
        part = 1 + (i % 7)
        if part == args.eval_fold:
            part = (part % 7) + 1
        rows.append((w.stem, part))

    with open(ann_root / "8-folds.split", "w") as f:
        for stem, part in rows:
            f.write(f"{stem}\t{part}\n")
    with open(ann_root / "single.split", "w") as f:
        for stem, _ in rows:
            f.write(f"{stem}\ttrain\n")

    print(f"[prep] {len(rows)} items registered under {ann_root}")
    print(f"[prep] fold assignment: parts 1-7, evaluation fold {args.eval_fold} excluded")

    if args.audio_paths_csv:
        p = Path(args.audio_paths_csv)
        existing = []
        if p.exists():
            existing = [r for r in csv.reader(open(p)) if r and r[0] != args.name]
        with open(p, "w", newline="") as f:
            wri = csv.writer(f)
            for r in existing:
                wri.writerow(r)
            wri.writerow([args.name, str(gen)])
        print(f"[prep] audio_paths.csv updated: {p}")


if __name__ == "__main__":
    main()
