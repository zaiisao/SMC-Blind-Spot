"""Stage the generated corpus as a Beat This dataset ("sa3gen").

The point is to add data to the VANILLA training pipeline rather than reimplement
it: BeatDataModule then gives us Beat This's own augmentation, positive weights,
padding masks, warmup and checkpointing for free.

Layout produced (mirrors every real dataset):
    <root>/data/annotations/sa3gen/info.json          {"has_downbeats": ...}
    <root>/data/annotations/sa3gen/8-folds.split      stem <TAB> fold
    <root>/data/annotations/sa3gen/annotations/beats/<stem>.beats
    <root>/audio_src/<stem>.wav                       input for preprocess_audio.py
    <root>/audio_paths.csv                            sa3gen,<root>/audio_src

Fold assignment: synthetic clips must NEVER land in the evaluation fold, or the
tracker trains on data derived from the songs it is scored on. They are dealt
round-robin over the non-eval folds, and the eval fold is asserted empty.
"""
import argparse, json, os, shutil
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", default="/disk1/jaehoon/sa3_gen_corpus_fold0")
    ap.add_argument("--root", default="/disk1/jaehoon/bt_sa3")
    ap.add_argument("--eval-fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=8)
    ap.add_argument("--keep-list", default=None,
                    help="optional file of stems to include (the stratified filter)")
    ap.add_argument("--link", action="store_true",
                    help="symlink audio instead of copying (saves ~20 GB)")
    args = ap.parse_args()

    gen = Path(args.gen_dir); root = Path(args.root)
    ann = root / "data" / "annotations" / "sa3gen"
    beats_dir = ann / "annotations" / "beats"
    audio = root / "audio_src"
    for d in (beats_dir, audio):
        d.mkdir(parents=True, exist_ok=True)

    keep = None
    if args.keep_list:
        keep = {l.strip() for l in open(args.keep_list) if l.strip()}

    folds = [f for f in range(args.n_folds) if f != args.eval_fold]
    rows, has_db, skipped = [], False, 0
    for i, w in enumerate(sorted(gen.glob("*.wav"))):
        b = w.with_suffix(".beats")
        if not b.exists() or b.stat().st_size == 0:
            skipped += 1; continue
        if keep is not None and w.stem not in keep:
            continue
        a = np.loadtxt(b, ndmin=2)
        if a.size == 0 or len(a) < 8:
            skipped += 1; continue
        if a.shape[1] > 1 and (a[:, 1] == 1).any():
            has_db = True
        shutil.copy(b, beats_dir / f"{w.stem}.beats")
        dst = audio / w.name
        if not dst.exists():
            os.symlink(w, dst) if args.link else shutil.copy(w, dst)
        rows.append((w.stem, folds[len(rows) % len(folds)]))

    (ann / "info.json").write_text(json.dumps({"has_downbeats": bool(has_db)}))
    (ann / "8-folds.split").write_text(
        "\n".join(f"{s}\t{f}" for s, f in rows) + "\n")
    (ann / "single.split").write_text(
        "\n".join(f"{s}\ttrain" for s, _ in rows) + "\n")
    (root / "audio_paths.csv").write_text(f"sa3gen,{audio}\n")

    assert all(f != args.eval_fold for _, f in rows), "synthetic clip in eval fold"
    counts = {f: sum(1 for _, x in rows if x == f) for f in folds}
    print(f"[stage] {len(rows)} clips staged, {skipped} skipped (empty/short)")
    print(f"[stage] has_downbeats={has_db}  eval fold {args.eval_fold} empty (asserted)")
    print(f"[stage] per-fold: {counts}")
    print(f"[stage] root={root}")


if __name__ == "__main__":
    main()
