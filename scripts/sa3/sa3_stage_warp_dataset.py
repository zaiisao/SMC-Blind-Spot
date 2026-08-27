"""Stage the warp-augment wav corpus as a Beat This dataset ("warpaug").

Mirrors sa3_stage_bt_dataset.py, with one decisive difference: fold
assignment is NOT round-robin. Every warped clip derives from a real
annotated song, so it must INHERIT its source song's fold from
/disk1/jaehoon/dataset_store/beat_this_annotations/<corpus>/8-folds.split.
Any clip whose source sits in the eval fold (fold 0) of ANY dataset is
excluded entirely -- a warped copy of an evaluation song in the training
set is leakage. This is asserted at the end.

Layout produced (mirrors every real dataset):
    <root>/data/annotations/warpaug/info.json
    <root>/data/annotations/warpaug/8-folds.split      stem <TAB> inherited fold
    <root>/data/annotations/warpaug/annotations/beats/<stem>.beats
    <root>/audio_src/<stem>.wav                        symlinks
    <root>/audio_paths.csv                             warpaug,<root>/audio_src
"""
import argparse, json, os, re, shutil
from collections import Counter
from pathlib import Path
import numpy as np

SPLIT_ROOT = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
VAR_RE = re.compile(r"__w\d+$")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", default="/disk1/jaehoon/warp_wav_corpus")
    ap.add_argument("--root", default="/disk1/jaehoon/bt_warp")
    ap.add_argument("--eval-fold", type=int, default=0)
    args = ap.parse_args()

    gen = Path(args.gen_dir); root = Path(args.root)
    ann = root / "data" / "annotations" / "warpaug"
    beats_dir = ann / "annotations" / "beats"
    audio = root / "audio_src"
    for d in (beats_dir, audio):
        d.mkdir(parents=True, exist_ok=True)

    fold_maps = {}
    def source_fold(corpus, stem):
        if corpus not in fold_maps:
            m = {}
            for l in open(SPLIT_ROOT / corpus / "8-folds.split"):
                l = l.strip()
                if l:
                    s, f = l.split("\t")
                    m[s] = int(f)
            fold_maps[corpus] = m
        return fold_maps[corpus].get(stem)

    rows, has_db = [], False
    skipped = Counter()
    for w in sorted(gen.glob("*.wav")):
        m = VAR_RE.search(w.stem)
        if not m:
            skipped["bad-name"] += 1; continue
        corpus, stem = w.stem[:m.start()].split("__", 1)
        fold = source_fold(corpus, stem)
        if fold is None:
            skipped["no-fold-entry"] += 1; continue
        if fold == args.eval_fold:
            skipped["fold0-source"] += 1; continue
        b = w.with_suffix(".beats")
        if not b.exists() or b.stat().st_size == 0:
            skipped["missing/empty-beats"] += 1; continue
        a = np.loadtxt(b, ndmin=2)
        if a.size == 0 or len(a) < 8:
            skipped["short-beats"] += 1; continue
        if a.shape[1] > 1 and (a[:, 1] == 1).any():
            has_db = True
        shutil.copy(b, beats_dir / f"{w.stem}.beats")
        dst = audio / w.name
        if not dst.exists():
            os.symlink(w, dst)
        rows.append((w.stem, fold))

    (ann / "info.json").write_text(json.dumps({"has_downbeats": bool(has_db)}))
    (ann / "8-folds.split").write_text(
        "\n".join(f"{s}\t{f}" for s, f in rows) + "\n")
    (ann / "single.split").write_text(
        "\n".join(f"{s}\ttrain" for s, _ in rows) + "\n")
    (root / "audio_paths.csv").write_text(f"warpaug,{audio}\n")

    assert all(f != args.eval_fold for _, f in rows), "fold-0-source clip staged"
    counts = dict(sorted(Counter(f for _, f in rows).items()))
    print(f"[stage] {len(rows)} clips staged, skipped: {dict(skipped)}")
    print(f"[stage] has_downbeats={has_db}  fold {args.eval_fold} empty (asserted)")
    print(f"[stage] inherited fold distribution: {counts}")
    print(f"[stage] root={root}")


if __name__ == "__main__":
    main()
