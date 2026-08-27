"""Score every generated clip against its own labels, and build a filtered subset.

Adherence declines monotonically with grid irregularity (+0.347 for the
steadiest quartile down to +0.069 for the most rubato), so a GLOBAL threshold
would preferentially delete rubato clips -- entangling label quality with the
very property under test. A null result on such a subset could not distinguish
"synthetic data does not help" from "we removed the material that mattered".

So the filter is stratified: keep the top --keep fraction by adherence WITHIN
each cv quartile. Label quality improves; the timing distribution is preserved.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "sa3"))


def score_one(args):
    import soundfile as sf
    from sa3_beat_adherence import beat_adherence
    w = Path(args)
    # The 2026-08-23 hard lock left zero-byte sidecars mid-write; loadtxt returns
    # shape (1,0) for those and [:, 0] raises, which killed the whole worker pool.
    bf = w.with_suffix(".beats")
    if bf.stat().st_size == 0:
        return None
    b = np.loadtxt(bf, ndmin=2)
    if b.size == 0 or b.shape[1] < 1 or len(b) < 8:
        return None
    b = b[:, 0]
    y, _ = sf.read(str(w), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    ibi = float(np.median(np.diff(b)))
    return dict(stem=w.stem,
                adherence=float(beat_adherence(y, b)["phase_align"]),
                half=float(beat_adherence(y, b + ibi / 2)["phase_align"]),
                cv=float(np.std(np.diff(b)) / np.mean(np.diff(b))),
                bpm=60.0 / ibi, n_beats=int(len(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", default="/disk1/jaehoon/sa3_gen_corpus_fold0")
    ap.add_argument("--keep", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="cache/gen_corpus_scores.json")
    args = ap.parse_args()

    import multiprocessing as mp
    wavs = [str(p) for p in sorted(Path(args.gen_dir).glob("*.wav"))
            if p.with_suffix(".beats").exists()]
    print(f"[score] {len(wavs)} clips, {args.workers} workers", flush=True)
    with mp.Pool(args.workers) as pool:
        rows = [r for r in pool.imap_unordered(score_one, wavs, chunksize=8) if r]

    a = np.array([r["adherence"] for r in rows])
    cv = np.array([r["cv"] for r in rows])
    print(f"[score] n={len(rows)} adherence mean={a.mean():+.4f} "
          f"median={np.median(a):+.4f} frac>0={100*(a>0).mean():.1f}%")

    q = np.quantile(cv, [0.25, 0.5, 0.75])
    keep = np.zeros(len(rows), dtype=bool)
    for lo, hi in zip([-1, *q], [*q, 1e9]):
        m = (cv > lo) & (cv <= hi)
        if not m.sum():
            continue
        thr = np.quantile(a[m], 1 - args.keep)
        keep |= m & (a >= thr)
        print(f"   cv in ({max(lo,0):.3f},{min(hi,9):.3f}] n={m.sum():4d} "
              f"thr={thr:+.4f} kept={int((m & (a>=thr)).sum()):4d} "
              f"adherence {a[m].mean():+.4f} -> {a[m & (a>=thr)].mean():+.4f}")

    kept = [r["stem"] for r, k in zip(rows, keep) if k]
    print(f"[score] kept {len(kept)}/{len(rows)}; "
          f"adherence {a.mean():+.4f} -> {a[keep].mean():+.4f}; "
          f"cv median {np.median(cv):.4f} -> {np.median(cv[keep]):.4f}")
    json.dump({"per_clip": rows, "kept": kept, "keep_frac": args.keep},
              open(args.out, "w"))
    Path(args.out).with_suffix(".txt").write_text("\n".join(sorted(kept)) + "\n")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
