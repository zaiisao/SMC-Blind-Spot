"""Where does SMC's missing F actually live? Fold-honest oracle decomposition.

Every intervention in this project has targeted a property we never confirmed was
the bottleneck. The tag analysis (216 tracks) says failures are dominated by
ambiguity of metrical LEVEL (t=-3.38) and beat PHASE (t=-2.98). This measures how
much F each of those is actually worth, by giving the tracker an oracle for one
factor at a time and seeing how far F moves.

The one previous level-oracle measurement here gave +0.0104 and was taken on fold
0 -- which contains ZERO metrical-ambiguity tracks and ZERO m4 tracks, so it
measured the headroom of a fold that does not have the problem. This runs every
fold with the released checkpoint that held that fold out.

Oracles, applied to the tracker's own estimate (never to the truth):
  LEVEL   rescale the estimated grid by 1/3,1/2,1,2,3 -- keeps the tracker's phase
          and tempo evidence, changes only which metrical level it committed to.
  PHASE   shift the estimated grid by a fraction of its own IBI.
  BOTH    the two jointly.

Residual after BOTH is error the decoder cannot fix by reinterpreting the grid:
it needs better frame activations, or the track is genuinely ambiguous.
"""
import argparse, csv, os, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TAGDIR = Path("/home/sogang/jaehoon/Analyze-SMC/smc_metadata/tags")
LEVELS = [1/3, 1/2, 1.0, 2.0, 3.0]
# 1/12 IBI is 125 ms at 40 bpm -- wider than mir_eval's +/-70 ms f_measure window,
# so the optimal phase was routinely stepped over. The gain saturates by ~1/64
# (measured: 1/12 +0.0242, 1/16 +0.0310, 1/24 +0.0399, 1/32 +0.0455, 1/64 +0.0503,
# 1/128 +0.0532), so this is a genuine per-track best phase, not oracle over-search.
PHASES = np.arange(-0.5, 0.5, 1/64)


def load_tags():
    out = {}
    for f in sorted(TAGDIR.glob("*.tag")):
        n = int(re.search(r"(\d+)", f.stem).group(1))
        tags = set()
        for part in re.split(r"[,\n]", f.read_text()):
            t = part.strip().lower().strip("()").replace(" ", "")
            if t and not re.fullmatch(r"[aoj]\d", t):
                tags.add(t)
        out[f"smc_{n:03d}"] = tags
    return out


def relevel(est, factor, span):
    """DEPRECATED metronome regrid -- kept only to reproduce the pre-fix numbers.

    Rebuilding the grid as est[0] + k*median_ibi discards all local timing. On a
    corpus that is 124/217 `expressivetiming` this is catastrophic: the IDENTITY
    case (factor=1.0) scores F=0.1937 against a base of 0.6270, losing 0.43 F on a
    transform that should change nothing. The LEVEL oracle built on this could not
    express a metrical-level change on the very tracks it was meant to probe, so
    the published +LEVEL 0.0124 / +BOTH 0.0282 / residual 92.4% were artifacts of a
    broken candidate set. Use `relevel_preserving`.
    """
    if len(est) < 4:
        return est
    ibi = float(np.median(np.diff(est))) * factor
    if ibi <= 0.05:
        return est
    lo, hi = span
    n = int((hi - lo) / ibi) + 2
    return est[0] + np.arange(-n, n) * ibi


def relevel_preserving(est, factor, offset=0):
    """Change metrical level while KEEPING the estimate's local timing.

    Halving/thirding take every k-th estimated beat (each `offset` is a different
    phase of that reading). Doubling/tripling insert interpolated points between
    consecutive estimates, so the inserted points follow the local tempo curve
    rather than a global median. Both are what a decoder could actually emit.
    """
    e = np.asarray(est, dtype=float)
    if len(e) < 4:
        return e
    if factor == 1.0:
        return e
    if factor in (2.0, 3.0):                      # coarser grid: subsample
        k = int(factor)
        return e[offset % k::k]
    if factor in (0.5, 1 / 3):                    # finer grid: interpolate
        m = int(round(1 / factor))
        out = []
        for i in range(len(e) - 1):
            for j in range(m):
                out.append(e[i] + (e[i + 1] - e[i]) * j / m)
        out.append(e[-1])
        return np.asarray(out)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cache", default="/disk1/jaehoon/smc_estimates.npz")
    a = ap.parse_args()

    import mir_eval.beat as mb

    # ---- collect fold-honest estimates once, then the oracle search is CPU only
    if os.path.exists(a.cache):
        z = np.load(a.cache, allow_pickle=True)
        data = {k: z[k].item() for k in z.files}
        print(f"[cache] loaded {len(data)} tracks from {a.cache}")
    else:
        import torch, soundfile as sf, soxr
        from beat_this.inference import load_model, split_predict_aggregate
        from beat_this.model.postprocessor import Postprocessor
        from beat_this.preprocessing import LogMelSpect
        from smc_eval import fold_songs

        dev = f"cuda:{a.gpu}"
        spect = LogMelSpect(device=dev)
        post = Postprocessor(type="minimal", fps=50)
        data = {}
        for fold in range(8):
            net = load_model(f"fold{fold}", device=dev).eval()
            for stem, wav, truth in fold_songs(fold):
                y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
                y = y.mean(axis=1)
                if sr != 22050:
                    y = soxr.resample(y, sr, 22050)
                with torch.inference_mode():
                    pr = split_predict_aggregate(
                        spect(torch.from_numpy(y).to(dev)), 1500, 6, "keep_first", net)
                    est, _ = post(pr["beat"].float(), pr["downbeat"].float())
                data[stem] = {"truth": np.asarray(truth), "est": np.asarray(est)}
            print(f"[fold {fold}] {len(list(fold_songs(fold)))} tracks", flush=True)
        np.savez(a.cache, **{k: np.array(v, dtype=object) for k, v in data.items()})
        print(f"[cache] wrote {a.cache}")

    tags = load_tags()
    rows = []
    for stem, d in data.items():
        truth, est = d["truth"], d["est"]
        if len(truth) < 8 or len(est) < 4:
            continue
        rt = mb.trim_beats(truth)
        span = (float(truth[0]) - 1.0, float(truth[-1]) + 1.0)
        base = mb.f_measure(rt, mb.trim_beats(est))

        # every level reading, each at every phase of that reading
        level_cands = []
        for f in LEVELS:
            offs = int(f) if f in (2.0, 3.0) else 1
            for o in range(offs):
                e = relevel_preserving(est, f, o)
                e = e[(e >= span[0]) & (e <= span[1])]
                if len(e) >= 4:
                    level_cands.append(e)

        best_lvl = max([base] + [mb.f_measure(rt, mb.trim_beats(e)) for e in level_cands])

        ibi = float(np.median(np.diff(est)))
        best_ph = base
        for p in PHASES:
            e = est + p * ibi
            if len(e) >= 4:
                best_ph = max(best_ph, mb.f_measure(rt, mb.trim_beats(e)))

        best_both = base
        for e0 in level_cands:
            i2 = float(np.median(np.diff(e0))) if len(e0) > 2 else ibi
            for p in PHASES:
                e = e0 + p * i2
                e = e[(e >= span[0]) & (e <= span[1])]
                if len(e) >= 4:
                    best_both = max(best_both, mb.f_measure(rt, mb.trim_beats(e)))

        # SPLIT-HALF ATTAINABILITY: fit the best (level, phase) on the first half of
        # the track, apply it to the second, score there. An oracle chosen per track
        # on a 30 s excerpt is a fit with ~25 observations, not a ceiling -- this is
        # what a decoder could actually reach without seeing the answer.
        mid = float(truth[len(truth) // 2])
        fit_m = truth < mid
        rt_fit = mb.trim_beats(truth[fit_m]) if fit_m.sum() >= 8 else None
        rt_hold = mb.trim_beats(truth[~fit_m]) if (~fit_m).sum() >= 8 else None
        attain = base
        if rt_fit is not None and rt_hold is not None and len(rt_fit) and len(rt_hold):
            best_fit, best_pair = -1.0, None
            for e0 in level_cands:
                i2 = float(np.median(np.diff(e0))) if len(e0) > 2 else ibi
                for p in PHASES:
                    e = e0 + p * i2
                    ef = mb.trim_beats(e[e < mid])
                    if len(ef) >= 4:
                        s = mb.f_measure(rt_fit, ef)
                        if s > best_fit:
                            best_fit, best_pair = s, (e0, p, i2)
            if best_pair is not None:
                e0, p, i2 = best_pair
                eh = mb.trim_beats((e0 + p * i2)[(e0 + p * i2) >= mid])
                base_h = mb.trim_beats(est[est >= mid])
                if len(eh) >= 4 and len(base_h) >= 4:
                    attain = (mb.f_measure(rt_hold, eh)
                              - mb.f_measure(rt_hold, base_h) + base)

        rows.append((stem, base, best_lvl, best_ph, best_both, attain))

    arr = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows])
    n = len(arr)
    print(f"\nfold-honest, n={n} SMC tracks\n")
    print(f"{'':<26}{'F':>9}{'gain':>9}{'% of gap to 1.0':>18}")
    base = arr[:, 0].mean()
    gap = 1.0 - base
    for name, i in (("as predicted", 0), ("+ LEVEL oracle", 1),
                    ("+ PHASE oracle", 2), ("+ BOTH oracles", 3),
                    ("BOTH, split-half attainable", 4)):
        v = arr[:, i].mean()
        print(f"{name:<26}{v:9.4f}{v-base:+9.4f}{100*(v-base)/gap:17.1f}%")
    print(f"\nresidual after both oracles: {1-arr[:,3].mean():.4f} "
          f"({100*(1-arr[:,3].mean())/gap:.1f}% of the original gap)")

    # per-tag: where is the level/phase headroom concentrated?
    print(f"\n{'tag':<30}{'n':>4}{'base':>8}{'+both':>8}{'gain':>8}")
    by = defaultdict(list)
    for (stem, b, l, p, bo, at) in rows:
        for t in tags.get(stem, ()):
            by[t].append((b, bo))
    for t, v in sorted(by.items(), key=lambda kv: -(np.mean([x[1] - x[0] for x in kv[1]]))):
        if len(v) < 5:
            continue
        b = np.mean([x[0] for x in v]); bo = np.mean([x[1] for x in v])
        print(f"{t:<30}{len(v):4d}{b:8.4f}{bo:8.4f}{bo-b:+8.4f}")


if __name__ == "__main__":
    main()
