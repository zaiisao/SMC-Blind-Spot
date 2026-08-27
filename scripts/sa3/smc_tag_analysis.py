"""Which of SMC's own difficulty tags actually predict beat-tracking failure?

SMC was curated by difficulty descriptor, not by genre -- its metadata carries
tags like "expressive timing" (99 tracks), "lack of transient sounds" (59),
"slow tempo" (47). Those are an enumerable, falsifiable target list, but only if
they predict anything. This ranks them.

Fold-honest across all 217 tracks: each fold is scored with the released
checkpoint that held that fold out, so no track is evaluated by a model that
trained on it. That matters -- a previous analysis in this project used the
fold-0 checkpoint on all 217 songs, ~7/8 of which were in its training data, and
voided its own numbers.

n=217 rather than fold 0's 27 is the point: with 27 tracks the standard error on
mean F is ~0.024, which cannot resolve a per-tag difference worth acting on.
"""
import sys, os, glob, re
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smc_eval import score_fold

TAGDIR = Path("/home/sogang/jaehoon/Analyze-SMC/smc_metadata/tags")


# The dataset's own README (SMC_MIREX_Readme.txt:44): "a code, e.g. f1 where 'f' is
# the name of the annotator and '1' is a confidence level". So these are NOT content
# tags. The previous filter was r"[aoj]\d", which caught three annotator letters and
# let m* and f* through -- 'm4' was then reported as a difficulty tag predicting
# failure at t=-3.88, when it is annotator m at confidence 4. Pooling by LETTER shows
# no signal (all |t| <= 0.96); all the signal is in the DIGIT.
ANNOTATOR = re.compile(r"[a-z]\d")

# Spelling variants in the raw files split one concept across two keys and halve n.
SYNONYMS = {
    "lowfamiliaritywiththesong/style": "lowfamiliarity",
    "tempoincontinuity": "tempodiscontinuity",
    "excerptistheintroductionofasong": "excerptistheintroduction",
    "excerptistheintroductionofthesong": "excerptistheintroduction",
    "pause": "pauses",
    "ternary": "ternarymeter",
}


def load_tags(with_confidence=False):
    """stem like 'smc_001' -> set of normalised tags (annotator codes removed).

    with_confidence=True instead returns the annotator CONFIDENCE digit per track,
    which is the quantity that actually predicts difficulty.
    """
    out, conf = {}, {}
    for f in sorted(TAGDIR.glob("*.tag")):
        n = int(re.search(r"(\d+)", f.stem).group(1))
        stem = f"smc_{n:03d}"
        tags = set()
        for part in re.split(r"[,\n]", f.read_text()):
            t = part.strip().lower().strip("()").replace(" ", "")
            if not t:
                continue
            if ANNOTATOR.fullmatch(t):
                conf[stem] = int(t[1])
                continue
            tags.add(SYNONYMS.get(t, t))
        out[stem] = tags
    return conf if with_confidence else out


def main():
    dev = f"cuda:{sys.argv[1]}" if len(sys.argv) > 1 else "cuda:0"
    res = {}
    for fold in range(8):
        r = score_fold(f"fold{fold}", fold=fold, device=dev)
        for k, v in r.items():
            if not v["flagged"]:
                res[k] = v
        print(f"[fold {fold}] {len(r)} tracks, mean F={np.mean([x['F'] for x in r.values()]):.4f}",
              flush=True)

    tags = load_tags()
    common = [k for k in res if k in tags]
    print(f"\nfold-honest tracks with tags: {len(common)}")
    F = {k: res[k]["F"] for k in common}
    allF = np.array(list(F.values()))
    print(f"overall mean F = {allF.mean():.4f}  (sd {allF.std(ddof=1):.4f})\n")

    counts = defaultdict(list)
    for k in common:
        for t in tags[k]:
            counts[t].append(F[k])

    rows = []
    for t, vals in counts.items():
        if len(vals) < 8:
            continue
        has = np.array(vals)
        hasnot = np.array([F[k] for k in common if t not in tags[k]])
        d = has.mean() - hasnot.mean()
        se = np.sqrt(has.var(ddof=1)/len(has) + hasnot.var(ddof=1)/len(hasnot))
        rows.append((d/se if se else 0, t, len(has), has.mean(), hasnot.mean(), d, se))

    # 26 tests at a fixed |t|>2 with no multiplicity control let through anything;
    # nothing here survives Bonferroni and the BH minimum q was 0.051. Report q, and
    # do NOT use "predicts nothing" as a kill rule -- poorsoundquality has a LARGER
    # point estimate (-0.0669) than slowtempo (-0.0634) and differs only in n.
    from scipy import stats as _st
    rows.sort()
    pvals = [2 * _st.norm.sf(abs(r[0])) for r in rows]
    order = np.argsort(pvals)
    m = len(pvals)
    q = [0.0] * m
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        prev = min(prev, pvals[idx] * m / (m - rank + 1))
        q[idx] = prev
    print(f"{'tag':<28}{'n':>5}{'F(has)':>9}{'F(not)':>9}{'delta':>9}{'se':>8}{'t':>7}{'BH q':>8}")
    for i, (tstat, t, n, m1, m0, d, se) in enumerate(rows):
        star = " <-- survives FDR" if q[i] < 0.10 else ""
        print(f"{t:<28}{n:5d}{m1:9.4f}{m0:9.4f}{d:+9.4f}{se:8.4f}{tstat:7.2f}{q[i]:8.3f}{star}")

    # The annotator confidence digit: one ordered trend, not 26 tests.
    conf = load_tags(with_confidence=True)
    print(f"\nannotator CONFIDENCE (1 = easy to annotate ... 4 = very hard)")
    print(f"{'conf':<8}{'n':>5}{'mean F':>10}")
    buckets = defaultdict(list)
    for k in common:
        if k in conf:
            buckets[conf[k]].append(F[k])
    for c in sorted(buckets):
        print(f"{c:<8}{len(buckets[c]):5d}{np.mean(buckets[c]):10.4f}")
    xs = np.array([conf[k] for k in common if k in conf], dtype=float)
    ys = np.array([F[k] for k in common if k in conf])
    r, p = _st.pearsonr(xs, ys)
    print(f"  trend: r={r:+.3f}  p={p:.2e}  n={len(xs)}")


if __name__ == "__main__":
    main()
