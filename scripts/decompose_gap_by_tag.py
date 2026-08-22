"""Where does the missing F (0.627 -> 1.0) actually live, by SMC's own difficulty tags and by
annotator confidence?

Never done before this: we tested tag ENRICHMENT among model failures, but never decomposed the
deficit itself. Deficit share of a tag = sum(1-F) over tagged tracks / sum(1-F) over all tracks.
Over-representation = deficit share / track share (>1 means the tag carries more than its headcount).
Tags overlap (a track carries several), so shares do not sum to 100%.

F is peak-pick @0.5 (the 0.6270 baseline); CMLt is the beat-only DBN (the paper's primary metric).
Both per-track tables are already on disk from scripts/exp_beat_only_dbn.py.
"""
import json
import numpy as np
from scipy.stats import spearmanr

per = json.load(open("/tmp/overnight/beat_only_dbn.json"))["per_track"]
peak, bo = per["peak"], per["beatonly_min30"]
tags = json.load(open("/tmp/smc_tags.json"))
ids = sorted(t for t in peak if t in tags and t in bo)
F = {t: peak[t]["F"] for t in ids}
C = {t: bo[t]["CMLt"] for t in ids}
defF = sum(1 - F[t] for t in ids)
defC = sum(1 - C[t] for t in ids)
print(f"n={len(ids)}  mean F {np.mean(list(F.values())):.4f}  "
      f"total F-deficit {defF:.1f}  total CMLt-deficit {defC:.1f}\n")

byw = {}
for t in ids:
    for w in tags[t]["words"]:
        byw.setdefault(w, []).append(t)

rows = []
for w, ts in byw.items():
    if len(ts) < 10:
        continue
    dF = sum(1 - F[t] for t in ts)
    dC = sum(1 - C[t] for t in ts)
    rows.append((dF / defF, len(ts), w,
                 float(np.mean([F[t] for t in ts])), float(np.mean([C[t] for t in ts])),
                 (dF / defF) / (len(ts) / len(ids)), dC / defC))
rows.sort(reverse=True)
print(f"{'tag':<42}{'n':>4}{'meanF':>8}{'meanCMLt':>9}{'F-def%':>8}{'CMLt-def%':>10}{'overrep':>8}")
for share, n, w, mf, mc, orr, cshare in rows:
    print(f"{w:<42}{n:>4}{mf:>8.3f}{mc:>9.3f}{share:>8.1%}{cshare:>10.1%}{orr:>8.2f}")

print("\nby annotator confidence (1 = most confident):")
print(f"{'conf':>5}{'n':>5}{'meanF':>8}{'meanCMLt':>9}{'F-def%':>8}{'track%':>8}")
for c in (1, 2, 3, 4):
    ts = [t for t in ids if tags[t]["conf"] == c]
    if not ts:
        continue
    dF = sum(1 - F[t] for t in ts)
    print(f"{c:>5}{len(ts):>5}{np.mean([F[t] for t in ts]):>8.3f}"
          f"{np.mean([C[t] for t in ts]):>9.3f}{dF/defF:>8.1%}{len(ts)/len(ids):>8.1%}")
cs = [tags[t]["conf"] for t in ids if tags[t]["conf"]]
fs = [F[t] for t in ids if tags[t]["conf"]]
cc = [C[t] for t in ids if tags[t]["conf"]]
r1, p1 = spearmanr(cs, fs)
r2, p2 = spearmanr(cs, cc)
print(f"\nSpearman(confidence, F)    = {r1:+.3f} (p={p1:.2g})")
print(f"Spearman(confidence, CMLt) = {r2:+.3f} (p={p2:.2g})")
