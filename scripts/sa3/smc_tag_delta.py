"""Break the aug-vs-control fold-0 difference down by SMC difficulty tag.

Fold 0 is 27 tracks, so most tag cells hold a handful of tracks and individual
numbers are noise. What is worth looking at is whether any gain lands on the tags
that actually predict failure (measured fold-honestly over 216 tracks in
`smc_tag_analysis.py`): ambiguity of metrical level (t=-3.38), beat phase
ambiguity (t=-2.98), low familiarity (t=-2.43), slow tempo (t=-2.07). Gains on
tags that predict nothing -- missing bass, ternary meter, quiet accompaniment --
would suggest we moved the model without touching the actual difficulty.
"""
import argparse, csv, glob, os, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TAGDIR = Path("/home/sogang/jaehoon/Analyze-SMC/smc_metadata/tags")
PREDICTIVE = {"ambiguityofmetricallevel", "beatphaseambiguity", "lowfamiliarity",
              "slowtempo", "m4", "m3"}


def load_tags():
    out = {}
    for f in sorted(TAGDIR.glob("*.tag")):
        n = int(re.search(r"(\d+)", f.stem).group(1))
        tags = set()
        for part in re.split(r"[,\n]", f.read_text()):
            t = part.strip().lower().strip("()").replace(" ", "")
            if not t or re.fullmatch(r"[aoj]\d", t):
                continue
            tags.add(t)
        out[f"smc_{n:03d}"] = tags
    return out


def load_run(run, step):
    p = f"/disk1/jaehoon/bt_sdedit_runs/{run}/pertrack_step{step:06d}.csv"
    r = list(csv.DictReader(open(p)))
    k = [c for c in r[0] if c.lower() in ("track", "name", "file", "id")][0]
    fk = [c for c in r[0] if c.lower() == "f"][0]
    return {x[k]: float(x[fk]) for x in r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="scr_sdaug_s0")
    ap.add_argument("--ctl", default="scr_sdctl_s0")
    ap.add_argument("--step", type=int, default=None, help="default: latest matched")
    a = ap.parse_args()

    if a.step is None:
        sa = {int(re.search(r"(\d+)", os.path.basename(p)).group(1))
              for p in glob.glob(f"/disk1/jaehoon/bt_sdedit_runs/{a.aug}/pertrack_step*.csv")}
        sb = {int(re.search(r"(\d+)", os.path.basename(p)).group(1))
              for p in glob.glob(f"/disk1/jaehoon/bt_sdedit_runs/{a.ctl}/pertrack_step*.csv")}
        both = sorted(sa & sb)
        if not both:
            print("no matched step yet"); return
        a.step = both[-1]

    A, B = load_run(a.aug, a.step), load_run(a.ctl, a.step)
    tags = load_tags()
    common = sorted(set(A) & set(B) & set(tags))
    d = {k: A[k] - B[k] for k in common}
    print(f"step {a.step}   n={len(common)} fold-0 tracks with tags")
    print(f"overall  aug={np.mean([A[k] for k in common]):.4f}  "
          f"ctl={np.mean([B[k] for k in common]):.4f}  delta={np.mean(list(d.values())):+.4f}\n")

    rows = []
    for t in {x for k in common for x in tags[k]}:
        has = [d[k] for k in common if t in tags[k]]
        if len(has) < 2:
            continue
        rows.append((np.mean(has), t, len(has)))
    rows.sort(reverse=True)
    print(f"{'tag':<34}{'n':>4}{'mean delta':>12}   predicts-failure?")
    for m, t, n in rows:
        star = "YES" if t in PREDICTIVE else ""
        print(f"{t:<34}{n:4d}{m:+12.4f}   {star}")


if __name__ == "__main__":
    main()
