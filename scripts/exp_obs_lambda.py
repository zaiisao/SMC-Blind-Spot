"""observation_lambda sweep on Beat This + SMC -- the parameter the project never swept.

WHY. scripts/_harness.py:make_dbn() never passes observation_lambda, so every DBN row in the
paper and in FOLLOWUP_FINDINGS.md inherits madmom's default of 16. Beat This's own shipped decode
uses 6. So the whole transition-lambda analysis was conditioned on an observation parameter that
(a) was never examined and (b) is not the value the frontend ships with.

WHAT IT IS. Not a likelihood scale -- a GEOMETRIC one. In the Boeck-2016 observation model a state
counts as "on the beat" iff its position within the beat is < 1/observation_lambda, i.e.

    beat_region_width = 1 / observation_lambda        (as a FRACTION of the beat)

so the effective tolerance in milliseconds SCALES WITH TEMPO, while the F-measure tolerance is a
fixed +-70 ms. At 48 BPM, obs=16 gives a 78 ms region (wider than the metric's window); at 120 BPM
it gives 31 ms (much narrower). Hypothesis: obs_lambda is mis-specified in a tempo-dependent way,
a second slow-tempo mis-specification alongside min_bpm=55, living in the OBSERVATION model.

DESIGN.
  1. obs sweep at fixed transition_lambda=100, all 5 metrics, split by GT tempo band.
  2. Joint (obs x transition) grid -- the two interact, and every transition-lambda conclusion in
     the project was drawn at obs=16, so the interaction must be measured, not assumed.
  3. Per-track oracle over obs, to bound what a learned/adaptive observation model could buy.
Validation: obs=16, lam=100 must reproduce the known wide-DBN row F 0.5851 / CMLt 0.5102 /
AMLt 0.6735 (that is the configuration _harness has been using all along).
"""
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import FPS, track_ids, load_gt, gt_bpm, bthis_beat_prob, bthis_downbeat_prob, ev

OUTDIR = Path("/tmp/overnight")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "obs_lambda_sweep.json"
JSONL = OUTDIR / "obs_lambda_sweep.per_track.jsonl"

OBS_GRID = [2, 4, 6, 8, 12, 16, 24, 32]
LAM_GRID = [5, 20, 50, 100, 200]
BANDS = [(0, 55), (55, 70), (70, 90), (90, 120), (120, 999)]
WORKERS = 6


def dbn_with(obs, lam, min_bpm=30.0, max_bpm=215.0):
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    return DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4], min_bpm=min_bpm, max_bpm=max_bpm, fps=FPS,
        transition_lambda=lam, observation_lambda=obs)


def run_cell(args):
    obs, lam, tids = args
    proc = dbn_with(obs, lam)
    eps = 1e-5
    out = {}
    for t in tids:
        bp = bthis_beat_prob(t) * (1 - eps) + eps / 2
        dp = bthis_downbeat_prob(t) * (1 - eps) + eps / 2
        comb = np.vstack((np.maximum(bp - dp, eps / 2), dp)).T
        est = proc(comb)[:, 0]
        out[t] = ev(load_gt(t), est)
    return obs, lam, out


def band_of(bpm):
    for lo, hi in BANDS:
        if lo <= bpm < hi:
            return f"{lo}-{hi}"
    return "unk"


def summarise(per, ids, bands):
    res = {"all": {m: float(np.mean([per[t][m] for t in ids]))
                   for m in ("F", "CMLt", "AMLt", "CMLc", "AMLc")}}
    by = defaultdict(list)
    for t in ids:
        by[bands[t]].append(t)
    for b, grp in sorted(by.items()):
        res[b] = {"n": len(grp),
                  **{m: float(np.mean([per[t][m] for t in grp]))
                     for m in ("F", "CMLt", "AMLt")}}
    return res


def main():
    ids = track_ids()
    bands = {t: band_of(gt_bpm(load_gt(t))) for t in ids}
    print(f"n={len(ids)} obs_grid={OBS_GRID} lam_grid={LAM_GRID}", flush=True)

    cells = [(o, 100) for o in OBS_GRID]
    cells += [(o, l) for o in (4, 6, 8, 16) for l in LAM_GRID if not (l == 100 and o in OBS_GRID)]
    cells = list(dict.fromkeys(cells))

    results, fh = {}, open(JSONL, "w")
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for obs, lam, per in pool.map(run_cell, [(o, l, ids) for o, l in cells]):
            results[(obs, lam)] = per
            for t, sc in per.items():
                fh.write(json.dumps({"obs": obs, "lam": lam, "track": t, **sc}) + "\n")
            fh.flush()
            s = summarise(per, ids, bands)["all"]
            print(f"  obs={obs:<3} lam={lam:<4} F {s['F']:.4f} CMLt {s['CMLt']:.4f} "
                  f"AMLt {s['AMLt']:.4f} AMLc {s['AMLc']:.4f}", flush=True)
    fh.close()

    ref = results.get((16, 100))
    val = summarise(ref, ids, bands)["all"] if ref else None
    if val:
        ok = abs(val["F"] - 0.5851) < 0.003
        print(f"\nVALIDATION obs=16 lam=100 -> F {val['F']:.4f} (want 0.5851) "
              f"CMLt {val['CMLt']:.4f} (want 0.5102) : {'PASS' if ok else 'FAIL'}", flush=True)

    # per-track oracle over obs at lam=100: ceiling for an adaptive observation model
    obs_cells = [(o, 100) for o in OBS_GRID if (o, 100) in results]
    orc = {m: float(np.mean([max(results[c][t][m] for c in obs_cells) for t in ids]))
           for m in ("F", "CMLt", "AMLt")}
    orc_lam = {t: max(obs_cells, key=lambda c: results[c][t]["F"])[0] for t in ids}

    payload = {
        "n": len(ids), "obs_grid": OBS_GRID, "lam_grid": LAM_GRID,
        "validation_obs16_lam100": val,
        "cells": {f"obs{o}_lam{l}": summarise(p, ids, bands) for (o, l), p in results.items()},
        "per_track_oracle_obs_at_lam100": orc,
        "oracle_obs_choice": {t: int(v) for t, v in orc_lam.items()},
        "oracle_obs_histogram": {str(o): sum(1 for v in orc_lam.values() if v == o) for o in OBS_GRID},
    }
    OUT.write_text(json.dumps(payload, indent=1))
    best = max(obs_cells, key=lambda c: summarise(results[c], ids, bands)["all"]["F"])
    print(f"\nbest obs at lam=100: obs={best[0]} -> {summarise(results[best], ids, bands)['all']}",
          flush=True)
    print(f"per-track ORACLE over obs (lam=100): {orc}", flush=True)
    print(f"oracle obs histogram: {payload['oracle_obs_histogram']}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
