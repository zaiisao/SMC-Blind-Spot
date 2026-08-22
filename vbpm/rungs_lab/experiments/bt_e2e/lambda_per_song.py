"""Per-song tempo-tolerance lambda (smooth-target, jitter-free) + correlation with BPM.

Pure tempo-annotation property: for each song, take the smoothest-in-tolerance beat frames, map
consecutive interval pairs onto the real (row-normalized) tempo kernel grid, and find the lambda
that maximizes the transition likelihood of that interval sequence. No frontend, no activations.
Constant-tempo songs push lambda to the grid ceiling (no penalty for staying); wandering-tempo
songs pull it down. Then: does lambda correlate with BPM? (Watch the confound: high BPM -> short
intervals -> coarser quantization granularity, a MECHANICAL push on lambda separate from musical.)"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.songs import iter_songs
from train_bt import FPS
from train_r2_frozen_smooth import smooth_beat_frames
from crf_baseline import CRFLearnedFactors
import torch

DEVICE = "cpu"
LAMBDAS = np.logspace(np.log10(2), np.log10(400), 80)


def main():
    r2 = CRFLearnedFactors(fps=FPS, device=DEVICE, observation_lambda=6)
    grid = r2.chassis.state_spaces[0].interval_frames          # [V] integer frame intervals
    # precompute the normalized log tempo kernel for each candidate lambda: [L, V, V]
    kernels = []
    for lam in LAMBDAS:
        with torch.no_grad():
            r2.log_transition_lambda.copy_(torch.log(torch.tensor(float(lam))))
            kernels.append(r2.log_tempo_transition().cpu().numpy())
    kernels = np.stack(kernels)                                 # [L, V, V]
    lam_ceiling = LAMBDAS[-1]

    rows = []
    for s in iter_songs():
        bt, _ = s.beats()
        if len(bt) < 12:
            continue
        frames = smooth_beat_frames(bt)
        intervals = np.diff(frames)
        if intervals.min() < grid[0] or intervals.max() > grid[-1]:
            continue
        idx = np.searchsorted(grid, intervals)                 # nearest grid interval index
        idx = np.clip(idx, 0, len(grid) - 1)
        frm, to = idx[:-1], idx[1:]
        nll = -kernels[:, frm, to].sum(axis=1)                 # [L] NLL per candidate lambda
        lam = float(LAMBDAS[np.argmin(nll)])
        bpm = 60.0 / float(np.median(np.diff(bt)))
        # fraction of boundaries that are a genuine tempo change (post-smooth)
        change_rate = float((np.diff(intervals) != 0).mean())
        rows.append((s.dataset, bpm, lam, change_rate, len(bt)))

    ds_names = sorted(set(r[0] for r in rows))
    lams = np.array([r[2] for r in rows]); bpms = np.array([r[1] for r in rows])
    print(f"{len(rows)} songs\n")
    print(f"{'dataset':12s} {'n':>4} {'med BPM':>8} {'med lam':>8} {'lam IQR':>14} "
          f"{'%@ceiling':>9} {'med change%':>11}")
    print("  " + "-" * 72)
    for ds in ds_names + ["ALL"]:
        r = rows if ds == "ALL" else [x for x in rows if x[0] == ds]
        L = np.array([x[2] for x in r]); B = np.array([x[1] for x in r])
        C = np.array([x[3] for x in r])
        q1, q3 = np.percentile(L, [25, 75])
        print(f"{ds:12s} {len(r):>4} {np.median(B):>8.1f} {np.median(L):>8.1f} "
              f"[{q1:>5.1f},{q3:>6.1f}] {100*np.mean(L >= lam_ceiling*0.99):>8.1f}% "
              f"{100*np.median(C):>10.1f}%")

    print(f"\nCORRELATION lambda vs BPM (Spearman rho, p):")
    rho, p = stats.spearmanr(bpms, lams)
    print(f"  overall (n={len(rows)}): rho {rho:+.3f}  p {p:.2e}")
    for ds in ds_names:
        r = [x for x in rows if x[0] == ds]
        rho, p = stats.spearmanr([x[1] for x in r], [x[2] for x in r])
        print(f"  {ds:12s} (n={len(r)}): rho {rho:+.3f}  p {p:.2e}")
    # confound check: BPM vs post-smooth change-rate (quantization granularity proxy)
    rho_c, p_c = stats.spearmanr(bpms, [r[3] for r in rows])
    print(f"\nCONFOUND check -- BPM vs post-smooth change-rate: rho {rho_c:+.3f} p {p_c:.2e}")
    print("  (if strongly +, high-BPM 'lower lambda' is partly quantization granularity, not music)")

    import csv
    with open(Path(__file__).resolve().parent / "lambda_per_song.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "bpm", "lambda", "change_rate", "n_beats"])
        w.writerows(rows)
    print(f"\nwrote lambda_per_song.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
