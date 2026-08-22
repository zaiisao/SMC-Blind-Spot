"""STEP 0: can the beat activation's own autocorrelation recover the head's octave errors?

HYPOTHESIS (from scripts/diagnostics/diag_tempo_head_failures.py). Of the 103 tracks the head gets wrong,
**37 (35.9%) are group A**: the annotated tempo IS the peak of the autocorrelation of the model's own
beat activation, and the head outputs something else. The head reads only 512-d embeddings -- it
never sees the activation and never sees an ACF, so it must learn periodicity estimation implicitly
while the ACF computes it in closed form. That is a FEATURE gap, not a loss gap, which is consistent
with both sandbox loss variants failing and with every decode-time repair failing: none of them gave
the head information it did not already have.

This tests the hypothesis WITHOUT retraining, by blending the two distributions in log space:

    log q_blend  =  (1 - w) * log q_head  +  w * log q_acf,   renormalised

`q_acf` is built from the unbiased, mean-centred autocorrelation of the beat activation over the
DBN's own tempo grid, softmaxed at temperature `tau` -- the same readout that was GATED against the
hard peak-pick measurement (sign agreement, |soft-hard| <= 0.25, matching flat regions; all passed).

WHAT WOULD FALSIFY IT. Group A being 37 tracks does not mean the ACF is *reliably* right; a blend
could lose more on the 114 tracks the head already gets right than it gains on the 37. The metric is
therefore the OCTAVE-ERROR RATE over all 217 tracks, which counts both directions:

    current head        24.9%  (54/217)
    break-even          24.4%  (measured, §4.9)
    target              ~15%   (worth ~+0.07 CMLt at +0.007/point)

Reported alongside: how many of group A are recovered, and how many previously-correct tracks are
broken. A blend that trades 1:1 is worthless even if group-A recovery looks good.

No training, no SMC fitting -- w and tau are swept and the whole curve is reported, so the reader can
see the shape rather than a single chosen point.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from _harness import FPS, track_ids, load_gt, bthis_beat_prob
from tempo_readout import lag_grid
from train_tempo_observation import load_net

SMC_EMB = Path("/disk1/jaehoon/smc_bt_embeddings")
HEAD = Path("/disk1/jaehoon/tempo_obs_head.pt")
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
TOL = 0.15
WS = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
TAUS = [10.0, 30.0, 60.0]


def acf_dist(act, lags, tau):
    """Unbiased, mean-centred, scale-free ACF over the tempo grid -> a distribution."""
    a = torch.as_tensor(act, dtype=torch.float64)
    a = a - a.mean()
    T = len(a)
    r = []
    for l in lags.tolist():
        r.append((a[:T - l] * a[l:]).sum() / float(T - l) if T - l >= 8
                 else torch.tensor(-1e9, dtype=torch.float64))
    r = torch.stack(r)
    r = r / ((a * a).sum() / T + 1e-12)
    return torch.log_softmax(tau * r, dim=0)


def cls(x):
    for v, n in ((0, "correct"), (1, "2x"), (-1, "0.5x"),
                 (np.log2(3), "3x"), (-np.log2(3), "1/3x")):
        if abs(x - v) < TOL:
            return n
    return "other"


def main():
    dev = torch.device(DEV)
    net = load_net(str(HEAD), DEV)
    ck = torch.load(str(HEAD), map_location="cpu", weights_only=False)
    grid = np.asarray(ck["grid"], dtype=np.float64)
    # The ACF must be evaluated on the HEAD's grid, not lag_grid()'s uniform 30-300 BPM lags:
    # the DBN uses num_tempi=60 non-uniformly spaced integer intervals (14..100 frames), so a
    # uniform 91-lag grid is a different support and the two distributions could not be blended.
    lags = torch.tensor(grid.astype(np.int64))
    assert len(lags) == len(grid)

    rows = {}
    for t in track_ids():
        f = SMC_EMB / f"{t}.npz"
        if not f.exists():
            continue
        gt = np.asarray(load_gt(t), float)
        if len(gt) < 4:
            continue
        true_iv = float(np.median(np.diff(gt)) * FPS)
        z = np.load(f, allow_pickle=True)
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(z["emb"], np.float32)).permute(1, 0).unsqueeze(0).to(dev)
            lq = torch.log_softmax(net(x).float(), 1)[0].T.double()            # [T, V] per frame
        rows[t] = {"true_iv": true_iv, "lq": lq.cpu(),
                   "act": np.asarray(bthis_beat_prob(t), dtype=np.float64)}
    print(f"n={len(rows)} tracks", flush=True)

    # AGGREGATION must match scripts/train_tempo_observation.py:372-379 -- per-frame argmax, then
    # MEDIAN over frames. Averaging log-probs first and taking one argmax is a DIFFERENT estimator
    # and gave 33.2% octave error vs the deployed 24.9% on the same head and tracks: an 8-point
    # swing from aggregation alone. All arms below use the deployed aggregation.
    def agg(lp):
        return float(np.median(grid[lp.argmax(1).numpy()]))

    def octave_rate(pred_iv):
        c = [cls(np.log2(pred_iv[t] / rows[t]["true_iv"])) for t in rows]
        oct_ = sum(1 for x in c if x in ("2x", "0.5x", "3x", "1/3x"))
        return oct_ / len(c), sum(1 for x in c if x == "correct") / len(c), c

    base_pred = {t: agg(rows[t]["lq"]) for t in rows}
    b_oct, b_acc, b_cls = octave_rate(base_pred)
    print(f"head alone:   octave-error {b_oct:.1%}   correct {b_acc:.1%}")

    best = None
    print(f"\n{'tau':>6}{'w':>6}{'octErr':>9}{'correct':>9}{'A recov':>9}{'broken':>8}")
    out = {}
    for tau in TAUS:
        acf = {t: acf_dist(rows[t]["act"], lags, tau) for t in rows}
        a_pred = {t: agg(acf[t][None, :].expand(rows[t]["lq"].shape[0], -1)) for t in rows}
        a_oct, a_acc, _ = octave_rate(a_pred)
        if tau == TAUS[0]:
            print(f"  (ACF alone at tau={tau:g}: octave-error {a_oct:.1%}, correct {a_acc:.1%})")
        for w in WS:
            pred = {}
            for t in rows:
                lb = (1 - w) * rows[t]["lq"] + w * acf[t][None, :]     # broadcast track-level ACF
                pred[t] = agg(lb)
            o, acc, c = octave_rate(pred)
            recov = sum(1 for i, t in enumerate(rows)
                        if b_cls[i] != "correct" and c[i] == "correct")
            broke = sum(1 for i, t in enumerate(rows)
                        if b_cls[i] == "correct" and c[i] != "correct")
            out[f"tau{tau:g}_w{w:g}"] = {"octave": o, "correct": acc,
                                         "recovered": recov, "broken": broke}
            print(f"{tau:>6.0f}{w:>6.2f}{o:>9.1%}{acc:>9.1%}{recov:>9d}{broke:>8d}")
            if best is None or o < best[0]:
                best = (o, tau, w, acc, recov, broke)
    print(f"\nBEST: octave-error {best[0]:.1%} at tau={best[1]:g}, w={best[2]:.2f} "
          f"(correct {best[3]:.1%}, recovered {best[4]}, broken {best[5]})")
    print(f"  head alone {b_oct:.1%}   break-even 24.4%   target ~15%")
    verdict = ("PROMISING -- retraining with ACF features is justified (Step 1)"
               if best[0] < b_oct - 0.03 else
               "NO GAIN -- the ACF's answer is not reliably usable; do not retrain on it")
    print(f"  VERDICT: {verdict}")
    json.dump({"head_octave": b_oct, "grid_search": out},
              open("/tmp/overnight/head_acf_blend.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
