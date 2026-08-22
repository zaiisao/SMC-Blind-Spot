"""FPS-resolution probe for the R2 generative lambda.

Hypothesis test: how much of the lambda_MLE ~= 40 looseness is FRAME-GRID artifact
(quantization jitter + forced integer-interval alternation) vs real musical timing?
The kernel acts on interval RATIOS, so lambda is directly comparable across fps.

Prediction: if grid effects dominate, EM at 2x fps recovers a substantially STIFFER lambda;
if lambda stays ~40, the looseness is mostly genuine microtiming/drift.

Same 300 crops as train_r2_em.py; activations upsampled 2x by linear interpolation
(sub-frame peak structure is partially recoverable by interpolation since the underlying
activation is a smooth function sampled on the grid). EM only -- no gradient arm, no decode.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import final_eval; final_eval.DEVICE = "cuda:1"
from train_bt import FPS, BT_SHIPPED_DECODE, load_songs
from rungs.r2_em_dbn import R2GenerativeLambda
from crf_baseline import CRFLearnedFactors

NUM_CROPS, CROP_FRAMES, EM_ITERS, UPSAMPLE = 300, 700, 8, 2
DEVICE = "cuda:1"


def upsample(a, factor):
    """Linear interpolation along time: [T, 2] -> [factor*(T-1)+1, 2]."""
    t = np.arange(a.shape[0], dtype=np.float64)
    t_new = np.linspace(0, a.shape[0] - 1, factor * (a.shape[0] - 1) + 1)
    return np.stack([np.interp(t_new, t, a[:, c]) for c in range(a.shape[1])], axis=1)


def main():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    probe = CRFLearnedFactors(fps=FPS, device=DEVICE,
                              observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    train, _, _ = load_songs(probe)
    model = final_eval.load_model(Path(__file__).resolve().parent / "vanilla_best_prelim.pt")
    train_acts = final_eval.activations_for(model, train[:NUM_CROPS])

    for factor, fps in ((1, FPS), (UPSAMPLE, UPSAMPLE * FPS)):
        crops = []
        for e in train[:NUM_CROPS]:
            a = train_acts[e["stem"]]
            if a.shape[0] > CROP_FRAMES + 1:
                s = rng.integers(0, a.shape[0] - CROP_FRAMES)
                a = a[s:s + CROP_FRAMES]
            if factor > 1:
                a = upsample(a, factor)
            crops.append(torch.from_numpy(np.ascontiguousarray(a)).float().to(DEVICE))
        r2 = R2GenerativeLambda(fps=fps, device=DEVICE,
                                observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
        n_states = r2.chassis.state_spaces[0].num_states
        print(f"\n=== fps {fps:.2f} (x{factor}) | {len(crops)} crops | "
              f"{n_states} states (meter 3) ===", flush=True)
        for it in range(EM_ITERS):
            lam = r2.em_step(crops)
            print(f"  EM iter {it}: lambda = {lam:.2f}", flush=True)
        print(f"fps x{factor} FINAL lambda = {r2.transition_lambda:.2f}", flush=True)


if __name__ == "__main__":
    main()
