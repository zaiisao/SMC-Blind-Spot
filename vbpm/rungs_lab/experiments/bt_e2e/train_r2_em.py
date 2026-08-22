"""R2 experiment: learn transition_lambda (and optionally observation_lambda) by UNSUPERVISED
generative MLE on frozen activations.

Runs exact EM and direct forward-gradient ascent SIDE BY SIDE on the same marginal likelihood --
Fisher's identity (grad log p(x) = E_post[grad log p(x,z)]) says they must converge to the same
lambda; watching them do so on the real model is the demonstration. NO annotations are used for
learning (meter marginalized); annotations only score the final decode.

--learn-obs adds the exact discrete coordinate update for observation_lambda (argmax of the
marginal over integer candidates; see rungs/r2_em_dbn.py::observation_step). The gradient arm is
lambda-only either way (obs has no gradient -- that is the point).

Ladder comparison at decode (val fold, shipped-BT decode):
    R0 madmom  vs  R1 hand-set  vs  R2 generative  vs  crf_baseline discriminative (98.6)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import final_eval; final_eval.DEVICE = "cuda:0"
import mir_eval
from train_bt import FPS, BT_SHIPPED_DECODE, load_songs
from rungs.r2_em_dbn import R2GenerativeLambda
from rungs.r1_2016_dbn import DBN2016
from rungs.r0_madmom_dbn import MadmomDBN
from crf_baseline import CRFLearnedFactors

NUM_CROPS, CROP_FRAMES, EM_ITERS, GRAD_STEPS = 300, 700, 8, 250


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learn-obs", action="store_true",
                        help="also learn observation_lambda (exact discrete coordinate argmax)")
    args = parser.parse_args()

    torch.manual_seed(0); rng = np.random.default_rng(0)
    r2 = R2GenerativeLambda(fps=FPS, device="cuda:0",
                            observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    probe = CRFLearnedFactors(fps=FPS, device="cuda:0",
                              observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    train, val, _ = load_songs(probe)
    model = final_eval.load_model(Path(__file__).resolve().parent / "vanilla_best_prelim.pt")
    train_acts = final_eval.activations_for(model, train[:NUM_CROPS])
    val_acts = final_eval.activations_for(model, val)

    crops = []
    for e in train[:NUM_CROPS]:
        a = train_acts[e["stem"]]
        if a.shape[0] <= CROP_FRAMES + 1:
            crops.append(torch.from_numpy(a).float().to("cuda:0"))
        else:
            s = rng.integers(0, a.shape[0] - CROP_FRAMES)
            crops.append(torch.from_numpy(a[s:s + CROP_FRAMES]).float().to("cuda:0"))
    print(f"{len(crops)} unsupervised crops ({CROP_FRAMES} frames) | learn-obs={args.learn_obs}",
          flush=True)

    # --- exact EM ---
    print("EM (exact E-step via autograd counts, 1-D M-step"
          + (", + discrete obs argmax" if args.learn_obs else "") + "):", flush=True)
    for it in range(EM_ITERS):
        lam = r2.em_step(crops, learn_observation_lambda=args.learn_obs)
        print(f"  EM iter {it}: lambda = {lam:.2f}, obs_lambda = {r2.observation_lambda}",
              flush=True)
    lam_em, obs_em = r2.transition_lambda, r2.observation_lambda

    # --- direct gradient on the same marginal (Fisher's identity arm; lambda only, at the EM
    # arm's final obs so the two arms optimize the SAME objective) ---
    print(f"Direct forward-gradient ascent on the same marginal (lambda only, obs={obs_em}):",
          flush=True)
    log_lam = torch.tensor(float(np.log(100.0)), device="cuda:0", requires_grad=True)
    opt = torch.optim.Adam([log_lam], lr=0.05)
    grad_chassis = r2._chassis_for(obs_em)
    for step in range(GRAD_STEPS):
        idx = rng.choice(len(crops), size=32, replace=False)
        loss = -torch.stack([r2.marginal_log_likelihood(crops[i], r2.log_kernel(log_lam.exp()),
                                                        grad_chassis)
                             for i in idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 50 == 0:
            print(f"  grad step {step + 1}: lambda = {float(log_lam.exp()):.2f}", flush=True)
    lam_grad = float(log_lam.exp())
    print(f"\nFISHER CHECK -- EM lambda {lam_em:.2f} vs gradient lambda {lam_grad:.2f} "
          f"(same objective at obs={obs_em}; must agree)", flush=True)

    # --- ladder decode comparison ---
    def score(rung):
        fs = []
        for e in val:
            ev = rung.predict(val_acts[e["stem"]])
            est = mir_eval.beat.trim_beats(ev["beats"])
            fs.append(mir_eval.beat.f_measure(mir_eval.beat.trim_beats(e["beat_times"]), est)
                      if len(est) else 0.0)
        return float(np.mean(fs))

    def F_torch(lam, obs):
        decode = dict(BT_SHIPPED_DECODE); decode["observation_lambda"] = obs
        return score(DBN2016(fps=FPS, device="cuda:0", dtype=torch.float32, bounding="none",
                             transition_lambda=lam, **decode))

    obs0 = BT_SHIPPED_DECODE["observation_lambda"]
    print("\nLadder decode comparison (val fold, shipped-BT decode):", flush=True)
    r0 = MadmomDBN(fps=FPS, bounding="none", transition_lambda=100, **BT_SHIPPED_DECODE)
    print(f"  {'R0   madmom (100, obs=' + str(obs0) + ')':44s}: beatF {score(r0):.4f}", flush=True)
    rows = [(f"R1   hand-set (100, obs={obs0})", 100.0, obs0),
            (f"R2   generative ({lam_em:.1f}, obs={obs_em})", lam_em, obs_em),
            (f"crf  discriminative (98.6, obs={obs0})", 98.6, obs0)]
    for label, lam, obs in rows:
        print(f"  {label:44s}: beatF {F_torch(lam, obs):.4f}", flush=True)


if __name__ == "__main__":
    main()
