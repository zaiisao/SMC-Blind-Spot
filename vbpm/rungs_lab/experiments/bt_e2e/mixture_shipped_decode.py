"""Does the mixture kernel's model-level win survive DEPLOYMENT decode?

The +0.0093 (mixture 0.9193 vs R1 0.9100) was bare-vs-bare. Shipped R1 adds madmom's
deployment heuristics (threshold crop + peak snap) worth ~+0.04 on this frontend. Both
heuristics are already behavior-copied in our certified code (rungs/deployment.threshold_crop,
readout.state_path_to_events(snap_to_activations=...)); this script applies the IDENTICAL
wrapper to both kernels:

  row 1  R1 shipped (DBN2016, threshold=0.2, correct)  -- reference 0.9506
  row 2  R1 exponential lam=100, OUR wrapper           -- calibration: must ~= row 1
  row 3  MIXTURE (w=0.370, lam=93.1), OUR wrapper      -- the question
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import final_eval; final_eval.DEVICE = "cuda:1"
import mir_eval
from train_bt import FPS, BT_SHIPPED_DECODE, load_songs
from rungs.r1_2016_dbn import DBN2016
from rungs.deployment import threshold_crop
from rungs.bar_pointer.readout import state_path_to_events
from mixture_kernel_probe import MixtureLambda
from crf_baseline import CRFLearnedFactors

DEVICE = "cuda:1"
W_MIX, LAMBDA_MIX = 0.370, 93.1
THRESHOLD = BT_SHIPPED_DECODE["threshold"]


def decode_wrapped(mix: MixtureLambda, kernel: torch.Tensor, activations: np.ndarray) -> dict:
    cropped, first = threshold_crop(activations, THRESHOLD)
    if not cropped.size:
        return {"beats": np.empty(0), "downbeats": np.empty(0)}
    acts = torch.from_numpy(np.ascontiguousarray(cropped)).float().to(DEVICE)
    densities = mix.log_class_densities(acts)
    best = None
    for mi in range(len(mix.chassis.state_spaces)):
        dp = mix.chassis.dynamic_programs[mi]
        path, score = dp.viterbi(mix.chassis.log_initial_distributions[mi], kernel, densities,
                                 state_to_class=mix.chassis.state_to_classes[mi],
                                 return_log_score=True)
        if best is None or score > best[0]:
            best = (score, path.cpu().numpy(), mix.chassis.state_spaces[mi])
    _, path, space = best
    return state_path_to_events(path, space, mix.chassis.fps,
                                snap_to_activations=cropped, first_frame=first)


def main():
    torch.manual_seed(0)
    mix = MixtureLambda(fps=FPS, device=DEVICE,
                        observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    probe = CRFLearnedFactors(fps=FPS, device=DEVICE,
                              observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    _, val, _ = load_songs(probe)
    model = final_eval.load_model(Path(__file__).resolve().parent / "vanilla_best_prelim.pt")
    val_acts = final_eval.activations_for(model, val)

    def score(get_events):
        bf, df = [], []
        for e in val:
            ev = get_events(e)
            est_b = mir_eval.beat.trim_beats(ev["beats"])
            bf.append(mir_eval.beat.f_measure(mir_eval.beat.trim_beats(e["beat_times"]), est_b)
                      if len(est_b) else 0.0)
            est_d = mir_eval.beat.trim_beats(ev["downbeats"])
            df.append(mir_eval.beat.f_measure(mir_eval.beat.trim_beats(e["downbeat_times"]), est_d)
                      if len(est_d) else 0.0)
        return float(np.mean(bf)), float(np.mean(df))

    r1_shipped = DBN2016(fps=FPS, device=DEVICE, dtype=torch.float32, bounding="none",
                         transition_lambda=100.0, **BT_SHIPPED_DECODE)
    b, d = score(lambda e: r1_shipped.predict(val_acts[e["stem"]]))
    print(f"R1 shipped (DBN2016 heuristics)        : beatF {b:.4f}  dbF {d:.4f}", flush=True)

    exp_kernel = mix.log_mixture_kernel(1e-9, 100.0)   # w->0: pure exponential lam=100
    b, d = score(lambda e: decode_wrapped(mix, exp_kernel, val_acts[e["stem"]]))
    print(f"R1 lam=100, OUR wrapper (calibration)  : beatF {b:.4f}  dbF {d:.4f}", flush=True)

    mix_kernel = mix.log_mixture_kernel(W_MIX, LAMBDA_MIX)
    b, d = score(lambda e: decode_wrapped(mix, mix_kernel, val_acts[e["stem"]]))
    print(f"MIXTURE (w=.370, lam=93.1), wrapper    : beatF {b:.4f}  dbF {d:.4f}", flush=True)


if __name__ == "__main__":
    main()
