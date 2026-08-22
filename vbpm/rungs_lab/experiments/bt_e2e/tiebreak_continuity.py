"""Tiebreaker: beat F ties near 0.950 for every sensible deployed config -- do the CONTINUITY
metrics (CMLt: correct metrical level, sustained; AMLt: allowed levels) separate them?
Same val fold, same activations; every custom kernel goes through the SAME validated wrapper
(threshold_crop + peak snap); bare rows for the model-level comparison."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import final_eval; final_eval.DEVICE = "cuda:1"
import mir_eval
from train_bt import FPS, BT_SHIPPED_DECODE, load_songs
from rungs.deployment import threshold_crop
from rungs.bar_pointer.readout import state_path_to_events
from mixture_kernel_probe import MixtureLambda
from crf_baseline import CRFLearnedFactors

DEVICE = "cuda:1"
THRESHOLD = BT_SHIPPED_DECODE["threshold"]


def decode(mix, kernel, activations, deployed, chassis=None):
    chassis = chassis or mix.chassis
    if deployed:
        cropped, first = threshold_crop(activations, THRESHOLD)
    else:
        cropped, first = activations, 0
    if not cropped.size:
        return {"beats": np.empty(0), "downbeats": np.empty(0)}
    acts = torch.from_numpy(np.ascontiguousarray(cropped)).float().to(DEVICE)
    densities = mix.log_class_densities(acts, chassis)
    best = None
    for mi in range(len(chassis.state_spaces)):
        dp = chassis.dynamic_programs[mi]
        path, score = dp.viterbi(chassis.log_initial_distributions[mi], kernel, densities,
                                 state_to_class=chassis.state_to_classes[mi],
                                 return_log_score=True)
        if best is None or score > best[0]:
            best = (score, path.cpu().numpy(), chassis.state_spaces[mi])
    _, path, space = best
    return state_path_to_events(path, space, chassis.fps,
                                snap_to_activations=cropped if deployed else None,
                                first_frame=first)


def main():
    torch.manual_seed(0)
    mix = MixtureLambda(fps=FPS, device=DEVICE,
                        observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    probe = CRFLearnedFactors(fps=FPS, device=DEVICE,
                              observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    _, val, _ = load_songs(probe)
    model = final_eval.load_model(Path(__file__).resolve().parent / "vanilla_best_prelim.pt")
    val_acts = final_eval.activations_for(model, val)

    def score(kernel, deployed, chassis=None):
        bf, cml, aml, df, dcml, daml = [], [], [], [], [], []
        for e in val:
            ev = decode(mix, kernel, val_acts[e["stem"]], deployed, chassis)
            ref = mir_eval.beat.trim_beats(e["beat_times"])
            est = mir_eval.beat.trim_beats(ev["beats"])
            if len(est):
                bf.append(mir_eval.beat.f_measure(ref, est))
                _, cmlt, _, amlt = mir_eval.beat.continuity(ref, est)
                cml.append(cmlt); aml.append(amlt)
            else:
                bf.append(0.0); cml.append(0.0); aml.append(0.0)
            ref_d = mir_eval.beat.trim_beats(e["downbeat_times"])
            est_d = mir_eval.beat.trim_beats(ev["downbeats"])
            if len(est_d) and len(ref_d) > 1:
                df.append(mir_eval.beat.f_measure(ref_d, est_d))
                _, dcmlt, _, damlt = mir_eval.beat.continuity(ref_d, est_d)
                dcml.append(dcmlt); daml.append(damlt)
            else:
                df.append(0.0); dcml.append(0.0); daml.append(0.0)
        return (np.mean(bf), np.mean(cml), np.mean(aml),
                np.mean(df), np.mean(dcml), np.mean(daml))

    rows = [
        ("DEPLOYED lam=100 hand-set      ", mix.log_mixture_kernel(1e-9, 100.0), True, None),
        ("DEPLOYED crf lam=98.6          ", mix.log_mixture_kernel(1e-9, 98.6), True, None),
        ("DEPLOYED MIXTURE w=.37 lam=93.1", mix.log_mixture_kernel(0.370, 93.1), True, None),
        ("DEPLOYED em-single lam=40.3    ", mix.log_mixture_kernel(1e-9, 40.3), True, None),
        ("DEPLOYED learn-obs lam=30.9 o=4", mix.log_mixture_kernel(1e-9, 30.9), True,
         mix._chassis_for(4)),
        ("BARE     lam=100 hand-set      ", mix.log_mixture_kernel(1e-9, 100.0), False, None),
        ("BARE     MIXTURE w=.37 lam=93.1", mix.log_mixture_kernel(0.370, 93.1), False, None),
        ("BARE     em-single lam=40.3    ", mix.log_mixture_kernel(1e-9, 40.3), False, None),
    ]
    print(f"{'config':34s}  beatF   CMLt    AMLt    dbF     dbCMLt  dbAMLt", flush=True)
    for name, kernel, deployed, chassis in rows:
        b, c, a, d, dc, da = score(kernel, deployed, chassis)
        print(f"{name}  {b:.4f}  {c:.4f}  {a:.4f}  {d:.4f}  {dc:.4f}  {da:.4f}", flush=True)


if __name__ == "__main__":
    main()
