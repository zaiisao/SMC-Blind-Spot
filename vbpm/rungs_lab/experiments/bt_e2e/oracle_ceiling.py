"""Oracle ceiling probe: how much could ANY decoder improvement buy on these [T,2] activations?

Per-song oracle selection (uses annotations to pick, per song, the best of a decode-variant
family). The gap oracle-minus-shipped bounds every possible decoder gain; the gap 1.0-minus-
oracle is frontend-only territory.

Variants per song: lambda in {40, 100, 150, 300} x meter restriction {both, 3-only, 4-only},
all through the deployed wrapper. Downbeat oracle additionally allows the half-bar flip
(re-assigning downbeats to the opposite beat parity) -- the dominant downbeat error mode.
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
from rungs.deployment import threshold_crop
from rungs.bar_pointer.readout import state_path_to_events
from mixture_kernel_probe import MixtureLambda
from crf_baseline import CRFLearnedFactors

DEVICE = "cuda:1"
THRESHOLD = BT_SHIPPED_DECODE["threshold"]
LAMBDAS = (40.0, 100.0, 150.0, 300.0)


def decode(mix, kernel, activations, meters):
    cropped, first = threshold_crop(activations, THRESHOLD)
    if not cropped.size:
        return {"beats": np.empty(0), "downbeats": np.empty(0)}
    acts = torch.from_numpy(np.ascontiguousarray(cropped)).float().to(DEVICE)
    densities = mix.log_class_densities(acts)
    best = None
    for mi in meters:
        dp = mix.chassis.dynamic_programs[mi]
        path, score = dp.viterbi(mix.chassis.log_initial_distributions[mi], kernel, densities,
                                 state_to_class=mix.chassis.state_to_classes[mi],
                                 return_log_score=True)
        if best is None or score > best[0]:
            best = (score, path.cpu().numpy(), mix.chassis.state_spaces[mi])
    _, path, space = best
    return state_path_to_events(path, space, mix.chassis.fps,
                                snap_to_activations=cropped, first_frame=first)


def half_bar_flip(ev):
    """Downbeats moved to the beat half a bar away (parity flip on the beat list)."""
    beats = ev["beats"]
    if not len(beats) or not len(ev["downbeats"]):
        return ev["downbeats"]
    db_idx = np.searchsorted(beats, ev["downbeats"])
    db_idx = np.clip(db_idx, 0, len(beats) - 1)
    gaps = np.diff(db_idx)
    period = int(np.round(np.median(gaps))) if len(gaps) else 4
    flipped = db_idx + period // 2
    return beats[flipped[flipped < len(beats)]]


def main():
    torch.manual_seed(0)
    mix = MixtureLambda(fps=FPS, device=DEVICE,
                        observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    probe = CRFLearnedFactors(fps=FPS, device=DEVICE,
                              observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    _, val, _ = load_songs(probe)
    model = final_eval.load_model(Path(__file__).resolve().parent / "vanilla_best_prelim.pt")
    val_acts = final_eval.activations_for(model, val)

    kernels = {lam: mix.log_mixture_kernel(1e-9, lam) for lam in LAMBDAS}
    meter_sets = {"both": (0, 1), "3": (0,), "4": (1,)}

    ship_b, ship_d, orac_b, orac_d = [], [], [], []
    for e in val:
        ref_b = mir_eval.beat.trim_beats(e["beat_times"])
        ref_d = mir_eval.beat.trim_beats(e["downbeat_times"])
        best_b, best_d, ship_beat, ship_db = 0.0, 0.0, None, None
        for lam in LAMBDAS:
            for mname, meters in meter_sets.items():
                ev = decode(mix, kernels[lam], val_acts[e["stem"]], meters)
                est_b = mir_eval.beat.trim_beats(ev["beats"])
                fb = mir_eval.beat.f_measure(ref_b, est_b) if len(est_b) else 0.0
                best_b = max(best_b, fb)
                for est_d_arr in (ev["downbeats"], half_bar_flip(ev)):
                    est_d = mir_eval.beat.trim_beats(est_d_arr)
                    fd = mir_eval.beat.f_measure(ref_d, est_d) if len(est_d) else 0.0
                    best_d = max(best_d, fd)
                if lam == 100.0 and mname == "both":
                    ship_beat, ship_db = fb, (mir_eval.beat.f_measure(
                        ref_d, mir_eval.beat.trim_beats(ev["downbeats"]))
                        if len(ev["downbeats"]) else 0.0)
        ship_b.append(ship_beat); ship_d.append(ship_db)
        orac_b.append(best_b); orac_d.append(best_d)

    print(f"shipped-equivalent (lam=100, both) : beatF {np.mean(ship_b):.4f}  "
          f"dbF {np.mean(ship_d):.4f}", flush=True)
    print(f"PER-SONG ORACLE (12 variants+flip) : beatF {np.mean(orac_b):.4f}  "
          f"dbF {np.mean(orac_d):.4f}", flush=True)
    print(f"decoder-side headroom              : beat +{np.mean(orac_b)-np.mean(ship_b):.4f}  "
          f"db +{np.mean(orac_d)-np.mean(ship_d):.4f}", flush=True)
    print(f"frontend-only territory            : beat {1-np.mean(orac_b):.4f}  "
          f"db {1-np.mean(orac_d):.4f}", flush=True)


if __name__ == "__main__":
    main()
