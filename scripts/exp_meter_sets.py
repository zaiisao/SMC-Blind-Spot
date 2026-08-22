"""Does the joint DBN help if it is allowed meters beyond madmom's default [3, 4]?

MOTIVATION. The joint decoder's deficit on SMC is caused by the DOWNBEAT OBSERVATIONS, not by the
bar-level state space: keeping beats_per_bar=[3,4] but making the downbeat channel uninformative
(dp := bp/4) matches beat-only within noise (F +0.0006 p=0.59, CMLt +0.0006 p=0.60, n=217), and
substituting Beat Transformer's downbeats is likewise neutral (CMLt +0.0025, p=0.64). So the state
space is free -- which means ENLARGING it is also free, and a meter set that actually matches the
music could constrain phase and tempo usefully.

SMC's own difficulty tags say the default is too narrow: "ternary meter" is tagged on 68/217 tracks,
and "changing time signature" is significantly enriched among our failures (OR 4.42, Fisher p=0.024).
madmom's [3, 4] cannot represent 5, 6, 7 or compound meters at all.

SMC is beat-only annotated, so downbeats are never scored -- a wider meter set can only help through
the constraint it places on the BEAT sequence, and can only hurt by spending likelihood on meter
hypotheses the music does not support.

ARMS: meter sets x downbeat source (Beat This, Beat Transformer, uninformative bp/4), all against
the beat-only decoder (beats_per_bar=1, class-corrected) which is our current best.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import (FPS, BT_FPS, track_ids, load_gt, ev, bthis_beat_prob,
                      bthis_downbeat_prob, btrans_downbeat_prob, resample_to)
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events

METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
METER_SETS = {
    "[3,4] default": [3, 4],
    "[2,3,4]": [2, 3, 4],
    "[2,3,4,5,6,7]": [2, 3, 4, 5, 6, 7],
    "[2,3,4,6,8,9,12]": [2, 3, 4, 6, 8, 9, 12],
}
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"


def main():
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    bo = make_beatonly_dbn(30.0, 16, 100, DEV)
    space = bo.state_spaces[0]

    procs = {k: DBNDownBeatTrackingProcessor(beats_per_bar=v, min_bpm=30.0, max_bpm=215.0,
                                             fps=FPS, transition_lambda=100)
             for k, v in METER_SETS.items()}
    print("meter sets: " + ", ".join(f"{k}" for k in METER_SETS), flush=True)

    SRC = ("bthis", "btrans", "uninf")
    arms = {"beat-only": {}}
    for k in METER_SETS:
        for s in SRC:
            arms[f"{k} / {s}"] = {}

    for t in track_ids():
        bp = np.asarray(bthis_beat_prob(t), np.float64)
        gt = load_gt(t)
        got = emissions_beatonly(bo, bp)
        if got is None:
            continue
        lcd, dec, ff = got
        path, _ = bo.dynamic_programs[0].viterbi(
            bo.log_initial_distributions[0], bo.log_tempo_transition, lcd,
            state_to_class=bo.state_to_classes[0], return_log_score=True)
        arms["beat-only"][t] = ev(gt, state_path_to_events(
            path.cpu().numpy(), space, FPS, snap_to_activations=dec, first_frame=ff)["beats"])

        dsrc = {}
        dsrc["bthis"] = np.asarray(bthis_downbeat_prob(t), np.float64)
        try:
            d2 = np.asarray(btrans_downbeat_prob(t), np.float64)
            dsrc["btrans"] = d2 if len(d2) == len(bp) else resample_to(d2, BT_FPS, FPS, len(bp))
        except Exception:
            pass
        dsrc["uninf"] = bp / 4.0

        eps = 1e-5
        for k, proc in procs.items():
            for s, dp in dsrc.items():
                b = bp * (1 - eps) + eps / 2
                d = dp * (1 - eps) + eps / 2
                comb = np.vstack((np.maximum(b - d, eps / 2), d)).T
                try:
                    est = proc(comb)[:, 0]
                except Exception:
                    continue
                arms[f"{k} / {s}"][t] = ev(gt, est)

    ids = sorted(set.intersection(*[set(v) for v in arms.values() if v]))
    A = lambda r: {k: float(np.mean([r[t][k] for t in ids])) for k in METRICS}
    base = arms["beat-only"]
    print(f"\nn={len(ids)}   baseline = beat-only decoder")
    print(f"{'arm':<26}" + "".join(f"{k:>9}" for k in METRICS) + f"{'dCMLt':>9}{'p':>10}")
    for k, v in arms.items():
        if not v:
            continue
        d = np.array([v[t]["CMLt"] - base[t]["CMLt"] for t in ids])
        p = wilcoxon(d).pvalue if np.any(d) else 1.0
        print(f"{k:<26}" + "".join(f"{A(v)[m]:>9.4f}" for m in METRICS)
              + f"{d.mean():>+9.4f}{p:>10.2g}")
    json.dump({"n": len(ids), "summary": {k: A(v) for k, v in arms.items() if v}},
              open("/tmp/overnight/meter_sets.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
