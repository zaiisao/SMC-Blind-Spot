"""Two UNTRIED, DEPLOYABLE, NO-ORACLE decoder levers: the tempo grid density and the tempo floor.

WHY THESE TWO, GIVEN EVERYTHING ELSE IS CLOSED.
  * GRID (num_tempi): madmom ships a 60-point log-spaced tempo grid; the bare model supports the
    FULL integer grid (num_tempi=None). The shipped grid omits 27 of the 87 integer intervals in
    [14, 100] frames -- nearly all in the slow half (43, 48, 52, 55, 58, 60, 63, ...). A track whose
    true tempo sits between grid points forces the pointer to alternate between adjacent intervals,
    drifting each beat by up to half the gap (1-2 frames = 20-40 ms at slow tempi). That is pure
    LOCALIZATION error: invisible to CMLt, charged directly to F's +-70 ms window, and worst
    exactly on slow soft-onset material where the snap-to-activation read-out has no clear peak to
    rescue it. Nobody has tested this because everyone inherits madmom's 60 as a constant.
  * FLOOR (min_bpm): 30 of 217 SMC tracks dip below 30 BPM during ritardandi/pauses; the decoder
    cannot follow them down. min_bpm=20 makes those excursions representable. The known risk is the
    opposite error: a deeper floor invites half-tempo interpretations on normal tracks.

ARMS: {grid60, gridFULL} x {min30, min20}, beat-only decoder, raw Beat This activations, n=217 --
then the winning cell re-run as the FULL STACK row (rebalanced weights + TTA activations) so the
composed, deployable number is measured rather than summed. No oracle anywhere: every arm is one
global configuration applied to all tracks.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob
from exp_oracle_tempo_beatonly import emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events
from exp_best_stack import tta_prob

METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
W_REBAL = [1.6, 0.0, 0.0, 0.6]


def make_bo(min_bpm, num_tempi):
    """Beat-only DBN2016 with the class-assignment fix, selectable grid density."""
    from rungs.r1_2016_dbn import DBN2016 as _D
    mm = _D(fps=FPS, min_bpm=min_bpm, max_bpm=215.0, device=DEV, beats_per_bar=1,
            observation_lambda=16, transition_lambda=100, num_tempi=num_tempi)
    sp = mm.state_spaces[0]
    pc = sp.position_classes.copy()
    pc[pc == 2] = 1
    mm.state_to_classes = [torch.from_numpy(pc).to(DEV)]
    return mm


def decode(mm, prob, w=None):
    got = emissions_beatonly(mm, np.clip(prob, 1e-6, 1 - 1e-6))
    if got is None:
        return None
    lcd, dec, ff = got
    dp0, s2c = mm.dynamic_programs[0], mm.state_to_classes[0]
    w = w or [1.0, 0.0, 0.0, 1.0]
    em = w[0] * lcd[:, s2c]
    path, _ = dp0.viterbi(mm.log_initial_distributions[0], w[3] * mm.log_tempo_transition, em,
                          state_to_class=None, return_log_score=True)
    return state_path_to_events(path.cpu().numpy(), mm.state_spaces[0], FPS,
                                snap_to_activations=dec, first_frame=ff)["beats"]


def main():
    cfgs = {
        "grid60_min30": make_bo(30.0, 60),
        "gridFULL_min30": make_bo(30.0, None),
        "grid60_min20": make_bo(20.0, 60),
        "gridFULL_min20": make_bo(20.0, None),
    }
    for k, mm in cfgs.items():
        sp = mm.state_spaces[0]
        print(f"  {k:<16} tempi {len(sp.interval_frames):>3}  states {sp.num_states:>5}", flush=True)

    arms = {k: {} for k in cfgs}
    arms["stack_gridFULL_min30"] = {}
    arms["stack_grid60_min30"] = {}
    bpm = {}
    for n, tid in enumerate(track_ids()):
        raw = np.asarray(bthis_beat_prob(tid), np.float64)
        tta = tta_prob(tid)
        gt = np.asarray(load_gt(tid), float)
        if len(gt) < 4:
            continue
        bpm[tid] = 60.0 / float(np.median(np.diff(gt)))
        for k, mm in cfgs.items():
            e = decode(mm, raw)
            if e is not None:
                arms[k][tid] = ev(gt, e)
        if tta is not None:
            for k, mm in (("stack_grid60_min30", cfgs["grid60_min30"]),
                          ("stack_gridFULL_min30", cfgs["gridFULL_min30"])):
                e = decode(mm, tta, W_REBAL)
                if e is not None:
                    arms[k][tid] = ev(gt, e)
        if (n + 1) % 40 == 0:
            print(f"  {n+1}/217", flush=True)

    ids = sorted(set.intersection(*[set(v) for v in arms.values() if v]))
    A = lambda a, S=None: {k: float(np.mean([arms[a][t][k] for t in (S or ids)])) for k in METRICS}
    base = "grid60_min30"
    print(f"\nn={len(ids)}   PEAK-PICK reference F 0.6270   (all arms: one global config, no oracle)")
    print(f"{'arm':<22}" + "".join(f"{k:>9}" for k in METRICS) + f"{'dF':>10}{'dCMLt':>9}{'pF':>9}")
    for a in arms:
        if not arms[a]:
            continue
        dF = np.array([arms[a][t]["F"] - arms[base][t]["F"] for t in ids])
        dC = np.array([arms[a][t]["CMLt"] - arms[base][t]["CMLt"] for t in ids])
        p = wilcoxon(dF).pvalue if np.any(dF) else 1.0
        print(f"{a:<22}" + "".join(f"{A(a)[k]:>9.4f}" for k in METRICS)
              + f"{dF.mean():>+10.4f}{dC.mean():>+9.4f}{p:>9.2g}")
    slow = [t for t in ids if bpm[t] < 55]
    print(f"\nslow subset (<55 BPM, n={len(slow)}):")
    for a in ("grid60_min30", "gridFULL_min30", "grid60_min20", "gridFULL_min20"):
        print(f"  {a:<20} F {A(a, slow)['F']:.4f}  CMLt {A(a, slow)['CMLt']:.4f}")
    json.dump({"n": len(ids), "summary": {a: A(a) for a in arms if arms[a]},
               "per_track": arms}, open("/tmp/overnight/grid_floor.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
