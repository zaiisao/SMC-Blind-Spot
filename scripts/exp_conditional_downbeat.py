"""Recover the BAR-LEVEL state space by fixing how the downbeat channel enters the emission.

WHY. Validation established that the joint beat+downbeat DBN's loss on SMC is NOT caused by the
bar-level state space. Control: keep beats_per_bar=[3,4] but make the downbeat channel
uninformative (dp := bp/4) and it matches beat-only within noise (F +0.0006 p=0.59, CMLt +0.0006
p=0.60, n=217). 100% of the loss comes from injecting Beat This's REAL downbeat probabilities.

THE MECHANISM, measured: madmom's convention needs mutually exclusive classes, so `_harness.dbn_beats`
builds [max(bp - dp, eps), dp]. Beat This's two heads are INDEPENDENT sigmoids, not a partition, and
its beat channel already fires on downbeats. Result: dp > bp on 19.0% of all frames and **28.0% of
beat-peak frames (bp>0.5)** -- the beat likelihood is truncated to eps at many TRUE beats. Removing
the subtraction entirely is not an option either (F collapses to 0.062).

THE FIX NEVER TESTED: a CONDITIONAL parameterisation. Treat the heads as
    p(beat)                 = bp
    p(downbeat | beat)      = clip(dp / bp, 0, 1)
so the two observation classes become
    BEAT-not-downbeat  =  bp * (1 - dp/bp)  =  bp - dp     ... but floored MULTIPLICATIVELY
    DOWNBEAT           =  bp * (dp/bp)      =  dp
The difference from the shipped form is what happens when dp > bp: the additive version floors to
eps (destroying the beat evidence), while the conditional version clips the RATIO to 1, which keeps
the frame's total beat mass bp and merely assigns all of it to the downbeat class. Beat evidence is
preserved; only its split between the two classes changes.

WHY THIS MIGHT HELP THE OCTAVE PROBLEM, which is the real target. Metrical level is a HIERARCHY
question, and the bar-level states are the only part of the model that represents hierarchy. We
disabled them (§3/P5) because their input channel was miscalibrated. If the bar states can be used
without the truncation artefact, the decoder gains genuine evidence about the metrical level rather
than relying on the emission's weak ~93-nat phase preference. Break-even octave error is 24.4% and
the learned head sits at 24.9% (§4.9), so even a small improvement in the decoder's own level
discrimination changes what the head is worth.

ARMS (all n=217, DBN2016 min_bpm=30, obs_lambda=16, lambda=100, Beat This activations, cached):
    beatonly        beats_per_bar=1, class-corrected      -- the current best decoder (CMLt 0.5685)
    joint_additive  beats_per_bar=(3,4), max(bp-dp, eps)  -- the shipped form (CMLt ~0.5102)
    joint_cond      beats_per_bar=(3,4), conditional      -- THE TEST
    joint_uninf     beats_per_bar=(3,4), dp := bp/4       -- validation's control, sanity check
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob, bthis_downbeat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.r1_2016_dbn import DBN2016
from rungs.deployment import threshold_crop
from rungs.bar_pointer.readout import state_path_to_events

METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"


def joint_emissions(m, bp, dp, mode):
    """Class densities for the bar-level space under a given beat/downbeat parameterisation."""
    if mode == "uninf":
        dp = bp / 4.0
    elif mode == "cond":
        # p(downbeat|beat) clipped to [0,1]: preserves total beat mass bp when dp > bp,
        # where the additive form floors the beat class to eps instead.
        r = np.clip(np.divide(dp, np.maximum(bp, 1e-8)), 0.0, 1.0)
        dp = bp * r
    bb, dd, fl = m._bound(bp, dp)
    dec = m._decorrelate(bb, dd, fl)
    dec, ff = threshold_crop(dec, m.threshold)
    if not len(dec):
        return None
    return m._log_class_densities(bp[ff:ff + len(dec)], dp[ff:ff + len(dec)]), dec, ff


def main():
    beat_only = make_beatonly_dbn(30.0, 16, 100, DEV)
    joint = DBN2016(fps=FPS, min_bpm=30.0, max_bpm=215.0, device=DEV, beats_per_bar=(3, 4),
                    observation_lambda=16, transition_lambda=100)
    print(f"beat-only states {int(beat_only.state_spaces[0].interval_frames.sum()):,}; "
          f"joint spaces {len(joint.state_spaces)}", flush=True)

    def decode(m, lcd, dec, ff, gt):
        best = None
        for space, dp_, li, s2c in zip(m.state_spaces, m.dynamic_programs,
                                       m.log_initial_distributions, m.state_to_classes):
            path, sc = dp_.viterbi(li, m.log_tempo_transition, lcd,
                                   state_to_class=s2c, return_log_score=True)
            if best is None or sc > best[0]:
                best = (sc, space, path)
        _, space, path = best
        return ev(gt, state_path_to_events(path.cpu().numpy(), space, FPS,
                                           snap_to_activations=dec, first_frame=ff)["beats"])

    ARMS = ("beatonly", "joint_additive", "joint_cond", "joint_uninf")
    per = {a: {} for a in ARMS}
    trunc = []
    for t in track_ids():
        bp, dp = bthis_beat_prob(t), bthis_downbeat_prob(t)
        gt = load_gt(t)
        pk = bp > 0.5
        if pk.sum():
            trunc.append(float((dp[pk] > bp[pk]).mean()))
        got = emissions_beatonly(beat_only, bp)
        if got is None:
            continue
        per["beatonly"][t] = decode(beat_only, *got, gt)
        for arm, mode in (("joint_additive", "add"), ("joint_cond", "cond"), ("joint_uninf", "uninf")):
            g = joint_emissions(joint, bp, dp, mode)
            if g is None:
                continue
            per[arm][t] = decode(joint, *g, gt)

    ids = sorted(set.intersection(*[set(per[a]) for a in ARMS]))
    A = lambda r: {k: float(np.mean([r[t][k] for t in ids])) for k in METRICS}
    print(f"\nn={len(ids)}   dp>bp on {np.mean(trunc):.1%} of beat-peak frames (the artefact)")
    print(f"{'arm':<16}" + "".join(f"{k:>9}" for k in METRICS))
    for a in ARMS:
        print(f"{a:<16}" + "".join(f"{A(per[a])[k]:>9.4f}" for k in METRICS))
    print("\npaired vs beatonly (the current best decoder):")
    for a in ARMS[1:]:
        out = []
        for k in METRICS:
            d = np.array([per[a][t][k] - per["beatonly"][t][k] for t in ids])
            p = wilcoxon(d).pvalue if np.any(d) else 1.0
            out.append(f"{k} {d.mean():+.4f}(p={p:.2g})")
        print(f"  {a:<15}" + "  ".join(out))
    print("\npaired: joint_cond vs joint_additive (isolates the parameterisation):")
    for k in METRICS:
        d = np.array([per["joint_cond"][t][k] - per["joint_additive"][t][k] for t in ids])
        p = wilcoxon(d).pvalue if np.any(d) else 1.0
        print(f"  {k} {d.mean():+.4f} (p={p:.2g})")
    json.dump({"n": len(ids), "summary": {a: A(per[a]) for a in ARMS}, "per_track": per},
              open("/tmp/overnight/conditional_downbeat.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
