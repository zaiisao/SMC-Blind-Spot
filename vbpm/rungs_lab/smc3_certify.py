"""Certification inside the user's Analyze-SMC protocol (repo READ-ONLY, imported):
reproduce raw peak-pick F=0.627 and wide-DBN [30,215] oracle-lambda F=0.642.
Also: our DBN2016 engine at the same wide config, fixed lambda=100 -- parity check vs
their make_dbn(30,215,50,100) row (engine drop-in gate for the ladder)."""
import sys
import numpy as np
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts")
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/vbpm/rungs_lab")
from _harness import (track_ids, load_gt, bthis_beat_prob, bthis_downbeat_prob,
                      peakpick, make_dbn, dbn_beats, FPS)
import mir_eval, torch
TRIM = mir_eval.beat.trim_beats
def fastF(ref, est):
    if len(est) == 0 or len(ref) == 0: return 0.0
    r, e = TRIM(ref), TRIM(est)
    if len(r) == 0 or len(e) == 0: return 0.0
    return mir_eval.beat.f_measure(r, e)

LAMBDAS = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
ids = track_ids(); refs = {t: load_gt(t) for t in ids}
dbns = {L: make_dbn(30, 215, FPS, L) for L in LAMBDAS}

from rungs.r1_2016_dbn import DBN2016
ours = DBN2016(fps=FPS, device="cuda", dtype=torch.float32, bounding="none",
               observation_lambda=16, transition_lambda=100.0, min_bpm=30.0, max_bpm=215.0,
               num_tempi=60, threshold=0.05, correct=True)

raw, lamO, fixed100, ours100, identical = [], [], [], [], 0
for t in ids:
    ref = refs[t]
    bp, dp = bthis_beat_prob(t), bthis_downbeat_prob(t)
    raw.append(fastF(ref, peakpick(bp, 0.5, FPS)))
    lamO.append(max(fastF(ref, dbn_beats(bp, dp, dbns[L])) for L in LAMBDAS))
    est_madmom = dbn_beats(bp, dp, dbns[100])
    fixed100.append(fastF(ref, est_madmom))
    # ours: same decorrelation happens inside Rung.predict; feed raw (bp,dp) prob pair
    eps = 1e-5
    acts = np.vstack((bp*(1-eps)+eps/2, dp*(1-eps)+eps/2)).T
    est_ours = ours.predict(acts)["beats"]
    ours100.append(fastF(ref, est_ours))
    if len(est_ours) == len(est_madmom) and np.allclose(est_ours, est_madmom, atol=1e-6):
        identical += 1

print(f"raw peak-pick @0.5        F = {np.mean(raw):.4f}   (target 0.627)")
print(f"wide-DBN oracle lambda    F = {np.mean(lamO):.4f}   (target 0.642)")
print(f"madmom wide fixed 100     F = {np.mean(fixed100):.4f}")
print(f"OUR engine wide fixed 100 F = {np.mean(ours100):.4f}  event-identical {identical}/{len(ids)}")
