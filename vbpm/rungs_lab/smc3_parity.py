"""Isolate the wide-regime R0/R1 divergence: parity matrix over configs, 60 songs."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts")
from _harness import track_ids, load_gt, bthis_beat_prob, bthis_downbeat_prob, make_dbn, dbn_beats, FPS
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from rungs.r1_2016_dbn import DBN2016
import mir_eval
TRIM = mir_eval.beat.trim_beats
def fastF(ref, est):
    if not len(est) or not len(ref): return 0.0
    r, e = TRIM(ref), TRIM(est)
    return mir_eval.beat.f_measure(r, e) if len(r) and len(e) else 0.0

ids = track_ids()[:60]
CONFIGS = {
 "min55_num60_shipped": dict(min_bpm=55, num_tempi=60, threshold=0.05, correct=True),
 "min30_num60_shipped": dict(min_bpm=30, num_tempi=60, threshold=0.05, correct=True),
 "min30_numNone_shipped": dict(min_bpm=30, num_tempi=None, threshold=0.05, correct=True),
 "min30_numNone_bare": dict(min_bpm=30, num_tempi=None, threshold=0.0, correct=False),
}
eps = 1e-5
for name, kw in CONFIGS.items():
    mad = DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], max_bpm=215.0, fps=FPS,
                                       transition_lambda=100, **{k:v for k,v in kw.items()})
    ours = DBN2016(fps=FPS, device="cuda", dtype=torch.float32, bounding="squeeze",
                   observation_lambda=16, transition_lambda=100.0, max_bpm=215.0, **kw)
    same = 0; f_m = []; f_o = []
    for t in ids:
        ref = load_gt(t)
        bp, dp = bthis_beat_prob(t), bthis_downbeat_prob(t)
        combined = np.vstack((np.maximum(bp*(1-eps)+eps/2 - (dp*(1-eps)+eps/2), eps/2),
                              dp*(1-eps)+eps/2)).T
        em = mad(combined)[:,0]
        eo = ours.predict(np.vstack((bp, dp)).T)["beats"]
        f_m.append(fastF(ref, em)); f_o.append(fastF(ref, eo))
        if len(em)==len(eo) and np.allclose(em, eo, atol=1e-6): same += 1
    print(f"{name:24s} identical {same}/60  madmomF {np.mean(f_m):.4f} oursF {np.mean(f_o):.4f}")
