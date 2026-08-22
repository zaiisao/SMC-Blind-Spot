"""
EXPERIMENT 5: is F=0.673 really 'an upper bound on any decoder operating on
the real activations'?  Recompute every decoder with ONE consistent harness
(my _harness), on Beat This real activations, no GT side-info except oracle
hyperparameter selection (the same kind the paper's threshold sweep uses).

Decoders, each with per-track oracle hyperparameter:
  A  peak-pick, oracle threshold
  B  wide DBN [30,215], oracle transition_lambda in {1..500}
  C  per-track BEST of {A,B}   <- the real decoder envelope
"""
import sys, numpy as np
sys.path.insert(0, "scripts")
from _harness import (track_ids, load_gt, bthis_beat_prob, bthis_downbeat_prob,
                      peakpick, make_dbn, dbn_beats, FPS)
from scipy.special import expit
import mir_eval
TRIM = mir_eval.beat.trim_beats
def fastF(ref, est):
    if len(est) == 0 or len(ref) == 0: return 0.0
    r = TRIM(ref); e = TRIM(est)
    if len(r) == 0 or len(e) == 0: return 0.0
    return mir_eval.beat.f_measure(r, e)

LAMBDAS = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]

def main():
    ids = track_ids(); refs = {t: load_gt(t) for t in ids}
    fine = expit(np.arange(-4.0, 4.5, 0.25))
    dbns = {L: make_dbn(30, 215, FPS, L) for L in LAMBDAS}

    raw, thrO, lamO, both = [], [], [], []
    for t in ids:
        ref = refs[t]
        bp = bthis_beat_prob(t); dp = bthis_downbeat_prob(t)
        raw.append(fastF(ref, peakpick(bp, 0.5, FPS)))
        # A threshold oracle
        fa = max(fastF(ref, peakpick(bp, th, FPS)) for th in fine)
        thrO.append(fa)
        # B lambda oracle
        fb = max(fastF(ref, dbn_beats(bp, dp, dbns[L])) for L in LAMBDAS)
        lamO.append(fb)
        both.append(max(fa, fb))
    print("Beat This real activations, one consistent harness:")
    print(f"  raw peak-pick @0.5         F = {np.mean(raw):.4f}  (paper 0.627)")
    print(f"  A peak-pick oracle thresh  F = {np.mean(thrO):.4f}  (paper 0.673 'upper bound')")
    print(f"  B wide-DBN oracle lambda   F = {np.mean(lamO):.4f}  (paper 0.642)")
    print(f"  C per-track best of {{A,B}}  F = {np.mean(both):.4f}  <- TRUE decoder envelope")
    n_b_wins = sum(1 for a, b in zip(thrO, lamO) if b > a + 1e-9)
    print(f"  tracks where DBN-oracle beats threshold-oracle: {n_b_wins}/{len(ids)}")

if __name__ == "__main__":
    main()
