"""
EXPERIMENT 2: realistic decoders on the fused activation.

Pipelines (all eval'd identically: mir_eval 5 s trim, full evaluate):
  1  Beat This  raw peak-pick               (paper baseline 0.627)
  2  Ensemble(mean) raw peak-pick @best-global
  3  Beat This  + wide DBN lambda=100
  4  Ensemble   + wide DBN lambda=100
  5  Ensemble   + wide DBN lambda=5  (global-best lambda)
  6  Beat This  + BT-predicted-tempo DBN (+/-20%)
  7  Ensemble   + BT-predicted-tempo DBN (+/-20%)
  8  Oracle model-selection upper bound (best of the two single models per track)

BT-predicted tempo is REALISTIC (Beat Transformer's own tempo head, ~59% acc).
"""
import sys, numpy as np, scipy.special
sys.path.insert(0, "scripts")
from _harness import (track_ids, load_gt, bthis_beat_prob, bthis_downbeat_prob,
                      btrans_beat_prob, btrans_downbeat_prob, btrans_tempo,
                      resample_to, peakpick, make_dbn, dbn_beats, ev, mean_scores,
                      FPS, BT_FPS)
import mir_eval
TRIM = mir_eval.beat.trim_beats

def fastF(ref, est):
    if len(est) == 0 or len(ref) == 0: return 0.0
    r = TRIM(ref); e = TRIM(est)
    if len(r) == 0 or len(e) == 0: return 0.0
    return mir_eval.beat.f_measure(r, e)

def main():
    ids = track_ids()
    refs = {t: load_gt(t) for t in ids}
    # aligned probs (50 fps grid)
    Ab, Ad, Bb, Bd, btempo = {}, {}, {}, {}, {}
    for t in ids:
        Ab[t] = bthis_beat_prob(t); Ad[t] = bthis_downbeat_prob(t)
        n = len(Ab[t])
        Bb[t] = resample_to(btrans_beat_prob(t), BT_FPS, FPS, n)
        Bd[t] = resample_to(btrans_downbeat_prob(t), BT_FPS, FPS, n)
        btempo[t] = btrans_tempo(t)[0]
    # ensemble = mean (beat & downbeat)
    Eb = {t: 0.5 * (Ab[t] + Bb[t]) for t in ids}
    Ed = {t: 0.5 * (Ad[t] + Bd[t]) for t in ids}

    fine = scipy.special.expit(np.arange(-4.0, 4.5, 0.25))

    rows = {}  # name -> list of ev dicts

    # 1 Beat This raw peak-pick @0.5
    rows["1 BeatThis peak@0.5"] = [ev(refs[t], peakpick(Ab[t], 0.5, FPS)) for t in ids]
    # 2 ensemble peak best-global (0.325 from prior run; reconfirm quickly)
    rows["2 Ens(mean) peak@0.325"] = [ev(refs[t], peakpick(Eb[t], 0.325, FPS)) for t in ids]

    # DBN pipelines
    dbn100 = make_dbn(30, 215, FPS, 100)
    dbn5 = make_dbn(30, 215, FPS, 5)
    rows["3 BeatThis+DBN l=100"] = [ev(refs[t], dbn_beats(Ab[t], Ad[t], dbn100)) for t in ids]
    dbn100b = make_dbn(30, 215, FPS, 100)
    rows["4 Ens+DBN l=100"] = [ev(refs[t], dbn_beats(Eb[t], Ed[t], dbn100b)) for t in ids]
    dbn5b = make_dbn(30, 215, FPS, 5)
    rows["5 Ens+DBN l=5"] = [ev(refs[t], dbn_beats(Eb[t], Ed[t], dbn5b)) for t in ids]

    # 6,7 BT-predicted-tempo-constrained DBN (+/-20%)
    def tempo_dbn(beat, down):
        out = []
        for t in ids:
            bpm = btempo[t]
            mn, mx = bpm * 0.8, bpm * 1.2
            d = make_dbn(mn, mx, FPS, 100)
            out.append(ev(refs[t], dbn_beats(beat[t], down[t], d)))
        return out
    rows["6 BeatThis+BTtempoDBN"] = tempo_dbn(Ab, Ad)
    rows["7 Ens+BTtempoDBN"] = tempo_dbn(Eb, Ed)

    # 8 oracle model-selection upper bound (per-track best of the two, oracle threshold each)
    sel = []
    for t in ids:
        fa = max(fastF(refs[t], peakpick(Ab[t], th, FPS)) for th in fine)
        fb = max(fastF(refs[t], peakpick(Bb[t], th, FPS)) for th in fine)
        sel.append({"F": max(fa, fb), "CMLt": 0, "AMLt": 0, "CMLc": 0, "AMLc": 0})
    rows["8 ORACLE model-select"] = sel

    print(f"{'pipeline':26s} {'F':>7s} {'CMLt':>7s} {'AMLt':>7s}")
    print("-" * 52)
    for name, lst in rows.items():
        m = mean_scores(lst)
        print(f"{name:26s} {m['F']:>7.4f} {m['CMLt']:>7.4f} {m['AMLt']:>7.4f}", flush=True)

if __name__ == "__main__":
    main()
