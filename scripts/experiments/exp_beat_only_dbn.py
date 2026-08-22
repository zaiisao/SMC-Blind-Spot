"""Joint beat+downbeat DBN vs BEAT-ONLY DBN on SMC.

MOTIVATED BY PRIOR ART, not by a hunch. Boeck & Davies, "Deconstruct, Analyse, Reconstruct"
(ISMIR 2020), Sec 3.3, report that joint modelling of beats and downbeats in the DBN

    "can be beneficial for music with constant meter and steady tempo (e.g. Ballroom), whereas it
     negatively impacts performance for expressive music as contained in Hainsworth and SMC."

Every DBN row in this project uses madmom's DBNDownBeatTrackingProcessor with beats_per_bar=[3,4]
-- i.e. exactly the joint decoding they report as harmful on SMC. SMC is beat-only annotated, so
the bar-level state space is spending states on a quantity we never evaluate and that the music
often does not support. This script tests whether dropping it recovers the F that structured
decoding currently costs (fold-0 example: peak-pick 0.6510 vs joint DBN 0.5677).

ARMS, all on the same cached fold-honest Beat This activations, all 217 tracks:
    peak        peak-picking at 0.5, no structured decoding (decoder-free reference)
    joint       DBNDownBeatTrackingProcessor, beats_per_bar=[3,4]   <- what the project uses now
    beatonly    DBNBeatTrackingProcessor                            <- the Boeck/Davies suggestion

min_bpm is swept because the two processors do not necessarily share an optimum, and because 21% of
SMC lies below 55 BPM: a min_bpm that excludes those tracks collapses CMLt on them (measured
0.0528), which is the single largest decoder effect found in this project.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from _harness import (FPS, track_ids, load_gt, ev, peakpick, make_dbn, dbn_beats,
                      bthis_beat_prob, bthis_downbeat_prob)

METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
MIN_BPMS = [30.0, 40.0, 55.0]


def beat_only_dbn(min_bpm, max_bpm=215.0, transition_lambda=100):
    from madmom.features.beats import DBNBeatTrackingProcessor
    return DBNBeatTrackingProcessor(min_bpm=min_bpm, max_bpm=max_bpm, fps=FPS,
                                    transition_lambda=transition_lambda)


def main():
    ids = track_ids()
    gt = {t: load_gt(t) for t in ids}
    bp = {t: bthis_beat_prob(t) for t in ids}
    dp = {t: bthis_downbeat_prob(t) for t in ids}
    print(f"n = {len(ids)} SMC tracks, cached Beat This activations", flush=True)

    res = {}
    res["peak"] = {t: ev(gt[t], peakpick(bp[t], 0.5, FPS)) for t in ids}
    for mb in MIN_BPMS:
        dj = make_dbn(min_bpm=mb, transition_lambda=100)
        res[f"joint_min{mb:g}"] = {t: ev(gt[t], dbn_beats(bp[t], dp[t], dj)) for t in ids}
        db = beat_only_dbn(min_bpm=mb)
        res[f"beatonly_min{mb:g}"] = {t: ev(gt[t], db(bp[t])) for t in ids}
        print(f"  min_bpm {mb:g} done", flush=True)

    A = lambda r: {m: float(np.mean([r[t][m] for t in ids])) for m in METRICS}
    print(f"\n{'arm':<18}" + "".join(f"{m:>9}" for m in METRICS))
    for nm in res:
        a = A(res[nm])
        print(f"{nm:<18}" + "".join(f"{a[m]:>9.4f}" for m in METRICS))

    print("\nbeat-only minus joint, paired, at each min_bpm:")
    for mb in MIN_BPMS:
        j, b = res[f"joint_min{mb:g}"], res[f"beatonly_min{mb:g}"]
        out = []
        for m in METRICS:
            d = np.array([b[t][m] - j[t][m] for t in ids])
            p = stats.wilcoxon(d).pvalue if np.any(d) else 1.0
            out.append(f"{m} {d.mean():+.4f} (p={p:.2g})")
        print(f"  min_bpm {mb:g}: " + "  ".join(out))

    print("\nbest arm minus peak-picking, paired:")
    best = max((nm for nm in res if nm != "peak"), key=lambda n: A(res[n])["F"])
    for m in METRICS:
        d = np.array([res[best][t][m] - res["peak"][t][m] for t in ids])
        p = stats.wilcoxon(d).pvalue if np.any(d) else 1.0
        print(f"  {best} vs peak  {m} {d.mean():+.4f} (p={p:.2g})")

    json.dump({"n": len(ids), "summary": {nm: A(res[nm]) for nm in res},
               "per_track": res}, open("/tmp/overnight/beat_only_dbn.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
