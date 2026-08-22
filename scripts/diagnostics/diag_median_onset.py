"""Is there beat evidence in a MEDIAN-aggregated onset function that Beat This discards?

MOTIVATION (McFee & Ellis, ICASSP 2014, "Better Beat Tracking Through Robust Onset Aggregation").
The classic onset strength function SUMS spectral flux across frequency:

    w_sum(t) = sum_f  max(0, log S_f,t - log S_f,t-1)

so one loud asynchronous source (their example: a vocalist entering late for dramatic effect)
dominates the envelope. Replacing the sum with a MEDIAN,

    w_med(t) = median_f max(0, log S_f,t - log S_f,t-1)

fires only when more than half the bands rise together. On SMC they measured AMLt 0.290 -> 0.340
and F 0.347 -> 0.375 from that single operator substitution.

WHY IT MIGHT STILL MATTER IN 2026. A median across frequency is a RANK statistic. Beat This's
front end is convolutions over a mel spectrogram -- linear maps plus pointwise nonlinearities --
which has SUM semantics and cannot express an order statistic no matter how it is trained. If
robustness to loud asynchronous events genuinely requires a rank operator, that capability is
architecturally absent, not merely undertrained. Our tag decomposition puts 59.3% of the F-deficit
on "expressive timing" and the highest per-track intensity on "beat phase ambiguity" (overrep 1.59),
which is exactly the failure mode McFee & Ellis describe.

WHAT THIS SCRIPT DOES *NOT* DO. It does not build a beat tracker. It asks the prior question:
among annotated beats that Beat This misses, does w_med land a peak there more often than chance?
If no, median aggregation is a 2014 fix for a 2014 front end and the route is closed cheaply.

PROTOCOL, and the three ways this measurement could lie to us:

(1) THRESHOLD CONFOUND. Detectors are not compared at their own operating points. Every detector
    (Beat This activation, w_sum, w_med) is reduced by the SAME operator: local maxima over a
    140 ms neighbourhood, keep the N tallest, where N = number of annotated beats. Equal budget,
    so coverage differences cannot come from one detector simply firing more often.
    Audited at hop=64: every detector has 227-241 candidates per track against N~85, so none is
    budget-starved (the sparse w_med least of all), and all peak heights are unique on 30/30
    tracks, so the index-order tie-break in top_peaks() never fires and cannot bias selection
    toward early peaks in the sparse median.

(2) CHANCE COVERAGE IS LARGE AND MUST BE SUBTRACTED. With ~80 peaks over a 40 s clip and a +-70 ms
    window, a peak set with NO phase information still covers ~1-exp(-80*0.14/40) = 24% of beats.
    Any raw "rescue rate" is therefore meaningless. The null is the SAME peak set circularly shifted
    by fixed offsets modulo track duration: this destroys beat phase while preserving peak count,
    peak heights and the entire inter-peak interval distribution. Reported numbers are
    rescue-minus-null, and w_med is tested against w_sum as well -- if the median matches the sum,
    the rank operator contributes nothing and the McFee mechanism does not transfer.

(3) FLAT-ZERO PLATEAUS. w_med is mostly exactly zero (that is the point -- it is sparse). Under
    max-pooling every frame of a zero plateau ties for the local max, which would manufacture
    thousands of spurious "peaks". Candidates are therefore required to be strictly positive.

(4) TIME RESOLUTION OF THE FLUX. McFee & Ellis difference frames HOP=64 SAMPLES (~2.9 ms) apart.
    A first version of this script used hop=441 (20 ms) to put the OSF on Beat This's 50 fps grid --
    which was never necessary, since peaks are emitted as continuous times. That choice is not
    neutral: the median fires only when more than half the bands rise SIMULTANEOUSLY, and
    differencing frames 20 ms apart smears the simultaneity it keys on, suppressing the median
    specifically while leaving the sum largely intact. That is a mechanism capable of manufacturing
    a null. --hop is therefore swept, and the peak-picking neighbourhood is pinned to a fixed
    duration (--peak-width-ms) rather than a fixed frame count, so the effective peak budget is
    comparable across resolutions.

Beat This stays on its own 50 fps grid (librosa's centred frames put frame t at t/50 s, matching the
activation exactly). Annotations before 5 s are dropped to match mir_eval's default trim, so this is
scored on the same beats the paper's tables are.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from _harness import FPS, track_ids, load_gt, bthis_beat_prob

SMC_AUD = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
TOL = 0.070          # mir_eval F-measure tolerance
TRIM = 5.0           # mir_eval default trim
# Deterministic null offsets (seconds). Chosen coprime-ish with plausible beat periods (0.28-2.0 s)
# so no offset accidentally re-aligns a peak train with the annotations.
NULL_OFFSETS = (3.1, 5.7, 8.3, 11.9, 14.3, 17.9, 21.7, 25.3)


def onset_strengths(y, sr, hop, power=1.0):
    """Return (w_sum, w_med).

    power=1.0 is McFee & Ellis's MAGNITUDE spectrogram. librosa's default is power=2.0, and the
    difference is NOT a harmless rescaling: `power` is applied to the linear spectrogram BEFORE the
    mel filterbank, and the filterbank sums over bins, so mel(|S|^2) != mel(|S|)^2 and the factor
    does not cancel out of the log-difference. Verified empirically: the two settings produce
    different peak sets on 30/30 (track, detector) pairs.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop,
                                       n_mels=128, fmax=8000, power=power)
    L = np.log(np.maximum(S, 1e-10))
    D = np.maximum(0.0, np.diff(L, axis=1))
    D = np.pad(D, ((0, 0), (1, 0)))                 # frame 0 has no predecessor
    return D.sum(0), np.median(D, axis=0)


def odd_width(fps, ms):
    """Peak-picking neighbourhood of `ms` milliseconds, in frames, forced odd and >= 3."""
    w = int(round(fps * ms / 1000.0))
    return max(3, w + 1 - w % 2)


def top_peaks(x, n, fps, width):
    """N tallest strictly-positive local maxima over `width` frames -> times (s), sorted."""
    x = np.asarray(x, np.float64)
    t = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0)
    ismax = (t == F.max_pool1d(t, width, 1, width // 2)).squeeze(0).numpy()
    idx = np.nonzero(ismax & (x > 0))[0]
    if len(idx) == 0:
        return np.zeros(0)
    return np.sort(idx[np.argsort(-x[idx])][:n]) / fps


def covered(refs, est, tol=TOL):
    """Boolean per reference beat: is there an estimate within tol?"""
    if len(est) == 0:
        return np.zeros(len(refs), bool)
    return np.array([np.min(np.abs(est - r)) <= tol for r in refs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hop", type=int, default=441,
                    help="OSF hop in samples at 22050 Hz. 441 = 20 ms (Beat This grid); "
                         "64 = 2.9 ms (McFee & Ellis 2014).")
    ap.add_argument("--peak-width-ms", type=float, default=140.0,
                    help="peak-picking neighbourhood in ms, held FIXED across hops so the "
                         "effective peak budget does not change with resolution")
    ap.add_argument("--power", type=float, default=1.0,
                    help="1.0 = magnitude spectrogram (McFee & Ellis); 2.0 = librosa default")
    ap.add_argument("--out", default="/tmp/overnight/median_onset.json")
    args = ap.parse_args()
    osf_fps = 22050.0 / args.hop
    w_osf = odd_width(osf_fps, args.peak_width_ms)
    w_act = odd_width(FPS, args.peak_width_ms)
    print(f"OSF hop {args.hop} samples ({1000*args.hop/22050:.2f} ms, {osf_fps:.1f} fps), "
          f"peak neighbourhood {args.peak_width_ms:g} ms = {w_osf} OSF frames / "
          f"{w_act} activation frames", flush=True)

    tags = json.load(open("/tmp/smc_tags.json"))
    DET = ("bthis", "sum", "med")
    rows = {}
    ids = track_ids()
    if args.limit:
        ids = ids[:args.limit]

    for n, tid in enumerate(ids):
        w = SMC_AUD / f"{tid.upper()}.wav"
        if not w.exists():
            continue
        refs = np.asarray(load_gt(tid), float)
        refs = refs[refs >= TRIM]
        if len(refs) < 8:
            continue
        y, sr = librosa.load(str(w), sr=22050, mono=True)
        dur = len(y) / sr
        ws, wm = onset_strengths(y, sr, args.hop, args.power)
        act = np.asarray(bthis_beat_prob(tid), np.float64)

        N = len(load_gt(tid))                             # equal peak budget for every detector
        est = {"bthis": top_peaks(act, N, FPS, w_act),
               "sum": top_peaks(ws, N, osf_fps, w_osf),
               "med": top_peaks(wm, N, osf_fps, w_osf)}
        cov = {d: covered(refs, est[d]) for d in DET}
        # circular-shift null: same peaks, phase destroyed, everything else preserved
        null = {d: np.mean([covered(refs, np.sort((est[d] + o) % dur)).astype(float)
                            for o in NULL_OFFSETS], axis=0) for d in DET}

        miss = ~cov["bthis"]
        rows[tid] = {
            "n_ref": int(len(refs)), "n_miss": int(miss.sum()),
            "recall": {d: float(cov[d].mean()) for d in DET},
            "recall_null": {d: float(null[d].mean()) for d in DET},
            # rescue = coverage restricted to the beats Beat This missed
            "rescue": {d: (float(cov[d][miss].mean()) if miss.any() else None) for d in DET},
            "rescue_null": {d: (float(null[d][miss].mean()) if miss.any() else None)
                            for d in DET},
            "cov": {d: cov[d].astype(int).tolist() for d in DET},
        }
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(ids)}", flush=True)

    ids = sorted(rows)
    print(f"\nn={len(ids)} tracks, {sum(rows[t]['n_ref'] for t in ids)} scored beats "
          f"(annotations >= {TRIM:g} s), tolerance +-{TOL*1000:g} ms, equal peak budget\n")

    # ---- 1. plain recall at equal budget, against its own phase-scrambled null ----
    print(f"{'detector':<10}{'recall':>9}{'null':>9}{'above-null':>12}{'p':>10}")
    for d in DET:
        r = np.array([rows[t]["recall"][d] for t in ids])
        z = np.array([rows[t]["recall_null"][d] for t in ids])
        p = wilcoxon(r - z).pvalue if np.any(r - z) else 1.0
        print(f"{d:<10}{r.mean():>9.3f}{z.mean():>9.3f}{(r-z).mean():>+12.3f}{p:>10.2g}")

    # ---- 2. THE QUESTION: among beats Beat This missed, who covers them above chance? ----
    have = [t for t in ids if rows[t]["n_miss"] > 0]
    tot_miss = sum(rows[t]["n_miss"] for t in have)
    print(f"\nrescue of the {tot_miss} beats Beat This missed  ({len(have)} tracks have >=1 miss)")
    print(f"{'detector':<10}{'rescue':>9}{'null':>9}{'above-null':>12}{'p':>10}")
    resc = {}
    for d in DET:
        r = np.array([rows[t]["rescue"][d] for t in have])
        z = np.array([rows[t]["rescue_null"][d] for t in have])
        resc[d] = r - z
        p = wilcoxon(r - z).pvalue if np.any(r - z) else 1.0
        print(f"{d:<10}{r.mean():>9.3f}{z.mean():>9.3f}{(r-z).mean():>+12.3f}{p:>10.2g}")
    d_ms = resc["med"] - resc["sum"]
    p_ms = wilcoxon(d_ms).pvalue if np.any(d_ms) else 1.0
    print(f"\nmedian minus sum (does the RANK operator add anything?): "
          f"{d_ms.mean():+.4f}  p={p_ms:.3g}")

    # ---- 3. per-tag, on the subsets the deficit actually lives in ----
    print(f"\nabove-null rescue by SMC tag:")
    print(f"{'tag':<32}{'n':>5}{'med':>9}{'sum':>9}{'med-sum':>10}")
    bytag = defaultdict(list)
    for t in have:
        for word in tags.get(t, {}).get("words", []):
            bytag[word].append(t)
        bytag["ALL"].append(t)
    for word, ts in sorted(bytag.items(), key=lambda kv: -len(kv[1])):
        if len(ts) < 10:
            continue
        m = np.array([rows[t]["rescue"]["med"] - rows[t]["rescue_null"]["med"] for t in ts])
        s = np.array([rows[t]["rescue"]["sum"] - rows[t]["rescue_null"]["sum"] for t in ts])
        print(f"{word:<32}{len(ts):>5}{m.mean():>+9.3f}{s.mean():>+9.3f}{(m-s).mean():>+10.3f}")

    json.dump({"n": len(ids), "tol": TOL, "trim": TRIM, "hop": args.hop,
               "peak_width_ms": args.peak_width_ms, "power": args.power, "rows": rows},
              open(args.out, "w"), indent=1)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
