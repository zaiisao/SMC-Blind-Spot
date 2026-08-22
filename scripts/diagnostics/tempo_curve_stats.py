"""What does SMC's expressive timing actually look like, as a distribution we can SAMPLE FROM?

WHY THIS COMES FIRST. The plan is to time-warp well-annotated corpus audio along SMC-like tempo
curves so the labels transform exactly. That requires a generative description of SMC's rubato --
not "it varies a lot", but numbers we can draw curves from. Guessing the curve shape would make the
whole augmentation an untested assumption.

WHY WARPING IS THE RIGHT INSTRUMENT AT ALL. The deficit decomposition puts 59.3% of SMC's F-deficit
on 'expressive timing' and 35.8% on 'slow tempo', and the fold checkpoints score ~0.71 F on SMC
tracks they TRAINED on versus 0.63 held out -- so the architecture can represent this material, it
just has almost no labelled examples of it. Time-warping is the one augmentation whose labels are
exact by construction (beat times map through the warp), which is what generation cannot promise.

WHAT IS MEASURED, per track, from beat annotations alone:
  * local tempo   T_i = 60 / (t_{i+1} - t_i)
  * log-tempo increments  d_i = log2(T_{i+1}/T_i)      -- scale-free, so fast and slow tracks pool
  * sigma(d)      volatility: how much tempo moves beat-to-beat
  * AR(1) rho of d: is rubato SMOOTH (rho>0, sustained accel/decel as in ritardando) or JITTERY
                    (rho<0, alternating)? This single number decides the shape of curve to sample:
                    smooth curves need a correlated process, jitter needs white noise.
  * drift range   max/min of the smoothed tempo curve over a track -- how far it wanders overall
Compared against the CORPUS (ballroom/beatles/hainsworth/hjdb/rwc/asap), because the augmentation
only needs to supply what the corpus LACKS. Whatever axis SMC and the corpus already agree on is not
worth synthesising.

FOLD DISCIPLINE. Statistics are reported per SMC fold as well as pooled, so a downstream user can
draw curves from folds != test fold and stay honest. Nothing here touches audio.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from _harness import track_ids, load_gt

ANN = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
SPLIT = ROOT / "third-party" / "beat_this_annotations" / "smc" / "8-folds.split"
CORPORA = ("ballroom", "beatles", "hainsworth", "hjdb", "rwc", "asap")


def curve_stats(beats):
    """Per-track rubato descriptors from a beat sequence."""
    b = np.asarray(beats, float)
    b = b[np.isfinite(b)]
    if len(b) < 12:
        return None
    ibi = np.diff(b)
    ibi = ibi[(ibi > 0.05) & (ibi < 4.0)]
    if len(ibi) < 10:
        return None
    T = 60.0 / ibi
    d = np.diff(np.log2(T))
    if len(d) < 8:
        return None
    # AR(1) on the increments: >0 means sustained accel/decel (ritardando-like), <0 means alternation
    rho = float(np.corrcoef(d[:-1], d[1:])[0, 1]) if len(d) > 3 else np.nan
    # smoothed curve to measure overall wander, robust to single-beat jitter
    k = min(9, (len(T) // 2) * 2 + 1)
    sm = np.convolve(T, np.ones(k) / k, mode="valid") if k >= 3 else T
    return {
        "median_bpm": float(np.median(T)),
        "sigma_d": float(np.std(d)),
        "mad_d": float(np.median(np.abs(d - np.median(d)))),
        "rho1": rho,
        "drift_octaves": float(np.log2(sm.max() / max(sm.min(), 1e-6))) if len(sm) else np.nan,
        "n_beats": int(len(b)),
    }


def load_corpus():
    out = {}
    for ds in CORPORA:
        d = ANN / ds / "annotations" / "beats"
        if not d.exists():
            continue
        for p in sorted(d.glob("*.beats")):
            try:
                a = np.loadtxt(str(p))
            except Exception:
                continue
            t = a if a.ndim == 1 else a[:, 0]
            s = curve_stats(t)
            if s:
                out[f"{ds}/{p.stem}"] = s
    return out


def summarise(name, rows):
    if not rows:
        print(f"{name:<14} (none)")
        return {}
    g = lambda k: np.array([r[k] for r in rows.values() if np.isfinite(r.get(k, np.nan))])
    s = {k: (float(np.median(g(k))), float(np.percentile(g(k), 10)), float(np.percentile(g(k), 90)))
         for k in ("median_bpm", "sigma_d", "rho1", "drift_octaves")}
    print(f"{name:<14}{len(rows):>5}"
          f"{s['median_bpm'][0]:>10.1f}{s['sigma_d'][0]:>10.4f}{s['rho1'][0]:>9.3f}"
          f"{s['drift_octaves'][0]:>10.3f}"
          f"   [sigma_d p10-p90 {s['sigma_d'][1]:.4f}-{s['sigma_d'][2]:.4f}]")
    return s


def main():
    smc = {}
    for t in track_ids():
        s = curve_stats(load_gt(t))
        if s:
            smc[t] = s
    cor = load_corpus()
    print(f"SMC {len(smc)} tracks, corpus {len(cor)} tracks\n")
    print(f"{'set':<14}{'n':>5}{'med BPM':>10}{'sigma_d':>10}{'AR1 rho':>9}{'drift oct':>10}")
    out = {"smc": summarise("SMC", smc)}
    by_ds = {}
    for k, v in cor.items():
        by_ds.setdefault(k.split("/")[0], {})[k] = v
    for ds, rows in sorted(by_ds.items()):
        out[ds] = summarise(ds, rows)
    out["corpus_all"] = summarise("CORPUS all", cor)

    sd_s = np.array([r["sigma_d"] for r in smc.values()])
    sd_c = np.array([r["sigma_d"] for r in cor.values()])
    bp_s = np.array([r["median_bpm"] for r in smc.values()])
    bp_c = np.array([r["median_bpm"] for r in cor.values()])
    print(f"\nWHAT THE CORPUS LACKS (the augmentation's job):")
    print(f"  rubato volatility sigma_d : SMC median {np.median(sd_s):.4f} vs corpus "
          f"{np.median(sd_c):.4f}  ({np.median(sd_s)/max(np.median(sd_c),1e-9):.1f}x)")
    print(f"  corpus tracks as volatile as the median SMC track: "
          f"{100*np.mean(sd_c >= np.median(sd_s)):.1f}%")
    print(f"  slow tempo <55 BPM        : SMC {100*np.mean(bp_s < 55):.1f}% vs corpus "
          f"{100*np.mean(bp_c < 55):.1f}%")
    print(f"  BOTH slow and volatile    : SMC {100*np.mean((bp_s < 55) & (sd_s >= np.median(sd_s))):.1f}%"
          f" vs corpus {100*np.mean((bp_c < 55) & (sd_c >= np.median(sd_s))):.1f}%")
    rho_s = np.array([r["rho1"] for r in smc.values() if np.isfinite(r["rho1"])])
    print(f"\n  SMC AR(1) rho median {np.median(rho_s):+.3f} -> rubato is "
          f"{'SMOOTH (sustained accel/decel): sample CORRELATED curves' if np.median(rho_s) > 0.05 else 'JITTERY/alternating: sample near-WHITE increments'}")

    splits = {}
    for line in open(SPLIT):
        p = line.split()
        if len(p) >= 2:
            splits[p[0]] = int(p[-1])
    print(f"\nper-SMC-fold sigma_d median (for fold-honest curve sampling):")
    for f in sorted(set(splits.values())):
        v = [smc[t]["sigma_d"] for t in smc if splits.get(t) == f]
        print(f"  fold {f}: n={len(v):>3}  sigma_d {np.median(v):.4f}")
    json.dump({"smc_per_track": smc, "summary": out,
               "smc_folds": {t: splits.get(t) for t in smc}},
              open("/tmp/overnight/tempo_curve_stats.json", "w"), indent=1)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
