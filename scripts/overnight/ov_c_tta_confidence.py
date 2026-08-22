"""
EXPERIMENT C (overnight batch): likelihood-free per-track confidence from TTA instability,
used to ROUTE each track between peak-picking and the wide DBN.

Motivation: the DBN likelihood is known-corrupted as a confidence signal on SMC -- the
ML-selected transition_lambda ANTI-correlates with the oracle lambda (rho = -0.23). So we
need a confidence that never touches the likelihood. Time-stretch TTA gives one for free:
if a track's beat activation is stable under resampling, peak-picking is trustworthy; if
the per-rate activations disagree, the track needs the DBN's global tempo constraint.

Two instability definitions (both reported):
  S1 "var_at_peaks"  : variance across rates of the per-rate probability sampled at the
                       peak frames of the rate-averaged activation (mean over peaks).
  S2 "peakset_dist"  : 1 - mean pairwise F-measure between the per-rate peak-pick sets
                       (all rate pairs), i.e. mean pairwise disagreement of peak LOCATIONS.
Both computed from cache/ts_tta_cache/ un-warped per-rate logits (rates 0.85, 1.0, 1.2 for
the mild set; the full 7-rate set is also scored).

Routing: sort tracks by score, sweep a threshold, send the unstable side to the DBN and
the stable side to peak-picking (and the reverse direction, as a control).

Baselines: always-peak-pick (0.6270), always-DBN (0.5851), and the ORACLE per-track
best-of-two envelope (recomputed here; recorded as ~0.691 F elsewhere).

VALIDATION (aborts if off by > 0.002): peak-pick F ~ 0.6270 and wide-DBN F ~ 0.5851
recomputed from this script's own code.
"""
import sys, json, time, itertools
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from multiprocessing import Pool

ROOT = Path("/home/sogang/jaehoon/Analyze-SMC")
sys.path.insert(0, str(ROOT / "scripts"))
from _harness import (track_ids, load_gt, bthis_beat_prob, bthis_downbeat_prob,
                      peakpick, make_dbn, dbn_beats, ev, FPS)
from ov_stats import summarize, paired, METRICS
import mir_eval

OUTDIR = Path("/tmp/overnight"); OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "C_tta_confidence.json"
JSONL = OUTDIR / "C_tta_confidence.per_track.jsonl"
CACHE = ROOT / "data" / "ts_tta_cache"
RATES_ALL = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
RATES_MILD = [0.85, 1.0, 1.2]
TOL = 0.002
NPROC = 6


def _decode(a):
    """per-track: peak-pick score, DBN score, and both instability scores."""
    tid = a
    ref = load_gt(tid)
    bp = bthis_beat_prob(tid)
    pp_est = peakpick(bp, 0.5, FPS)
    dbn = make_dbn(30.0, 215.0, FPS, 100)
    dbn_est = dbn_beats(bp, bthis_downbeat_prob(tid), dbn)

    scores = {}
    for tag, rates in (("mild", RATES_MILD), ("all7", RATES_ALL)):
        L = min(len(np.load(CACHE / f"{tid}__r{r}.npz")["logits"]) for r in rates)
        P = np.stack([expit(np.load(CACHE / f"{tid}__r{r}.npz")["logits"][:L]) for r in rates])
        mean_p = P.mean(0)
        pk = peakpick(mean_p, 0.5, FPS)
        fr = np.round(pk * FPS).astype(int)
        fr = fr[(fr >= 0) & (fr < L)]
        if len(fr):
            # S1: dispersion across rates at the consensus peak frames
            s1 = float(np.mean(P[:, fr].std(axis=0)))
        else:
            s1 = 1.0
        # S2: mean pairwise disagreement of per-rate peak LOCATIONS
        sets = [peakpick(P[i], 0.5, FPS) for i in range(len(rates))]
        ds = []
        for i, j in itertools.combinations(range(len(rates)), 2):
            if len(sets[i]) and len(sets[j]):
                ds.append(1.0 - mir_eval.beat.f_measure(sets[i], sets[j]))
            else:
                ds.append(1.0)
        s2 = float(np.mean(ds)) if ds else 1.0
        scores[f"S1_var_at_peaks_{tag}"] = s1
        scores[f"S2_peakset_dist_{tag}"] = s2

    return tid, ev(ref, pp_est), ev(ref, dbn_est), scores


def main():
    t0 = time.time()
    ids = track_ids()
    assert len(ids) == 217, len(ids)
    fh = open(JSONL, "a")
    PP, DB, SC = {}, {}, {}
    with Pool(NPROC) as pool:
        for tid, pp, db, sc in pool.imap_unordered(_decode, ids, chunksize=2):
            PP[tid], DB[tid], SC[tid] = pp, db, sc
            fh.write(json.dumps({"track": tid, "peakpick": pp, "dbn": db, "scores": sc}) + "\n")
            fh.flush()
    fh.close()
    print(f"[C] decoded 217 tracks in {time.time()-t0:.0f}s", flush=True)

    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "n": 217}
    out["always_peakpick"] = summarize([PP[t] for t in ids])
    out["always_dbn"] = summarize([DB[t] for t in ids])

    bad = []
    if abs(out["always_peakpick"]["F"] - 0.6270) > TOL:
        bad.append(f"peakpick F={out['always_peakpick']['F']:.4f} != 0.6270")
    if abs(out["always_dbn"]["F"] - 0.5851) > TOL:
        bad.append(f"dbn F={out['always_dbn']['F']:.4f} != 0.5851")
    out["validation"] = {"ok": not bad, "problems": bad}
    OUT.write_text(json.dumps(out, indent=2))
    if bad:
        print("[C] *** VALIDATION FAILED ***\n  " + "\n  ".join(bad), flush=True)
        sys.exit(2)
    print("[C] validation OK", flush=True)

    # ---- oracle per-track best-of-two envelope (selected on F) ----
    orc = [PP[t] if PP[t]["F"] >= DB[t]["F"] else DB[t] for t in ids]
    out["oracle_best_of_two"] = summarize(orc)
    out["oracle_pick_dbn_count"] = int(sum(1 for t in ids if DB[t]["F"] > PP[t]["F"]))
    print(f"[C] oracle envelope F={out['oracle_best_of_two']['F']:.4f} "
          f"(DBN wins on {out['oracle_pick_dbn_count']}/217)", flush=True)
    OUT.write_text(json.dumps(out, indent=2))

    # ---- routing sweeps ----
    score_names = sorted(SC[ids[0]].keys())
    out["score_diagnostics"] = {}
    out["routing"] = {}
    gap = np.array([DB[t]["F"] - PP[t]["F"] for t in ids])   # >0 means DBN is the right call
    for sn in score_names:
        v = np.array([SC[t][sn] for t in ids])
        rho, p = spearmanr(v, gap)
        out["score_diagnostics"][sn] = {
            "spearman_rho_vs_dbn_minus_pp_F": float(rho), "p": float(p),
            "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}
        rows = []
        for direction in ("high_to_dbn", "low_to_dbn"):
            for q in np.round(np.arange(0.0, 1.001, 0.05), 3):
                thr = float(np.quantile(v, q))
                if direction == "high_to_dbn":
                    use_dbn = v > thr
                else:
                    use_dbn = v <= thr
                sel = [DB[t] if u else PP[t] for t, u in zip(ids, use_dbn)]
                m = summarize(sel)
                st = paired([s["F"] for s in sel], [PP[t]["F"] for t in ids])
                rows.append({"direction": direction, "quantile": float(q), "threshold": thr,
                             "n_routed_to_dbn": int(use_dbn.sum()), **m,
                             "vs_always_peakpick": st})
        out["routing"][sn] = rows
        best = max(rows, key=lambda r: r["F"])
        recovered = ((best["F"] - out["always_peakpick"]["F"]) /
                     max(out["oracle_best_of_two"]["F"] - out["always_peakpick"]["F"], 1e-9))
        out["routing"][sn + "__best"] = {**best, "fraction_of_oracle_gain_recovered": float(recovered)}
        print(f"[C] {sn}: best routed F={best['F']:.4f} at q={best['quantile']} "
              f"({best['direction']}), recovers {recovered*100:.1f}% of oracle gain "
              f"| rho vs (DBN-PP) = {rho:+.3f}", flush=True)
        OUT.write_text(json.dumps(out, indent=2))

    print(f"[C] DONE in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
