"""
EXPERIMENT B (overnight batch): min_bpm=30 vs min_bpm=55, split at gt_bpm = 55.

Claim to reproduce and extend: pooled over all 217 SMC tracks the off-the-shelf madmom
DBN (min_bpm=55) looks like a CONTINUITY LOSS versus peak-picking, but on the 172 tracks
whose correct metrical level is REPRESENTABLE at min_bpm=55 (gt_bpm >= 55) the DBN WINS
both continuity metrics. The pooled figure is Simpson's paradox manufactured by the 45
tracks whose true level lies below the search floor.

Every DBN configuration is run at min_bpm=30 AND min_bpm=55 on all 217, then reported
  - pooled (n=217)
  - representable subset  gt_bpm >= 55  (expected n=172)
  - unrepresentable subset gt_bpm <  55 (expected n=45)
gt_bpm is read from data/bt_transformer_tempo_constrained_results.csv.

Full 5-metric rows everywhere, plus paired Wilcoxon + 10k paired bootstrap 95% CI
within each stratum.

VALIDATION (aborts if off by > 0.002):
  dbn min55 lambda=100 : F 0.5764 / CMLt 0.4738 / AMLt 0.6556
  dbn min30 lambda=100 : F 0.5851 / CMLt 0.5102 / AMLt 0.6735
  peak-pick @0.5       : F 0.6270 / CMLt 0.5138 / AMLt 0.6100
"""
import sys, json, time, csv
from pathlib import Path
import numpy as np
from multiprocessing import Pool

ROOT = Path("/home/sogang/jaehoon/Analyze-SMC")
sys.path.insert(0, str(ROOT / "scripts"))
from _harness import (track_ids, load_gt, bthis_beat_prob, bthis_downbeat_prob,
                      peakpick, make_dbn, dbn_beats, ev, FPS)
from ov_stats import summarize, paired, METRICS

OUTDIR = Path("/tmp/overnight"); OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "B_minbpm_split.json"
JSONL = OUTDIR / "B_minbpm_split.per_track.jsonl"
CSVP = ROOT / "data" / "bt_transformer_tempo_constrained_results.csv"
TOL = 0.002
NPROC = 6

LAMBDAS = [10, 20, 50, 100, 200, 500]
MINBPMS = [30, 55]
MAXBPM = 215.0


def _job(a):
    tid, minb, lam = a
    ref = load_gt(tid)
    if minb == 0:                       # peak-pick sentinel
        est = peakpick(bthis_beat_prob(tid), 0.5, FPS)
    else:
        dbn = make_dbn(float(minb), MAXBPM, FPS, lam)
        est = dbn_beats(bthis_beat_prob(tid), bthis_downbeat_prob(tid), dbn)
    return tid, minb, lam, ev(ref, est)


def main():
    t0 = time.time()
    ids = track_ids()
    assert len(ids) == 217, len(ids)

    gtbpm = {}
    with open(CSVP) as f:
        for r in csv.DictReader(f):
            gtbpm[r["track_id"]] = float(r["gt_bpm"])
    miss = [t for t in ids if t not in gtbpm]
    assert not miss, f"no gt_bpm for {miss[:5]}"
    hi = [t for t in ids if gtbpm[t] >= 55.0]
    lo = [t for t in ids if gtbpm[t] < 55.0]
    print(f"[B] n=217  representable(gt_bpm>=55)={len(hi)}  unrepresentable={len(lo)}", flush=True)

    # reference configs first so validation fires early
    configs = [("peakpick", 0, 0), ("dbn30_l100", 30, 100), ("dbn55_l100", 55, 100)]
    configs += [(f"dbn{m}_l{l}", m, l) for m in MINBPMS for l in LAMBDAS if l != 100]

    fh = open(JSONL, "a")
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n_all": len(ids), "n_representable": len(hi), "n_unrepresentable": len(lo),
           "strata": {"all": ids, "representable_ge55": hi, "unrepresentable_lt55": lo},
           "rows": {}, "tests": [], "validation": None}

    def dump():
        OUT.write_text(json.dumps(out, indent=2))

    per = {}
    with Pool(NPROC) as pool:
        for name, minb, lam in configs:
            res = {}
            for tid, mb, l, sc in pool.imap_unordered(
                    _job, [(t, minb, lam) for t in ids], chunksize=4):
                res[tid] = sc
                fh.write(json.dumps({"config": name, "track": tid, **sc}) + "\n"); fh.flush()
            per[name] = res
            out["rows"][name] = {
                "all": summarize([res[t] for t in ids]),
                "representable_ge55": summarize([res[t] for t in hi]),
                "unrepresentable_lt55": summarize([res[t] for t in lo]),
            }
            r = out["rows"][name]["all"]
            print(f"[B] {name:14s} pooled " + " ".join(f"{m}={r[m]:.4f}" for m in METRICS)
                  + f"  [{time.time()-t0:.0f}s]", flush=True)
            dump()

            # validation as soon as the three reference configs land
            if name in ("peakpick", "dbn30_l100", "dbn55_l100") and out["validation"] is None:
                want = {"peakpick": (0.6270, 0.5138, 0.6100),
                        "dbn30_l100": (0.5851, 0.5102, 0.6735),
                        "dbn55_l100": (0.5764, 0.4738, 0.6556)}
                have = {k: out["rows"][k] for k in want if k in out["rows"]}
                if len(have) == 3:
                    bad = []
                    for k, (f_, c_, a_) in want.items():
                        g = out["rows"][k]["all"]
                        for mname, gotv, wantv in (("F", g["F"], f_), ("CMLt", g["CMLt"], c_),
                                                   ("AMLt", g["AMLt"], a_)):
                            if abs(gotv - wantv) > TOL:
                                bad.append(f"{k}.{mname}={gotv:.4f} != {wantv:.4f}")
                    out["validation"] = {"ok": not bad, "problems": bad}
                    dump()
                    if bad:
                        print("[B] *** VALIDATION FAILED ***\n  " + "\n  ".join(bad), flush=True)
                        sys.exit(2)
                    print("[B] validation OK", flush=True)

    fh.close()

    # ---- paired tests, within each stratum ----
    strata = {"all": ids, "representable_ge55": hi, "unrepresentable_lt55": lo}
    pairs = [(f"dbn55_l{l}", "peakpick") for l in LAMBDAS] \
        + [(f"dbn30_l{l}", "peakpick") for l in LAMBDAS] \
        + [(f"dbn30_l{l}", f"dbn55_l{l}") for l in LAMBDAS]
    for a, b in pairs:
        for sname, sids in strata.items():
            row = {"a": a, "b": b, "stratum": sname, "n": len(sids)}
            for m in METRICS:
                row[m] = paired([per[a][t][m] for t in sids], [per[b][t][m] for t in sids])
            out["tests"].append(row)
    dump()
    print(f"[B] DONE in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
