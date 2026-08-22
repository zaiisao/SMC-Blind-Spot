"""
EXPERIMENT A (overnight batch): time-stretch TTA activations decoded with the WIDE DBN.

The paper's deployable, ORACLE-FREE system: every stacked row on record so far uses
ground-truth tempo. This one uses none.

  TTA  = mean over rates {0.85, 1.0, 1.2} of the un-warped 50 fps LOGITS
         (beat and downbeat channels, from cache/tta_full/*.npz)
  decode = DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], 30-215 bpm, lambda=100)

Reports the full 5-metric row + paired Wilcoxon and 10k paired bootstrap CI against
  (1) raw peak-pick @0.5, (2) TTA peak-pick @0.5, (3) raw + wide DBN.
Also sweeps transition_lambda over {5, 20, 50, 100, 200} on the TTA activations.

VALIDATION (aborts if off by > 0.002):
  TTA peak-pick F  ~ 0.6450
  raw wide-DBN F   ~ 0.5851

Incremental: every per-track score is appended to the JSONL sidecar as it is produced.
"""
import sys, json, time, os
from pathlib import Path
import numpy as np
from scipy.special import expit
from multiprocessing import Pool

ROOT = Path("/home/sogang/jaehoon/Analyze-SMC")
sys.path.insert(0, str(ROOT / "scripts"))
from _harness import (track_ids, load_gt, bthis_beat_prob, bthis_downbeat_prob,
                      peakpick, make_dbn, dbn_beats, ev, FPS)
from ov_stats import summarize, compare_row, METRICS

OUTDIR = Path("/tmp/overnight")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "A_tta_dbn.json"
JSONL = OUTDIR / "A_tta_dbn.per_track.jsonl"
TTA_FULL = ROOT / "data" / "tta_full"
TOL = 0.002
NPROC = 6


def tta_probs(tid):
    z = np.load(TTA_FULL / f"{tid}.npz")
    return expit(z["ts_beat"].astype(np.float64)), expit(z["ts_db"].astype(np.float64))


def _job(args):
    """One (config, track) cell. Runs in a worker so madmom stays out of the parent."""
    kind, tid, lam = args
    ref = load_gt(tid)
    if kind == "tta_pp":
        b, _ = tta_probs(tid)
        est = peakpick(b, 0.5, FPS)
    elif kind == "raw_pp":
        est = peakpick(bthis_beat_prob(tid), 0.5, FPS)
    elif kind == "raw_dbn":
        dbn = make_dbn(30.0, 215.0, FPS, lam)
        est = dbn_beats(bthis_beat_prob(tid), bthis_downbeat_prob(tid), dbn)
    elif kind == "tta_dbn":
        b, d = tta_probs(tid)
        dbn = make_dbn(30.0, 215.0, FPS, lam)
        est = dbn_beats(b, d, dbn)
    else:
        raise ValueError(kind)
    return kind, lam, tid, ev(ref, est)


def run(pool, fh, kind, ids, lam=100):
    res = {}
    for k, l, tid, sc in pool.imap_unordered(_job, [(kind, t, lam) for t in ids], chunksize=4):
        res[tid] = sc
        fh.write(json.dumps({"config": kind, "lambda": lam, "track": tid, **sc}) + "\n")
        fh.flush()
    return [res[t] for t in ids]


def main():
    t0 = time.time()
    ids = track_ids()
    print(f"[A] {len(ids)} tracks", flush=True)
    assert len(ids) == 217, f"expected 217 tracks, got {len(ids)}"
    missing = [t for t in ids if not (TTA_FULL / f"{t}.npz").exists()]
    assert not missing, f"missing tta_full for {len(missing)} tracks"

    fh = open(JSONL, "a")
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "n": len(ids), "rows": {}, "tests": []}

    def dump():
        OUT.write_text(json.dumps(out, indent=2))

    with Pool(NPROC) as pool:
        # ---------------- validation ----------------
        print("[A] validation ...", flush=True)
        per = {}
        per["tta_pp"] = run(pool, fh, "tta_pp", ids)
        per["raw_pp"] = run(pool, fh, "raw_pp", ids)
        per["raw_dbn_l100"] = run(pool, fh, "raw_dbn", ids, 100)
        for k in per:
            out["rows"][k] = summarize(per[k])
        dump()
        v1 = out["rows"]["tta_pp"]["F"]
        v2 = out["rows"]["raw_dbn_l100"]["F"]
        v3 = out["rows"]["raw_pp"]["F"]
        print(f"[A] VALIDATION tta_pp F={v1:.4f} (want 0.6450) "
              f"raw_dbn F={v2:.4f} (want 0.5851) raw_pp F={v3:.4f} (want 0.6270)", flush=True)
        bad = []
        if abs(v1 - 0.6450) > TOL: bad.append(f"tta_pp F={v1:.4f} != 0.6450")
        if abs(v2 - 0.5851) > TOL: bad.append(f"raw_dbn F={v2:.4f} != 0.5851")
        if abs(v3 - 0.6270) > TOL: bad.append(f"raw_pp F={v3:.4f} != 0.6270")
        out["validation"] = {"ok": not bad, "problems": bad,
                             "tta_pp_F": v1, "raw_dbn_F": v2, "raw_pp_F": v3}
        dump()
        if bad:
            print("[A] *** VALIDATION FAILED ***\n  " + "\n  ".join(bad), flush=True)
            sys.exit(2)
        print("[A] validation OK", flush=True)

        # ---------------- headline + lambda sweep ----------------
        for lam in [100, 5, 20, 50, 200]:
            key = f"tta_dbn_l{lam}"
            per[key] = run(pool, fh, "tta_dbn", ids, lam)
            out["rows"][key] = summarize(per[key])
            print(f"[A] {key}: " + "  ".join(f"{m}={out['rows'][key][m]:.4f}" for m in METRICS)
                  + f"   [{time.time()-t0:.0f}s]", flush=True)
            dump()

        # ---------------- paired tests (headline = lambda 100) ----------------
        for base in ["raw_pp", "tta_pp", "raw_dbn_l100"]:
            out["tests"].append(compare_row("tta_dbn_l100", per["tta_dbn_l100"], base, per[base]))
        # and each sweep point vs TTA peak-pick
        for lam in [5, 20, 50, 200]:
            k = f"tta_dbn_l{lam}"
            out["tests"].append(compare_row(k, per[k], "tta_pp", per["tta_pp"]))
        dump()
    fh.close()
    print(f"[A] DONE in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
