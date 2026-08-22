"""
EXPERIMENT E (overnight batch): score the VBPM tempo-prior sweep through BEAT metrics.

vbpm/docs/octave_error_audit.md reports a log-normal bar-period prior degrading
exact-period accuracy   81.37 -> 77.54 -> 73.01 -> 69.59 %   at weights 0 / 0.5 / 1.0 / 2.0
(the 81.37 anchor is the CORRECTED `acf-argmax` rule, NOT the published 79.3, which used
`acf-smallest`). The audit admits no F, CMLt or AMLt exists for any prior-weight row.

This script converts each weight's PERIOD PICKS into an actual beat sequence and scores
it through the harness protocol (mir_eval, 5 s trim, +/-70 ms).

  picks   : /home/sogang/jaehoon/VBPM/tempo_lab_{acf-argmax, acf+prior0.5/1.0/2.0}.npy
            (one BAR period in seconds per analysis window; 1232 windows)
  windows : /home/sogang/jaehoon/VBPM/tempo_acts.npz  (act[T,2] beat/downbeat logits,
            mask, truth bar-period, dataset, stem, split; fps = 50, T = 2250 -> 45 s)
  refs    : /disk1/jaehoon/dataset_store/beat_this_annotations/<dataset>/annotations/beats
            truncated to the analysis window [0, T/fps)

Because a pick is a BAR period, the beat grid is bar/divisor for divisor in {1,2,3,4}
(all four reported). PHASE IS NOT ORACLE: it is chosen to maximise the mean BEAT-channel
activation on the click grid, i.e. read off the same frontend the period came from.

Expected and worth confirming: the prior's harm shows in F and CMLt but NOT in AMLt,
because AMLt is octave-invariant by construction -- so the prior is buying nothing that
an octave-invariant metric can see, while paying in the metrics that matter.

VALIDATION: recompute exact-period accuracy (5 % tolerance) for the four weights and
require the four numbers to reproduce 81.37 / 77.54 / 73.01 / 69.59 within 0.5 pp on at
least one split ('val', 'gtzan' or pooled). Aborts otherwise.
"""
import sys, json, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool

ROOT = Path("/home/sogang/jaehoon/Analyze-SMC")
sys.path.insert(0, str(ROOT / "scripts"))
from _harness import ev
from ov_stats import summarize, paired, METRICS

VBPM = Path("/home/sogang/jaehoon/VBPM")
ANN = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
OUTDIR = Path("/tmp/overnight"); OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "E_prior_beat_metrics.json"
JSONL = OUTDIR / "E_prior_beat_metrics.per_window.jsonl"
NPROC = 6
TOLR = 0.05
WEIGHTS = [("w0.0", "tempo_lab_acf-argmax.npy", 81.37),
           ("w0.5", "tempo_lab_acf+prior0.5.npy", 77.54),
           ("w1.0", "tempo_lab_acf+prior1.0.npy", 73.01),
           ("w2.0", "tempo_lab_acf+prior2.0.npy", 69.59)]
DIVISORS = [1, 2, 3, 4]

_B = None


def _init():
    global _B
    _B = np.load(VBPM / "tempo_acts.npz", allow_pickle=True)


def load_ref(dataset, stem, tmax):
    p = ANN / dataset / "annotations" / "beats" / f"{stem}.beats"
    if not p.exists():
        return None
    d = np.loadtxt(str(p))
    if d.size == 0:
        return np.array([])
    t = d if d.ndim == 1 else d[:, 0]
    t = np.atleast_1d(t)
    return t[t < tmax]


def _job(args):
    i, picks = args           # picks: {wname: bar_period_seconds}
    b = _B
    fps = float(b["fps"])
    mask = b["mask"][i].astype(bool)
    T = int(mask.sum())
    tmax = T / fps
    ds, stem = str(b["dataset"][i]), str(b["stem"][i])
    ref = load_ref(ds, stem, tmax)
    if ref is None or len(ref) < 2:
        return i, ds, stem, None
    beat_act = 1.0 / (1.0 + np.exp(-b["act"][i, :T, 0].astype(np.float64)))

    res = {}
    for wname, bar in picks.items():
        for dv in DIVISORS:
            period = float(bar) / dv
            if period < 0.15:
                continue
            # phase from the frontend's beat channel, not from the annotations
            step = int(round(period * fps))
            if step < 2:
                continue
            best_ph, best_v = 0, -1.0
            for ph in range(step):
                idx = np.arange(ph, T, step)
                v = float(beat_act[idx].mean())
                if v > best_v:
                    best_v, best_ph = v, ph
            est = np.arange(best_ph, T, step) / fps
            res[f"{wname}_div{dv}"] = ev(ref, est)
    return i, ds, stem, res


def main():
    t0 = time.time()
    blob = np.load(VBPM / "tempo_acts.npz", allow_pickle=True)
    truth = blob["truth"].astype(np.float64)
    split = np.array([str(x) for x in blob["split"]])
    n = len(truth)
    print(f"[E] {n} windows, fps={float(blob['fps'])}", flush=True)

    P = {}
    for wname, fn, _ in WEIGHTS:
        a = np.load(VBPM / fn).astype(np.float64)
        assert len(a) == n, f"{fn}: {len(a)} != {n}"
        P[wname] = a

    # ---------------- validation: exact-period accuracy ----------------
    acc = {}
    for sname, keep in (("val", split == "val"), ("gtzan", split == "gtzan"),
                        ("pooled", np.ones(n, bool))):
        acc[sname] = {w: float(100.0 * np.mean(
            np.abs(P[w][keep] / truth[keep] - 1.0) <= TOLR)) for w, _, _ in WEIGHTS}
    want = [w[2] for w in WEIGHTS]
    names = [w[0] for w in WEIGHTS]
    ok_split = None
    for sname, d in acc.items():
        if all(abs(d[nm] - wv) <= 0.5 for nm, wv in zip(names, want)):
            ok_split = sname
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_windows": n,
           "exact_period_accuracy_pct": acc,
           "reference_accuracy_pct": dict(zip(names, want)),
           "validation": {"ok": ok_split is not None, "matching_split": ok_split,
                          "problems": [] if ok_split else
                          ["no split reproduces 81.37/77.54/73.01/69.59 within 0.5 pp"]}}
    OUT.write_text(json.dumps(out, indent=2))
    for sname, d in acc.items():
        print(f"[E] accuracy[{sname}]: " + "  ".join(f"{k}={v:.2f}%" for k, v in d.items()),
              flush=True)
    if ok_split is None:
        print("[E] *** VALIDATION FAILED *** (see accuracies above vs 81.37/77.54/73.01/69.59)",
              flush=True)
        sys.exit(2)
    print(f"[E] validation OK on split '{ok_split}'", flush=True)
    out["scored_on_split"] = ok_split
    keep_idx = [i for i in range(n) if (ok_split == "pooled" or split[i] == ok_split)]

    # ---------------- beat metrics ----------------
    fh = open(JSONL, "a")
    tasks = [(i, {w: float(P[w][i]) for w, _, _ in WEIGHTS}) for i in keep_idx]
    per = {}
    done = 0
    with Pool(NPROC, initializer=_init) as pool:
        for i, ds, stem, res in pool.imap_unordered(_job, tasks, chunksize=4):
            done += 1
            if res is None:
                continue
            per[i] = res
            fh.write(json.dumps({"idx": int(i), "dataset": ds, "stem": stem,
                                 "scores": res}) + "\n")
            fh.flush()
            if done % 100 == 0:
                print(f"[E]  {done}/{len(tasks)} windows [{time.time()-t0:.0f}s]", flush=True)
    fh.close()
    idxs = sorted(per)
    print(f"[E] scored {len(idxs)}/{len(tasks)} windows (rest had no usable annotation)",
          flush=True)
    out["n_scored"] = len(idxs)

    keys = sorted({k for i in idxs for k in per[i]})
    out["rows"] = {k: summarize([per[i][k] for i in idxs if k in per[i]]) for k in keys}
    for k in keys:
        r = out["rows"][k]
        print(f"[E] {k:12s} " + " ".join(f"{m}={r[m]:.4f}" for m in METRICS), flush=True)
    # paired tests: every weight vs weight 0, at each divisor
    out["tests"] = []
    for dv in DIVISORS:
        base = f"w0.0_div{dv}"
        if base not in keys:
            continue
        for wname, _, _ in WEIGHTS[1:]:
            k = f"{wname}_div{dv}"
            if k not in keys:
                continue
            ii = [i for i in idxs if k in per[i] and base in per[i]]
            row = {"a": k, "b": base, "n": len(ii)}
            for m in METRICS:
                row[m] = paired([per[i][k][m] for i in ii], [per[i][base][m] for i in ii])
            out["tests"].append(row)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[E] DONE in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
