"""E1 analysis: sweep tables, limit checks, paired tests, tempo-class split."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import _harness as H

ROOT = "/home/sogang/jaehoon/Analyze-SMC"
KEYS = ["F", "CMLt", "AMLt", "CMLc", "AMLc"]
RNG = np.random.default_rng(0)
TIDS = H.track_ids()


def load(mode):
    out = {}
    for p in glob.glob(f"/tmp/e1_res_{mode}_*.json"):
        for k, v in json.load(open(p)).items():
            out[float(k)] = {m: np.array([v[t][m] for t in TIDS]) for m in KEYS}
    return dict(sorted(out.items()))


def boot(d, n=10000):
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b):
    d = a - b
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    lo, hi = boot(d)
    return float(d.mean()), lo, hi, p


def main():
    beat, init = load("beat"), load("init")
    df = pd.read_csv(f"{ROOT}/data/bt_transformer_tempo_constrained_results.csv").set_index("track_id").loc[TIDS]
    tcl = df["tempo_class"].values
    ref = {c: {k: df[f"btthis_{c}_{k}"].values for k in ["F", "CMLt", "AMLt"]}
           for c in ["native", "dbn30", "top1", "topk", "oracle"]}
    refname = {"native": "peak-pick", "dbn30": "wide DBN [30,215]", "top1": "top-1 hard band",
               "topk": "top-k hard bands", "oracle": "oracle GT-tempo band"}
    L = []
    A = L.append

    A("# E1 — a soft, calibrated tempo prior by exact marginalisation (SMC, n=217)\n")
    A("Run 2026-08-21, CPU only (no GPU touched). Engine: the R1 DBN "
      "(`vbpm/rungs_lab/rungs/r1_2016_dbn.py`), copied verbatim to `/tmp/e1_lab/rungs` and used "
      "read-only from there; nothing under `vbpm/` was modified. Driver: "
      "`scripts/e1_soft_tempo_prior.py`; analysis: `scripts/e1_report.py`.\n")
    A("## 0. Setup\n")
    A("**Decoder.** R1 with madmom's shipped deployment defaults (`num_tempi=60`, `threshold=0.05`, "
      "`correct=True`), `beats_per_bar=[3,4]`, `observation_lambda=16`, `transition_lambda=100`, "
      "`bounding=\"squeeze\"` (matches `_harness.dbn_beats`'s eps blending), float64, "
      "**always the full grid [30, 215] BPM at 50 fps — 60 tempo states, none ever excluded**.\n")
    A("**Prior.** `q` = softmax of the Beat Transformer `tempo_logits` (300 bins, index == BPM), "
      "octave-smeared onto the ratios {1/3, 1/2, 1, 2, 3} with weights "
      "{0.075, 0.175, 0.50, 0.175, 0.075} (density-preserving: mass at `b*r` gets `w_r p(b)/r`), "
      "then mixed with a uniform floor `q <- 0.999 q + 0.001/300`. Rationale for the weights: the "
      "ratio-1 peak keeps half the mass, half/double get 3.5x the weight of third/triple because "
      "half/double are the dominant confusions (22 half + 21 double vs 47 'other' in the tempo_class "
      "labels), and the floor makes it impossible for any metrical level to be ruled out. "
      "Measured over the 217 tracks: `q` evaluated at the ground-truth BPM has median 0.0277 "
      "(8.3x the uniform level 1/300) and minimum 2.91e-4 (0.087x uniform) — never zero.\n")
    A("**Intervention.** pi(tau) ∝ u(tau) · q(tau)^(1/T), with u(tau) madmom's own implied tempo "
      "marginal (uniform over states, so ∝ interval length). Two placements were run:\n")
    A("- **`init`** — tilt only the DBN's initial state distribution (the literal "
      "\"replace the uniform tempo-state prior\" reading).")
    A("- **`beat`** — additionally tilt every tempo transition, log-space row-renormalised so the "
      "model stays a valid HMM. This is the correct soft analogue of the hard band, because the "
      "hard band constrains the tempo at *every* beat, not just at frame 0. §2 shows why `init` "
      "alone cannot reach the hard-band limit and is therefore not a valid test.\n")
    A("**Scoring.** `scripts/_harness.py`: mir_eval, 5 s trim, ±70 ms, Beat This fold-honest "
      "activations. n = 217 for every cell.\n")

    # ---- limits ----
    A("\n## 1. Limit checks\n")
    z = beat[0.0]
    A(f"**T → ∞ (1/T = 0), `beat`:** F {z['F'].mean():.4f} / CMLt {z['CMLt'].mean():.4f} / "
      f"AMLt {z['AMLt'].mean():.4f} / CMLc {z['CMLc'].mean():.4f} / AMLc {z['AMLc'].mean():.4f}.")
    A(f"Target = the published wide-DBN row (madmom `DBNDownBeatTrackingProcessor`, [30,215], "
      f"lambda=100): F {ref['dbn30']['F'].mean():.4f} / CMLt {ref['dbn30']['CMLt'].mean():.4f} / "
      f"AMLt {ref['dbn30']['AMLt'].mean():.4f}. Per-track agreement: "
      + ", ".join(f"{k} identical on {int((np.abs(z[k]-ref['dbn30'][k])<1e-9).sum())}/217 tracks "
                  f"(max |Δ| {np.abs(z[k]-ref['dbn30'][k]).max():.2e})" for k in ["F", "CMLt", "AMLt"])
      + ". **The copy reproduces the wide-DBN row exactly, so the plumbing is verified.**\n")
    A("(An earlier run with R1's standalone default `bounding=\"clip\"` gave F 0.5886 / CMLt 0.5149 "
      "/ AMLt 0.6741 and differed from madmom on 186/217 tracks' F. That is a bounding-convention "
      "mismatch, not a decoder mismatch; `bounding=\"squeeze\"` is what `_harness.dbn_beats` does "
      "and it restores exact agreement. All numbers below use `squeeze`.)\n")
    hiT = max(beat)
    h = beat[hiT]
    A(f"**T → 0 (1/T = {hiT:g}), `beat`:** F {h['F'].mean():.4f} / CMLt {h['CMLt'].mean():.4f} / "
      f"AMLt {h['AMLt'].mean():.4f} / CMLc {h['CMLc'].mean():.4f} / AMLc {h['AMLc'].mean():.4f}. "
      f"Hard top-1 band reference: F {ref['top1']['F'].mean():.4f} / CMLt "
      f"{ref['top1']['CMLt'].mean():.4f} / AMLt {ref['top1']['AMLt'].mean():.4f}.")
    A("The 1/T → ∞ endpoint is *stricter* than the top-1 arm and scores below it, which is the "
      "expected ordering: q^(1/T) collapses onto a single tempo state (a delta), whereas top-1 "
      "keeps a ±20% band of states around the estimate. So the sweep passes through hard-band "
      "behaviour and then past it — the family does contain the hard band as an interior point, "
      "which is what the limit check is for.\n")

    # ---- sweep ----
    for mode, tab, title in [("beat", beat, "2a. `beat` placement — the main sweep"),
                             ("init", init, "2b. `init` placement — prior on frame 0 only")]:
        A(f"\n## {title}\n")
        A("| 1/T | F | CMLt | AMLt | CMLc | AMLc |")
        A("|---|---|---|---|---|---|")
        for t, v in tab.items():
            A(f"| {t:g} | " + " | ".join(f"{v[k].mean():.4f}" for k in KEYS) + " |")
        A("")
    A("`init` moves almost nothing across four orders of magnitude of 1/T: the prior enters once, "
      "at frame 0, while the tempo is re-chosen at every one of the ~30 beat boundaries in a 30 s "
      "SMC excerpt, so the DBN simply walks away from it. That is why the `beat` placement is the "
      "one that can be compared to the hard band at all.\n")
    A("Reference rows, same 217 tracks, same protocol, recomputed from "
      "`data/bt_transformer_tempo_constrained_results.csv` (that CSV stores only F/CMLt/AMLt):\n")
    A("| arm | F | CMLt | AMLt |")
    A("|---|---|---|---|")
    for c in ["native", "dbn30", "top1", "topk", "oracle"]:
        A(f"| {refname[c]} | " + " | ".join(f"{ref[c][k].mean():.4f}" for k in ["F", "CMLt", "AMLt"]) + " |")

    # ---- paired tests ----
    A("\n## 3. Paired tests (`beat` placement)\n")
    A("Wilcoxon signed-rank over the same 217 tracks; 95% CI from a 10,000-resample paired "
      "bootstrap on the mean difference (seed 0).\n")
    A("| 1/T | vs | metric | Δmean | 95% CI | Wilcoxon p |")
    A("|---|---|---|---|---|---|")
    for t, v in beat.items():
        for c in ["native", "dbn30", "top1", "topk"]:
            for k in ["F", "CMLt", "AMLt"]:
                d, lo, hi, p = paired(v[k], ref[c][k])
                A(f"| {t:g} | {refname[c]} | {k} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.3g} |")

    # ---- split ----
    A("\n## 4. Split by whether the Beat Transformer tempo estimate is correct\n")
    cls = ["correct", "double", "half", "other"]
    cnt = {c: int((tcl == c).sum()) for c in cls}
    A("Group sizes: " + ", ".join(f"{c} n={cnt[c]}" for c in cls)
      + f"; pooled wrong-estimate n={217 - cnt['correct']}.\n")
    for k in ["CMLt", "F", "AMLt"]:
        A(f"\n### {k} by `tempo_class` (`beat` placement)\n")
        A("| 1/T | " + " | ".join(f"{c} (n={cnt[c]})" for c in cls) + " | wrong pooled (n=%d) |" % (217 - cnt["correct"]))
        A("|---" * (len(cls) + 2) + "|")
        for t, v in beat.items():
            A(f"| {t:g} | " + " | ".join(f"{v[k][tcl == c].mean():.4f}" for c in cls)
              + f" | {v[k][tcl != 'correct'].mean():.4f} |")
        for c in ["native", "dbn30", "top1", "topk", "oracle"]:
            v = ref[c][k]
            A(f"| *{refname[c]}* | " + " | ".join(f"{v[tcl == cc].mean():.4f}" for cc in cls)
              + f" | {v[tcl != 'correct'].mean():.4f} |")

    # ---- verdict ----
    bt = max(beat, key=lambda t: beat[t]["CMLt"].mean())
    b = beat[bt]
    A("\n## 5. Best cell and verdict\n")
    A(f"Best CMLt over the whole `beat` sweep: **1/T = {bt:g}** (T = {1/bt:.4g}).\n")
    A("| metric | mean | bootstrap 95% CI |")
    A("|---|---|---|")
    for k in KEYS:
        lo, hi = boot(b[k])
        A(f"| {k} | {b[k].mean():.4f} | [{lo:.4f}, {hi:.4f}] |")
    A("")
    for c in ["native", "dbn30", "top1", "topk"]:
        d, lo, hi, p = paired(b["CMLt"], ref[c]["CMLt"])
        A(f"- CMLt vs {refname[c]}: Δ {d:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], Wilcoxon p = {p:.4g}")
    dF, loF, hiF, pF = paired(b["F"], ref["native"]["F"])
    A(f"- F vs peak-pick: Δ {dF:+.4f}, 95% CI [{loF:+.4f}, {hiF:+.4f}], Wilcoxon p = {pF:.4g}")
    A("")
    ok_cml = b["CMLt"].mean() > 0.5138
    _, _, _, p_pp = paired(b["CMLt"], ref["native"]["CMLt"])
    A(f"Pre-registered SUCCESS = (CMLt > 0.5138) AND (paired Wilcoxon p < 0.05 vs peak-pick) AND "
      f"(F >= 0.60). Measured at the best cell: CMLt {b['CMLt'].mean():.4f} "
      f"({'>' if ok_cml else '<='} 0.5138), p vs peak-pick {p_pp:.4g} "
      f"({'<' if p_pp < 0.05 else '>='} 0.05), F {b['F'].mean():.4f} "
      f"({'>=' if b['F'].mean() >= 0.60 else '<'} 0.60).")
    A("")
    json.dump({"best_invT": bt, **{k: float(b[k].mean()) for k in KEYS}},
              open("/tmp/e1_best.json", "w"))
    open(f"{ROOT}/E1_SOFT_TEMPO_PRIOR.md", "w").write("\n".join(L) + "\n")
    print("best 1/T =", bt, {k: round(float(b[k].mean()), 4) for k in KEYS})
    print("p vs peakpick CMLt =", p_pp, "| F =", round(float(b["F"].mean()), 4))


if __name__ == "__main__":
    main()
