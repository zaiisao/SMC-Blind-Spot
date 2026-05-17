#!/usr/bin/env python3
"""
Combined GT-tempo constraint + per-track transition_lambda sweep.

For each track:
  - Compute GT tempo (60 / median IBI)
  - Constrain DBN to [gt_bpm*0.8, gt_bpm*1.2] (floor at 30)
  - Sweep transition_lambda across 13 values
  - Record optimal lambda and scores at optimal and default (λ=100)

Comparison: does GT-tempo + optimal λ compound beyond either lever alone?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit  # sigmoid

# ── Paths ──
ROOT = Path(__file__).parent.parent
BT_ACT_DIR = ROOT / "beat_this_activations_cache"
GT_DIR = ROOT / "beat_this_annotations" / "smc" / "annotations" / "beats"
EXISTING_SWEEP = ROOT / "transition_lambda_sweep_results.csv"

FPS = 50  # beat_this frame rate
LAMBDAS = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]


def load_gt(gt_path):
    """Load ground truth beat times and compute GT BPM."""
    gt_data = np.loadtxt(str(gt_path))
    if gt_data.ndim == 1:
        ref = gt_data
    else:
        ref = gt_data[:, 1]  # second column is beat time
    gt_bpm = 60.0 / np.median(np.diff(ref)) if len(ref) >= 2 else 0.0
    return ref, gt_bpm


def prepare_combined_act(beat_logits, downbeat_logits):
    """Convert beat_this logits to combined activation for DBNDownBeatTrackingProcessor."""
    beat_prob = expit(beat_logits.astype(np.float64))
    downbeat_prob = expit(downbeat_logits.astype(np.float64))
    epsilon = 1e-5
    beat_prob = beat_prob * (1 - epsilon) + epsilon / 2
    downbeat_prob = downbeat_prob * (1 - epsilon) + epsilon / 2
    combined = np.vstack((
        np.maximum(beat_prob - downbeat_prob, epsilon / 2),
        downbeat_prob,
    )).T
    return combined


def run_dbn_downbeat(combined_act, min_bpm, max_bpm, transition_lambda):
    """Run DBNDownBeatTrackingProcessor, return beat times."""
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    min_bpm = max(30.0, float(min_bpm))
    max_bpm = min(300.0, float(max_bpm))
    if max_bpm <= min_bpm + 5:
        max_bpm = min_bpm + 5
    dbn = DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4],
        min_bpm=min_bpm, max_bpm=max_bpm,
        fps=FPS,
        transition_lambda=transition_lambda,
    )
    out = dbn(combined_act)
    return out[:, 0]  # beat times


def eval_track(ref, est):
    import mir_eval
    if len(ref) == 0 or len(est) == 0:
        return {"F": 0.0, "CMLt": 0.0, "AMLt": 0.0}
    r = mir_eval.beat.evaluate(ref, est)
    return {
        "F": r["F-measure"],
        "CMLt": r["Correct Metric Level Total"],
        "AMLt": r["Any Metric Level Total"],
    }


def main():
    # Load error categories from existing sweep
    existing = pd.read_csv(EXISTING_SWEEP)
    cat_map = dict(zip(existing["track_id"], existing["error_category"]))

    # Collect track IDs
    act_files = sorted(BT_ACT_DIR.glob("*.npz"))
    print(f"Found {len(act_files)} activation files")

    rows = []
    for i, af in enumerate(act_files):
        track_id = af.stem

        # Load activations
        d = np.load(str(af))
        combined_act = prepare_combined_act(d["beat"], d["downbeat"])

        # Load GT
        gt_path = GT_DIR / f"{track_id}.beats"
        if not gt_path.exists():
            gt_path = GT_DIR / f"{track_id.upper()}.beats"
        if not gt_path.exists():
            print(f"  SKIP {track_id}: no GT")
            continue
        ref, gt_bpm = load_gt(gt_path)
        if gt_bpm <= 0:
            print(f"  SKIP {track_id}: no GT tempo")
            continue

        # GT-tempo constraint: ±20% envelope
        env_min = gt_bpm * 0.8
        env_max = gt_bpm * 1.2

        # Sweep lambdas
        best_f = -1
        best_lam = 100
        lam_scores = {}
        for lam in LAMBDAS:
            beats = run_dbn_downbeat(combined_act, env_min, env_max, lam)
            scores = eval_track(ref, beats)
            lam_scores[lam] = scores
            if scores["F"] > best_f:
                best_f = scores["F"]
                best_lam = lam

        opt = lam_scores[best_lam]
        default = lam_scores[100]

        row = {
            "track_id": track_id,
            "error_category": cat_map.get(track_id, "unknown"),
            "gt_bpm": round(gt_bpm, 2),
            "optimal_lambda": best_lam,
            "optimal_F": round(opt["F"], 4),
            "optimal_CMLt": round(opt["CMLt"], 4),
            "optimal_AMLt": round(opt["AMLt"], 4),
            "default_lambda_F": round(default["F"], 4),
            "default_lambda_CMLt": round(default["CMLt"], 4),
            "default_lambda_AMLt": round(default["AMLt"], 4),
        }
        # Also store per-lambda F for analysis
        for lam in LAMBDAS:
            row[f"F_lam{lam}"] = round(lam_scores[lam]["F"], 4)
        rows.append(row)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(act_files)}")

    df = pd.DataFrame(rows)
    n = len(df)
    print(f"\nProcessed {n} tracks")

    # ── Load existing wide-DBN sweep for comparison ──
    ex = pd.read_csv(EXISTING_SWEEP)

    # ── Beat_this raw (no DBN) scores from results.csv ──
    raw_csv = ROOT / "results.csv"
    raw_df = pd.read_csv(raw_csv)
    raw_f = raw_df["F-measure"].mean()
    raw_cmlt = raw_df["CMLt"].mean()
    raw_amlt = raw_df["AMLt"].mean()

    # ── Comparison table ──
    print(f"\n{'='*90}")
    print(f"COMPARISON TABLE — {n} tracks")
    print(f"{'='*90}")
    print(f"  {'System':<40} {'F':>8} {'CMLt':>8} {'AMLt':>8}")
    print("  " + "-" * 66)

    systems = [
        ("Beat This raw (no DBN)", raw_f, raw_cmlt, raw_amlt),
        ("Wide DBN, fixed λ=100", ex["default_F"].mean(), ex["default_CMLt"].mean(), ex["default_AMLt"].mean()),
        ("Wide DBN, per-track optimal λ", ex["optimal_F"].mean(), ex["optimal_CMLt"].mean(), ex["optimal_AMLt"].mean()),
        ("GT-tempo DBN, fixed λ=100", df["default_lambda_F"].mean(), df["default_lambda_CMLt"].mean(), df["default_lambda_AMLt"].mean()),
        ("GT-tempo DBN, per-track optimal λ", df["optimal_F"].mean(), df["optimal_CMLt"].mean(), df["optimal_AMLt"].mean()),
    ]
    for name, f, c, a in systems:
        print(f"  {name:<40} {f:>8.4f} {c:>8.4f} {a:>8.4f}")

    # ── Key comparisons ──
    gt_fix = df["default_lambda_F"].mean()
    gt_opt = df["optimal_F"].mean()
    wide_fix = ex["default_F"].mean()
    wide_opt = ex["optimal_F"].mean()

    print(f"\n{'='*90}")
    print("KEY COMPARISONS (mean F)")
    print(f"{'='*90}")
    print(f"  GT-tempo fixed λ=100 vs Wide fixed λ=100:    {gt_fix - wide_fix:+.4f}")
    print(f"  GT-tempo optimal λ vs Wide optimal λ:        {gt_opt - wide_opt:+.4f}")
    print(f"  GT-tempo optimal λ vs GT-tempo fixed λ=100:  {gt_opt - gt_fix:+.4f}")
    print(f"  GT-tempo optimal λ vs Wide fixed λ=100:      {gt_opt - wide_fix:+.4f}")
    print(f"  Compounding check:")
    gt_tempo_gain = gt_fix - wide_fix
    lambda_gain = wide_opt - wide_fix
    combined_gain = gt_opt - wide_fix
    expected_sum = gt_tempo_gain + lambda_gain
    print(f"    GT-tempo gain alone:       {gt_tempo_gain:+.4f}")
    print(f"    Optimal-λ gain alone:      {lambda_gain:+.4f}")
    print(f"    Sum (if independent):      {expected_sum:+.4f}")
    print(f"    Actual combined gain:      {combined_gain:+.4f}")
    print(f"    Interaction:               {combined_gain - expected_sum:+.4f}")

    # ── Optimal lambda distribution comparison ──
    print(f"\n{'='*90}")
    print("OPTIMAL LAMBDA DISTRIBUTION")
    print(f"{'='*90}")
    print(f"  {'Lambda':>8} {'Wide DBN':>10} {'GT-tempo':>10}")
    print("  " + "-" * 30)
    wide_lam_counts = ex["optimal_lambda"].value_counts()
    gt_lam_counts = df["optimal_lambda"].value_counts()
    for lam in LAMBDAS:
        wc = wide_lam_counts.get(lam, 0)
        gc = gt_lam_counts.get(lam, 0)
        print(f"  {lam:>8} {wc:>10} {gc:>10}")

    # ── Error category breakdown ──
    print(f"\n{'='*90}")
    print("BREAKDOWN BY ERROR CATEGORY")
    print(f"{'='*90}")
    print(f"  {'Category':<20} {'N':>4}  {'Wide fix':>9} {'Wide opt':>9} {'GT fix':>9} {'GT opt':>9} {'GT opt−Wide fix':>15}")
    print("  " + "-" * 80)
    for cat in ["good", "octave_error", "continuity_drift", "total_failure", "other"]:
        gt_sub = df[df["error_category"] == cat]
        ex_sub = ex[ex["error_category"] == cat]
        if len(gt_sub) == 0:
            continue
        wf = ex_sub["default_F"].mean() if len(ex_sub) > 0 else 0
        wo = ex_sub["optimal_F"].mean() if len(ex_sub) > 0 else 0
        gf = gt_sub["default_lambda_F"].mean()
        go = gt_sub["optimal_F"].mean()
        print(f"  {cat:<20} {len(gt_sub):>4}  {wf:>9.4f} {wo:>9.4f} {gf:>9.4f} {go:>9.4f} {go - wf:>+15.4f}")

    # ── Biggest gains from combination ──
    print(f"\n{'='*90}")
    print("TOP 15 TRACKS: LARGEST GAINS FROM GT-TEMPO + OPTIMAL λ vs WIDE DBN λ=100")
    print(f"{'='*90}")
    wide_subset = ex[["track_id", "default_F", "optimal_F"]].rename(
        columns={"default_F": "wide_fix_F", "optimal_F": "wide_opt_F"})
    merged = df.merge(wide_subset, on="track_id")
    merged["combo_gain"] = merged["optimal_F"] - merged["wide_fix_F"]
    merged["gt_only_gain"] = merged["default_lambda_F"] - merged["wide_fix_F"]
    merged["lam_only_gain"] = merged["wide_opt_F"] - merged["wide_fix_F"]
    top = merged.nlargest(15, "combo_gain")
    print(f"  {'Track':<10} {'Cat':<18} {'GT BPM':>7} {'λ*':>5} {'Wide fix':>9} {'GT fix':>9} {'GT opt':>9} {'Combo gain':>10}")
    print("  " + "-" * 80)
    for _, r in top.iterrows():
        print(f"  {r['track_id']:<10} {r['error_category']:<18} {r['gt_bpm']:>7.1f} {r['optimal_lambda']:>5} "
              f"{r['wide_fix_F']:>9.4f} {r['default_lambda_F']:>9.4f} {r['optimal_F']:>9.4f} {r['combo_gain']:>+10.4f}")

    # ── Save CSV ──
    csv_path = ROOT / "gt_tempo_lambda_sweep_results.csv"
    out_cols = ["track_id", "error_category", "gt_bpm", "optimal_lambda",
                "optimal_F", "optimal_CMLt", "optimal_AMLt",
                "default_lambda_F", "default_lambda_CMLt", "default_lambda_AMLt"]
    df[out_cols].to_csv(csv_path, index=False)
    print(f"\nPer-track results saved to {csv_path}")

    # ── Save report ──
    report_path = ROOT / "gt_tempo_lambda_sweep_report.md"
    with open(report_path, "w") as f:
        f.write("# GT-Tempo Constraint + Per-Track Lambda Sweep\n\n")
        f.write("## Setup\n\n")
        f.write("- Dataset: SMC (217 tracks)\n")
        f.write("- Activations: beat_this cached logits (50 fps)\n")
        f.write("- Decoder: `DBNDownBeatTrackingProcessor` (beats_per_bar=[3,4])\n")
        f.write("- GT tempo: 60 / median(IBI) from ground truth annotations\n")
        f.write("- BPM range: [GT×0.8, GT×1.2], floor at 30\n")
        f.write("- Swept transition_lambda: [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]\n")
        f.write("- Evaluation: mir_eval.beat.evaluate (5s trim, ±70ms tolerance)\n\n")

        f.write("## Comparison Table\n\n")
        f.write("| System | F | CMLt | AMLt |\n")
        f.write("|--------|---|------|------|\n")
        for name, fv, cv, av in systems:
            f.write(f"| {name} | {fv:.4f} | {cv:.4f} | {av:.4f} |\n")

        f.write("\n## Key Comparisons (mean F)\n\n")
        f.write(f"- GT-tempo fixed λ=100 vs Wide fixed λ=100: **{gt_fix - wide_fix:+.4f}**\n")
        f.write(f"- GT-tempo optimal λ vs Wide optimal λ: **{gt_opt - wide_opt:+.4f}**\n")
        f.write(f"- GT-tempo optimal λ vs GT-tempo fixed λ=100: **{gt_opt - gt_fix:+.4f}**\n")
        f.write(f"- GT-tempo optimal λ vs Wide fixed λ=100: **{gt_opt - wide_fix:+.4f}**\n\n")
        f.write("### Compounding Analysis\n\n")
        f.write(f"- GT-tempo gain alone (fixed λ=100): {gt_tempo_gain:+.4f}\n")
        f.write(f"- Optimal-λ gain alone (wide DBN): {lambda_gain:+.4f}\n")
        f.write(f"- Sum if independent: {expected_sum:+.4f}\n")
        f.write(f"- Actual combined gain: {combined_gain:+.4f}\n")
        f.write(f"- Interaction term: {combined_gain - expected_sum:+.4f}\n\n")

        f.write("## Optimal Lambda Distribution\n\n")
        f.write("| Lambda | Wide DBN | GT-tempo |\n")
        f.write("|--------|----------|----------|\n")
        for lam in LAMBDAS:
            wc = wide_lam_counts.get(lam, 0)
            gc = gt_lam_counts.get(lam, 0)
            f.write(f"| {lam} | {wc} | {gc} |\n")

        f.write("\n## Breakdown by Error Category\n\n")
        f.write("| Category | N | Wide fix | Wide opt | GT fix | GT opt | GT opt−Wide fix |\n")
        f.write("|----------|---|----------|----------|--------|--------|----------------|\n")
        for cat in ["good", "octave_error", "continuity_drift", "total_failure", "other"]:
            gt_sub = df[df["error_category"] == cat]
            ex_sub = ex[ex["error_category"] == cat]
            if len(gt_sub) == 0:
                continue
            wf = ex_sub["default_F"].mean() if len(ex_sub) > 0 else 0
            wo = ex_sub["optimal_F"].mean() if len(ex_sub) > 0 else 0
            gf = gt_sub["default_lambda_F"].mean()
            go = gt_sub["optimal_F"].mean()
            f.write(f"| {cat} | {len(gt_sub)} | {wf:.4f} | {wo:.4f} | {gf:.4f} | {go:.4f} | {go - wf:+.4f} |\n")

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
