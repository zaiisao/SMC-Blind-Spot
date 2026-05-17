"""
Build a CSV combining:
- beat_this predictions vs ground truth (mir_eval metrics) — both no-DBN and DBN
- SMC difficulty tags
- annotator confidence scores
"""
import csv
import re
import numpy as np
import mir_eval
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR = ROOT / "beat_this_annotations" / "smc" / "annotations" / "beats"
PRED_DIR = ROOT / "beat_this_output"
PRED_DBN_DIR = ROOT / "beat_this_output_dbn"
TAG_DIR = ROOT / "SMC_MIREX" / "SMC_MIREX_Tags"
FOLD_FILE = ROOT / "beat_this_annotations" / "smc" / "8-folds.split"
CONFIDENCE_CSV = ROOT / "dbn_confidence.csv"
OUT_CSV = DATA_DIR / "results.csv"


def load_gt(path):
    times = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                times.append(float(line))
    return np.array(times)


def load_pred(path):
    times = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                times.append(float(line.split("\t")[0]))
    return np.array(times)


def parse_tag_file(path):
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return [], None, None

    annotator = None
    confidence = None
    for line in reversed(lines):
        m = re.fullmatch(r"([a-z])(\d+)", line)
        if m:
            annotator = m.group(1)
            confidence = int(m.group(2))
            break

    descriptors = []
    for line in lines:
        if not re.fullmatch(r"[a-z]\d+", line):
            cleaned = line.strip("()")
            if cleaned:
                descriptors.append(cleaned)

    return descriptors, annotator, confidence


def evaluate_pred(gt_path, pred_path):
    """Evaluate a single prediction against ground truth. Returns dict or None."""
    if not pred_path.exists():
        return None
    ref_beats = load_gt(gt_path)
    est_beats = load_pred(pred_path)
    if len(ref_beats) == 0 or len(est_beats) == 0:
        return None
    return mir_eval.beat.evaluate(ref_beats, est_beats)


METRIC_KEYS = [
    "F-measure", "Cemgil", "Cemgil Best Metric Level",
    "Goto", "P-score", "Information gain",
    "Correct Metric Level Continuous", "Correct Metric Level Total",
    "Any Metric Level Continuous", "Any Metric Level Total",
]
METRIC_SHORT = [
    "F-measure", "Cemgil", "Cemgil_BML",
    "Goto", "P-score", "Info_gain",
    "CMLc", "CMLt", "AMLc", "AMLt",
]


def load_confidence_data():
    """Load the DBN confidence CSV into a dict keyed by track_id."""
    conf = {}
    with open(CONFIDENCE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            conf[row["track_id"]] = {
                "n_frames": int(row["n_frames"]),
                "n_beats_dbn": int(row["n_beats"]),
                "best_bar": int(row["best_bar"]),
                "dbn_log_prob": float(row["log_prob"]),
                "dbn_norm_log_prob": float(row["norm_log_prob"]),
                "dbn_log_prob_3_4": float(row["log_prob_3_4"]),
                "dbn_log_prob_4_4": float(row["log_prob_4_4"]),
            }
    return conf


def main():
    # Load fold assignments
    fold_map = {}
    with open(FOLD_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            track_id, fold_num = line.split("\t")
            fold_map[track_id] = int(fold_num)

    # Load DBN confidence data
    confidence_data = load_confidence_data()

    # Collect all unique descriptors
    all_descriptors = set()
    tag_data = {}
    for tag_path in sorted(TAG_DIR.glob("SMC_*.tag")):
        num = tag_path.stem.split("_")[1]
        track_id = f"smc_{num}"
        descriptors, annotator, confidence = parse_tag_file(tag_path)
        tag_data[track_id] = (descriptors, annotator, confidence)
        all_descriptors.update(descriptors)
    all_descriptors = sorted(all_descriptors)

    # Build rows
    rows = []
    gt_files = sorted(GT_DIR.glob("*.beats"))

    for gt_path in gt_files:
        track_id = gt_path.stem
        num = track_id.split("_")[1]
        pred_name = track_id.upper() + ".beats"

        # Evaluate no-DBN
        scores = evaluate_pred(gt_path, PRED_DIR / pred_name)
        # Evaluate DBN
        scores_dbn = evaluate_pred(gt_path, PRED_DBN_DIR / pred_name)

        if scores is None and scores_dbn is None:
            continue

        # Tag info
        descriptors, annotator, confidence = tag_data.get(track_id, ([], None, None))
        is_easy = int(num) >= 271

        row = {
            "track_id": track_id,
            "fold": fold_map.get(track_id, ""),
            "annotator": annotator or "",
            "confidence": confidence if confidence is not None else "",
            "is_easy": is_easy,
            "n_descriptors": len(descriptors),
            "descriptors": "; ".join(descriptors),
        }

        # No-DBN metrics
        if scores:
            for key, short in zip(METRIC_KEYS, METRIC_SHORT):
                row[short] = round(scores[key], 4)
        else:
            for short in METRIC_SHORT:
                row[short] = ""

        # DBN metrics
        if scores_dbn:
            for key, short in zip(METRIC_KEYS, METRIC_SHORT):
                row[f"dbn_{short}"] = round(scores_dbn[key], 4)
        else:
            for short in METRIC_SHORT:
                row[f"dbn_{short}"] = ""

        # DBN confidence data
        cdata = confidence_data.get(track_id, {})
        for k in ["n_frames", "n_beats_dbn", "best_bar", "dbn_log_prob",
                   "dbn_norm_log_prob", "dbn_log_prob_3_4", "dbn_log_prob_4_4"]:
            row[k] = cdata.get(k, "")

        # One-hot tags
        for d in all_descriptors:
            row[f"tag:{d}"] = 1 if d in descriptors else 0

        rows.append(row)

    # Write CSV
    confidence_fields = ["n_frames", "n_beats_dbn", "best_bar", "dbn_log_prob",
                         "dbn_norm_log_prob", "dbn_log_prob_3_4", "dbn_log_prob_4_4"]
    fieldnames = (
        ["track_id", "fold", "annotator", "confidence", "is_easy",
         "n_descriptors", "descriptors"]
        + METRIC_SHORT
        + [f"dbn_{s}" for s in METRIC_SHORT]
        + confidence_fields
        + [f"tag:{d}" for d in all_descriptors]
    )

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    print(f"Columns: {len(fieldnames)}")

    # ── Comparison summary ──
    both_rows = [r for r in rows if r["F-measure"] != "" and r["dbn_F-measure"] != ""]

    print(f"\n{'='*80}")
    print("OVERALL: NO-DBN vs DBN")
    print(f"{'='*80}")
    print(f"{'Metric':<15} {'No-DBN':>10} {'DBN':>10} {'Diff':>10}")
    print("-" * 50)
    for short in METRIC_SHORT:
        no_dbn = np.mean([r[short] for r in both_rows])
        dbn = np.mean([r[f"dbn_{short}"] for r in both_rows])
        diff = dbn - no_dbn
        sign = "+" if diff >= 0 else ""
        print(f"{short:<15} {no_dbn:>10.4f} {dbn:>10.4f} {sign}{diff:>9.4f}")

    # ── Easy vs Hard comparison ──
    print(f"\n{'='*80}")
    print("EASY vs HARD: NO-DBN vs DBN (F-measure / CMLc / AMLt)")
    print(f"{'='*80}")
    for label, filt in [("Hard", lambda r: not r["is_easy"]),
                         ("Easy", lambda r: r["is_easy"])]:
        subset = [r for r in both_rows if filt(r)]
        if not subset:
            continue
        f_no = np.mean([r["F-measure"] for r in subset])
        f_dbn = np.mean([r["dbn_F-measure"] for r in subset])
        c_no = np.mean([r["CMLc"] for r in subset])
        c_dbn = np.mean([r["dbn_CMLc"] for r in subset])
        a_no = np.mean([r["AMLt"] for r in subset])
        a_dbn = np.mean([r["dbn_AMLt"] for r in subset])
        print(f"  {label:<6} (n={len(subset):>3}):")
        print(f"    F-measure: {f_no:.3f} -> {f_dbn:.3f} ({f_dbn-f_no:+.3f})")
        print(f"    CMLc:      {c_no:.3f} -> {c_dbn:.3f} ({c_dbn-c_no:+.3f})")
        print(f"    AMLt:      {a_no:.3f} -> {a_dbn:.3f} ({a_dbn-a_no:+.3f})")

    # ── By confidence ──
    print(f"\n{'='*80}")
    print("BY CONFIDENCE: NO-DBN vs DBN (F-measure)")
    print(f"{'='*80}")
    conf_groups = defaultdict(list)
    for row in both_rows:
        if row["confidence"] != "":
            conf_groups[row["confidence"]].append(row)
    for conf in sorted(conf_groups.keys()):
        subset = conf_groups[conf]
        f_no = np.mean([r["F-measure"] for r in subset])
        f_dbn = np.mean([r["dbn_F-measure"] for r in subset])
        print(f"  Confidence {conf} (n={len(subset):>3}):  {f_no:.3f} -> {f_dbn:.3f} ({f_dbn-f_no:+.3f})")

    # ── By descriptor ──
    print(f"\n{'='*80}")
    print("BY DESCRIPTOR: NO-DBN vs DBN F-MEASURE (sorted by diff)")
    print(f"{'='*80}")
    print(f"{'Descriptor':<45} {'Count':>5} {'No-DBN':>8} {'DBN':>8} {'Diff':>8}")
    print("-" * 80)

    desc_scores = defaultdict(list)
    for row in both_rows:
        for d in all_descriptors:
            if row[f"tag:{d}"] == 1:
                desc_scores[d].append(row)

    desc_summary = []
    for d in all_descriptors:
        if desc_scores[d]:
            f_no = np.mean([r["F-measure"] for r in desc_scores[d]])
            f_dbn = np.mean([r["dbn_F-measure"] for r in desc_scores[d]])
            desc_summary.append((d, len(desc_scores[d]), f_no, f_dbn, f_dbn - f_no))

    desc_summary.sort(key=lambda x: x[4])
    for d, count, f_no, f_dbn, diff in desc_summary:
        sign = "+" if diff >= 0 else ""
        print(f"{d:<45} {count:>5} {f_no:>8.3f} {f_dbn:>8.3f} {sign}{diff:>7.3f}")

    # ── DBN Confidence Analysis ──
    from scipy.stats import pearsonr, spearmanr

    conf_rows = [r for r in rows if r.get("dbn_norm_log_prob", "") != ""
                 and r.get("dbn_F-measure", "") != ""]

    if conf_rows:
        norm_lp = np.array([r["dbn_norm_log_prob"] for r in conf_rows])
        dbn_f = np.array([r["dbn_F-measure"] for r in conf_rows])
        no_dbn_f = np.array([r["F-measure"] for r in conf_rows])
        dbn_cmlc = np.array([r["dbn_CMLc"] for r in conf_rows])
        dbn_amlt = np.array([r["dbn_AMLt"] for r in conf_rows])

        print(f"\n{'='*80}")
        print("DBN CONFIDENCE (norm_log_prob) CORRELATION WITH METRICS")
        print(f"{'='*80}")
        for name, vals in [("dbn_F-measure", dbn_f),
                           ("no-dbn_F-measure", no_dbn_f),
                           ("dbn_CMLc", dbn_cmlc),
                           ("dbn_AMLt", dbn_amlt)]:
            pr, pp = pearsonr(norm_lp, vals)
            sr, sp = spearmanr(norm_lp, vals)
            print(f"  {name:<25} Pearson r={pr:+.3f} (p={pp:.1e})  "
                  f"Spearman rho={sr:+.3f} (p={sp:.1e})")

        # Correlation with human confidence score
        conf_with_human = [r for r in conf_rows if r["confidence"] != ""]
        if conf_with_human:
            human_conf = np.array([r["confidence"] for r in conf_with_human])
            machine_conf = np.array([r["dbn_norm_log_prob"] for r in conf_with_human])
            pr, pp = pearsonr(machine_conf, human_conf)
            sr, sp = spearmanr(machine_conf, human_conf)
            print(f"\n  vs human_confidence       Pearson r={pr:+.3f} (p={pp:.1e})  "
                  f"Spearman rho={sr:+.3f} (p={sp:.1e})")
            print("  (human confidence: 1=easy, 4=hard; norm_log_prob: higher=more confident)")

        # Quartile analysis
        print(f"\n{'='*80}")
        print("DBN CONFIDENCE QUARTILE ANALYSIS")
        print(f"{'='*80}")
        sorted_rows = sorted(conf_rows, key=lambda r: r["dbn_norm_log_prob"])
        q_size = len(sorted_rows) // 4
        quartiles = [
            ("Q1 (lowest conf)", sorted_rows[:q_size]),
            ("Q2", sorted_rows[q_size:2*q_size]),
            ("Q3", sorted_rows[2*q_size:3*q_size]),
            ("Q4 (highest conf)", sorted_rows[3*q_size:]),
        ]
        print(f"  {'Quartile':<22} {'n':>4} {'norm_lp range':>22} "
              f"{'dbn_F':>8} {'no-dbn_F':>10} {'dbn_AMLt':>10}")
        print("  " + "-" * 82)
        for label, subset in quartiles:
            nlps = [r["dbn_norm_log_prob"] for r in subset]
            f_dbn = np.mean([r["dbn_F-measure"] for r in subset])
            f_no = np.mean([r["F-measure"] for r in subset])
            amlt = np.mean([r["dbn_AMLt"] for r in subset])
            print(f"  {label:<22} {len(subset):>4} "
                  f"[{min(nlps):>8.3f}, {max(nlps):>8.3f}] "
                  f"{f_dbn:>8.3f} {f_no:>10.3f} {amlt:>10.3f}")

        # Easy vs Hard confidence
        print(f"\n{'='*80}")
        print("DBN CONFIDENCE: EASY vs HARD")
        print(f"{'='*80}")
        for label, filt in [("Hard", lambda r: not r["is_easy"]),
                             ("Easy", lambda r: r["is_easy"])]:
            subset = [r for r in conf_rows if filt(r)]
            if subset:
                nlp = np.mean([r["dbn_norm_log_prob"] for r in subset])
                nlp_std = np.std([r["dbn_norm_log_prob"] for r in subset])
                print(f"  {label:<6} (n={len(subset):>3}):  "
                      f"mean norm_log_prob = {nlp:.4f} +/- {nlp_std:.4f}")


if __name__ == "__main__":
    main()
