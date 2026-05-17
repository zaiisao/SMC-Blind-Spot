"""
Analyze SMC_MIREX .tag files.

Each .tag file contains difficulty descriptors for a music excerpt,
with the last meaningful line being a code like 'a1', 'm3', 'j2', etc.
- The letter = annotator initial (person's name)
- The number = confidence score

This script parses all .tag files and provides summary statistics.
"""

import os
import re
from collections import Counter
from pathlib import Path

TAG_DIR = Path(__file__).parent.parent / "SMC_MIREX" / "SMC_MIREX_Tags"


def parse_tag_file(filepath):
    """Extract the annotator code (e.g. 'a1') from a .tag file.
    Returns (letter, number, raw_code) or None if unparseable.
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        return None

    # Walk backwards to find the last line matching letter+digit pattern
    for line in reversed(lines):
        match = re.fullmatch(r"([a-z])(\d+)", line)
        if match:
            return match.group(1), int(match.group(2)), line
    return None


def main():
    tag_files = sorted(TAG_DIR.glob("SMC_*.tag"))
    print(f"Total .tag files found: {len(tag_files)}\n")

    # Parse all files
    results = []  # (filename, letter, number, raw_code, descriptors)
    unparseable = []

    for f in tag_files:
        parsed = parse_tag_file(f)
        if parsed is None:
            unparseable.append(f.name)
            continue

        letter, number, raw = parsed
        # Also collect the descriptor lines (everything except the code line)
        with open(f) as fh:
            all_lines = [line.strip() for line in fh.readlines() if line.strip()]
        descriptors = [l for l in all_lines if not re.fullmatch(r"[a-z]\d+", l)]
        results.append((f.name, letter, number, raw, descriptors))

    if unparseable:
        print(f"Could not parse {len(unparseable)} file(s): {unparseable}\n")

    # ─── 1. Confidence score distribution ───
    score_counts = Counter(r[2] for r in results)
    print("=" * 55)
    print("CONFIDENCE SCORE DISTRIBUTION")
    print("=" * 55)
    for score in sorted(score_counts):
        count = score_counts[score]
        bar = "█" * count
        print(f"  Score {score}: {count:>4} samples  ({100*count/len(results):5.1f}%)  {bar}")
    print()

    # ─── 2. Annotator distribution ───
    annotator_counts = Counter(r[1] for r in results)
    print("=" * 55)
    print("ANNOTATOR DISTRIBUTION")
    print("=" * 55)
    for ann in sorted(annotator_counts, key=lambda x: -annotator_counts[x]):
        count = annotator_counts[ann]
        print(f"  Annotator '{ann}': {count:>4} samples  ({100*count/len(results):5.1f}%)")
    print()

    # ─── 3. Annotator × Score cross-tabulation ───
    all_scores = sorted(score_counts.keys())
    all_annotators = sorted(annotator_counts.keys())
    cross = Counter((r[1], r[2]) for r in results)

    print("=" * 55)
    print("ANNOTATOR × CONFIDENCE SCORE CROSS-TABLE")
    print("=" * 55)
    header = f"  {'Ann':>4}" + "".join(f"  Score {s}" for s in all_scores) + "   Total"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ann in all_annotators:
        row = f"  {ann:>4}"
        for s in all_scores:
            row += f"  {cross[(ann, s)]:>7}"
        total = annotator_counts[ann]
        row += f"   {total:>5}"
        print(row)
    # Score totals row
    row = f"  {'Tot':>4}"
    for s in all_scores:
        row += f"  {score_counts[s]:>7}"
    row += f"   {len(results):>5}"
    print(row)
    print()

    # ─── 4. Average confidence score per annotator ───
    print("=" * 55)
    print("AVERAGE CONFIDENCE SCORE PER ANNOTATOR")
    print("=" * 55)
    for ann in sorted(all_annotators):
        scores = [r[2] for r in results if r[1] == ann]
        avg = sum(scores) / len(scores)
        print(f"  Annotator '{ann}': avg = {avg:.2f}  (n={len(scores)})")
    all_scores_list = [r[2] for r in results]
    overall_avg = sum(all_scores_list) / len(all_scores_list)
    print(f"  {'Overall':>13}: avg = {overall_avg:.2f}  (n={len(results)})")
    print()

    # ─── 5. Difficulty descriptor frequency ───
    desc_counter = Counter()
    for _, _, _, _, descriptors in results:
        for d in descriptors:
            # Normalize: strip parentheses used for optional/uncertain tags
            cleaned = d.strip("()")
            if cleaned:
                desc_counter[cleaned] += 1

    print("=" * 55)
    print("TOP 20 DIFFICULTY DESCRIPTORS")
    print("=" * 55)
    for desc, count in desc_counter.most_common(20):
        print(f"  {count:>4}  {desc}")
    print()

    # ─── 6. Average number of descriptors per score ───
    print("=" * 55)
    print("AVG NUMBER OF DESCRIPTORS PER CONFIDENCE SCORE")
    print("=" * 55)
    from collections import defaultdict
    descs_by_score = defaultdict(list)
    for _, _, number, _, descriptors in results:
        descs_by_score[number].append(len(descriptors))
    for score in sorted(descs_by_score):
        vals = descs_by_score[score]
        avg = sum(vals) / len(vals)
        print(f"  Score {score}: avg {avg:.2f} descriptors  (n={len(vals)})")
    print()
    print("(More descriptors may indicate the excerpt is harder to")
    print(" beat-track, which could help interpret the confidence score.)")

    # ─── 7. Descriptor frequency by confidence score ───
    print()
    print("=" * 55)
    print("TOP DESCRIPTORS BY CONFIDENCE SCORE")
    print("=" * 55)
    descs_by_score_counter = defaultdict(Counter)
    for _, _, number, _, descriptors in results:
        for d in descriptors:
            cleaned = d.strip("()")
            if cleaned:
                descs_by_score_counter[number][cleaned] += 1
    for score in sorted(descs_by_score_counter):
        print(f"\n  Score {score} — top 5 descriptors:")
        for desc, count in descs_by_score_counter[score].most_common(5):
            print(f"    {count:>3}  {desc}")


if __name__ == "__main__":
    main()
