"""
Run madmom's TCN-based beat tracker (Davies & Böck, 2019 + Böck et al. ISMIR 2019)
on SMC as a baseline, AND use its tempo estimate to constrain the DBN when
postprocessing beat_this activations.

The TCN is multi-task: it outputs both beat activations and a tempo histogram.
We use:
  - TCN beats + DBN → madmom TCN baseline
  - TCN tempo → constrains the DBN applied to beat_this activations

Caveat: SMC was almost certainly in the TCN's training data. Aggregate comparison
is not fair; focus on per-track failure patterns.
"""
import csv
import numpy as np
import torch
import mir_eval
from pathlib import Path
from tqdm import tqdm

from madmom.features.beats import TCNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features.tempo import TCNTempoHistogramProcessor, TempoEstimationProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor

from beat_this.inference import Audio2Frames
from beat_this.preprocessing import load_audio

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR = ROOT / "beat_this_annotations" / "smc" / "annotations" / "beats"
AUDIO_DIR = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
SPLIT_FILE = ROOT / "beat_this_annotations" / "smc" / "8-folds.split"
BT_PRED_DIR = ROOT / "beat_this_output"
BT_DBN_DIR = ROOT / "beat_this_output_dbn"

TCN_MADMOM_OUT = ROOT / "madmom_tcn_output"
TCN_TEMPO_DIR = ROOT / "madmom_tcn_tempo"  # cached per-track tempo estimates
BT_ACT_DIR = ROOT / "beat_this_activations_cache"  # cached beat_this logits
TCN_CONSTRAINED_OUT = ROOT / "beat_this_output_tcn_constrained"
OUT_CSV = DATA_DIR / "tcn_tempo_constrained_results.csv"

BT_FPS = 50
MADMOM_FPS = 100


def load_gt(path):
    return np.array([float(line.strip()) for line in open(path) if line.strip()])


def load_pred(path):
    return np.array([float(line.strip().split("\t")[0]) for line in open(path) if line.strip()])


def evaluate(ref, est):
    if len(ref) == 0 or len(est) == 0:
        return {k: 0.0 for k in ["F-measure", "Correct Metric Level Total",
                                   "Any Metric Level Total", "P-score",
                                   "Correct Metric Level Continuous",
                                   "Any Metric Level Continuous"]}
    return mir_eval.beat.evaluate(ref, est)


def gt_tempo(ref_beats):
    """Ground truth tempo (median) in BPM."""
    if len(ref_beats) < 2:
        return 0.0
    return 60.0 / np.median(np.diff(ref_beats))


def _tcn_worker(args):
    """Worker: run TCN on one track, save beats and tempo to cache.
    Each worker lazily loads its own TCN model on first call.
    """
    track_id, wav_path_str, beats_cache_str, tempo_cache_str = args
    # Lazy init per-process (only once, cached via module globals)
    global _W_TCN, _W_DBN, _W_TEMPO_EST
    if '_W_TCN' not in globals():
        from madmom.features.beats import TCNBeatProcessor, DBNBeatTrackingProcessor
        from madmom.features.tempo import (
            TCNTempoHistogramProcessor, TempoEstimationProcessor,
        )
        _W_TCN = TCNBeatProcessor(tasks=(0, 1))
        _W_DBN = DBNBeatTrackingProcessor(
            fps=100, min_bpm=55.0, max_bpm=215.0, transition_lambda=100
        )
        hist = TCNTempoHistogramProcessor(min_bpm=40, max_bpm=250, fps=100)
        _W_TEMPO_EST = TempoEstimationProcessor(
            method=None, histogram_processor=hist, fps=100,
            interpolate=True, hist_smooth=15, act_smooth=None
        )
    tcn_out = _W_TCN(wav_path_str)
    madmom_beats = _W_DBN(tcn_out[0])
    tempo_pairs = _W_TEMPO_EST(tcn_out)
    tempo_bpm = float(tempo_pairs[0, 0]) if len(tempo_pairs) > 0 else 0.0
    np.savetxt(beats_cache_str, madmom_beats, fmt="%.4f")
    with open(tempo_cache_str, "w") as f:
        f.write(f"{tempo_bpm:.4f}\n")
    return track_id


def run_bt_constrained_dbn(beat_logits, downbeat_logits, min_bpm, max_bpm):
    """Apply a DBN constrained to [min_bpm, max_bpm] to beat_this logits."""
    # Clamp
    min_bpm = max(30.0, min_bpm)
    max_bpm = min(300.0, max_bpm)
    if max_bpm <= min_bpm + 5:
        max_bpm = min_bpm + 5

    dbn = DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4],
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        fps=BT_FPS,
        transition_lambda=100,
    )
    # Convert logits to probabilities (same as beat_this postprocessor)
    beat_prob = beat_logits.double().sigmoid()
    downbeat_prob = downbeat_logits.double().sigmoid()
    epsilon = 1e-5
    beat_prob = beat_prob * (1 - epsilon) + epsilon / 2
    downbeat_prob = downbeat_prob * (1 - epsilon) + epsilon / 2
    combined_act = np.vstack((
        np.maximum(beat_prob.cpu().numpy() - downbeat_prob.cpu().numpy(), epsilon / 2),
        downbeat_prob.cpu().numpy(),
    )).T
    dbn_out = dbn(combined_act)
    return dbn_out[:, 0]  # beat times


def main():
    TCN_MADMOM_OUT.mkdir(exist_ok=True)
    TCN_TEMPO_DIR.mkdir(exist_ok=True)
    BT_ACT_DIR.mkdir(exist_ok=True)
    TCN_CONSTRAINED_OUT.mkdir(exist_ok=True)

    # DBN for madmom TCN beats (100 fps) — used in main process only
    dbn_madmom = DBNBeatTrackingProcessor(
        fps=MADMOM_FPS, min_bpm=55.0, max_bpm=215.0, transition_lambda=100
    )

    # ── Load fold assignments for beat_this ──
    fold_map = {}
    with open(SPLIT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            track_id, fold_num = line.split("\t")
            fold_map[track_id] = int(fold_num)

    # Group by fold
    from collections import defaultdict
    folds = defaultdict(list)
    for gt_path in sorted(GT_DIR.glob("*.beats")):
        track_id = gt_path.stem
        fold = fold_map.get(track_id)
        if fold is not None:
            folds[fold].append(track_id)

    # ── Step 1: Get TCN beats + tempo for all tracks (resumable) ──
    print(f"\n[Step 1] Running TCN on {sum(len(v) for v in folds.values())} tracks (skipping cached)...")
    tcn_data = {}  # track_id -> (madmom_tcn_beats, tempo_bpm)
    tracks_to_process = []
    for track_id in sorted(fold_map.keys()):
        beats_cache = TCN_MADMOM_OUT / f"{track_id}.beats"
        tempo_cache = TCN_TEMPO_DIR / f"{track_id}.bpm"
        if beats_cache.exists() and tempo_cache.exists():
            # Load from cache
            madmom_beats = np.loadtxt(str(beats_cache))
            if madmom_beats.ndim == 0:
                madmom_beats = np.array([float(madmom_beats)])
            tempo_bpm = float(open(tempo_cache).read().strip())
            tcn_data[track_id] = (madmom_beats, tempo_bpm)
        else:
            tracks_to_process.append(track_id)

    print(f"  {len(tcn_data)} tracks loaded from cache, {len(tracks_to_process)} to process")
    if tracks_to_process:
        worker_args = []
        for track_id in tracks_to_process:
            wav_path = AUDIO_DIR / f"{track_id.upper()}.wav"
            if not wav_path.exists():
                continue
            worker_args.append((
                track_id,
                str(wav_path),
                str(TCN_MADMOM_OUT / f"{track_id}.beats"),
                str(TCN_TEMPO_DIR / f"{track_id}.bpm"),
            ))
        # Parallelize across 4 CPU workers (madmom TCN is CPU-bound)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"  Running {len(worker_args)} tracks with 4 parallel workers...")
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_tcn_worker, a): a[0] for a in worker_args}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                _ = fut.result()

        # Reload all cached data
        for track_id in tracks_to_process:
            beats_cache = TCN_MADMOM_OUT / f"{track_id}.beats"
            tempo_cache = TCN_TEMPO_DIR / f"{track_id}.bpm"
            if beats_cache.exists() and tempo_cache.exists():
                madmom_beats = np.loadtxt(str(beats_cache))
                if madmom_beats.ndim == 0:
                    madmom_beats = np.array([float(madmom_beats)])
                tempo_bpm = float(open(tempo_cache).read().strip())
                tcn_data[track_id] = (madmom_beats, tempo_bpm)

    # ── Step 2: Get beat_this activations (per fold) with caching ──
    print(f"\n[Step 2] Loading beat_this activations (cached where possible)...")
    bt_data = {}  # track_id -> (beat_logits, downbeat_logits)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for fold_num in sorted(folds.keys()):
        tracks = folds[fold_num]
        model_name = f"fold{fold_num}"
        # Find which tracks need computing
        to_compute = []
        for track_id in tracks:
            cache_path = BT_ACT_DIR / f"{track_id}.npz"
            if cache_path.exists():
                d = np.load(str(cache_path))
                bt_data[track_id] = (
                    torch.from_numpy(d["beat"]),
                    torch.from_numpy(d["downbeat"]),
                )
            else:
                to_compute.append(track_id)
        if to_compute:
            print(f"  Fold {fold_num}: computing {len(to_compute)}/{len(tracks)} tracks with {model_name}")
            audio2frames = Audio2Frames(checkpoint_path=model_name, device=device)
            for track_id in to_compute:
                wav_path = AUDIO_DIR / f"{track_id.upper()}.wav"
                signal, sr = load_audio(str(wav_path))
                beat_logits, downbeat_logits = audio2frames(signal, sr)
                beat_cpu = beat_logits.cpu()
                db_cpu = downbeat_logits.cpu()
                bt_data[track_id] = (beat_cpu, db_cpu)
                # Cache
                np.savez(
                    str(BT_ACT_DIR / f"{track_id}.npz"),
                    beat=beat_cpu.numpy(),
                    downbeat=db_cpu.numpy(),
                )
        else:
            print(f"  Fold {fold_num}: all {len(tracks)} tracks cached")

    # ── Step 3: Evaluate everything ──
    print(f"\n[Step 3] Evaluating all variants...")
    rows = []
    for track_id in tqdm(sorted(fold_map.keys())):
        if track_id not in tcn_data or track_id not in bt_data:
            continue

        gt_path = GT_DIR / f"{track_id}.beats"
        ref = load_gt(gt_path)
        gt_bpm = gt_tempo(ref)

        madmom_beats, tcn_tempo = tcn_data[track_id]
        beat_logits, downbeat_logits = bt_data[track_id]

        # Load no-DBN and DBN beat_this predictions
        bt_beats = load_pred(BT_PRED_DIR / f"{track_id.upper()}.beats")
        bt_dbn_beats = load_pred(BT_DBN_DIR / f"{track_id.upper()}.beats")

        # Tempo-constrained DBN on beat_this (± 20% window)
        if tcn_tempo > 0:
            min_bpm = tcn_tempo * 0.8
            max_bpm = tcn_tempo * 1.2
            bt_tcnconstr = run_bt_constrained_dbn(beat_logits, downbeat_logits, min_bpm, max_bpm)
        else:
            bt_tcnconstr = bt_dbn_beats  # fallback

        # Oracle: constrain around GT tempo
        if gt_bpm > 0:
            min_bpm_orc = gt_bpm * 0.8
            max_bpm_orc = gt_bpm * 1.2
            bt_oracle = run_bt_constrained_dbn(beat_logits, downbeat_logits, min_bpm_orc, max_bpm_orc)
        else:
            bt_oracle = bt_dbn_beats

        # Save outputs
        np.savetxt(str(TCN_CONSTRAINED_OUT / f"{track_id}.beats"), bt_tcnconstr, fmt="%.4f")

        # Evaluate
        mm_scores = evaluate(ref, madmom_beats)
        bt_scores = evaluate(ref, bt_beats)
        bt_dbn_scores = evaluate(ref, bt_dbn_beats)
        bt_constr_scores = evaluate(ref, bt_tcnconstr)
        bt_oracle_scores = evaluate(ref, bt_oracle)

        # Tempo ratio: TCN estimate vs GT
        tempo_ratio = tcn_tempo / gt_bpm if gt_bpm > 0 else 0

        # Classify tempo estimator accuracy
        if 0.8 <= tempo_ratio <= 1.2:
            tempo_class = "correct"
        elif 0.4 <= tempo_ratio < 0.8:
            tempo_class = "half"  # TCN says half-tempo
        elif 1.2 < tempo_ratio <= 2.5:
            tempo_class = "double"  # TCN says double-tempo
        else:
            tempo_class = "other"

        row = {
            "track_id": track_id,
            "gt_bpm": round(gt_bpm, 2),
            "tcn_bpm": round(tcn_tempo, 2),
            "tempo_ratio": round(tempo_ratio, 4),
            "tempo_class": tempo_class,
            # madmom TCN (full baseline)
            "mm_tcn_F": round(mm_scores["F-measure"], 4),
            "mm_tcn_CMLt": round(mm_scores["Correct Metric Level Total"], 4),
            "mm_tcn_AMLt": round(mm_scores["Any Metric Level Total"], 4),
            # beat_this no-DBN
            "bt_F": round(bt_scores["F-measure"], 4),
            "bt_CMLt": round(bt_scores["Correct Metric Level Total"], 4),
            "bt_AMLt": round(bt_scores["Any Metric Level Total"], 4),
            # beat_this + unconstrained DBN
            "bt_dbn_F": round(bt_dbn_scores["F-measure"], 4),
            "bt_dbn_CMLt": round(bt_dbn_scores["Correct Metric Level Total"], 4),
            "bt_dbn_AMLt": round(bt_dbn_scores["Any Metric Level Total"], 4),
            # beat_this + TCN-tempo-constrained DBN
            "bt_tcnconstr_F": round(bt_constr_scores["F-measure"], 4),
            "bt_tcnconstr_CMLt": round(bt_constr_scores["Correct Metric Level Total"], 4),
            "bt_tcnconstr_AMLt": round(bt_constr_scores["Any Metric Level Total"], 4),
            # beat_this + oracle-tempo-constrained DBN
            "bt_oracle_F": round(bt_oracle_scores["F-measure"], 4),
            "bt_oracle_CMLt": round(bt_oracle_scores["Correct Metric Level Total"], 4),
            "bt_oracle_AMLt": round(bt_oracle_scores["Any Metric Level Total"], 4),
        }
        rows.append(row)

    # ── Write CSV ──
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print("AGGREGATE COMPARISON")
    print(f"  (caveat: madmom TCN trained on SMC; tempo constraints are post-hoc)")
    print(f"{'='*80}")
    print(f"  {'System':<30} {'Mean F':>8} {'Med F':>8} {'CMLt':>8} {'AMLt':>8}")
    print("  " + "-" * 70)
    for name, fkey, ckey, akey in [
        ("madmom TCN (RNN+DBN)", "mm_tcn_F", "mm_tcn_CMLt", "mm_tcn_AMLt"),
        ("beat_this (no DBN)", "bt_F", "bt_CMLt", "bt_AMLt"),
        ("beat_this + DBN", "bt_dbn_F", "bt_dbn_CMLt", "bt_dbn_AMLt"),
        ("beat_this + TCN-constr DBN", "bt_tcnconstr_F", "bt_tcnconstr_CMLt", "bt_tcnconstr_AMLt"),
        ("beat_this + oracle-constr DBN", "bt_oracle_F", "bt_oracle_CMLt", "bt_oracle_AMLt"),
    ]:
        f_vals = [r[fkey] for r in rows]
        c_vals = [r[ckey] for r in rows]
        a_vals = [r[akey] for r in rows]
        print(f"  {name:<30} {np.mean(f_vals):>8.4f} {np.median(f_vals):>8.4f} "
              f"{np.mean(c_vals):>8.4f} {np.mean(a_vals):>8.4f}")

    # ── Tempo estimator accuracy ──
    print(f"\n{'='*80}")
    print("TCN TEMPO ESTIMATOR ACCURACY")
    print(f"{'='*80}")
    from collections import Counter
    tempo_counts = Counter(r["tempo_class"] for r in rows)
    for cls in ["correct", "double", "half", "other"]:
        n = tempo_counts[cls]
        print(f"  {cls:<10} {n:>4} ({100*n/len(rows):>5.1f}%)")

    # ── Does the constraint help conditional on tempo correctness? ──
    print(f"\n{'='*80}")
    print("CONSTRAINT EFFECT CONDITIONED ON TEMPO ACCURACY")
    print(f"{'='*80}")
    print(f"  {'Tempo class':<12} {'N':>4} {'bt_F':>8} {'bt_dbn_F':>10} {'bt_constr_F':>12} "
          f"{'bt_oracle_F':>12}")
    print("  " + "-" * 62)
    for cls in ["correct", "half", "double", "other"]:
        subset = [r for r in rows if r["tempo_class"] == cls]
        if not subset:
            continue
        bt_f = np.mean([r["bt_F"] for r in subset])
        dbn_f = np.mean([r["bt_dbn_F"] for r in subset])
        constr_f = np.mean([r["bt_tcnconstr_F"] for r in subset])
        orc_f = np.mean([r["bt_oracle_F"] for r in subset])
        print(f"  {cls:<12} {len(subset):>4} {bt_f:>8.4f} {dbn_f:>10.4f} "
              f"{constr_f:>12.4f} {orc_f:>12.4f}")

    # ── Per-track win analysis ──
    improvements = []
    hurts = []
    for r in rows:
        delta = r["bt_tcnconstr_F"] - r["bt_dbn_F"]
        if delta > 0.05:
            improvements.append((r, delta))
        elif delta < -0.05:
            hurts.append((r, delta))

    print(f"\n{'='*80}")
    print("TCN-CONSTRAINED vs UNCONSTRAINED DBN (per track)")
    print(f"{'='*80}")
    print(f"  Tracks improved by constraint (delta F > 0.05): {len(improvements)}")
    print(f"  Tracks hurt by constraint (delta F < -0.05):    {len(hurts)}")

    print(f"\n  Top 10 improvements:")
    for r, d in sorted(improvements, key=lambda x: -x[1])[:10]:
        print(f"    {r['track_id']}: tempo_class={r['tempo_class']} "
              f"F {r['bt_dbn_F']:.3f}->{r['bt_tcnconstr_F']:.3f} ({d:+.3f}) "
              f"(tcn_bpm={r['tcn_bpm']:.1f} gt_bpm={r['gt_bpm']:.1f})")

    print(f"\n  Top 10 regressions:")
    for r, d in sorted(hurts, key=lambda x: x[1])[:10]:
        print(f"    {r['track_id']}: tempo_class={r['tempo_class']} "
              f"F {r['bt_dbn_F']:.3f}->{r['bt_tcnconstr_F']:.3f} ({d:+.3f}) "
              f"(tcn_bpm={r['tcn_bpm']:.1f} gt_bpm={r['gt_bpm']:.1f})")


if __name__ == "__main__":
    main()
