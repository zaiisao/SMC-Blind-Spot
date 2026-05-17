#!/usr/bin/env python3
"""
Activation Function Visualization and Diagnostic Analysis for SMC Tracks.

Extracts and visualizes raw beat activation functions from beat_this and
Beat Transformer, computes quantitative diagnostics, and classifies
activation failure modes.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.special import expit as sigmoid
from scipy.stats import spearmanr, entropy as scipy_entropy
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
BT_ACT_CACHE = ROOT / "beat_this_activations_cache"
BTR_ACT_CACHE = ROOT / "bt_transformer_cache"
GT_DIR = ROOT / "beat_this_annotations" / "smc" / "annotations" / "beats"
BT_PRED_DIR = ROOT / "beat_this_output"
BTR_PRED_DIR = ROOT / "bt_transformer_beats"
RESULTS_CSV = DATA_DIR / "results.csv"
BTR_RESULTS_CSV = DATA_DIR / "bt_transformer_tempo_constrained_results.csv"
OUT_DIR = ROOT / "activation_plots"
OUT_DIR.mkdir(exist_ok=True)

BT_FPS = 50.0
BTR_FPS = 44100.0 / 1024.0  # ≈ 43.066 Hz


# ── Helpers ────────────────────────────────────────────────────────────────

def load_gt_beats(track_id):
    """Load ground truth beat times from .beats file."""
    p = GT_DIR / f"{track_id}.beats"
    beats = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                beats.append(float(line.split()[0].rstrip(',')))
    return np.array(beats)


def load_bt_activation(track_id):
    """Load beat_this activation (apply sigmoid to cached logits)."""
    p = BT_ACT_CACHE / f"{track_id}.npz"
    d = np.load(p)
    logits = d['beat']
    act = sigmoid(logits)
    times = np.arange(len(act)) / BT_FPS
    return act, times


def load_btr_activation(track_id):
    """Load Beat Transformer activation (already sigmoid)."""
    p = BTR_ACT_CACHE / f"{track_id}.npz"
    d = np.load(p)
    act = d['beat_act']
    times = np.arange(len(act)) / BTR_FPS
    return act, times


def load_predictions(pred_dir, track_id, uppercase=False):
    """Load predicted beat times from .beats file."""
    fname = track_id.upper().replace('SMC_', 'SMC_') if uppercase else track_id
    # beat_this uses uppercase SMC_XXX, BT uses lowercase smc_XXX
    p = pred_dir / f"{fname}.beats"
    if not p.exists():
        # Try alternate casing
        for candidate in pred_dir.glob(f"*{track_id.split('_')[1]}*"):
            p = candidate
            break
    beats = []
    if p.exists():
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    beats.append(float(line.split()[0].rstrip(',')))
    return np.array(beats)


def compute_local_f(gt_beats, pred_beats, window_start, window_end, tol=0.07):
    """Compute F-measure within a time window."""
    gt_w = gt_beats[(gt_beats >= window_start) & (gt_beats < window_end)]
    pred_w = pred_beats[(pred_beats >= window_start) & (pred_beats < window_end)]
    if len(gt_w) == 0 and len(pred_w) == 0:
        return 1.0
    if len(gt_w) == 0 or len(pred_w) == 0:
        return 0.0
    # Simple matching
    matched = 0
    used = set()
    for g in gt_w:
        for j, p in enumerate(pred_w):
            if j not in used and abs(g - p) <= tol:
                matched += 1
                used.add(j)
                break
    prec = matched / len(pred_w) if len(pred_w) > 0 else 0
    rec = matched / len(gt_w) if len(gt_w) > 0 else 0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def find_worst_5s_window(gt_beats, pred_beats, track_duration, step=0.5):
    """Find the 5-second window with the worst local F-measure."""
    worst_f = 1.0
    worst_start = 0.0
    for start in np.arange(0, max(track_duration - 5, 0.1), step):
        f = compute_local_f(gt_beats, pred_beats, start, start + 5)
        if f < worst_f:
            worst_f = f
            worst_start = start
    return worst_start, worst_f


def activation_at_positions(act, times, beat_times, tolerance_frames=2):
    """Get activation values at beat positions (±tolerance frames)."""
    vals = []
    for bt in beat_times:
        idx = np.argmin(np.abs(times - bt))
        lo = max(0, idx - tolerance_frames)
        hi = min(len(act), idx + tolerance_frames + 1)
        vals.append(np.max(act[lo:hi]))
    return np.array(vals) if vals else np.array([0.0])


def compute_periodicity(act, fps, bpm_lo=30, bpm_hi=215):
    """Compute autocorrelation periodicity strength."""
    act_centered = act - act.mean()
    n = len(act_centered)
    # Autocorrelation via FFT
    fft = np.fft.rfft(act_centered, n=2*n)
    acf = np.fft.irfft(fft * np.conj(fft))[:n]
    acf = acf / (acf[0] + 1e-12)  # Normalize

    # Lags corresponding to BPM range
    lag_lo = int(fps * 60.0 / bpm_hi)  # Fastest tempo → shortest lag
    lag_hi = int(fps * 60.0 / bpm_lo)  # Slowest tempo → longest lag
    lag_hi = min(lag_hi, n - 1)
    if lag_lo >= lag_hi:
        return 0.0
    segment = acf[lag_lo:lag_hi+1]
    return float(np.max(segment))


def compute_activation_entropy(act):
    """Compute entropy of activation treated as probability distribution."""
    p = act.copy()
    p = np.clip(p, 0, None)
    total = p.sum()
    if total < 1e-12:
        return np.log(len(p))  # Max entropy for flat
    p = p / total
    return float(scipy_entropy(p))


def compute_diagnostics(act, times, fps, gt_beats, pred_beats, prefix="bt_act"):
    """Compute all diagnostic metrics for one activation function."""
    d = {}
    d[f'{prefix}_mean'] = float(act.mean())
    d[f'{prefix}_max'] = float(act.max())
    d[f'{prefix}_median'] = float(np.median(act))

    # Peak analysis
    peaks, properties = find_peaks(act, prominence=0.05, height=0.1)
    prominences = properties.get('prominences', np.array([]))

    d[f'{prefix}_peak_prominence_mean'] = float(prominences.mean()) if len(prominences) > 0 else 0.0
    d[f'{prefix}_peak_prominence_std'] = float(prominences.std()) if len(prominences) > 0 else 0.0
    d[f'{prefix}_n_peaks'] = int(np.sum(act[peaks] > 0.3)) if len(peaks) > 0 else 0

    # Periodicity
    d[f'{prefix}_periodicity'] = compute_periodicity(act, fps)

    # Entropy
    d[f'{prefix}_entropy'] = compute_activation_entropy(act)

    # Activation at GT positions
    gt_vals = activation_at_positions(act, times, gt_beats)
    d[f'{prefix}_gt_peak_mean'] = float(gt_vals.mean())
    d[f'{prefix}_gt_peak_std'] = float(gt_vals.std())

    # False positive activation (pred beats not matching GT)
    if len(pred_beats) > 0 and len(gt_beats) > 0:
        fp_beats = []
        for pb in pred_beats:
            if np.min(np.abs(gt_beats - pb)) > 0.07:
                fp_beats.append(pb)
        if fp_beats:
            fp_vals = activation_at_positions(act, times, np.array(fp_beats))
            d[f'{prefix}_nongt_peak_mean'] = float(fp_vals.mean())
        else:
            d[f'{prefix}_nongt_peak_mean'] = 0.0
    else:
        d[f'{prefix}_nongt_peak_mean'] = 0.0

    return d


def classify_failure(diag, prefix="bt_act"):
    """Classify activation failure mode for total failure tracks."""
    act_max = diag.get(f'{prefix}_max', 0)
    prom_mean = diag.get(f'{prefix}_peak_prominence_mean', 0)
    gt_peak_mean = diag.get(f'{prefix}_gt_peak_mean', 0)
    periodicity = diag.get(f'{prefix}_periodicity', 0)
    n_peaks = diag.get(f'{prefix}_n_peaks', 0)
    f_measure = diag.get('bt_F', 0)

    if act_max < 0.5 and prom_mean < 0.2:
        return "flat_activations"
    elif act_max > 0.5 and gt_peak_mean < 0.3:
        return "wrong_peaks"
    elif periodicity < 0.3 and n_peaks > 0:
        # Check if too many peaks relative to expected
        return "ambiguous_peaks"
    elif gt_peak_mean > 0.4 and f_measure < 0.3:
        return "reasonable_but_misaligned"
    else:
        return "other"


# ── Track Selection ────────────────────────────────────────────────────────

def select_tracks(df, df_btr):
    """Select 12 representative tracks spanning the error taxonomy."""
    # Merge key BT transformer metrics
    df = df.merge(
        df_btr[['track_id', 'bt_transformer_F', 'bt_transformer_CMLt', 'bt_transformer_AMLt',
                 'bt_wide_F', 'bt_wide_CMLt', 'bt_wide_AMLt']],
        on='track_id', how='left'
    )

    selected = {}

    # Category 1: Good tracks (F ≥ 0.8 for both systems)
    good = df[(df['F-measure'] >= 0.8) & (df['bt_transformer_F'] >= 0.8)]
    good = good.sort_values('F-measure', ascending=False)
    selected['good'] = good.head(3)['track_id'].tolist()

    # Category 2: Octave errors (AMLt - F > 0.25)
    octave_candidates = df[(df['AMLt'] - df['F-measure']) > 0.25]
    must_include = ['smc_274', 'smc_276']
    octave = [t for t in must_include if t in df['track_id'].values]
    remaining = octave_candidates[~octave_candidates['track_id'].isin(octave)]
    remaining = remaining.sort_values(by='AMLt', ascending=False)
    for t in remaining['track_id']:
        if len(octave) >= 3:
            break
        if t not in octave:
            octave.append(t)
    selected['octave_error'] = octave[:3]

    # Category 3: Continuity drift (CMLt - CMLc > 0.2, F between 0.3-0.6)
    # CMLc is in results.csv as 'CMLc'
    drift = df[(df['CMLt'] - df['CMLc'] > 0.2) & (df['F-measure'] >= 0.3) & (df['F-measure'] <= 0.6)]
    drift = drift.sort_values(by='F-measure')
    selected['continuity_drift'] = drift.head(3)['track_id'].tolist()

    # Category 4: Total failure (F < 0.3, AMLt < 0.3)
    fail = df[(df['F-measure'] < 0.3) & (df['AMLt'] < 0.3)]
    must_include_fail = ['smc_194', 'smc_148', 'smc_203', 'smc_064']
    total_fail = [t for t in must_include_fail if t in fail['track_id'].values]
    # Fill up if needed
    remaining_fail = fail[~fail['track_id'].isin(total_fail)].sort_values('F-measure')
    for t in remaining_fail['track_id']:
        if len(total_fail) >= 4:
            break
        total_fail.append(t)
    selected['total_failure'] = total_fail[:4]

    return selected


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_activation(track_id, bt_act, bt_times, btr_act, btr_times,
                    gt_beats, bt_pred, btr_pred,
                    bt_F, bt_AMLt, btr_F, confidence, descriptors,
                    xlim=None, suffix="full", category=""):
    """Create activation plot."""
    fig, ax = plt.subplots(figsize=(16, 4))

    # Ground truth beats as vertical lines
    for gb in gt_beats:
        if xlim and (gb < xlim[0] or gb > xlim[1]):
            continue
        ax.axvline(gb, color='green', linestyle='--', alpha=0.5, linewidth=0.8)

    # beat_this activation
    ax.plot(bt_times, bt_act, color='royalblue', linewidth=0.7, alpha=0.8, label='beat_this act')
    # Beat Transformer activation
    ax.plot(btr_times, btr_act, color='darkorange', linewidth=0.7, alpha=0.8, label='BeatTransformer act')

    # beat_this predicted beats as dots on the activation curve
    for pb in bt_pred:
        if xlim and (pb < xlim[0] or pb > xlim[1]):
            continue
        idx = np.argmin(np.abs(bt_times - pb))
        ax.plot(pb, bt_act[idx], 'o', color='red', markersize=4, zorder=5)

    # BT predicted beats as dots on activation curve
    for pb in btr_pred:
        if xlim and (pb < xlim[0] or pb > xlim[1]):
            continue
        idx = np.argmin(np.abs(btr_times - pb))
        ax.plot(pb, btr_act[idx], 'o', color='purple', markersize=4, zorder=5)

    # Add dummy artists for legend
    ax.plot([], [], 'o', color='red', markersize=4, label='beat_this pred')
    ax.plot([], [], 'o', color='purple', markersize=4, label='BT pred')
    ax.axvline(0, color='green', linestyle='--', alpha=0.5, linewidth=0.8, label='GT beats')

    desc_short = descriptors[:60] if isinstance(descriptors, str) else ""
    title = (f"{track_id} [{category}] | bt_F={bt_F:.3f} bt_AMLt={bt_AMLt:.3f} | "
             f"BT_F={btr_F:.3f} | conf={confidence} | {desc_short}")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Activation')
    ax.set_ylim(-0.05, 1.05)
    if xlim:
        ax.set_xlim(xlim)
    ax.legend(loc='upper right', fontsize=7, ncol=5)
    fig.tight_layout()
    outpath = OUT_DIR / f"{track_id}_{suffix}.png"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    return outpath


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    df = pd.read_csv(RESULTS_CSV)
    df_btr = pd.read_csv(BTR_RESULTS_CSV)

    # Select representative tracks
    selected = select_tracks(df, df_btr)
    print("\nSelected tracks:")
    for cat, tracks in selected.items():
        print(f"  {cat}: {tracks}")

    # Build category map for all tracks
    all_track_ids = df['track_id'].tolist()

    # We'll run diagnostics on ALL 217 tracks
    print(f"\nRunning diagnostics on all {len(all_track_ids)} tracks...")

    all_diags = []
    for i, track_id in enumerate(all_track_ids):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i+1}/{len(all_track_ids)}...")

        row = df[df['track_id'] == track_id].iloc[0]
        btr_row = df_btr[df_btr['track_id'] == track_id].iloc[0]

        gt_beats = load_gt_beats(track_id)
        bt_act, bt_times = load_bt_activation(track_id)
        btr_act, btr_times = load_btr_activation(track_id)
        bt_pred = load_predictions(BT_PRED_DIR, track_id, uppercase=True)
        btr_pred = load_predictions(BTR_PRED_DIR, track_id, uppercase=False)

        # Determine category
        f_bt = row['F-measure']
        amlt_bt = row['AMLt']
        cmlt_bt = row['CMLt']
        cmlc_bt = row['CMLc']
        btr_f = btr_row['bt_transformer_F']

        if track_id in selected.get('good', []):
            cat = 'good'
        elif track_id in selected.get('octave_error', []):
            cat = 'octave_error'
        elif track_id in selected.get('continuity_drift', []):
            cat = 'continuity_drift'
        elif track_id in selected.get('total_failure', []):
            cat = 'total_failure'
        else:
            # Auto-classify for the full dataset
            if f_bt >= 0.8 and btr_f >= 0.8:
                cat = 'good'
            elif (amlt_bt - f_bt) > 0.25:
                cat = 'octave_error'
            elif (cmlt_bt - cmlc_bt) > 0.2 and 0.3 <= f_bt <= 0.6:
                cat = 'continuity_drift'
            elif f_bt < 0.3 and amlt_bt < 0.3:
                cat = 'total_failure'
            else:
                cat = 'other'

        # Compute diagnostics
        diag = {
            'track_id': track_id,
            'bt_F': f_bt,
            'bt_AMLt': amlt_bt,
            'bt_CMLt': cmlt_bt,
            'btr_F': btr_f,
            'category': cat,
            'descriptors': row['descriptors'],
            'confidence': row['confidence'],
        }
        diag.update(compute_diagnostics(bt_act, bt_times, BT_FPS, gt_beats, bt_pred, prefix="bt_act"))
        diag.update(compute_diagnostics(btr_act, btr_times, BTR_FPS, gt_beats, btr_pred, prefix="btr_act"))

        # Failure mode classification (for total failure tracks)
        if cat == 'total_failure' or (f_bt < 0.3 and amlt_bt < 0.3):
            diag['bt_failure_mode'] = classify_failure(diag, "bt_act")
            diag['btr_failure_mode'] = classify_failure(
                {**diag, 'bt_F': btr_f},  # Use BT's F for classification
                "btr_act"
            )
        else:
            diag['bt_failure_mode'] = ''
            diag['btr_failure_mode'] = ''

        all_diags.append(diag)

    diag_df = pd.DataFrame(all_diags)

    # Save diagnostics CSV
    csv_path = DATA_DIR / "activation_diagnostics.csv"
    diag_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nSaved diagnostics for {len(diag_df)} tracks to {csv_path}")

    # ── Generate plots for selected tracks ────────────────────────────────
    print("\nGenerating plots for selected tracks...")
    selected_flat = []
    for cat, tracks in selected.items():
        for t in tracks:
            selected_flat.append((t, cat))

    for track_id, cat in selected_flat:
        print(f"  Plotting {track_id} ({cat})...")
        row = df[df['track_id'] == track_id].iloc[0]
        btr_row = df_btr[df_btr['track_id'] == track_id].iloc[0]

        gt_beats = load_gt_beats(track_id)
        bt_act, bt_times = load_bt_activation(track_id)
        btr_act, btr_times = load_btr_activation(track_id)
        bt_pred = load_predictions(BT_PRED_DIR, track_id, uppercase=True)
        btr_pred = load_predictions(BTR_PRED_DIR, track_id, uppercase=False)

        f_bt = row['F-measure']
        amlt_bt = row['AMLt']
        btr_f = btr_row['bt_transformer_F']
        conf = row['confidence']
        desc = row['descriptors']

        # Full track plot
        plot_activation(
            track_id, bt_act, bt_times, btr_act, btr_times,
            gt_beats, bt_pred, btr_pred,
            f_bt, amlt_bt, btr_f, conf, desc,
            suffix="full", category=cat
        )

        # Zoomed plot: find worst 5s window
        track_dur = bt_times[-1]
        if cat == 'good':
            # For good tracks, find best 5s window
            best_f = 0.0
            best_start = 0.0
            for start in np.arange(0, max(track_dur - 5, 0.1), 0.5):
                f = compute_local_f(gt_beats, bt_pred, start, start + 5)
                if f > best_f:
                    best_f = f
                    best_start = start
            zoom_start = best_start
        else:
            zoom_start, _ = find_worst_5s_window(gt_beats, bt_pred, track_dur)

        plot_activation(
            track_id, bt_act, bt_times, btr_act, btr_times,
            gt_beats, bt_pred, btr_pred,
            f_bt, amlt_bt, btr_f, conf, desc,
            xlim=(zoom_start, zoom_start + 5),
            suffix="zoom", category=cat
        )

    # ── Summary Analysis ──────────────────────────────────────────────────
    print("\n" + "="*80)
    print("ACTIVATION DIAGNOSTICS SUMMARY")
    print("="*80)

    summary_lines = []

    # Per-category means
    metric_cols = [c for c in diag_df.columns if c.startswith('bt_act_') or c.startswith('btr_act_')]
    categories = ['good', 'octave_error', 'continuity_drift', 'total_failure', 'other']

    for cat in categories:
        subset = diag_df[diag_df['category'] == cat]
        if len(subset) == 0:
            continue
        header = f"\n── {cat.upper()} (n={len(subset)}) ──"
        print(header)
        summary_lines.append(header)
        for col in metric_cols:
            mean_val = subset[col].mean()
            std_val = subset[col].std()
            line = f"  {col:40s}: {mean_val:8.4f} ± {std_val:.4f}"
            print(line)
            summary_lines.append(line)

    # Spearman correlations with F-measure
    print("\n── SPEARMAN CORRELATIONS: F-measure vs activation metrics (all 217 tracks) ──")
    summary_lines.append("\n── SPEARMAN CORRELATIONS: F-measure vs activation metrics (all 217 tracks) ──")
    for col in metric_cols:
        valid = diag_df[[col, 'bt_F']].dropna()
        if len(valid) > 5:
            rho, pval = spearmanr(valid['bt_F'], valid[col])
            line = f"  {col:40s}: rho={rho:+.3f}  p={pval:.2e}"
            print(line)
            summary_lines.append(line)

    # Failure mode classification summary
    fail_tracks = diag_df[diag_df['bt_failure_mode'] != '']
    if len(fail_tracks) > 0:
        header = f"\n── FAILURE MODE CLASSIFICATION (n={len(fail_tracks)} tracks with F<0.3, AMLt<0.3) ──"
        print(header)
        summary_lines.append(header)

        print("\n  beat_this failure modes:")
        summary_lines.append("\n  beat_this failure modes:")
        for mode, count in fail_tracks['bt_failure_mode'].value_counts().items():
            line = f"    {mode:30s}: {count:3d} ({100*count/len(fail_tracks):.1f}%)"
            print(line)
            summary_lines.append(line)

        print("\n  Beat Transformer failure modes:")
        summary_lines.append("\n  Beat Transformer failure modes:")
        for mode, count in fail_tracks['btr_failure_mode'].value_counts().items():
            line = f"    {mode:30s}: {count:3d} ({100*count/len(fail_tracks):.1f}%)"
            print(line)
            summary_lines.append(line)

        # Print individual failure track details
        print("\n  Per-track failure details (total_failure category):")
        summary_lines.append("\n  Per-track failure details (total_failure category):")
        for _, r in fail_tracks.iterrows():
            line = (f"    {r['track_id']:10s} bt_F={r['bt_F']:.3f} "
                    f"act_max={r['bt_act_max']:.3f} prom={r['bt_act_peak_prominence_mean']:.3f} "
                    f"gt_peak={r['bt_act_gt_peak_mean']:.3f} period={r['bt_act_periodicity']:.3f} "
                    f"mode={r['bt_failure_mode']}")
            print(line)
            summary_lines.append(line)

    # Key findings
    print("\n── KEY FINDINGS ──")
    summary_lines.append("\n── KEY FINDINGS ──")

    good_df = diag_df[diag_df['category'] == 'good']
    fail_df = diag_df[diag_df['category'] == 'total_failure']

    if len(good_df) > 0 and len(fail_df) > 0:
        comparisons = [
            ('bt_act_peak_prominence_mean', 'Peak prominence'),
            ('bt_act_periodicity', 'Periodicity'),
            ('bt_act_gt_peak_mean', 'Activation at GT beats'),
            ('bt_act_entropy', 'Entropy'),
            ('bt_act_max', 'Max activation'),
        ]
        for col, label in comparisons:
            g_mean = good_df[col].mean()
            f_mean = fail_df[col].mean()
            line = f"  {label:30s}: good={g_mean:.3f}  fail={f_mean:.3f}  ratio={f_mean/(g_mean+1e-8):.2f}"
            print(line)
            summary_lines.append(line)

    # Save summary
    summary_path = ROOT / "activation_diagnostics_summary.txt"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\nSaved summary to {summary_path}")

    # Print plot file list
    print(f"\nPlot files saved to {OUT_DIR}/")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
