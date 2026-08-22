# Overnight batch — six experiments

Launched 2026-08-21. All outputs under `/tmp/overnight/` (`.json`/`.csv` + `.log` beside it).
Every script is standalone under `scripts/`, writes results **incrementally**, and performs a
**self-validation** against known-good numbers before spending time on new work.

Env: `/home/sogang/mnt/db_2/anaconda3/envs/analyze-smc/bin/python`, run from
`/home/sogang/jaehoon/Analyze-SMC`. Shared stats helper: `scripts/ov_stats.py`
(paired Wilcoxon signed-rank + 10,000-resample paired bootstrap 95% CI on the mean difference).
Every row reports **F, CMLt, AMLt, CMLc, AMLc**.

Untouched, as required: `scripts/e1_soft_tempo_prior.py`, `/tmp/dither_em/`, `vbpm/`.

---

## A. TTA + wide DBN, no oracle  — **COMPLETE**

| | |
|---|---|
| script | `scripts/ov_a_tta_dbn.py` |
| launch | `setsid nohup $PY scripts/ov_a_tta_dbn.py > /tmp/overnight/A_tta_dbn.log 2>&1 < /dev/null &` |
| output | `/tmp/overnight/A_tta_dbn.json`, per-track `/tmp/overnight/A_tta_dbn.per_track.jsonl`, log `/tmp/overnight/A_tta_dbn.log` |
| validation | recompute TTA peak-pick F (want 0.6450), raw wide-DBN F (want 0.5851), raw peak-pick F (want 0.6270); abort if any is off by > 0.002 |
| status | **COMPLETE** (235 s). Validation PASSED: 0.6449 / 0.5851 / 0.6270 |

Inputs: `data/tta_full/*.npz` (`ts_beat`, `ts_db` = mean logits over rates {0.85, 1.0, 1.2}).
Decoder: `beats_per_bar=[3,4]`, 30–215 bpm, λ swept over {5, 20, 50, 100, 200}.

## B. min_bpm 30 vs 55, split at gt_bpm = 55 — **COMPLETE**

| | |
|---|---|
| script | `scripts/ov_b_minbpm_split.py` |
| launch | `setsid nohup $PY scripts/ov_b_minbpm_split.py > /tmp/overnight/B_minbpm_split.log 2>&1 < /dev/null &` |
| output | `/tmp/overnight/B_minbpm_split.json`, per-track `.per_track.jsonl`, log `.log` |
| validation | reproduce peak-pick (0.6270/0.5138/0.6100), dbn30 λ=100 (0.5851/0.5102/0.6735), dbn55 λ=100 (0.5764/0.4738/0.6556) to within 0.002 on F, CMLt and AMLt; those three configs run FIRST so the abort is early |
| status | **COMPLETE** (397 s). Validation PASSED |

12 DBN configs (min_bpm ∈ {30, 55} × λ ∈ {10, 20, 50, 100, 200, 500}) + peak-pick, each reported
pooled (217), representable `gt_bpm ≥ 55` (172) and unrepresentable `gt_bpm < 55` (45), with paired
tests inside every stratum. `gt_bpm` from `data/bt_transformer_tempo_constrained_results.csv`.

## C. TTA-instability confidence for per-track decoder routing — **COMPLETE**

| | |
|---|---|
| script | `scripts/ov_c_tta_confidence.py` |
| launch | `setsid nohup $PY scripts/ov_c_tta_confidence.py > /tmp/overnight/C_tta_confidence.log 2>&1 < /dev/null &` |
| output | `/tmp/overnight/C_tta_confidence.json`, per-track `.per_track.jsonl`, log `.log` |
| validation | recompute peak-pick F (0.6270) and wide-DBN F (0.5851) from this script's own code; abort if off by > 0.002 |
| status | **COMPLETE** (91 s). Validation PASSED |

Two likelihood-free instability scores, each on the mild {0.85,1,1.2} and the full 7-rate set:
`S1_var_at_peaks` (across-rate SD of the per-rate probability at the consensus peak frames) and
`S2_peakset_dist` (1 − mean pairwise F between per-rate peak sets). 21-point quantile threshold
sweep in both routing directions, Spearman ρ against the per-track (DBN − peak-pick) F gap, and the
recomputed oracle best-of-two envelope.

## D. Metronome ceiling across all corpora — **RUNNING**

| | |
|---|---|
| script | `scripts/ov_d_metronome_ceiling.py` |
| launch | `setsid nohup $PY scripts/ov_d_metronome_ceiling.py > /tmp/overnight/D_metronome_ceiling.log 2>&1 < /dev/null &` |
| output | `/tmp/overnight/D_metronome_ceiling.json` (per-corpus rows, rewritten after every corpus), `/tmp/overnight/D_metronome_ceiling.per_track.csv` (appended per track), log `.log` |
| validation | HARD: ballroom mean F must land within 0.02 of 0.984, checked after the first corpus, `sys.exit(2)` otherwise. hjdb (1.000), hainsworth (0.894), beatles (0.715), asap (0.164) are checked as recorded WARNINGS in `warnings` / `reference_delta` |
| status | **RUNNING**. ballroom already in: F = 0.9659, n = 685 → validation PASSED |

Annotations only, no inference. Search: coarse (period, phase) grid (0.20–4.00 s @ 10 ms; phase @ 20 ms),
then ICP-style least-squares refit of (period, phase) on the matched beats from the top-40 coarse seeds
plus seeds at median-IBI × {1/4, 1/3, 1/2, 1, 2, 3, 4}. Selection uses a vectorised F surrogate;
**reported numbers are recomputed with the harness `ev()`** (mir_eval, 5 s trim, ±70 ms).
Reports F/CMLt/AMLt/CMLc/AMLc, `frac_F_eq_1` and n per corpus.
Corpus order puts the reference corpora first: ballroom, hjdb, hainsworth, beatles, asap, then the rest.

*Caveat for the reader:* a search-based ceiling is a LOWER bound on the true ceiling. Where
`reference_delta` is negative the search, not the corpus, fell short — treat those rows as conservative.

## E. Tempo-prior sweep scored through beat metrics — **COMPLETE**

| | |
|---|---|
| script | `scripts/ov_e_prior_beat_metrics.py` |
| launch | `setsid nohup $PY scripts/ov_e_prior_beat_metrics.py > /tmp/overnight/E_prior_beat_metrics.log 2>&1 < /dev/null &` |
| output | `/tmp/overnight/E_prior_beat_metrics.json`, per-window `.per_window.jsonl`, log `.log` |
| validation | recompute exact-period accuracy (5 % tolerance) for weights 0/0.5/1.0/2.0 and require 81.37 / 77.54 / 73.01 / 69.59 within 0.5 pp on at least one split; abort otherwise |
| status | **COMPLETE** (156 s). Validation PASSED on the `gtzan` split: 81.37 / 77.54 / 73.01 / 69.59 |

Picks read from the already-saved `/home/sogang/jaehoon/VBPM/tempo_lab_{acf-argmax, acf+prior0.5/1.0/2.0}.npy`
(the weight-0 anchor is `acf-argmax`, NOT the published `acf-smallest` 79.3). Windows and activations from
`/home/sogang/jaehoon/VBPM/tempo_acts.npz` (50 fps, 2250 frames = 45 s). A pick is a BAR period, so beat
grids are bar/divisor for divisor ∈ {1,2,3,4}, all four reported. **Phase is not oracle** — it maximises the
mean BEAT-channel activation on the click grid. References are the corpus `.beats` files truncated to the
45 s analysis window. `vbpm/` was not modified or imported.

## F. Fold-honest Beat This activations for asap (only GPU job) — **COMPLETE**

| | |
|---|---|
| script | `scripts/ov_f_asap_activations.py` |
| launch | `setsid nohup $PY scripts/ov_f_asap_activations.py --device cuda:3 > /tmp/overnight/F_asap_activations.log 2>&1 < /dev/null &` |
| output | activations `/disk1/jaehoon/asap_bt_fold_acts/<stem>.npz` (keys `beat`, `downbeat` = logits @ 50 fps, `fold`); state `/tmp/overnight/F_asap_activations.json`; log `/tmp/overnight/F_asap_activations.log` |
| validation | before touching the GPU: split lists exactly 473 stems; every stem has a `<stem>/track.npy` entry in the spectrogram archive; the probe spectrogram is `[T, 128]`. `sys.exit(2)` with an explicit "input format unusable — skipping rather than improvising" message otherwise |
| status | **COMPLETE** (319 s on cuda:3, one device). Validation PASSED. 473/473 written, 0 failures |

Per-fold `Spect2Frames(checkpoint_path=f"fold{fold}")`, each track decoded only by the checkpoint whose
fold holds it out. Input is the precomputed log-mel cache `/disk1/jaehoon/beat_this_spect/done/asap.npz`
(`<stem>/track.npy`, `[T,128]` float16 @ 50 fps) — exactly what `Spect2Frames` consumes, so no audio decode
was needed. Output layout matches `beat_this_activations_cache/`, so the harness loaders work unchanged.
Restartable: existing `.npz` files are skipped.

---

## Headline numbers already on disk

**A** (SMC n=217, no ground-truth tempo anywhere):

| system | F | CMLt | AMLt | CMLc | AMLc |
|---|---|---|---|---|---|
| raw peak-pick | 0.6270 | 0.5138 | 0.6100 | 0.3655 | 0.4214 |
| raw + wide DBN λ100 | 0.5851 | 0.5102 | 0.6735 | 0.4162 | 0.5371 |
| TTA peak-pick | 0.6449 | 0.5504 | 0.6486 | 0.4146 | 0.4808 |
| **TTA + wide DBN λ100** | **0.6047** | **0.5413** | **0.6954** | **0.4509** | **0.5623** |

TTA+wide-DBN is the best AMLt/CMLc/AMLc row of the four and beats raw+DBN on every metric
(ΔF +0.0195, p = 0.12; ΔCMLt +0.0311, p = 0.074), but it loses F to TTA peak-picking
(ΔF −0.0403, p = 3.1e-6) at no CMLt gain (−0.0091, p = 0.66). λ sweep: F falls monotonically
with λ (0.6214 at λ=5 → 0.5833 at λ=200) while AMLt rises — the usual F/continuity trade.

**B** — the Simpson's paradox reproduces exactly. dbn55 λ100 vs peak-pick:

| stratum | n | ΔCMLt | p | ΔAMLt | p |
|---|---|---|---|---|---|
| pooled | 217 | −0.0400 | 0.101 | +0.0440 | 6.3e-4 |
| representable gt_bpm ≥ 55 | 172 | **+0.0438** | 0.030 | **+0.0543** | 5.8e-5 |
| unrepresentable gt_bpm < 55 | 45 | −0.3603 | 2.0e-7 | +0.0044 | 0.965 |

**C** — oracle per-track best-of-two envelope is **F = 0.6493** (DBN is the right call on 65/217),
not 0.691; both instability scores anti-correlate weakly with the (DBN − peak-pick) gap
(ρ ≈ −0.17 to −0.20) and recover only ~2–3 % of the oracle gain. Likelihood-free TTA instability
is **not** a usable routing signal as defined here.

**E** — the prior's harm is real in F and CMLt (bar/4 grid: F 0.5343 → 0.5210 → 0.5075 → 0.4964;
CMLt 0.5073 → 0.4767 → 0.4447 → 0.4188) and much weaker in AMLt (0.6077 → 0.6044 → 0.5954 → 0.5771),
confirming the prediction: the prior buys nothing an octave-invariant metric can see.
