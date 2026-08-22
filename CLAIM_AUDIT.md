# CLAIM_AUDIT.md — the ledger

Compiled 2026-08-21 from a multi-agent verification pass over `/home/sogang/jaehoon/Analyze-SMC`,
`/home/sogang/jaehoon/Analyze-SMC/vbpm/` (moved VBPM assets), `/home/sogang/jaehoon/VBPM`,
`/home/sogang/jaehoon/VBPM_reintegration`, and the VBPM memory store. Every number below is
marked CONFIRMED (recomputed or read from a surviving artifact), CONTRADICTED, REQUIRES-RERUN or
UNVERIFIABLE. Nothing under `vbpm/` was modified.

Late corrections from `LATE_CORRECTIONS.md` (written after the audit workflow launched) are merged
in and marked **[LATE]**. Corrections I established myself during synthesis are marked **[SYNTH]**.

---

## 1. BOTTOM LINE

**No — the ICASSP argument as currently worded cannot be made from verified numbers alone, but a
restated version of it can, and the restatement is stronger.** Clause 1 ("Beat This-style
factorized-loss models win F-measure but lose continuity") is CONFIRMED twice on real corpora with
paired confidence intervals. Clause 3 ("the madmom DBN's apparent cost is inflated by an
integer-frame quantisation artifact that can be repaired") is half-confirmed: the artifact is real
and I reproduced it on a machine and a day the original author did not choose, but **the repair is
a measured null** — on SMC (n=217) the dither-corrected mixture kernel moves F by +0.0050
(p=0.505), CMLt by +0.0076 (p=0.931) and AMLt by −0.0038 (p=0.584), and in-domain it is a dead tie
(0.9503/0.9188/0.9407 vs 0.9504/0.9191/0.9410).

Clause 2 ("the fix is an explicit probabilistic sequence model with latent tempo state") is the one
that must change. The evidence says the sequence model is **not** the missing piece — madmom already
is one, and on SMC it already delivers a deployable, no-oracle **+0.0618 AMLt (p=8.28e-07)** and
**+0.1179 AMLc (p=4.25e-12)** over peak-picking. What fails is the *hard, homogeneous, uncalibrated
tempo prior*: give the same decoder the ground-truth tempo band and CMLt jumps +0.1909 (p=2.2e-16);
give it an estimated band (correct on only 133/217 tracks) and CMLt falls to 0.0395 on the 83
misses — worse than having no prior at all (0.2799). The defensible thesis is therefore: **F is a
placement metric and hides a tracking failure; the decoder was never the problem; the homogeneous
tempo/metrical-level prior was, and the fix is a calibrated, soft, music-conditioned posterior over
tempo and metrical level inside an exact-inference sequence model.** Every number in that sentence
is in the SAFE LIST.

---

## 2. BUILD RECOMMENDATION

### Build this: a neural HMM with a **soft, audio-conditioned prior over the tempo/metrical-level state**, on the rungs_lab bar-pointer chassis.

Not a conditioned *transition kernel* — that axis is measured exhausted (see below). A conditioned
*state prior*, replacing madmom's hard `[min_bpm, max_bpm]` band with a temperature-scaled,
octave-smeared distribution that is marginalised exactly, never truncated.

**Why this one, decisively:**

1. **The chassis runs today and self-certifies.** `vbpm/rungs_lab/tests/test_structured_dp.py`
   executed 2026-08-20 in the `analyze-smc` env, CPU, ~40 s, unmodified:
   `forward LL: structured −852.486829 | dense −852.486829 | |gap| 0.00e+00`, `viterbi: 100.0% path
   match`, gradients finite through `−inf` transitions, `STRUCTURED_DP_CERTIFIED`. The
   blind suites pass too: `test_r2_em.py` 16/16 (471.78 s), `test_r3_generative_mixture.py` 15/15,
   `test_r4_5_emission.py` 9/9, `test_structured_dp.py` — 31/31 in one run, 371.39 s.
2. **The headroom is measured, large, and on the target corpus.** Perfect tempo prior vs no prior on
   SMC n=217: **+0.1909 CMLt (p=2.2e-16)**. On the 134 tracks where an *estimated* band happens to
   contain the true tempo, a fully deployable structured decoder with zero ground truth already wins
   **CMLt +0.0933 (p=6.88e-06)**, **AMLt +0.0635 (p=7.11e-05)**, recovering 97.1% of the oracle
   band's CMLt.
3. **The failure mode of every previous attempt is diagnosed and avoidable.** Prior attempts hardened
   a point tempo estimate into a support; on the 83 tracks where the band misses, CMLt is 0.0395 —
   an order of magnitude below both peak-picking (0.2883) and the unconstrained DBN (0.2799). A soft
   prior cannot do that, by construction. **This exact experiment has never been run.**
4. **Cost is trivial.** ~1,381 LOC reusable verbatim (`rungs/bar_pointer/` 508 + `r1_2016_dbn.py`
   279 + `base.py` 186 + `r2_em_dbn.py` 182 + `deployment.py` 30 + `mixture.py` 75 + `__init__.py`
   15 + `r3_conditioned_dbn.py` 151), ~150 LOC of new loader glue, dependencies numpy+torch only, no
   madmom on the path. The decisive gate (E1 in §5) is **CPU-only, 4–8 wall-clock hours, zero GPU**,
   so it cannot contend with the recertification agent.
5. **It is the only candidate that fits clause 2 honestly.** The chassis marginalises the full
   (bar-position × tempo) latent chain exactly *and* marginalises meter {3,4} by logsumexp over
   per-meter forwards, using no annotation of any kind. That is a real latent tempo state with real
   marginalisation, not a periodicity penalty.

**Ranking, all four, on (runs today) × (verified result) × (rebuild cost) × (fit):**

| # | method | runs today | verified result | rebuild cost | fit to argument | verdict |
|---|---|---|---|---|---|---|
| 1 | **Neural HMM, conditioned state prior on the bar-pointer chassis** | YES — DP certified today, |gap| 0.00e+00, 31/31 tests | Mixed but *informative*: the transition-conditioning arms are null (R3 spread 0.0015 F; R4 head degenerate, `lambda_t_std` exactly 0.0, zero-trunk decode bit-identical at 0.9471657091592964); the **state-prior** arm was never tried and has +0.1909 CMLt of measured headroom | ~1,381 LOC reused, ~150 written, <10 GPU-h total, gate is CPU-only | Exact marginalisation over a genuine latent tempo state; directly answers the restated clause 2 | **BUILD** |
| 2 | **Kalman VAE** | YES — I trained 300 steps + evaluated end-to-end (431 s on one A6000, in-domain F 0.6048 at 30% of the recorded budget) | ONE artifact-verified number: SMC F 0.654038, n=217, recomputed exactly from a committed per-song CSV — but on a `final0` frontend the project's own record calls contaminated; the full continuity table (SMC CMLt 0.587 / AMLt 0.616) is prose-only, checkpoint destroyed | ~1,359 LOC (714 copied MIT + 645 written), <10 GPU-h; **[SYNTH]** the 86 fps 512-d SMC feature caches survive (see §5), so only a retrain is needed | **Half-fit.** `z` is an unstructured 8-dim Euclidean vector; `A`/`C` are free matrices at `0.9·I + 0.1·randn`; K=5 corresponds to nothing musical; the read-out is `sigmoid(head(E[z]))`, i.e. the posterior covariance is computed and then thrown away. **No tempo state, no marginalisation over tempo.** Making it fit needs a switching LGSSM, which destroys the exact linear-Gaussian recursion the method rests on | **RUNNER-UP** |
| 3 | **R2 exact-EM global transition λ** | YES — module imports, EM converged live (55.76→46.70→43.90 in 3 iters, 20.5 s), 16/16 blind tests | Its headline **does not survive**: "unsupervised EM beats hand-set λ, 0.9132 vs 0.9033" was measured with madmom's peak-snap OFF; with `correct=True` (what madmom ships) R1 0.9622 vs R2 0.9623, ΔF +0.0000, 95% CI [−0.0015, +0.0014]. And on SMC its own objective anti-correlates with the oracle (ρ = −0.230, p = 6.42e-4) | ~1,100 LOC, ~0 GPU (8 EM iters on 24 SMC crops = 128 s CPU) | Learns exactly one scalar inside a decoder that already had the latent state. It is a **measured counterexample**, not a method — and a valuable one | KEEP AS CONTROL |
| 4 | **Dither-corrected mixture kernel** | YES — SMC row bit-reproduced today (4 tracks, ΔF = 0.000000 vs `results_smc3.json`) | The **diagnosis** is confirmed by rerun (interval 21.0 → λ_MLE 133.39; 21.4 → 39.68; 21.5 → 37.56, perfect alternation). The **repair** is a null: SMC ΔF +0.0050 (p=0.505), ΔCMLt +0.0076 (p=0.931), ΔAMLt −0.0038 (p=0.584), 110/217 decodes bit-identical; in-domain a dead tie | Cheapest of all: 75 LOC + the same chassis, ~25 min CPU per fit | Supports clause 3's diagnosis only. One paragraph + one figure in the paper | KEEP AS ONE PARAGRAPH |

**Condition under which we switch to the runner-up:** run E1 (§5, CPU, ≤8 h) with its
pre-registered criterion — some temperature `T` must beat *both* the hard band and the unconstrained
wide grid on CMLt at p<0.05. If **no** `T` does, the discrete integer-frame tempo grid itself is the
binding limit, softening the prior cannot help, and the route becomes a continuous latent-tempo
model — at which point the Kalman VAE line is the starting point. Budget honestly: that is a
switching-LGSSM redesign with intractable exact inference, **not** the 1,359-LOC port, and it should
not be started on four weeks' notice without E1's answer first.

**One thing to do regardless of which model wins.** The measured decoder-side envelopes say the
prize is bigger in continuity than in F, on the same tracks: decoder share of the SMC gap is 13.2%
on F, 24.9% on CMLt, 27.9% on AMLt (and 54.1% / 46.1% / 13.3% if the tempo band counts as a decoder
hyperparameter). Grade on CMLt/AMLt primary, F secondary, or you will select against your own
mechanism — which is exactly what the retired e2e experiment did.

---

## 3. THE SAFE LIST

Every number below is CONFIRMED: recomputed independently, or read verbatim from a surviving
artifact whose path is given. **Rows may only share a table if the harness column matches.**

### 3.1 Analyze-SMC, Beat This @50 fps, `scripts/_harness.py` (mir_eval, 5 s trim, ±70 ms), SMC n=217

**[LATE] This branch is fold-honest.** `SMC_Beat_Tracking_Analysis.ipynb` and `scripts/run_folds.py`
build `beat_this_activations_cache/` per fold from `beat_this_annotations/smc/8-folds.split`, so
every SMC track is predicted by a checkpoint that never saw it. F = 0.6270 is a fold-honest
reproduction of Beat This's published SMC 62.7. **The `final0` contamination warning does not apply
to any Analyze-SMC SMC number.**

| row | F | CMLt | AMLt | n | artifact | oracle? |
|---|---|---|---|---|---|---|
| peak-pick @0.5 (baseline) | 0.6270 | 0.5138 | 0.6100 | 217 | `beat_this_activations_cache/`; `data/results.csv`; `data/threshold_sweep_results.csv`; `vbpm/rungs_lab/results_smc3.json`; `smc3_ladder.log` | no |
| — also CMLc / AMLc | 0.3655 | — | — | 217 | AMLc 0.4214 | no |
| pitch-shift TTA @0.5 | 0.6327 | 0.5295 | 0.6256 | 217 | `data/tta_full/*.npz:pitch_beat`; `data/tta_activations/*.npz:avg_logits` | no |
| **time-stretch TTA @0.5 {0.85,1.0,1.2}** | **0.6449** | **0.5504** | **0.6486** | 217 | `data/tta_full/*.npz:ts_beat`; `data/ts_tta_cache/` | **no — best deployable row in the repo** |
| pitch+TS TTA @0.5 | 0.6411 | — | — | 217 | `data/tta_full` mean-of-logits | no |
| BT+BTransformer fusion, global thr 0.325 | 0.6352 | — | — | 217 | `scripts/exp_ensemble.py` | threshold tuned on the set |
| GT-tempo band ±20%, λ=100 | 0.6179 | 0.7000 | 0.7290 | 217 | `data/gt_tempo_lambda_sweep_results.csv` (`default_lambda_*`) | **GT tempo** |
| GT-tempo band + per-track opt λ | 0.6665 | 0.7345 | 0.7551 | 217 | same CSV (`optimal_*`); mean opt λ 69.11 | **double oracle** |
| TS-TTA + GT-tempo + opt λ | 0.6812 | 0.7532 | 0.7726 | 217 | `scripts/eval_tta_oracle_tempo.py`, rerun end-to-end | **double oracle** |
| pitch TTA + GT-tempo + opt λ | 0.6722 | 0.7386 | 0.7643 | 217 | same rerun | double oracle |
| pitch+TS + GT-tempo + opt λ | 0.6800 | 0.7458 | 0.7681 | 217 | same rerun | double oracle |
| peak-pick @ per-track oracle threshold | 0.6696 | — | — | 217 | `scripts/exp_decoder_ceiling.py`, rerun | oracle threshold |
| wide DBN [30,215] @ per-track oracle λ | 0.6525 | — | — | 217 | same rerun (script prints "(paper 0.642)") | oracle λ |
| **per-track envelope of the two** | **0.6908** | — | — | 217 | same rerun | oracle |
| — DBN-oracle beats threshold-oracle on | **78/217** | — | — | — | same rerun | — |
| fusion @ per-track oracle threshold | 0.6856 | — | — | 217 | `scripts/exp_ensemble.py` | oracle |
| per-track oracle stretch rate (7 rates) | 0.7119 | — | — | 217 | `data/ts_tta_cache/` (1519 npz = 217×7) | oracle |
| **[SYNTH]** interaction of TTA × oracle-tempo | −0.0032 | — | — | 217 | additive expectation +0.0574 vs measured +0.0542 | — |

Realistic-decoder table (`scripts/eval_tta_realistic.py`, all four sets rerun, CPU ~15 min/set):

| activation set | peak-pick F | wide DBN best-λ F | BT-predicted-tempo DBN best-λ F |
|---|---|---|---|
| raw | 0.6270 | 0.6001 (λ=10) | 0.5651 (λ=30) |
| pitch TTA | 0.6327 | 0.6064 (λ=5) | 0.5736 (λ=20) |
| time-stretch TTA | 0.6449 | 0.6214 (λ=5) | 0.5828 (λ=50) |
| pitch+TS | **0.6411** (not 0.644) | 0.6201 (λ=10) | 0.5795 (λ=30) |

*Not in the source table but measured in the rerun and materially on-thesis:* wide-DBN **AMLt beats
peak-pick on every set** (e.g. TS 0.6954 vs 0.6486) while CMLt is roughly a tie; wide DBN at λ=100
loses F on every set (raw 0.5851, pitch 0.5914, ts 0.6047, both 0.5993). The document's F-only table
hides the AMLt win.

Per-band time-stretch competence (peak-pick F, banded by GT median-IBI BPM, rates 0.7/0.85/1.0/1.2/1.5):

| band | n | 0.7 | 0.85 | 1.0 | 1.2 | 1.5 |
|---|---|---|---|---|---|---|
| <55 BPM | 45 | 0.4473 | 0.4948 | 0.5365 | **0.5624** | 0.5337 |
| 55–90 | 110 | 0.5339 | 0.5983 | 0.6226 | **0.6253** | 0.5801 |
| >90 | 62 | 0.7045 | **0.7350** | 0.7004 | 0.6783 | 0.6283 |

Rate-1.0 un-warping identity check: 0.6270 vs cached baseline 0.6270, |diff| = 0.0000.

TTA gain by band (ΔF vs baseline; total ΔF = +0.0180):

| band | n | base | TTA | ΔF | share | +/=/− |
|---|---|---|---|---|---|---|
| 0–55 | 45 | 0.5365 | 0.5691 | +0.0326 | 37.6% | 28/3/14 |
| 55–70 | 58 | 0.5865 | 0.6218 | +0.0354 | 52.6% | 38/4/16 |
| 70–90 | 52 | 0.6629 | 0.6621 | −0.0009 | −1.1% | 23/6/23 |
| 90–120 | 44 | 0.7374 | 0.7431 | +0.0057 | 6.4% | 19/14/11 |
| 120+ | 18 | 0.6098 | 0.6198 | +0.0100 | 4.6% | 10/3/5 |

<64 BPM share 74.0%; <70 BPM share 90.2%; tracks made worse 69/217 = 31.8%; Spearman(baseline F, ΔF)
= −0.0749, p = 0.272 (a null — phrase as "no detectable relationship", not as mechanism).

SMC corpus statistics, recomputed from the raw annotations (`SMC_MIREX_Annotations_05_08_2014/`,
tempo = 60/median(diff(beats))): min 32.7082, p5 36.8815, Q1 57.3100, **median 70.9200**, Q3 92.7400,
p95 139.6825, max 206.8966 BPM; **45/217 (20.74%) below 55 BPM**, 61/217 (28.11%) below 60; 0 above
215. Independently corroborated: `results_smc_diagnosis.json` stores `bpm_below55 = 45`.

Other confirmed in-repo statistics: Spearman(mean BT activation at GT beats, per-track F) = **0.7839,
p = 2.136e-46**, n=217 (`data/activation_diagnostics.csv`). Spearman(BT per-track F, BTransformer
per-track F) = **0.7569, p = 1.385e-41** — *use the stored per-track `bt_F`/`btr_F` columns; a fresh
peak-pick at p=0.5 gives 0.624, a different quantity.* Error taxonomy over 217 tracks in
`data/transition_lambda_sweep_results.csv`: other 89 / good 62 / continuity_drift 38 / octave_error
17 / total_failure 11. Column means of the same CSV: F_lam5 0.6027, F_lam100 0.5917, optimal_F 0.6422.

### 3.2 rungs_lab SMC ladder — Beat This activations, pinned to `scripts/_harness.py`, n=217

`vbpm/rungs_lab/results_smc3.json` (217 per-song records) + `smc3_ladder.log`. Re-aggregated
independently; all 21 figures reproduce to 4 dp, and the peak-pick row is bit-identical to
Analyze-SMC's own harness — **a genuine cross-harness certificate for that one row.**

| arm | F | CMLt | AMLt | CMLc | AMLc |
|---|---|---|---|---|---|
| peak-pick | 0.6270 | 0.5138 | 0.6100 | 0.3655 | 0.4214 |
| fixed λ=100 | 0.5827 | 0.5091 | 0.6718 | 0.4192 | 0.5393 |
| R2mix (w=0.390, λ=104.43) | 0.5878 | 0.5166 | 0.6680 | 0.4206 | 0.5293 |
| R3 w_t | 0.5875 | 0.5174 | 0.6679 | — | — |
| R3 λ_t | 0.6008 | 0.4960 | 0.5815 | — | — |
| ML per-song λ (label-free) | 0.6021 | 0.4886 | 0.5618 | 0.3252 | 0.3707 |
| ORACLE per-song λ | 0.6495 | 0.5801 | 0.6584 | 0.4473 | 0.4991 |

Derived, all recomputed: DBN(λ=100) − peak = −0.0442 F / +0.0618 AMLt; fixed-100 worse on F than
peak on **122/217** and better on AMLt on **122/217** (two different comparisons landing on the same
count — verified, not a copy error). ML − fixed100 = **+0.0194 F, −0.0205 CMLt, −0.1100 AMLt** (5.67×
understatement by F). R3_lamt − fixed100 = **+0.0181 F, −0.0131 CMLt, −0.0903 AMLt** (4.99×).
ML recovers 29.04% of the fixed→oracle F gap = (0.6021−0.5827)/(0.6495−0.5827).
Spearman(log λ_ML, log λ_ORACLE) = **−0.2299, p = 6.416e-4** — λ_ML picks the grid floor (λ=1) on
116/217 tracks (53.5%) and λ≤2 on 141/217 (65.0%). λ_ML quartiles [1, 1, 1, 10, **500**];
λ_OR [1, 1, 10, 50, **500**].

Strata (same file, same harness):

| stratum | n | peak F/CMLt/AMLt | fixed100 F/CMLt/AMLt | ORACLE F/CMLt/AMLt |
|---|---|---|---|---|
| bpm < 55 | 45 | 0.5365 / 0.4130 / 0.5527 | 0.4690 / **0.3428** / **0.6255** | 0.5517 / 0.4726 / 0.5799 |
| bpm ≥ 55 | 172 | 0.6506 / 0.5402 / 0.6250 | 0.6125 / 0.5526 / 0.6839 | 0.6751 / 0.6082 / 0.6789 |
| tempo-change high | 72 | 0.6258 / 0.4226 / 0.5235 | 0.5388 / 0.3581 / 0.5224 | 0.6356 / 0.4555 / 0.5347 |
| tempo-change low | 145 | 0.6275 / 0.5591 / 0.6530 | 0.6046 / 0.5840 / 0.7460 | 0.6563 / 0.6420 / 0.7198 |

The bpm<55 row is the single cleanest octave-error signature in the inventory: **−0.0702 CMLt with
+0.0728 AMLt** — the damage is invisible to AMLt and largely invisible to F.

λ_oracle-stratified (`PHASE0_RECERT.md` §4.2–4.3, same JSON): λ_oracle==1 on **73/217 (33.6%)**,
λ_oracle≤2 on 84 (38.7%), λ_oracle≥100 on 34; only 2 tracks censored at the 500 ceiling. On the 34
stiff-kernel tracks the oracle wins **+0.0849 F / +0.1712 CMLt / +0.1588 AMLt**; on the 73 floor
tracks it **loses** −0.0096 F / −0.0181 CMLt / −0.0284 AMLt. Quote both or a reviewer will find the
selection effect. Spearman(λ_oracle, median BPM) = 0.019, p = 0.786; (λ_oracle, max tempo change) =
0.098, p = 0.15 — both null, so the mode is not predictable from observable tempo statistics.

Illustration only (n=1): `smc_001`, 48.15 BPM — peak-pick 0.5946/0.3696/0.3696 vs fixed-100
0.4107/**0.0000**/**0.0000**; R2mix, R3_wt, R3_lamt and ML are all 0.0 on every continuity metric for
this track. **`max_tempo_change` is in BPM units, not a ratio: this track's value is 1.57 BPM, not
1.57×.**

### 3.3 rungs_lab Beat Transformer branch @43.066 fps — DO NOT MIX WITH §3.1/§3.2

`vbpm/rungs_lab/results_smc.json` and `results_smc_diagnosis.json`, n=217, rungs_lab `score()`.

| config | F | CMLt | AMLt |
|---|---|---|---|
| R0_shipped (madmom as BT ships it) | 0.6265706943842514 | 0.5368910103589458 | 0.7110498000164516 |
| R1_matched_shipped | **bit-identical to R0 on all 16 digits** | | |
| R1_obs16_bare | 0.5955581605719443 | 0.5296284792510799 | 0.6992232074140610 |
| R1_obs6_shipped (**best F**) | 0.6389067314126663 | 0.5244879387699007 | 0.6906823452616172 |
| R1_obs6_thr02 (the crippled "R1") | 0.4689991007652502 | 0.5010427574813001 | 0.6569797488504635 |
| R1_wide30_shipped | 0.6276911812581689 | 0.5591476194921321 | 0.7011559315214574 |
| R1_wide30_notempi (**best CMLt**) | 0.6327820947666689 | 0.5666931785584285 | 0.7079037809514055 |

Best-F config vs shipped: **+0.0123 F, −0.0124 CMLt, −0.0204 AMLt** — F and continuity select
different configurations. `wide30_notempi` vs shipped: +0.0062 F, **+0.0298 CMLt**, −0.0031 AMLt.
Engine certificate: **216/217 SMC tracks event-identical to madmom** (`certificate = [216, 217]`).

BPM-extreme strata (`results_smc.json`): R0_shipped on ≤p15 or ≥p85, n=66: 0.5263/0.2477/0.6321;
bpm_mid n=151: 0.6704/0.6633/0.7456. F drops 0.1441 between strata, **CMLt drops 0.4156 — a 2.88×
sensitivity ratio.** Note these strata are SMC-internal percentiles, a different cut from the
absolute <55 BPM band (n=45); do not conflate n=66 and n=45.

Correlations in the same file (9 uncorrected tests; Bonferroni α = 0.00556, **only one survives**):
R3lamt−R1 vs bpm_dist_from_median ρ = 0.2450 (p = 2.679e-4) ✓; vs vol ρ = 0.1790 (p = 0.00823) ✗;
vs max_tempo_change ρ = 0.1256 (p = 0.0647) ✗; R2mix−R1 vs max_tempo_change ρ = 0.0746 (p = 0.274) ✗.

Campaign, 8 folds × 3 seeds = 24 runs (`results_campaign_r2mix.json`): n_val by fold
{0:146, 1:139, 2:148, 3:147, 4:150, 5:149, 6:142, 7:145}; w ∈ [0.3500, 0.4500] (**tight**);
λ ∈ [57.2042, 131.3456] (**not tight** — fold 1's three seeds give 80.7 / 117.1 / 62.3, a 1.9×
spread); deploy−bare beat F **+0.0134 mean, range +0.0115…+0.0157** (the best-replicated effect in
the inventory); deploy AMLt max **0.9721**; `smc_deploy` beat F 0.3992–0.5239, mean 0.4834 — the
mixture chassis never approaches R0_shipped 0.6266 on SMC in any of the 24 runs.

### 3.4 bt_e2e fold-0 val, n=146 (ballroom/beatles/hainsworth/hjdb) — rungs_lab decode, prose-only

Split: 1,020 train / 146 val; 139 songs skipped for no meter/grid representation; 1305 stems in the
four datasets (ballroom 672 + beatles 179 + hainsworth 222 + hjdb 232 — **[SYNTH]** counted directly
in the demix cache, and 1020+146+139 = 1305 exactly).

Continuity tiebreaker (`tiebreak_continuity.py`; **metric names verified correct at the source
blob**, see §7). Columns: beat F / CMLt / AMLt | db F / dbCMLt / dbAMLt.

| row | beat F | CMLt | AMLt | db F | dbCMLt | dbAMLt |
|---|---|---|---|---|---|---|
| DEPLOYED λ=100 hand-set | 0.9504 | 0.9191 | 0.9410 | 0.9057 | 0.8842 | 0.9366 |
| DEPLOYED CRF λ=98.6 | 0.9504 | 0.9191 | 0.9410 | 0.9057 | 0.8842 | 0.9366 |
| DEPLOYED MIXTURE w=.37 λ=93.1 | 0.9503 | 0.9188 | 0.9407 | 0.9053 | 0.8845 | 0.9369 |
| DEPLOYED em-single λ=40.3 | 0.9494 | 0.9144 | 0.9340 | 0.8924 | 0.8671 | 0.9154 |
| DEPLOYED learn-obs λ=30.9 o=4 | 0.9457 | 0.9034 | 0.9245 | 0.8876 | 0.8610 | 0.9089 |
| BARE λ=100 | 0.9114 | 0.9107 | 0.9325 | 0.8677 | 0.8831 | 0.9359 |
| BARE MIXTURE | 0.9193 | 0.9145 | 0.9363 | 0.8744 | 0.8843 | 0.9372 |
| BARE em-single λ=40.3 | 0.9180 | 0.9079 | 0.9274 | 0.8616 | 0.8682 | 0.9164 |

The two dissociations, measured on the same rows of the same table — this pairing is what makes
"F and continuity come apart" a finding rather than a metric artifact:

- **Loose kernel:** λ=40.3 costs **−0.0212 dbAMLt** while beat F moves **−0.0010** — a 21× ratio.
  learn-obs costs −0.0277 dbAMLt for −0.0047 F — 5.9×. *(Prefer the λ=40.3 row; and note the "21×"
  divides a downbeat-AMLt delta by a beat-F delta. Same-metric-family statements: dbAMLt is 3.0×
  worse than beat AMLt (−0.0212 vs −0.0070); dbAMLt is 1.6× the dbF cost (−0.0212 vs −0.0133).)*
- **Deployment wrappers (inverse):** bare→deployed buys **+0.0390 beat F** but only **+0.0084 CMLt**
  and **+0.0011 dbCMLt** — a 4.6× asymmetry. The wrappers repair *placement*, not *tracking*.

R4.5 rich-emission rows, same fold, same tooling: v0 DEPLOYED 0.9331/0.8768/0.8955 | 0.8635/0.8347/
0.8736; v1 DEPLOYED 0.9392/0.8840/0.9420 | 0.8826/0.8496/0.9096; v2 (LDA) 0.9382/0.8817/0.9281 |
0.8791/0.8465/0.9044; v1 BARE 0.9146/0.8776/0.9354 | 0.8554/0.8490/0.9091. v1 loses **−0.0351 CMLt
against −0.0112 beat F** (3×) while its **AMLt (0.9420) exceeds the reference (0.9410)** — the
signature of tracking continuously at the wrong metrical level.

Mechanism figures for the same line: emission gaps **22,600 nats** vs transition scores 0.2–272;
uniform-kernel decode ties the real kernel (ΔF 0.004, 91% frame-argmax agreement); 96/256 dims at the
1e-3 variance floor giving a +660-nat log-det bonus; v1 fix → 4.2 nats, v2 → 10.2 nats; validity gate
(real-vs-uniform ΔF) +0.0152 / +0.0098; EM movement ~85 nats; float32 DP noise ~0.5 nat/song.

Training-history artifacts (`experiments/bt_e2e/*_history.json`, all four read):

| run | best beat F | best db F | epoch | λ |
|---|---|---|---|---|
| vanilla_history_prelim | 0.9612674904550522 | 0.9395877314063663 | 21 | 100.0 |
| r2_history_prelim | 0.9470625179951031 | 0.9199828335448514 | 23 | **14.226956367492676** |
| vanilla_history (rematch) | 0.9657925959528175 | 0.9280965497485181 | 29 | 100.0 |
| r2_history (rematch) | 0.9115376639931124 | 0.8573449021883721 | 19 | **101.03919982910156** |

**The rematch RESULTS.md calls "launched, never reported" exists in artifact form.** At the corrected
observation model r2's learned λ converges near madmom's 100, and the arm loses *harder*
(−0.054 vs −0.006). But: these are 60-song training-time evals; the bias is **+0.0107 (vanilla) /
+0.0097 (r2)** vs the full 146-song fold; and **[SYNTH]** the first 60 val songs are **60/60
ballroom**. No epoch of any of the four histories logs CMLt or AMLt.

LR-anneal asymmetry, recovered from the same JSONs' per-epoch `lr` field: vanilla 1.00e-03 for
epochs 0–9 then **2.00e-04** from epoch 10; r2 **1.00e-03 for all 30 epochs, never annealed**.
Vanilla's val beat F gained +0.001 over epochs 1–9 and **+0.016 over epochs 11–23** — 16× more after
the anneal. `train_bt.py:213` `ReduceLROnPlateau(factor=0.2)` gives exactly 1e-3 → 2e-4; the in-code
comment at :283 names the bug. Epoch wall time: vanilla 71–75 s, r2 415–448 s (5.8×).

Oracle decode ceiling (`oracle_ceiling.py`, 12 variants + half-bar flip, n=146): beat F shipped
0.9504 → oracle 0.9596 (**+0.0092**); downbeat F 0.9057 → 0.9533 (**+0.0476**). *In-domain only, and
**SMC has no downbeat annotations**, so the downbeat prize cannot be claimed there.*

### 3.5 Cross-corpus, n=2304 — the strongest clause-1 artifact

`results_baseline.json` (R0 madmom shipped) vs `results_btminimal.json` (Beat This's own minimal
postprocessor), identical 2304 stems, both producers verified to unpack mir_eval continuity at
indices 1 and 3. Per-dataset means, beat F / CMLt / AMLt | db F / dbCMLt / dbAMLt:

| dataset | n | BTmin | R0 (DBN) |
|---|---|---|---|
| gtzan | 999 | 0.8909 / 0.7969 / 0.9006 \| 0.7823 / **0.6736** / **0.7943** | 0.8786 / 0.8023 / 0.9103 \| 0.7707 / **0.7297** / **0.8791** |
| ballroom | 672 | 0.9748 / 0.9634 / 0.9700 \| 0.9533 / 0.9313 / 0.9430 | 0.9684 / 0.9548 / 0.9707 \| 0.9551 / 0.9523 / 0.9685 |
| beatles | 179 | 0.9446 / 0.8707 / 0.9303 \| 0.8892 / 0.7407 / 0.8229 | 0.9380 / 0.8721 / 0.9311 \| 0.8780 / 0.7899 / 0.8785 |
| hainsworth | 222 | 0.9184 / 0.8400 / 0.9090 \| 0.8003 / **0.6362** / 0.7509 | 0.9058 / 0.8409 / 0.9070 \| 0.8044 / **0.7597** / 0.8478 |
| hjdb | 232 | 0.9830 / 0.9741 / 0.9807 \| 0.9667 / 0.9483 / 0.9565 | 0.9828 / 0.9771 / 0.9846 \| 0.9665 / 0.9669 / 0.9770 |

**Beat F up on 5/5 datasets; downbeat continuity down on 5/5.** gtzan +0.0123 F / −0.0561 dbCMLt /
−0.0848 dbAMLt; hainsworth +0.0126 F / −0.1235 dbCMLt / −0.0969 dbAMLt; beatles +0.0066 F /
−0.0492 dbCMLt / −0.0556 dbAMLt. Caveats: pooled mean over mixed checkpoints, **no SMC row**, not
re-run under `scripts/_harness.py`.

R0 == R1 to 4 dp on all 2304 rows; **[SYNTH] max absolute difference across all six metrics on all
rows is exactly 0.0 — bit-identical.** That is equally consistent with genuine parity and with an
accidental dict copy; cite the 216/217 event-identity certificate as the proof, this as corroboration.

### 3.6 tchain re-score — a learned structured model, GTZAN **downbeats**, n=993 (`TCHAIN_RESCORE.md`)

Re-scored from surviving checkpoints this month, one harness, fold-honesty established (GTZAN is
test-only; measured train/val/test overlap = 0 songs), paired per-song CIs. **All numbers are
downbeat numbers; SMC is beat-only and was correctly not evaluated.**

| system | db F | db CMLt | db AMLt |
|---|---|---|---|
| tchain_true | 0.7470 | 0.7038 | 0.8489 |
| tchain_s0 | 0.7498 | 0.7011 | 0.8483 |
| tchain_s1 | 0.7533 | 0.7017 | 0.8484 |
| Beat This final0, minimal | 0.7844 | 0.6717 | 0.7910 |
| Beat This, published, full songs | 0.783 | 0.673 | 0.791 |

Paired deltas (tchain − Beat This), mean ± 95% CI, n=993: `tchain_s0` **ΔF −0.0347 ± 0.0055,
ΔCMLt +0.0294 ± 0.0121, ΔAMLt +0.0573 ± 0.0101**; `tchain_s1` −0.0311 / +0.0300 / +0.0574;
`tchain_true` −0.0374 / +0.0321 / +0.0579. **All three exclude zero comfortably.** Seed-to-seed
spread of the corpus mean at n=993: 0.0035 F, 0.0006 CMLt, 0.0001 AMLt — so the +0.03 CMLt advantage
is ~50× the seed noise. This is the best-instrumented instance of clause 1 + clause 2 in the whole
inventory.

### 3.7 Forensics — new measurements made during this audit

Metronome ceilings (oracle constant period × oracle phase, 481×241 grid, 5 s trim, ±70 ms; fast F
validated exact against `mir_eval.beat.f_measure` on 8 spot checks):

| set | n | F | CMLt | AMLt | frac(F=1.0) |
|---|---|---|---|---|---|
| bt_e2e rematch eval set (`val_entries[:60]`, **60/60 ballroom**) | 60 | **0.9843** | 0.9597 | 0.9597 | 75.0% |
| ballroom (fold-0 subset) | 80 | 0.9866 | 0.9516 | 0.9516 | 78.7% |
| beatles | 20 | 0.7154 | 0.7523 | 0.7523 | 0.0% |
| hainsworth | 25 | 0.8942 | 0.9026 | 0.9026 | 52.0% |
| hjdb | 28 | 1.0000 | 0.9845 | 0.9845 | 100% |
| pooled reconstruction | 153 | 0.9385 | **0.9235** | 0.9235 | 68.0% |

**An oracle metronome's pooled CMLt (0.9235) exceeds the full deployed BT+madmom pipeline's CMLt
(0.9191) on this fold.** ballroom+hjdb = 108/153 = 70.6% of the fold. This is a protocol result in
its own right and it invalidates every in-domain "structure does not help" verdict.

Structured-vs-peak-pick on SMC under one harness, paired (Wilcoxon + 10k bootstrap), n=217, wide DBN
[30,215] λ=100 minus peak-pick @0.5:

| metric | Δ | 95% CI | p | win/loss/tie |
|---|---|---|---|---|
| F | −0.0442 | [−0.0604, −0.0283] | 4.22e-07 | 64/122/31 |
| CMLt | −0.0048 | [−0.0403, +0.0314] | **0.923 (NULL, not a loss)** | 86/75/56 |
| AMLt | **+0.0618** | [+0.0346, +0.0893] | **8.28e-07** | 122/50/45 |
| CMLc | +0.0537 | [+0.0207, +0.0874] | 0.0217 | 85/72/60 |
| AMLc | **+0.1179** | [+0.0862, +0.1506] | **4.25e-12** | 127/44/46 |

Off-the-shelf madmom (min_bpm=55) split by tempo representability: **bpm≥55, n=172 — F −0.0276
(p=1.06e-03), CMLt +0.0439 (p=0.0284), AMLt +0.0544 (p=5.51e-05)**, i.e. structured inference with
no oracle beats peak-picking on *both* continuity metrics; bpm<55, n=45 — CMLt −0.3607 (p=2.02e-07)
with AMLt +0.0119 (p=0.731, null). Pooled CMLt −0.0400 (p=0.121) is Simpson's paradox.

Tempo-prior isolation: GT-band λ=100 − peak = −0.0091 F (p=0.686, null) / **+0.1862 CMLt
(p=4.3e-18)** / +0.1190 AMLt; GT-band − wide DBN = +0.0352 F / **+0.1909 CMLt (p=2.2e-16)** /
+0.0573 AMLt. **102.5% of the CMLt dissociation is the tempo oracle; ~0% is structured decoding per
se.** On AMLt the deployable wide DBN already captures 51.9% of the oracle's gain.

Beat Transformer tempo head on SMC, n=217: acc1 (±4%) 43.8%, octave-tolerant acc2 59.0%, GT tempo
inside the predicted ±20% band on only **133–134/217 = 61.3%** (33.3% on the 45 slow tracks).
Split by band correctness — peak / wide DBN / BT-band DBN / GT-band DBN:

| subset | n | peak | wide DBN | BT-band DBN | GT-band DBN |
|---|---|---|---|---|---|
| band contains GT | 134 | 0.6887/0.6535/0.7008 | 0.6480/0.6510/0.7476 | 0.6610/**0.7467**/0.7643 | 0.6728/0.7687/0.7851 |
| band misses GT | 83 | 0.5274/0.2883/0.4634 | 0.4774/0.2799/0.5494 | 0.3873/**0.0395**/0.4394 | 0.5293/0.5890/0.6385 |

A **perfect binary gate** over {estimated band, wide grid} caps at F 0.5908 / CMLt 0.5682 / AMLt
0.6821, i.e. only 29.2% of the oracle CMLt gain. Band-correctness is weakly predictable from the
tempo head's entropy (AUC 0.655, p=9.85e-05) and max-prob (AUC 0.645). **Do not promise CMLt 0.70.**

Deployable best found with no ground truth anywhere: TS-TTA activations + wide DBN λ=100 vs raw
peak-pick, n=217 — F −0.0223 (p=0.00965), **CMLt +0.0275 (p=0.0809, marginal)**, **AMLt +0.0854
(p=5.65e-09)**. The levers compound on AMLt: 0.6100 → 0.6486 (TTA) → 0.6954 (TTA+DBN), the DBN
adding +0.0468 on top of TTA (p=1.85e-04).

AMLt-win decomposition (rules out octave laundering): 94.5% of the wide DBN's +0.0618 AMLt lands on
the 86 tracks where it *also* wins CMLt (mean +0.1474); only 3.5% on the 75 where it loses CMLt.

Degenerate wrong-level locks (CMLt<0.05 with AMLt>0.5): DBN 35/217 vs peak-pick 16/217; AMLt−CMLt
gap widens 0.0962 → 0.1627, and collapses to 0.0290 once GT tempo is supplied.

Difficulty split (SMC annotators' own flag): hard n=198 — AMLt +0.0616 (p=3.04e-06), CMLt −0.0081
(p=0.768), F −0.0492 (p=2.02e-07). The win is on the hard majority, not an easy-fold ceiling.

Corrected octave-prior sweep (**the published baseline row was the wrong rule**): the prior family's
true w=0 is `score.argmax`, not the shipped smallest-peak rule. gtzan n=993, exact-period accuracy at
±5%: **81.37% (w=0) → 77.54 → 73.01 → 69.59** at w = 0.5/1.0/2.0; halving **8.06 → 10.17 → 13.19 →
14.90%**; doubling (never reported) **4.73 → 6.45 → 7.85 → 7.55%** — the prior raises *both* octave
errors. McNemar exact vs the correct baseline: p = 5.34e-4 / 3.38e-10 / 4.53e-16. Bootstrap 95% CI on
the w=1 delta: [−10.98, −5.84] pp. Replicated on a discrete-ladder family (80.36 → 77.44 → 74.22,
p = 3.46e-3 / 1.19e-7), so it is not a grid-drift artifact. **Closed-form mechanism:** with
`LOG_MU = 0.6042`, `LOG_SD = 0.3808`, the prior prefers the halved candidate above
`exp(mu + ln2/2) = 2.5877 s` and the doubled candidate below `exp(mu − ln2/2) = 1.2939 s`. All 82
gtzan songs the prior flips to halved lie above the first threshold (100.0%); all 44 it flips to
doubled lie below the second (100.0%). In-band accuracy rises 87.9 → 94.3%; out-of-band collapses
65.9 → **23.0%**. **Best conceivable song-independent prior**, a 26-parameter nonparametric f(log p)
fit *directly on the gtzan test set*: 83.18%, i.e. +1.81 of the 15.11 pp to the 96.48% ladder
ceiling (12.0%), transferring **+0.00 pp** to held-out validation.

Annotation-path arithmetic behind the dithering artifact, model-free, over all fold-0 songs
(17,219 consecutive beat-interval ratios): RAW `round(t·fps)` mean|r−1| = 0.03754 with
frac(r == 1 exactly) = 0.4039 (1/mean = 26.6); after `--smooth`, mean|r−1| = 0.00845 with
**frac(r == 1) = 0.8537** (1/mean = 118.3). Because `exp(−λ|r−1|)` row-normalised is monotone
increasing in λ on such a path, **there is no interior maximum** — which is why the rematch's λ
climbed +0.4785/epoch over epochs 20–29 (p = 5.11e-03) and never converged.

Sub-frame tempo-increment kurtosis, recomputed from SMC annotations in seconds (10,267 increments,
217 tracks): excess kurtosis **42.10**, mean −0.00074, sd 0.1461, 8.29% of increments with |Δ|>0.2.
The heavy-tail *direction* survives sub-frame re-derivation; the retracted "~12.6" value does not.

### 3.8 Synthetic quantisation probe — rerun 2026-08-20 on a machine the author did not choose

`experiments/bt_e2e/synthetic_grid_probe.py` (copy at `/tmp/sgp2.py`, `vbpm/` untouched), cuda:1,
~13 min:

| interval (frames) | λ_MLE (claimed) | λ_MLE (rerun) | decoded path |
|---|---|---|---|
| 21.0 | 133.4 | **133.39** | flat at 21 — histogram {21: 30, 28: 1, 35: 1}, i.e. **two start-edge intervals the record omits** |
| 21.4 | 39.7 | **39.68** | dither {21: 18, 22: 13} = 58/42 |
| 21.5 | 37.6 | **37.56** | perfect alternation 22,21,22,21 |

EM trace at 21.0: 119.85 / 129.68 / 132.63 / 133.25 / 133.36 / 133.39. **The load-bearing leg of
clause 3 now has a fresh artifact.** Note `/tmp/sgp2.out` is ephemeral — copy it into the repo before
citing. A CPU run of the identical probe did not finish in 50 minutes; this needs a GPU.

### 3.9 Code capability — verified by execution, not by docstring

| asset | status | evidence |
|---|---|---|
| `rungs/bar_pointer/structured_dp.py` (216 LOC) | **RUNS, self-certifies** | `tests/test_structured_dp.py` in `analyze-smc`, CPU, ~40 s: forward-LL \|gap\| 0.00e+00, Viterbi 100% path match, compact emission \|gap\| 0.00e+00, grad finite with −inf entries, `STRUCTURED_DP_CERTIFIED` |
| `rungs.r2_em_dbn` (182 LOC) | RUNS | imports; EM ran live 55.76→46.70→43.90 (3 iters, 12 crops, 20.5 s); `test_r2_em.py` **16 passed in 471.78 s** |
| `rungs.r3_conditioned_dbn` (151 LOC) | RUNS | imports; `test_r3_generative_mixture.py` **15/15 passed in 8.81 s** |
| `rungs.r4_5_rich_emission` (206 LOC) | RUNS | imports; `test_r4_5_emission.py` **9/9** |
| `mixture.py` (75 LOC) | RUNS | imports clean anywhere torch+numpy exist; EM + bare/deploy decode smoke in 3.9 s; **SMC row bit-reproduced** (4 tracks, Δ = 0.000000 vs `results_smc3.json`) |
| minimal closure | RUNS | 9 files / **1,097 LOC**, no `training` package, no madmom import: 1 EM step → λ 75.29, decode → 8 beats |
| E-step exactness | VERIFIED | autograd d logZ/d log A[0,1] vs central finite difference: float64 0.0609307247 vs 0.0609307165, rel. err **1.35e-07** |
| Kalman VAE (git objects only) | RUNS | 300 real training steps + eval, 431 s on one A6000, in-domain bt_val_rich n=15 **F 0.6048 / CMLt 0.3909 / AMLt 0.5039**; measured throughput 1.32–1.44 s/step at batch 16 / crop 256 / z=8 / K=5 |
| `vbpm.variants.{tchain,schain,vbpm,vmchain}` | import OK | but the CLI does not — see §9 |
| Analyze-SMC `scripts/` | 24 `.py`, `py_compile` clean; four CPU evaluation paths executed end to end today | all caches present: `beat_this_activations_cache` 217, `bt_transformer_cache` 217, `data/tta_activations` 217, `data/tta_full` 217, `data/ts_tta_cache` 1519 = 217×7, GT annotations 217 |

`_harness.ev` maps `CMLt = r['Correct Metric Level Total']`, `AMLt = r['Any Metric Level Total']`,
`CMLc`/`AMLc` to the Continuous fields (lines 110–118), by **string key** — structurally immune to
the index bug.

### 3.10 Data on disk — the rerun blocker is smaller than the record thinks

- **[LATE]** `/disk1/jaehoon/beat_this_spect/done/` — **163 GB, 16 preprocessed spectrogram
  archives**: asap, ballroom, beatles, candombe, filosax, groove_midi, gtzan, guitarset, hainsworth,
  harmonix, hjdb, jaah, rwc, simac, smc, tapcorrect. Annotations at
  `/disk1/jaehoon/dataset_store/beat_this_annotations/`, **every dataset with its own
  `8-folds.split`, SMC included**. Verified present. Counts: gtzan 999, harmonix 911, ballroom 685,
  **asap 473**, hjdb 235, rwc 226, hainsworth 222, smc 217, beatles 180.
- **[SYNTH] The bt_demix cache is NOT empty.** `find cache -type f` returns 0 only because it does
  not follow the symlink. `cache/bt_demix -> /disk4/jaehoon/VBPM_cache/bt_demix` contains **1,522
  files**: ballroom 672, beatles 179, hainsworth 222, hjdb 232, smc 217. Every bt_e2e-era experiment
  is reconstructible *except* for its trained checkpoints.
- **[SYNTH] `rungs_lab/dataset_store` is NOT empty either.** It resolves to
  `/home/sogang/jaehoon/VBPM/dataset_store` and contains asap, audio_by_stem,
  beat_this_annotations, beat_tracking_db1, fma, groove, harmonix_zenodo, lmd, maestro, … The
  "dataset_store is 0 bytes" report in the inherited record is wrong.
- **[SYNTH] The 86 fps SMC feature cache the KVAE line needs SURVIVES.**
  `/disk4/jaehoon/VBPM_cache/acts/smc_rich_86fps/` — 217 `.pt` files, keys
  `{activations [3447,512] float32, beat_targets, downbeat_targets, fps 86.1328125}`; sibling
  `smc_rich_86fps_act2/` 217. And a fold-honest twin exists: `smc_rich_foldhonest/` 217, same schema
  plus `act2`, **numerically different from `smc_rich_86fps` (max |Δ| = 9.435)**, i.e. two different
  frontend checkpoints on the same audio. `smc_rich_heldout/` is a *different* extraction (keys
  `feat [2001,512] / act2 / tid / fold`, i.e. 50 fps) and is the one project memory marks
  "confounded, do-not-use". Also present: `foldhonest_train_rich` 2287, `foldhonest_val_rich` 327
  (which carries `extractor: beat_this-fold7`), `bt_train_rich` 3300, `bt_val_rich` 118.
- Surviving checkpoints: `checkpoints/bt_fold{0..7}_repacked.pt`, 37,238,722 B each (pretrained Beat
  Transformer frontends), plus decoder scalars `runs/r2mix_seed0.pt` 880 B, `runs/r3_wt_seed0.pt`
  33,648 B, `runs/r3_lamt_seed0.pt` **33,668 B**, `runs/r2_seed0/r2_em_best.pt` 1,068 B. **36 result
  JSONs.** No end-to-end trained frontend exists anywhere.

### 3.11 External literature — verified against primary sources

**Beat This! (Foscarin et al., arXiv:2407.21658), Table 2, GTZAN test, mean ± sd over 3 seeds**
(beat F1 / CMLt / AMLt | downbeat F1 / CMLt / AMLt):

| system | beat F1 | beat CMLt | beat AMLt | db F1 | db CMLt | db AMLt |
|---|---|---|---|---|---|---|
| Hung et al. | 88.7 | 81.2 | 92.0 | 75.6 | 71.5 | 88.1 |
| Beat This! | 89.1±0.3 | 79.8±0.6 | 89.8±0.4 | 78.3±0.4 | 67.3±0.8 | 79.1±0.6 |
| Beat This! + DBN | 88.1±0.3 | 80.5±0.4 | 91.1±0.2 | 77.4±0.2 | **73.3±0.2** | 87.8±0.5 |

Their §4.3, verbatim: *"our loss does not specifically penalise non-periodic predictions, but treats
each beat individually."* Conclusions: *"developing new losses that enforce periodicity during
training."* **Their +DBN downbeat CMLt 73.3 is the best in the table** — a fill-in for our record,
and it sharpens rather than weakens the "the DBN buys continuity" leg.

**Sa Pinto (2023) thesis** (`644518.pdf`, in this repo), verified against the PDF, not the brief.
Table 3.1: min tempo 55 → 35, max 215 → 135, transition-λ 100 → 10. Table 3.2, SMC 217 excerpts,
40 s, Davies & Böck metrics (F / CMLc / CMLt / AMLc / AMLt): Default [Böck 2016a] 0.563 / 0.350 /
0.472 / 0.459 / 0.629; Expressive 0.540 / 0.306 / 0.410 / 0.427 / 0.565; **Optimal Choice 0.624 /
0.456 / 0.611 / 0.545 / 0.703**. 23.1% relative improvement figure appears twice in the text.
**Subtlety the brief misses: the thesis makes the optimal choice "according to each individual
evaluation metric", so the 0.624 F and 0.611 CMLt come from *different* per-excerpt selections — it
is five oracles, not one decoder scored five ways.** Also: their expressive parameterisation *loses*
on SMC on all five metrics, which makes "no single DBN setting wins on SMC" model-independent.

**Schreiber, Urbano & Müller (2020)**, TISMIR 3(1):111–125, DOI 10.5334/tismir.43: a single global
tempo is not well-posed for much of SMC; recommends against SMC for global-tempo benchmarking.

`vbpm/docs/ELBO_for_DBN.md` recovered at commit **`60500e4`** (98 lines): tempo is a **per-frame
ungated Log-Normal random walk** with σ learned from audio and **no [min,max] tempo bound** — a
deliberate departure from madmom's beat-gated `exp(−λ|ratio−1|)`, and aligned with the SMC blind-spot
thesis. Live constants at `/home/sogang/jaehoon/VBPM/vbpm/constants.py`: `TEMPO_PRIOR_MU = −2.6827`,
`TEMPO_PRIOR_SIGMA = 0.3903` (the `_LEGACY` pair −2.5028 / 0.5005 is explicitly named legacy);
`TEMPO_WALK_SIGMA = 0.00212`; `WALK_MIX_W = (0.687, 0.313)`; `WALK_MIX_SIGMA = (0.00029, 0.00377)`.

---

## 4. THE KILL LIST

### 4.1 CONTRADICTED numbers — both values given, neither may be cited without a rerun

| id | claim | recorded value | actual value | source of truth |
|---|---|---|---|---|
| meta-02 / BUG-01 | "rungs_lab CMLt columns dated 2026-07-20 must be read as AMLc" | AMLc | **genuine CMLt** | `git show d65e316:experiments/bt_e2e/tiebreak_continuity.py` line 68 already reads `_, cmlt, _, amlt = mir_eval.beat.continuity(...)`; commit 1bf2fbe says so verbatim. **The bt_e2e audit is wrong.** |
| RET-14 | chain-arm rule-g F, ballroom / beatles | 0.973 / 0.943 | **0.931 / 0.865** | `/tmp/scratch/abl_chain.log`. The strings 0.973 and 0.943 appear nowhere as F; the log's *AMLt* for those datasets is 0.975 / 0.942. The record read the AMLt column for two of five datasets — a mixed-metric table. |
| dither-08 / C08 / meta-01 | mixture bare gain over R1 bare | +0.0093 | **+0.0079** | RESULTS.md:136 says bare R1 = 0.9100, RESULTS.md:214 says 0.9114, in the same document. 0.9193 − 0.9114 = 0.0079. VBPM memory propagates the error by pairing 0.9114 with the +0.0093 label. |
| C08 / dither-06 | mixture fit on real data | w = 0.370, λ = 93.1 | **not reproduced**: the 24-run campaign gives fold-0 (0.4100, 78.41) / (0.4300, 87.93) / (0.4100, 95.83); corpus-wide w 0.35–0.45, λ 57.20–131.35; and the M-step grid `linspace(0.01, 0.95, 48)` has spacing 0.02, so **0.370 is not even on the grid** | `results_campaign_r2mix.json`; `mixture_kernel_probe.py:43` |
| C39 | "9 ranked bugs in the current mainline model file" | 9, mainline | **10 bugs; the file is DELETED** | `vbpm/variants/` contains base/schain/tchain/vbpm/vmchain only; `archain.py` removed in HEAD `29e7c97`; `project_state.md` names tchain as mainline |
| HET-02 | "~45 s crops needed to separate truth from coasting" | ~45 s | **30 s**: truth−coast = +12.3 nats at 256 frames (5 s), +12.4 at 512 (10 s), **−46.6 at 1500 (30 s)**. No 45 s measurement exists | `project_crop_horizon_root_cause.md`; the "45 s" is a mis-citation living in `project_rigid_grid_ceiling.md` |
| KVAE-01 | "KVAE SMC F 0.654 is deployable; the filter adds more OOD than in-domain" | deployable | **provenance destroyed** — `/home/sogang/jaehoon/VBPM/vae_rescue/OVERNIGHT_REPORT.md`, cited as its origin, does not exist; the 2026-07-05 reproduction records the honest ladder as peak-pick 0.620 / structured-DBN 0.589 / **e2e KVAE 0.135** / PF-oracle-σ 0.654 | `project_smc_07_mission.md` vs `project_smc_mirex_ood_generalization.md` |
| CAL-03 | "ρ = −0.230, p = 6e-4 from `results_lamselect_smc.json`" | that file | **that file gives −0.21176, p = 1.707e-3**; −0.22993 / 6.416e-4 comes from `results_smc3.json` | recomputed from both |
| C17 | R3 arms 0.9193 / 0.9191 / 0.9206 | prose | the surviving JSONs give **r3_wt 0.9139, r3_lamt 0.9155** at (w₀ = 0.390, λ₀ = 104.43), not (0.370, 93.1) | `results_r3_wt_val.json`, `results_r3_lamt_val.json` |
| C14 | ladder oracle ceiling gtzan / val | 96.7% / 92.5% | **96.48% / 92.05%** | recomputed from `tempo_lab_oracle-over-nominees.npy` vs `octave_audit.npy` |
| C15 | reranker gtzan accuracy / halving / ceiling | 82.1 / 6.3 / 96.7 | **81.4 / 6.8 / 98.2** (±4% criterion; the true criterion is unrecorded); **there is no `train` split in the artifact**, so the 91.6% train figure is unverifiable | `tempo_ranker_est.npy` |
| AS-007 | good : total-failure activation ratio | 3.1× | **3.22× (mean-of-means) or 4.17× (median-of-medians)**; neither is 3.1× | `data/activation_diagnostics.csv` |
| AS-016 | "pick the better model per track" oracle | 0.664 | **0.6978** (oracle-threshold) or **0.6416** (fixed p=0.5); 0.664 matches neither — **and under the oracle definition selection (0.698) BEATS fusion (0.686), inverting the record's sentence** | `scripts/exp_ensemble_dbn.py` |
| AS-029 | TTA oracle ceiling row | 0.703 next to 0.645/0.550/0.649 | **the 0.703 is the sym-5 rate set; the F/CMLt/AMLt beside it are the mild 3-rate set, whose ceiling is 0.6957** | `data/ts_tta_cache/` |
| AS-034 | pitch+TS peak-pick F | 0.644 (one table) / 0.640 (another) | **0.6411** | rerun of `eval_tta_realistic.py --set both` |
| AS-040 | "Beat Transformer F 0.596 reproduced exactly" | 0.596 | committed `btr_F` mean is **0.58057** — a 0.0154 gap, two protocols presented as one | `data/activation_diagnostics.csv` |
| AS-037 | README shape of `octave_correction_results.csv` | 217 × 15 | **217 × 16** | pandas |
| PRE-06 | "separate training was found to be advantageous…", attributed to arXiv:1810.10002 | that paper | **the quotation is not in the paper** (full-text search of both the arXiv PDF and the TISMIR version) | curl + pdftotext |
| C45 | comparison arm `results_r4_full_feats.json` | λ_t 72.3–157.3, half_time 0.001–0.009 | **λ_t 59.8–157.3, half_time 0.001–0.025** | recomputed |
| C28 | deploy AMLt max | 0.9723 | **0.9721** (0.9723 belongs to the neighbouring `results_r2mix_val.json`) | recomputed |
| LM-25 | λ quartile maxima | 50 / 150 | **500 / 500** | recomputed |
| SMC-04 memory | bpm<55 CMLt collapse | 0.553 → 0.343 | **0.4130 → 0.3428**; the "0.553" is the peak-pick **AMLt** for that stratum (0.5527) — a column mix-up (not the AMLc one) | `results_smc3.json` |
| HET-01 | `GAMMA = 0.0363` = corpus median bar-to-bar \|Δlog rate\| | 0.0363 | constant confirmed at `nets.py:12`, but an independent recompute over **all 1067 ASAP performances** gives median-of-medians **0.0518** (mean 0.0618) — 43% larger, i.e. the deployed prior is *tighter* than the corpus it claims to fit. `asap468` could not be resolved to a piece | ASAP `performance_downbeats` |
| C41 / LM-38 / CODE-01 | file inventory | r3_lamt 33,648 B; 22 `.pyc`; ~40 JSONs; `runs/r2_seed0.pt` | **33,668 B; 21 `.pyc`; 36 JSONs; the file is `runs/r2_seed0/r2_em_best.pt` (1,068 B)** | `ls -la` |
| THEO-01 / MEMORY.md / HANDOVER:32 / PRIOR_ART:342 | recovery hash for `ELBO_for_DBN.md` | `43ecf34` | **does not exist** (`git cat-file -t 43ecf34` → fatal). Use **`60500e4`** | git |
| e2e-18 | vanilla "35 effective epochs" | 35 | **not reconstructible**: histories log 24 and 30; RESULTS.md:45 says vanilla resumed from epoch 7, giving 31 | four `*_history.json` |
| **[SYNTH]** CODE-01 / e2e forensics | "`cache/bt_demix/` is empty, so every bt_e2e-era number needs re-demixing" | empty | **1,522 files present**; `dataset_store` likewise non-empty | see §3.10 |
| **[SYNTH]** KVAE candidate | "the SMC feature cache the 0.654 was computed on is gone" | gone | **`smc_rich_86fps` (217 files, 86.1328125 fps, [T,512]) and a fold-honest twin both survive** | see §3.10 |
| **[LATE]** several agents | "Analyze-SMC SMC rows are `final0`-contaminated" | contaminated | **fold-honest by construction** | `scripts/run_folds.py`, the notebook |

### 4.2 RETRACTED claims — must never appear in the paper

- **λ ≈ 40 is musically interpretable / matches heavy-tailed tempo-increment statistics.** WITHDRAWN
  in RESULTS.md's own `## RETRACTION`; a metronome reproduces it. **Excess kurtosis ~12.6 / ~13 is
  unusable** (computed from frame-quantised annotation intervals).
- **"Zero variational gap / training tchain is maximum-likelihood training of a neural DBN."**
  REFUTED by `audit_2026-08-19.md` item O. `quasi_svae_mechanism.md` lines 10 and 37 **still assert
  the refuted version today** — the doc was never edited. Say "structured chain posterior with exact
  forward recursion".
- **"The 0.973-vs-0.000 chain-vs-factorized A/B establishes a representational floor."** REFUTED
  (item Q): the factorized arm does not even attain its own mean-field optimum (MAP bin-step 0 at
  **100.0%** of frames; prior log T/frame −1.5181 vs feasible ≈ −0.672). It is a kinetic trap. And
  per §4.1, two of its five numbers are wrong as well.
- **gtzan F 0.733 (rate_grid) and 0.752 (anchor_k v2)** — RETIRED by explicit user instruction
  2026-08-15. Keep the mechanisms, drop the numbers. The "amortization gap" framing built on 0.733
  loses its denominator.
- **"Ladder step 7: monotone convergence to truth, ratio 1.06→1.02, med|err| 499→129 ms."** REFUTED —
  spliced from two different logs; the quoted run goes to ratio 2.01 / 612 ms at ep250 and never
  recovers (ep399 615 ms). Tail-mean F 0.418 ± 0.213. `project_quasi_svae_mechanism.md` still
  narrates the un-refuted version.
- **The archain capstone "F 1.000 / 100% in-tolerance / 38 ms at ep375."** REFUTED — max of 17 draws
  from a distribution with mean 0.491, sd 0.299; the same run scored 0.154 two snapshots earlier.
- **"The ELBO defects to the double octave; the defection is the objective's."** REFUTED — a limit
  cycle of the single-sample von Mises gradient. The **+0.04 and +2.12 nat pricings go with it.**
- **"Sampled F 0.164 / max 0.467 / P(F>0.5)=0% vs deployed 0.733."** UNVERIFIABLE AND CONFOUNDED —
  all 64 draws share one sampled φ₀; an identically-configured replicate gave 0.600 at ep399.
- **κ_physical = 383 and the 4 ms Wing–Kristofferson jitter constant.** RETRACT ENTIRELY — the WK
  estimator is at its floor on 30.6 / 0.5 / 84.0 / 50.5 / 78.4 / 95.3% of ballroom / hjdb / beatles /
  hainsworth / rwc / asap; pooled median κ 4.71e13, pooled median jitter 0.0 ms. The surviving 4 ms
  comes only from ballroom and hjdb, whose in-window inter-downbeat CV is **exactly 0.000000**.
- **"Truth is the objective argmax by 2.0 nats."** Actual gap 0.1 nats inside an 8–12 nat wander —
  TIED. Cause: the scoring omitted the rate prior and refit the rate per candidate.
- **"The ELBO structurally prefers a diffuse posterior (312 vs 6.9 nats)."** Arithmetic confirmed,
  interpretation OVERSTATED and partly circular (κ_p read off the checkpoint it explains, single
  song). Honest residual: **~7 nats of phase information per 30 s window, ~190× short.**
- **"The rubato sweep tests rubato."** REFUTED — it ranked whole-song CV while training on a 30 s
  window (asap169 1.103 → 0.083 in-window; asap456 0.980 → 0.011). Only asap343 is genuinely rubato
  in-window. `asap254`'s bar IBIs are [0.93, 2.65, 9.33, 9.22] s — an annotation anomaly.
- **The MERT input advantage.** RETRACTED: the n=250 advantage (0.816 vs 0.772) is gone at n=920; the
  single-seed fused 0.9658/0.9200 was a lucky draw (honest 5-seed mean 0.9548); the F "win" is a tie
  (paired +0.0020 ± 0.0024, Wilcoxon **p = 0.96**).
- **The categorical q(l_t) "fix."** The pre-registered falsifier FIRED: 8 seeds/arm, shipped 0.3108
  vs categorical 0.2479 (perm p = 0.0514) and CAT2 0.2767 (p = 0.0466, significantly worse).
  *The memory's own YAML description still says "validated fix" — it contradicts its own body.*
- **"The e2e CRF arm loses by −0.006 / −0.015."** The setup is withdrawn by its own ADDENDUM on three
  measured asymmetries, and the rematch (which exists) went the same way harder on 60 ballroom songs.
  **Do not cite as a verdict on end-to-end structured training.**
- **"Likelihood-optimal but F-suboptimal λ = 14.2."** CONFOUNDED AND WITHDRAWN (obs 16 train / 6
  decode); at the corrected obs the learned λ converges to ~101.
- **The q-family ladder.** Say **"not yet testable"**, never "negative": the injected corruption
  vM(17) has sd 0.24 rad = 64 ms, **inside the 70 ms tolerance**, and the corrector receives exactly
  zero gradient toward undisplaced phase. Power to detect +10 in-tolerance points was **8%**. Project
  discipline: "the q-family ladder earns NOTHING and is a liability if cited."
- **The mixture-repair deployment gain.** There is none: dead tie in-domain, null on SMC.

---

## 5. THE RERUN QUEUE

Ordered by (load-bearing × cheapness). **BLOCKING** = the submission cannot go out with the claim in
it until this runs. Nothing here may be launched inside `vbpm/` while the recertification agent is
live — copy to `scripts/` or `/tmp`.

| # | item | why | cost | command / design | blocking |
|---|---|---|---|---|---|
| 1 | **Beat This training tempo range (AS-025)** | "~64–168 BPM, ±20% augmentation" is load-bearing for the entire tempo-mismatch framing and is an **unsourced assertion**. Nothing in this repo derives it, and `data/training_data_bpm_summary.csv` cannot produce it (hjdb's row is *downbeat* rate, mean 38.1) | minutes, CPU | read the augmentation range out of the vendored config: `grep -rn "tempo" /home/sogang/jaehoon/Analyze-SMC/beat_this/beat_this/ \| grep -i aug` | **YES** |
| 2 | **Port the metronome-ceiling protocol table into `scripts/`** | It is itself a paper contribution and it retires every ballroom/hjdb "structure does not help" verdict. Already computed; needs re-running under `scripts/_harness.py` for one-harness compliance | ~30 min CPU | port `/tmp/gridceil2.py`, `/tmp/first60.py` into `scripts/`, re-run against `_harness.ev`, emit F+CMLt+AMLt per corpus | **YES** |
| 3 | **E1 — soft tempo prior by exact marginalisation** | The decisive build gate; the experiment has never been run; +0.1909 CMLt of measured headroom | **4–8 h, CPU only, ZERO GPU** | copy `rungs/` out of `vbpm/` into `scripts/`; replace the uniform tempo-state prior in R1 with π(τ) ∝ Beat-Transformer tempo posterior, octave-smeared over {1/3,1/2,1,2,3}, temperature T swept on a log grid; decode on the **full** wide grid so no state is forbidden. Pre-registered success: **CMLt > 0.5138 at p<0.05 AND F ≥ 0.60**. Pre-registered falsification: no T beats both the hard band and the wide grid. Report the 134/83 in-band/out-of-band split every time | **YES** |
| 4 | **Fix the `min_bpm` baseline and publish the split** | Converts an unexplained pooled −0.0400 CMLt into a stated mechanism and removes the strongest reviewer objection ("your baseline decoder was misconfigured") | <1 day CPU; most already computed | re-run every DBN row at `min_bpm=30` as well as 55, split at gt_bpm=55, publish both; add the IBI-median-cleanup row (0.6235/0.5177/0.6256) and the TTA row as competitor baselines | **YES** |
| 5 | **Recompute AS-016, AS-027, AS-029** | Three FOLLOWUP_FINDINGS rows are wrong or unsourced, and one (AS-016) **inverts** the document's "fusion beats selection" sentence | ~1 h CPU, caches on disk | re-run `scripts/exp_ensemble_dbn.py` pinning the model-select definition; write the missing sweet-spot-BPM routing script (per track pick r minimising \|gt_bpm·r − target\|, evaluate on `data/ts_tta_cache`); split AS-029 into mild-3 (0.6957) and sym-5 (0.7029) rows | **YES** |
| 6 | **Close the novelty search (NOV-01, PRE-02, C31, prior-01)** | Three acknowledged gaps: repositorio-aberto.up.pt (Porto/Davies/Oliveira lineage), CNKI/non-English, and **Electronics 11(24):4186 "HMM with RNN observer" — unverified and could pre-empt directly**. Also read Krebs 4.3.2 and ch. 6 primary, and search madmom's issue tracker | 1 day, no compute | manual | **YES** for any novelty sentence |
| 7 | **Archive `smc3_certify.py` output** | The `0.642` wide-DBN oracle target is unbacked, and Analyze-SMC's own rerun gives **0.6525** for the nominally same quantity — a 0.0105 disagreement | GPU, small, 217 cached tracks | `cd vbpm/rungs_lab && $PY /tmp/smc3_certify_copy.py` and commit the output. **Coordinate — the recert agent owns that directory** | yes if any oracle-λ number is cited |
| 8 | **HET-01 corpus-wide bar-to-bar tempo statistics** | The "hand-set homogeneous prior is mis-specified" mechanism currently rests on **one Scriabin performance, 19 bars**, and my recompute does not reproduce its constant | hours, CPU; ASAP is on disk | recompute median/p90/max bar-to-bar \|Δlog rate\| across all 1067 ASAP performances and across SMC; report GAMMA's provenance | **YES** if the mechanism is claimed |
| 9 | **Re-derive `WALK_MIX` on sub-frame annotations** | A 13.1× two-component mixture with 31% weight on the loose component is *exactly* what integer-frame dithering manufactures; the fitting script is unnamed so we cannot tell which times it used | hours, CPU | refit on sub-frame-interpolated annotation times. If it *is* contaminated, that is a second independent instance of the artifact and **strengthens** clause 3 | no |
| 10 | **DITH-02 fps-scaling law (λ 40.27 → 80.08, ratio 1.988)** | The strongest empirical leg of clause 3 on *real* data, and it has zero output artifact | GPU, minutes | `cd vbpm/rungs_lab && PYTHONPATH=. $PY experiments/bt_e2e/train_r2_em_fps2.py` — **first edit the hardcoded `final_eval.DEVICE='cuda:1'` at line 24**. Note it also needs `vanilla_best_prelim.pt`, which is **deleted**; substitute `checkpoints/bt_fold0_repacked.pt` and say so | no |
| 11 | **`mixture_kernel_probe.py` (dither-07 / LM-05 / DITH-03)** | Recovers w and λ on real data; the published (0.370, 93.1) is not on the M-step grid | GPU, minutes | `cd vbpm/rungs_lab && PYTHONPATH=. $PY experiments/bt_e2e/mixture_kernel_probe.py` (hardcodes cuda:2 at lines 24 and 33) | no |
| 12 | **KVAE fold-honest rerun** | Would settle in one run whether the runner-up cell is worth a paragraph. **[SYNTH]** the feature caches survive, so only a retrain is needed | ~0.4 GPU-h train + ~50 min eval | `python train_kvae.py --train_feature_dir /disk4/jaehoon/VBPM_cache/acts/foldhonest_train_rich --val_feature_dir .../foldhonest_val_rich --num_train_songs 400 --num_steps 1000 --batch_size 16 --crop_length_frames 256 --a_dim 8 --z_dim 8 --K 5`, then evaluate on `.../smc_rich_foldhonest` (217) **through `scripts/_harness.py`**, alongside an `act2` peak-pick row from the identical cache. First verify that cache's extractor provenance — it carries no `extractor` key | no |
| 13 | **`tiebreak_continuity.py` → JSON** | The whole §3.4 table is prose-only with no artifact | GPU, n=146; demix cache present | `$PY experiments/bt_e2e/tiebreak_continuity.py`, dump per-song arrays, and **diff the λ=98.6 and λ=100 beat times** to confirm the six-way byte-identical tie is a genuine identical Viterbi path | no |
| 14 | **`oracle_ceiling.py` with continuity columns** | The +0.009 / +0.048 ceiling has **no CMLt/AMLt at all**, which disqualifies it from a continuity paper | GPU, n=146 | `$PY experiments/bt_e2e/oracle_ceiling.py`, add all six metrics | no |
| 15 | **r45-04: the D=4–8 "structure matters" control** | The **weakest-evidenced claim in the inventory** — one qualitative sentence with **no number of any kind**, and it is the only thing separating "our DP is broken" from "emission domination is a D=256 property" | GPU | re-run the blind structure test and report numbers | yes if emission domination is cited |
| 16 | **Quantify the fold_0 leak (e2e-09)** | 0.9622 is flagged LEAKY with the leak never measured; it is computable from stem lists alone | seconds, CPU | intersect the released fold_0 training list with our 146-song val list | no |
| 17 | **CONT-01 with seeds and full-song eval** | Its own record says "one seed, one protocol. Before any claim: seeds, full-song eval, and the deployment read-out swap" | GPU, large | branch `archain-anchor`, commit `15b5afa`, `vbpm/configs/baseline.yaml` | no |
| 18 | **Octave audit scored through the model (C16)** | The audit states outright that "nothing in this audit was scored through the model" and the run saved no checkpoint | GPU retrain | — | yes if "fixing octave improves tracking" is claimed |
| 19 | **PER-02 Beethoven-rubato pair (0.437 vs 0.810)** | **No n, no track list, no script, no artifact** — one parenthetical in a handover, second-hand, and 0.437 collides with an unrelated 0.437 nine lines later | re-measure or drop | re-measure on a named rubato subset (ASAP) with n stated | **YES** if cited |
| 20 | **E2 — implement the periodicity loss Foscarin et al. only proposed** | Without it, our octave-prior negative supports only the narrow claim about priors over *absolute period*. Their released `loss.py` contains **no periodicity term**, so nothing has actually been refuted | 8–16 GPU-h | add `L_per = −mean(normalised ACF peak of sigmoid(beat logits) at argmax lag ∈ [0.25, 2.0] s)` to `beat_this/model/loss.py`, sweep λ ∈ {0, 0.05, 0.2, 1.0}, fine-tune from `final0`, 3 seeds, report F/CMLt/AMLt on SMC and gtzan through `_harness.py`. Pre-registered prediction: AMLt−CMLt does not close, because a scale-equivariant loss cannot select an octave | no |
| 21 | **E3 — the learned music-conditioned prior** | The actual build | ~15–30 GPU-h | train a head on frozen Beat This activations emitting (i) tempo-state initial mass and (ii) a **two-component mixture over transition scales** (λ_oracle is bimodal at 1 and ≥100, so a two-component marginalised mixture is the minimal sufficient parameterisation and avoids mode *selection*, which has failed repeatedly). Train on non-SMC data, evaluate **zero-shot** on SMC. Hazards to design around: `hygiene_check.py` shows the R3 w_t head emits −inf on 6/20 seeds when w_t saturates — clamp or use logit-space; ≥3 seeds; run to convergence (40→1000 epochs has reversed five verdicts); pool checkpoint selection across datasets; **do not select on F**. Pre-registered success: **CMLt ≥ 0.57 and AMLt ≥ 0.68 at F ≥ 0.62, zero-shot, no oracle, mean over ≥3 seeds with sd reported** | after E1 |
| 22 | **E4 — no-harm guard** | A fix that repairs SMC and breaks ballroom is not a fix | ~2 h | run E1/E3's winner on ballroom + gtzan; criterion is non-inferiority: \|ΔF\| ≤ 0.005 and ΔAMLt ≥ −0.005 | before submission |

**[LATE] Re-cost every "requires data preparation" item.** The full preprocessed corpus is on disk
(§3.10) and **asap (n=473, metronome ceiling 0.164, >2× SMC's size, *with* downbeat annotations)** is
the corpus where the thesis can be demonstrated at full statistical power before transfer to SMC.
SMC alone cannot resolve a +0.02 effect confidently.

**[LATE] Baseline choice is a live decision, not a detail.** Show both rows in the same table or a
reviewer reads it as a self-chosen strawman: oracle-λ over fixed λ=100 on the paper grid = **+0.050**
(0.592 → 0.642); over fixed λ=100 on the wide grid = **+0.0668** (0.5827 → 0.6495); over Beat This
peak-picking = **+0.0225** (0.6270 → 0.6495). Recommended framing: the DBN at shipped settings loses
to peak-picking on SMC (0.5827 vs 0.6270) — *that* is what drove the field away from it — and a
per-song λ recovers +0.0668 F / +0.0663 CMLt / +0.0484 AMLt, overtaking peak-picking on all three.
Standing caveat: all oracle; the best label-free selector is 0.6021, *below* peak-picking; the one
deployable data point is the EM-learned λ transferred to SMC, **+0.0211 F / +0.0115 CMLt** over
hand-set — about a third of the oracle budget, with no labels.

### Unverifiable — record and stop asking

- `PER-02` (Beethoven rubato 0.437 vs 0.810): no n, no artifact, no producer.
- `MARG-02` (13% / 57% / 64% / 43% posterior-multimodality statistics): no n stated, no checkpoint
  named among 30+ run dirs.
- `C40` / `RET-06`: the two logs the refutation adjudicates (`ov_anchor_resid.log`,
  `ov_anchor_resid_save.log`) are **not on disk**. The refutation is itself un-auditable.
- `CODE-02`: `archain_lab.py` does not exist in the working tree **or** on the `archain-anchor`
  branch, so the `torch.manual_seed(0)` line cannot be checked. Only `.pyc` bytecode survives.
- `C38` / `LM-34` (the tchain scoreboard): no JSON, no log, no named checkpoint, **no SMC row**.
- `MARG-03`, `LM-37` (Fourier refinement 192/384 bins): no artifact; needs a checkpoint the record
  does not name.
- `LM-55` (`logs_faithful.txt`): the backing log is not in `/home/sogang/jaehoon/VBPM/logs/`.
- `RET-11`, `RET-12`, `OCT-01`: no producer script located; synthetic and re-derivable in principle.

---

## 6. HARNESS RECONCILIATION

**Definitive: Analyze-SMC's 0.6270 and VBPM's 0.6266 are TWO DIFFERENT MEASUREMENTS that agree by
coincidence. They must never be merged, averaged, or placed in the same table row.**

| | Analyze-SMC 0.6270 | rungs_lab R0_shipped 0.6266 |
|---|---|---|
| network | **Beat This** | **Beat Transformer** |
| frame rate | **50 fps** | **43.06640625 fps** (44100/1024) |
| decoder | peak-pick, max-pool width 7, threshold p = 0.5 | **madmom DBN as Beat Transformer ships it** (min_bpm 55, obs_lambda 6, num_tempi None, threshold 0.2, transition_lambda 100) |
| scorer | `scripts/_harness.py` → `mir_eval.beat.evaluate`, string keys, 5 s trim, ±70 ms | rungs_lab `score()` → `mir_eval.beat.continuity` indices 1 and 3 |
| checkpoints | 8-fold, fold-honest **[LATE]** | fold-honest Beat Transformer frontends |
| artifact | `beat_this_activations_cache/`, `data/results.csv` | `results_smc.json`, `results_smc_diagnosis.json` |

Decisive: **Beat Transformer activations peak-picked score 0.5055**, not 0.6266. `PHASE0_RECERT.md`
§3 establishes the same conclusion independently. The metric *definitions* do coincide
(`_harness.py` and rungs_lab `score()` both use 5 s trim and ±70 ms, verified by code read), which is
exactly why the mistake hides.

**One row IS cross-harness certified:** the peak-pick arm of `results_smc3.json` (Beat This, wide
grid) reproduces Analyze-SMC's own `_harness` recompute to 4–6 dp on all five metrics
(0.6270 / 0.5138 / 0.6100 / CMLc 0.3655 / AMLc 0.4214). `smc3_ladder.py` `sys.path.insert`s
`/home/sogang/jaehoon/Analyze-SMC/scripts` and imports `_harness` directly, so **`results_smc3.json`
is the one inherited SMC file that may enter a shared table.**

### Every comparison currently mixing harnesses

1. **The four-way peak-pick canary — the same nominal quantity on the same 217 SMC tracks appears as
   four different numbers:** **0.6852** (`results_lamselect_smc.json` — scipy `find_peaks`,
   height 0.2, Beat Transformer, mixture chassis w=0.390, λ grid **floored at 20**), **0.6270**
   (`results_smc3.json` / `_harness`), **0.620** (`CAMPAIGN.md` OOD table, which annotates itself
   "matches published 0.626 → harness valid"), **0.599** (fold-honest parity memory). Any table built
   by grepping for "peak-pick on SMC" is wrong.
2. `results_lamselect_smc.json` vs `results_smc3.json` — same four arms, same n=217, fixed100
   differs by 0.1037 F and ORACLE by 0.1221 F. Different network, frame rate, peak-picker, and λ
   grid. **43.78% (95/217) of that file's λ_oracle values sit at its grid floor of 20**, so its
   ORACLE 0.5274 is a truncation artifact and is not a ceiling.
3. `results_baseline.json` (shipped decode) vs `results_stratified.json` (bare decode) — gtzan R1
   beat F **0.8786 vs 0.7092** over the identical 999 songs.
4. `results_smc.json` fixed-λ vs `results_smc3.json` fixed-λ — **0.6266 vs 0.5827** for "the DBN on
   SMC". Different frontend, grid, obs_lambda and bounding.
5. `results_r2_val.json` (peak-snap **off**, `correct=False`, pretrained `bt_fold0_repacked.pt`) vs
   RESULTS.md's DEPLOYED rows (peak-snap on, from-scratch e2e frontend) — **0.9033 vs 0.9504** for
   "R1 on fold-0 val". This is the single largest unexplained family gap in the inheritance and it is
   why the "R2 beats R1" headline evaporates (§4.1).
6. RESULTS.md internal: bare λ=100 as **0.9100** (line 136, `mixture_kernel_probe.py`) and **0.9114**
   (line 214, `tiebreak_continuity.py`); R2-generative λ=40.3 as **0.9488** (lines 110, 162) and
   **0.9494** (line 212); learn-obs as **0.9459** (line 163) and **0.9457** (line 213). The wrapper
   is validated faithful to ±0.0004, so 0.9457/0.9459 is within fidelity — but the rows must carry a
   decoder label.
7. bt_e2e 60-song training-time evals vs the 146-song final eval: **+0.0107 / +0.0097** bias, and the
   60 songs are **60/60 ballroom**.
8. tchain / KVAE / CAMPAIGN.md scoring: untrimmed in-house `f_measure` at ±0.07 s (F column) beside
   `mir_eval.beat.continuity` **after** `trim_beats` (continuity columns) — F and continuity computed
   on different beat sets inside the same table.
9. Three frame rates coexist across the record — **43.066** (`results_stratified.json`), **50**
   (SMC ladder, tchain), **86.13** (bt_e2e docstrings, all `*_rich` feature caches). The dithering
   argument says λ scales *with* fps (40.27 → 80.08 at 2×), so **any λ quoted without its fps is
   meaningless.**

---

## 7. AMLc / CMLt EXPOSURE LIST

### The dispute is settled — against the bt_e2e audit, in favour of project memory.

Four independent lines of evidence, no rerun required:

1. `mir_eval.beat.continuity` returns **(CMLc, CMLt, AMLc, AMLt)** — confirmed by `inspect.getsource`
   in the `analyze-smc` env (mir_eval 0.8.2) and by an empirical check (a double-time estimate
   returns `(0.0, 0.0, 0.9955, 0.9955)`).
2. **Every** `mir_eval.beat.continuity` call site under `vbpm/rungs_lab/` unpacks indices **1 and 3**
   — `run_r2_mix.py:39`, `eval_smc.py:36`, `eval_smc2.py:42`, `r3_lab.py:54`, `eval_stratified.py:45`,
   `lambda_select.py:23`, `lambda_select2.py:23`, `campaign_r2mix.py:19`, `eval_r2_val.py:42`,
   `eval_btminimal.py:21`, `eval_baseline.py:21`, `smc_diagnose.py:30`, `r4/r4_train.py:126`,
   `mert/mert_r4_train.py:126`, `training/evaluation.py:64,77`, `experiments/bt_e2e/train_r4_5*.py`,
   `r0_continuity.py:29,37`, and crucially **`tiebreak_continuity.py:68,76`** — the producer of the
   disputed 2026-07-20 table. `r4_train.py` even carries the comment `(CMLc, CMLt, AMLc, AMLt)
   -- explicit`.
3. The fix commit itself, VBPM **`1bf2fbe`** (2026-08-14), states verbatim: *"The eleven call sites in
   rungs_lab/ already unpacked indices 1 and 3 and were never affected."* The buggy site was **only**
   `/home/sogang/jaehoon/VBPM/vbpm/scoring/evaluation.py`, which unpacked `(_cc, _ac, cmlt, amlt)`.
4. The residual doubt — that `tiebreak_continuity.py` might have been edited after RESULTS.md was
   written, since `rungs_lab` is not under git — is **closed**:
   `git show d65e316:experiments/bt_e2e/tiebreak_continuity.py` (the 2026-07-20 blob, in the VBPM
   repo) already reads `_, cmlt, _, amlt`.

**Consequence: the entire rungs_lab / bt_e2e side is CLEAN.** This unblocks, at zero compute cost:
the RESULTS.md tiebreaker table (§3.4) and everything derived from it (tie-01, tie-03…tie-11,
CTRL-02, C22, C23, C44, DITH-04, r45-01, r45-08, r45-09, LM-07, LM-15, LM-17), plus `results_smc.json`,
`results_smc3.json`, `results_btminimal.json`, `results_baseline.json`,
`results_campaign_r2mix.json`, `results_stratified.json`, the r4 JSONs, `smc3_ladder.log`, and every
Analyze-SMC number (which uses string keys and cannot be mis-indexed at all).

### Still exposed — the vbpm PACKAGE before 2026-08-14

Every "CMLt" printed by `vbpm/scoring/evaluation.py` before commit `1bf2fbe` is an **AMLc**. Severity:
a perfectly double-time estimate scored **0.9917** under the old unpacking and scores **0.0** now.
No re-scoring artifact exists anywhere.

| exposed number | where | reading |
|---|---|---|
| gtzan **CMLt 0.70** / AMLt 0.85 (tchain scoreboard) | `project_state.md:19`; MARG-05; C38 | **AMLc 0.70.** Arithmetically consistent (AMLc ≤ AMLt). **Do not print as CMLt.** Calibration: the post-fix schain run gives gtzan genuine CMLt 0.668 against AMLt 0.873 — the same shape |
| gtzan **CMLt 0.870** vs AMLt 0.876 (categorical-search q) | `q_family_case.md:411`; LM-41 | Almost certainly **AMLc**. CMLt ≈ AMLt is implausible at F 0.733 |
| gtzan **CMLt 0.115 → 0.467**, AMLt 0.184 → 0.556 (gauss_time repair) | `project_state.md:137`; LM-51; MARG-04 | Read as **AMLc**. This is exactly the figure the continuity argument would want, and it is both unverified and probably mislabelled |
| gtzan **CMLt 0.864** at F 0.501 (triangle emission champion) | `vbpm_decisions.md`; LM-55 | Dated 2026-08-05, pre-fix. F 0.501 beside "CMLt" 0.864 is itself the METH-01 warning signature |
| octave audit "true CMLt 0.844 → 0.710" | `octave_error_audit.md:4` | Pre-fix package output |
| `CMLt 0.000` at ep399 (archain) | RET-06 / C40 | Formally AMLc — but at exactly 0.000 the distinction is immaterial, so the conclusion is robust |
| KVAE SMC table CMLc 0.420 / **CMLt 0.587** / AMLc 0.440 / AMLt 0.616 | `project_smc_mirex_ood_generalization.md` | Dated 2026-07-01/02, pre-fix package scoring. **The KVAE-02 target metric is literally "CMLt − CMLc", the exact pair the bug conflates** — maximal exposure |
| anchor_k v2 "CMLt-TRUE 0.600 → 0.743" | MARG-07 | **Partly discharged**: the 0.600 baseline is independently corroborated by a manual re-derivation in `project_cmlt_is_amlc_bug.md` on the same 250-window protocol. The **0.743 treatment value is not corroborated and is uncomfortably close to the 0.744 that the bug memory names as the WRONG (AMLc) baseline.** Unresolved — do not cite the pair |

### Clean by date or by construction

- `CONT-01` gtzan **CMLt 0.616** / AMLt 0.817 (2026-08-19, post-fix) — genuine CMLt.
- `RET-14` `abl_chain.log` CMLt (ballroom 0.964, beatles 0.843, hainsworth 0.746, hjdb 0.993,
  rwc 0.571, gtzan 0.668; 2026-08-18, post-fix) — genuine.
- `TCHAIN_RESCORE.md` — re-scored this month with the fixed code; the report states the field
  indices explicitly.
- `C37` (`count_term_and_bernoulli.md`, 2026-08-15, **one day** after the fix) — probably clean, not
  establishable; its values are 0.000/1.000/0.852, insensitive to the distinction anyway.
- `OCT-02` — the memory *asserts* correct unpacking; consistent with the date but not independently
  dated.

### Two adjacent metric hazards that are NOT the AMLc bug

- **`results_stratified.json` stores per-song values as bare positional 4-lists** `[beatF, CMLt,
  AMLt, downbeatF]` with no field names. The reading rests entirely on `eval_stratified.py`'s write
  order, which was not audited. Check before publishing any continuity column from that file.
- **The SMC-04 memory conflated the AMLt column with the CMLt column** (0.553 is the peak-pick AMLt,
  not CMLt). A different mix-up, same consequence.

---

## 8. NEGATIVES

### 8.1 The three forensic verdicts

**(a) End-to-end CRF training of Beat Transformer through the differentiable bar-pointer DP —
FAILED FOR FIXABLE REASONS. Do not cite it; do not repeat it as designed.**

Five confounds, ranked by how much of the negative each accounts for:

1. **The eval set cannot discriminate a sequence model.** `train_bt.py:291` evaluates
   `val_entries[:60]` from an unshuffled, sorted-dataset-order list; ballroom is first with 85 fold-0
   members, so **every rematch number is 60 ballroom songs, 100%**. An oracle constant-period
   metronome scores **F 0.9843 / CMLt 0.9597** there with 75% of songs at exactly F = 1.0 — *above*
   the winning arm's 0.9658. There was no tempo variation to exploit.
2. **No continuity metric was ever computed.** `train_bt.py:149-168` computes
   `mir_eval.beat.f_measure` only. Zero CMLt/AMLt for either arm anywhere. The experiment that
   "settled" the question never looked at the quantity the thesis is about. The r2 arm's three
   catastrophic evals (0.7649 / 0.6277 / 0.7864 vs vanilla 0/15) are exactly the octave-flip
   signature AMLt would forgive — unresolvable without a rerun.
3. **Provably conflicting targets at an untuned 1:1 weight.** BCE is applied to `widened_targets`
   (1 / 0.5 / 0.25 at the beat and ±1, ±2 frames; `train_bt.py:101-111`) while the CRF is applied to
   the **sharp** annotated path where those same frames are non-beat states. At a ~21.5-frame beat
   interval that is **18.6% of all frames in explicit disagreement**, never swept. The joint optimum
   therefore has strictly higher BCE than the BCE-only optimum *by construction*.
4. **The gap is frontend fit, not structure.** Vanilla's val F is nearly a deterministic function of
   its own BCE: `F = −0.1849·log(BCE) + 0.6838`, R² = 0.938, residual sd 0.0059. Projecting r2 onto
   that curve, its residual over 8 converged evals is **−0.0037 ± 0.0032 sem, t(7) = −1.15, p = 0.290,
   95% CI [−0.0114, +0.0040]** — *conditioned on frontend fit, the CRF arm is statistically
   indistinguishable from vanilla.* r2 ends at BCE 0.3087, which vanilla passes at **epoch 3**.
5. **Both headlines are selection-inflated maxima** (max over 15 evals; vanilla 0.9658 = +2.12 sd
   above its own late-plateau mean 0.9542 ± 0.0055; r2 0.9115 = +1.95 sd above 0.8937 ± 0.0091),
   **single seed**, with a 5.8× wall-clock asymmetry masquerading as epoch parity.

**Mechanism — this specific objective is destined to fail, and I reproduced it.** A conditional
path-NLL (`logZ − score(path)`) with free per-frame emissions is degenerate: it is maximised by
making the emissions confident on the annotated path, the structured analogue of unregularised
logistic regression on separable data. In a self-contained 45-line toy that imports nothing from
`vbpm/` (Krebs-style (tempo, phase) space, exact forward DP, T=200, free logits, Adam lr 0.1):
NLL/frame **0.05073 → 0.00000 within 100 steps** with the gradient norm collapsing
2.128e-02 → 4.627e-07. The recorded "logits to |55|, 100% saturated within one epoch, CRF gradient
dies" is mechanism-confirmed. The same toy reproduces the **unbounded λ**: on an exactly
constant-tempo path λ climbs 20 → 86.1 → 95.6 → 102.2 → 107.1 with a decaying gradient and no sign
flip. This matches the rematch's monotone 88.61 → 105.51 (+0.4785/epoch over epochs 20–29,
p = 5.11e-03) and is caused by `--smooth` making **85.37% of annotated tempo transitions exactly unit
ratio**.

**Why this does not transfer to the build recommendation:** the degeneracy is specific to a
*discriminative conditional-path* objective with free emissions. `p(z|x)` can be maximised by
emission overconfidence; the **marginal** `p(x) = Σ_z p(z)p(x|z)` cannot. Both candidate ideas the
user named optimise a marginal/ELBO objective. The record reached the same conclusion by another
route (`crf_baseline.py:1-8`: "NOT A RUNG … discriminative conditional objective, off-program") —
that demotion is a correct objective-class distinction, not a rhetorical dodge.

*Retry:* fair-test design is §5 items 3, 4, 21, 22, with the six specific changes — sweep the hybrid
weight α ∈ {0, 0.1, 0.3, 1, 3}; kill the target conflict (sharp BCE anchor **or** a tolerance-widened
CRF path); report at **matched BCE**, not matched epochs; log ‖g‖ per loss component before and after
`clip_grad_norm_(0.5)`; ≥3 seeds reporting late-plateau mean ± sd, never the max; `--eval-songs all`
with all six metrics on a discriminating val set. **Cost: ~20 GPU-h; do not spend it until §5 items
2–4 report.**

**(b) The log-normal tempo prior on the octave decision — DESTINED TO FAIL, and larger than
advertised.** Corrected numbers in §3.7. The prior is a regression-to-the-median operator with two
hard octave thresholds: right inside the band (87.9 → 94.3%), catastrophic outside (65.9 → 23.0%).
The class result is exhaustive: any song-independent additive `f(log p)` is either octave-periodic
(contributes literally nothing to octave selection) or not (then it changes sign somewhere on
[1, 6] s, or is monotone and converts one octave error into the other one-for-one). The best possible
member, oracle-fit with 26 parameters on the test set, recovers 12.0% of the headroom and transfers
0.00 pp. **The doc's stated mechanism is false** and must not be repeated: the prior emphatically
*does* punish halving a 2 s bar (Δ = +1.231 nats), and at w=1 it *repairs* 29 of the 83 baseline
halvings. *No retry needed to establish the negative; a retry (§5 item 20) is needed to make it
citable against Foscarin et al., because the object we tested — a decode-time prior on absolute
period — is not the object they proposed.*

**(c) "Structured inference has never deployably beaten peak-picking on SMC (0.589 vs 0.620)" —
FAILED FOR FIXABLE REASONS, and the negative is a truncated positive.** The primary artifact
(`git show 7301b92:experiments/kvae_barpointer/CAMPAIGN.md`, lines 145–151) is a **two-column** table:
peak-pick 0.620 F / **0.605 AMLt**; structured DBN 0.589 F / **0.656 AMLt**, with its own summary
line reading *"improves continuity."* The blocking negative was formed by quoting one column. The
honest four-part restatement is in §1 and §3.7. Confounds: metric-selective attrition in the record;
`min_bpm=55` making the correct level unrepresentable on 20.7% of the corpus (Simpson's paradox);
oracle leakage in the +0.186 CMLt row (it is a tempo-oracle result, not a decoder result); a hard
constraint under a point estimate (61.3% band accuracy, CMLt 0.0395 on the misses); and aggregation
over a bimodal population (86 winners, 75 losers at mean −0.2325). *Retry: §5 items 3, 4, 21.*

### 8.2 The other negatives, with verdicts

| negative | verdict | retry / cost |
|---|---|---|
| **R3 per-frame transition conditioning** (spread 0.0015 F; NLL 1.50747 → 1.50583/1.50601 ≈ 0.0016 nats/frame) | **Real but below the noise floor.** The document's own `r45-05` states a ~1 nat/song float32 DP readability floor, and this effect is far under it. The two claims are in tension and the document never reconciles them. Retired "on effect-size grounds, not failure" is generous | Check the dtype in `r3_generative_mixture{,_v2}.py` and re-run the NLL comparison in **float64**. Cheap. Not worth more |
| **R4 audio-conditioned transition head** | **Degenerate by construction, not merely null.** `results_controls.json`: `lambda_t_std` **exactly 0.0**, `weight_std_max` 1.13e-09, `r4_zerotrunk_beatF == r2mix_beatF == 0.9471657091592964` with **100% event agreement at 70 ms**. The learned conditioner does nothing at decode. n=30 songs | Any reimplementation must ship **zero-trunk parity + `lambda_t_std`** as a pre-registered gate. Free |
| **R4 conditioned tempo prior** | **Two-sided.** Priors ARE learnable and readable (entropy 2.47 vs uniform 3.58; Spearman vs annotated tempo 0.460 unsupervised / 0.769 supervised / 0.797 short-crop), but deployed F is flat across the family (0.9115 / 0.9129 / 0.9132, spread 0.0017) and the unsupervised prior prefers the correct octave in **0 of 8** wrong-octave cases (3/8 supervised). n=8 is far too small a denominator to carry weight (binomial 95% upper bound on 0/8 is 0.37; 3/8 vs 0/8 is Fisher p = 0.20), and 6 of the 8 are hjdb drum-and-bass with downbeat-only annotations | In-domain only; **SMC untested**. That is the escape, and E3 is the test |
| **R4.5 rich Gaussian emissions** | **Well-earned negative** — arguably the best-run experiment in the inheritance: v0 correctly diagnosed as invalid (emission domination, 22,600-nat gaps), a validity gate defined *in advance*, two independent fixes built (tied covariance + PCA-16; LDA control), the gate shown to open in both, and only then the verdict read | The `D=4–8` scale control (§5 item 15) is the one missing number |
| **learn-obs (`observation_lambda` by plug-in marginal)** | **Ill-posed, correctly retired.** The `(obs−1)` normalisation plus obs-dependent state partition inject hundreds of nats favouring small obs; **pure noise selects obs=2 by ~1200 nats**. A blind black-box suite rediscovered it from outside the code (16/16, verified running today) | Do not present obs 6→4 as a learned finding. **Most transferable warning for any neural HMM that wants to learn observation weighting** |
| **Dither-corrected mixture repair on SMC** | **Measured null** on all three metrics, 110/217 decodes bit-identical | Keep the diagnosis; drop the repair claim |
| **KVAE process-noise levers (3 of 3)** | Genuine nulls (leak tests clean): Q_reg sweep gap 0.241→0.257 with SMC F 0.654→0.620; rubato augmentation gap 0.166→0.162, F→0.640; adaptive per-frame noise gap 0.166→0.159, F→0.630 | **Do not re-try the process-noise axis.** But note the target metric is "CMLt − CMLc" from the pre-fix package (§7), so its *meaning* is unverified even though the direction is |
| **IWAE / FIVO** | Dead, and dead for a reason that needs no measurement: at T=1500, **ESS = 1.08/32**, so IWAE degenerates to the ELBO of the luckiest draw. 18 runs, ANOVA p = 0.450 | Use as a **design argument for exact forward-backward over particle methods** — a genuine positive distinction for the paper. p=0.450 with no artifact is not citable |
| **Categorical q(l_t)** | Falsifier fired (§4.2). Caveat the record adds: at pooled seed sd 0.082, n=8 gives MDE ~0.115, so "worse" is better read as **undetermined-to-negative** | — |
| **Metrical-level selection via classifier likelihood** | Hurts everywhere (val 0.9506 → 0.9180; SMC zero-shot 0.483 → 0.403) because cross-grid likelihood comparison is invalid off the training manifold. **Independently replicated by a second criterion in the same campaign** | Forecloses the cheapest route to "let the model pick its own tempo hypothesis". Solid |
| **Per-song λ selection by marginal likelihood on SMC** | **Robust negative, triple-sourced**: ρ = −0.230 (p = 6.4e-4) on Beat This, ρ = −0.212 (p = 1.7e-3) on Beat Transformer, and `PHASE0_RECERT` §4.2 finds λ_oracle uncorrelated with median BPM (ρ = 0.019, p = 0.786) and with max tempo change (ρ = 0.098, p = 0.15) | **A per-song λ selector is not a viable direction.** This is also a warning for the build: "be more properly probabilistic about tempo stiffness" measurably points the wrong way on SMC |
| **Music-type-conditioned λ (the direct test)** | **Clean negative, measured during this audit.** Full 13-λ × 217-track sweep: best global λ=2 → F 0.6013; ORACLE 2-group BPM stratum 0.6019 (+0.0006); ORACLE 4-group BPM quartile 0.6047 (+0.0034); ORACLE 2-group tempo-change stratum 0.6056 (+0.0043); per-song oracle 0.6495 (+0.0482). **Even with oracle group labels AND oracle per-group λ, a coarse music-type prior captures ≤9% of the headroom.** Corroborated by `lambda_per_song.csv` (n=2180): per-song λ medians near-identical across ballroom/beatles/gtzan/hainsworth (57.20) with within-dataset IQR 43.7–70.0 far exceeding between-dataset spread | **Do not build the novelty on a genre/music-type-conditioned λ.** Build it on the per-song / heterogeneous-scale axis, which holds +0.0482 F and +0.0710 CMLt |
| **MERT as a conditioning input** | Retracted (§4.2), and independently: MERT's best layer loses to a plain ACF baseline at octave classification (0.5707 vs 0.8301 on n=200; 0.8344 vs 0.8679 on n=553), and at 3328 input dims and 2× wall time it scores **worse** val NLL (1.5182) than the 2-channel activation arm (1.5029) | **The project's precedent that a large pretrained music encoder did not beat a signal-processing baseline at exactly the conditioning task a music-conditioned HMM would need.** Weigh it |
| **Slow-tempo fine-tuning** | Clean-shaped negative: 45 slow SMC tracks split A(train)=22 / B(held-out)=23; held-out F never exceeds its step-0 value across 12 measurements and two learning rates (1e-5: 0.551 → 0.497 min → 0.519; 3e-5: 0.551 → 0.531) while train F stays 0.987 → 0.974 | Ran under the retired `chart` env with **no continuity metrics at all**. Re-run under `analyze-smc` with `_harness` scoring before use |
| **Gradient forensics** | All 15 pairwise group cosines are **positive** (0.5019–0.9097), so there is **no gradient conflict**; only the *starvation* reading survives — SMC-slow carries gradient norm **3.835** against train_mid's **0.229** (16.8×) at loss 0.244 vs 0.0186 (13.1×). Per-module: task_heads cosines ~0.98 while frontend cosines go **negative** (smc_slow\|train_mid −0.0955) | Directly supports the ICASSP tempo argument and is **cited by nothing**. Producer needs VBPM paths — re-run in-repo |

---

## 9. WHAT DOES NOT RUN

| item | error / status |
|---|---|
| `vbpm/variants/archain.py` | **DELETED** from the working tree (HEAD `29e7c97`) and absent from the `archain-anchor` branch. Only `__pycache__/archain.cpython-310.pyc` (25,469 B) survives. Every item in `audit_2026-08-19.md` (claims A–Q, bugs S1–S10, fidelity D1–D9) refers to code that cannot be read or run today. Recover with `git -C /home/sogang/jaehoon/VBPM show 29e7c97^:vbpm/variants/archain.py` |
| `archain_lab.py` | Does not exist in the tree **or** on the branch. `find /home/sogang/jaehoon -name archain_lab.py` → nothing. The `torch.manual_seed(0)` claim cannot be checked |
| `vbpm/variants/anchor_k.py` | Absent; `anchor_swap.py:23` `from vbpm.variants.anchor_k import AnchorKVAE` is a dangling import. Only `.pyc` remains |
| `rungs/r4_neural_hmm.py` | **Never existed.** `rungs/__init__.py` advertises it and `train.py:19` imports it, so **`train.py` is dead**. `run_r2.py:1` exists precisely because of this. Worse, `r4_5_rich_emission.py:1` says it was "folded into" that missing file — the deprecation pointer is also dangling. **The neural-HMM rung is not an inherited asset; it must be written** |
| `vbpm.run` / `vbpm.config` | `ModuleNotFoundError: No module named 'yaml'` in the `analyze-smc` env. Nothing in the VBPM package can be trained or evaluated from this project until `pyyaml` is installed |
| `pytest` in `analyze-smc` | `ModuleNotFoundError: No module named 'pytest'` — verified again today. **No rungs_lab test suite can be run in the nominated env.** They pass under `chart` (used for the 15/15 and 31/31 runs) and `vbpm` (pytest 9.1.1; the 16/16 run). `test_structured_dp.py` is the exception: it has a plain `main()` and runs anywhere |
| `experiments/bt_e2e/mixture_kernel_probe.py` | Under `jaehoon_bt`: `ModuleNotFoundError: No module named 'mir_eval'`. It also does `import final_eval; final_eval.DEVICE = "cuda:2"` **at module scope** and hardcodes `DEVICE='cuda:2'` at line 33, so merely importing it seizes a GPU |
| `experiments/bt_e2e/train_r2_em_fps2.py` | `import final_eval; final_eval.DEVICE = 'cuda:1'` at line 24; needs `vanilla_best_prelim.pt`, which **does not exist** |
| `experiments/bt_e2e/{train_r2_em,train_r2_frozen_perdataset}.py`, `final_eval.py`, `oracle_ceiling.py` | All need trained e2e checkpoints. **`ls experiments/bt_e2e/*.pt` → "No such file or directory"** — `vanilla_best.pt`, `vanilla_best_prelim.pt`, `r2_best.pt`, `r3_frozen_net.pt` are gone from both trees |
| `experiments/bt_e2e/launch.sh` | Points at the **retired** env `/home/sogang/mnt/db_2/anaconda3/envs/chart` and at `/home/sogang/jaehoon/VBPM/experiments/bt_e2e`, a path that no longer exists after the move |
| `scripts/grad_analysis.py` | Imports `beat_this` from `/home/sogang/jaehoon/VBPM/external/beat_this` and reads a mel cache at `/home/sogang/jaehoon/VBPM/cache/mel/real_train_mel50` — outside Analyze-SMC and outside the moved tree. `data/grad/grad_report.json` cannot be regenerated in-repo |
| `scripts/grad_ray.py` | Its own log header shows it ran under the `chart` env, not `analyze-smc`, and it depends on VBPM paths |
| `synthetic_grid_probe.py` on CPU | Hardcodes `DEVICE='cuda:1'`. Forced to CPU, the exact-EM leg (50 crops × T=700, 500-point λ grid) did not finish the first EM iteration in **>90 minutes**; a second attempt at 45 min also failed. **This probe needs a GPU** |
| `rungs_lab/mixture.py:17` | `RuntimeWarning: divide by zero encountered in log` on `np.log(adj / adj.sum(axis=1, keepdims=True))`. **Benign and intended** (−inf for non-adjacent tempo pairs, always logsumexp'd against a finite component since w ≤ 0.95), and it is the first line of `lamselect_smc.log`. Same warning from `r4_5_rich_emission.py:65`. But `hygiene_check.py` exists to police exactly this, and it separately reports the R3 `w_t` head producing −inf on **6/20 seeds** when `w_t` saturates — clamp before training |
| `/home/sogang/jaehoon/VBPM/vae_rescue/OVERNIGHT_REPORT.md` | **Does not exist.** No `vae_rescue/` directory at all. This is the cited origin of the KVAE SMC 0.654 |
| `ov_anchor_resid.log`, `ov_anchor_resid_save.log` | Not in `/home/sogang/jaehoon/VBPM/logs/` (27 entries, 778 files, none matching `ov_*`) or anywhere else. The splicing refutation (RET-06 / C40) is un-auditable |
| `vbpm/docs/audit_2026-08-19.md` | Deleted from the VBPM working tree; the copy in `Analyze-SMC/vbpm/docs/` is the surviving one |
| `results_stratified.json` keys | The record cites `val_fold0.table` / `gtzan.table`; **those keys do not exist**. The file has `per_song` plus fps/terciles/spearman metadata |
| `data/transition_lambda_sweep_results.csv`, `data/ibi_variability.csv` | **Producer scripts lost.** Only two `to_csv` writers exist in `scripts/` (`activation_analysis.py:435`, `gt_tempo_lambda_sweep.py:260`), neither writes these. `AS-004` (the error taxonomy) and `AS-005` (the λ sweep) both depend on a file that cannot be regenerated |
| `scripts/` — no producer for | sweet-spot-BPM routing, Beat Transformer tempo accuracy, or the `0.685–0.702` envelope range. `grep -E "sweet\|rout"` over `scripts/` returns nothing |
| Env note | The `vbpm` env's numpy is 2.2.6; `analyze-smc` has 1.24.3 and **cannot unpickle** `octave_audit.npy` (`ModuleNotFoundError: No module named 'numpy._core'`). Use the `vbpm` interpreter for those artifacts |

---

## 10. WHAT I COULD NOT DETERMINE

1. **Which of the 217 SMC tracks failed the engine certificate.** `results_smc_diagnosis.json` records
   `certificate = [216, 217]` and never names the disagreeing track. Any claim of "n = 217,
   event-identical" is off by one and the identity of the exception is unknown.
2. **The provenance of `smc_rich_86fps` vs `smc_rich_foldhonest`.** Both are 217 files at
   86.1328125 fps and they differ numerically (max |Δ| = 9.435), so they are two frontends, but
   **neither carries an `extractor` or `fold` key** — unlike `foldhonest_val_rich`, which does
   (`beat_this-fold7`). I cannot certify which is fold-honest, so I cannot certify the KVAE rerun's
   honesty in advance.
3. **Whether `WALK_MIX` was fitted on frame-quantised or sub-frame annotation times.** The fitting
   script is unnamed. This decides whether `(0.687, 0.313)` / `(0.00029, 0.00377)` is a musical
   statistic or a second instance of the dithering artifact.
4. **`asap468`.** It is an index into a VBPM song list I could not resolve; index 468 of the sorted
   ASAP annotation keys is a Chopin Ballade (263 bars, median ratio 1.119, p90 1.605, max 3.720), a
   different piece. The whole HET-01 single-song mechanism therefore rests on an unidentified track.
5. **The true source of the misattributed joint-training quotation (PRE-06).** It is not in
   arXiv:1810.10002 or its TISMIR version, and a web search for the exact string returns an unrelated
   NER paper. Either find the source or drop the sentence.
6. **The 0.642 vs 0.6525 wide-DBN oracle-λ disagreement.** `smc3_certify.py`'s docstring targets
   0.642, `smc3_ladder.py`'s docstring says 0.6525, the ladder log's ORACLE row is 0.6495, and
   `exp_decoder_ceiling.py` prints "(paper 0.642)" beside its own recomputed 0.6525. Four values, one
   nominal quantity. I did not run `smc3_certify.py` (recert agent).
7. **The 0.9033-vs-0.9504 family gap** between `results_r2_val.json` and RESULTS.md's DEPLOYED rows
   for nominally the same R1 on the same 146 songs. Most likely a different frontend (pretrained
   `bt_fold0_repacked.pt` vs a from-scratch e2e vanilla) plus the peak-snap flag, but it is
   **unexplained in the record** and it is upstream of the entire deployment half of RESULTS.md.
8. **Whether gradient clipping was active in the bt_e2e runs.** `clip_grad_norm_(params, 0.5)` is
   applied to the *sum* of both losses' gradients; at batch 1 it is almost certainly binding, which
   would dilute the BCE direction's share of each step. **No grad norms were ever logged.**
9. **Whether the r2 arm's three catastrophic evals were octave/half-bar flips or genuine tracking
   loss.** No checkpoints, no activations, no continuity metrics. Permanently unresolvable without a
   full retrain — and the retrain would use the fixed scheduler, so it would not reproduce the
   numbers anyway.
10. **The write order of `eval_stratified.py`'s positional 4-tuples.** I read the aggregated table's
    key names but did not audit the per-song write, so the continuity columns of
    `results_stratified.json` are labelled by assumption.
11. **Which checkpoint produced the tchain scoreboard in `project_state.md`.** 30+ run directories,
    609 MB of checkpoints, and the doc names none. The scoreboard has no SMC row and no artifact.
12. **The n behind `MARG-01`'s "F 0.000 peak-picked vs F 0.602 through the sweep".** The memory states
    no n; `VBPM_ASSETS` says ~100 gtzan songs, ~150k frames, sourced to a third document. Pin it
    before printing.
13. **Whether `LM-19`'s performance figures are real.** "7,605× fewer ops/frame", "madmom 1.04 s/song
    vs ours 1.95 s/song", "100% frame agreement with hmmlearn/librosa/pomegranate" are **unexecuted
    assertions inside triple-quoted strings**; the test that *does* run prints "~1 GB and 260M
    ops/frame" instead. Only the internal consistency is verified (280,763,536 / 36,920 = 7,605
    exactly). Re-measure before citing any of them.
14. **`LM-11`'s `TRAIN_DEFAULTS`.** `getattr(module, 'TRAIN_DEFAULTS')` returns `None` — the named
    object does not exist in `r3_conditioned_dbn.py`. The hyperparameters (epochs=12, lr=1e-3,
    clip 1.0) may live elsewhere; the claim was written from memory, not from the file.
15. **The E-step tolerance: 4.8e-6 or 4.8e-7?** RESULTS.md:101 says 4.8e-7 for the R2 EM core and
    :248 says 4.8e-6 for the R4.5 emission E-step "meter mixture included". They are plausibly two
    different certifications of two different models — but the record never says so, and I did not
    run either.
16. **Whether the "identical to control on every metric" p(N|φ) result used a pre- or post-fix
    scorer**, and where the "85–100% m=4, meter latent KL ≈ 0.02" figures actually live. I could not
    locate them in `count_term_and_bernoulli.md`; treat both as unsourced.
17. **The three null figures behind the octave nomination ceiling** (log-uniform 70.1%, train-prior
    78.1%, permutation 31.3%). **No saved artifact for any of them**, and the ceiling means nothing
    without them.
18. **`LM-52`'s full-song-vs-45 s triple** (asap 0.164 / rwc 0.258 / beatles 0.349 vs 0.713). Prose
    only. It matters directly because **SMC excerpts are ~30–40 s**, so the "rigidity is costly on
    long songs" argument may not transfer to SMC at all.
19. **Novelty.** A negative literature search is the weakest evidence type in this ledger and it is
    load-bearing. Three gaps remain open (§5 item 6), and **Electronics 11(24):4186 ("HMM with RNN
    observer") could pre-empt the method directly and is unverified** (MDPI returned 403).
20. **Whether the concurrent recertification agent's `PHASE0_RECERT.md` §2 was ever filled in.** Its
    section header "## 2. The core certification: R0 vs R1 on SMC" contains the literal placeholder
    `*(filled in below)*`. The 216/217 figure is real and on disk; the separately-claimed "30/30
    event-identical, re-earned twice, shipped AND bare" traces only to a memory record with no
    reachable artifact. **Ask that agent to fill in its own section rather than racing it.**
