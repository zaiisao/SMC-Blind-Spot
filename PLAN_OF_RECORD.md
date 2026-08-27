# PLAN OF RECORD — SMC metrical-level work, Aug 26 → Sep 16 2026

## 1. CORRECTIONS

**C1 wording — "tracked at ~double rate" (sub-55 band).** Wrong: "64.4% of tracks below 55 bpm are tracked at ~double rate." Right: "64.4% are tracked at a faster metrical level (r > 1.5); of the 59 sub-55 tracks, 44.1% are near 2x (1.5–2.25), 18.6% near 3x, 1.7% at 4x+, and only 37.3% lie in 1.8–2.2." The ≥140 side *is* clean halving (41 of 62 halved tracks in r ∈ 0.45–0.55).

**C1 pooling.** Quoting only pooled figures understates the effect. Report both: sub-55 doubling 0.644 pooled / **0.701 macro-averaged over datasets**; ≥140 halving 0.163 pooled / **0.224 macro**. The pooled halving is diluted by hjdb (120 drum-and-bass tracks, all ≥140, halving 0.000 = 31% of the band). Also state n: the sub-55 band is 59 tracks from 6 corpora, only 4 with n≥5; bootstrap CI on 0.644 is [0.525, 0.763].

**C1 SMC denominator.** `crossdataset.json` contributes **140** fold-honest SMC tracks, not 217, and SMC's own sub-55 doubling there is 14/29 = 0.483 — not the 0.356 quoted from `anatomy_honest.json`. Never mix the two denominators in one table.

**C2 endpoints.** Wrong: "(AMLt−CMLt)/AMLt is ~0.166 at epoch 9 and ~0.166 at epoch 99." Those are the true shipped values (0.1659, 0.1655) but they are cherry-picked endpoints of a non-monotone path (peak 0.2066 at ep49, trough 0.1391 at ep69), and the wide arm goes 0.1687 → 0.1771 — *up*. Right: "the relative gap stays at 0.176 ± 0.022 (shipped) and 0.176 ± 0.013 (wide) across ep 9–99; the fitted change is −0.026 [−0.076, +0.024] and −0.008 [−0.039, +0.022] while F rises 0.478 → 0.592 and 0.459 → 0.582. Over the F-plateau (ep ≥ 39) both arms sit at 0.174."

**C2 claim strength.** Wrong: "the loss is blind to metrical level." Contradicted by the data. The **absolute** gap AMLt−CMLt grows 1.8–2.1x over the same span (shipped 0.052 → 0.094; wide 0.046 → 0.097, slope p = 0.009); on the slow band 0.065 → 0.115 and 0.060 → 0.131. Right: "the objective never *reduces* the metrical-level error fraction, and the absolute octave-error mass grows while F improves." Caveat that must be stated: ten autocorrelated checkpoints from one seed per arm; between-arm difference at matched epochs (0.018) is the size of the within-arm sd. Read the CIs as "no trend detectable", not "trend bounded".

**C3 numbers — invalid comparison, must be restated.** Wrong: "0.098 → 0.822 → 0.978." This mixes a **post-fix** `anat_pw32.json` (11:43) against **pre-fix** `anat_pw8.json` (02:44) and `anat_pw128.json` (02:50), across the 09:45 `error_anatomy.py` boundary — the one comparison the hazard note forbids. It also uses inconsistent denominators: `anat_pw8.json` has 209 tracks (8 dropped as degenerate, 4 of them sub-55), so its sub-55 denominator is 41, not 45. Right, on the 209-track intersection with denominator 41: **0.098 → 0.829 → 0.976** (post-fix pw32), or **0.098 → 0.878 → 0.976** if all three points are taken from the pre-fix code. Re-running pw8/pw128 post-fix is inference-only and should be done before publication. Direction and magnitude of the effect are unharmed (the fix moves pw32 by −0.045 against a +0.88 slope).

**C3 framing.** Missing and necessary: lowering pos_weight does not fix slow tracks, it halves everything. pw8 puts 51.7% of all tracks and 34.1% of sub-55 tracks into HALVED, and F falls: 0.4952 (pw8) / 0.5436 (pw32) / 0.5043 (pw128), CMLt 0.325 / 0.346 / 0.233. Metrical level is monotone in pos_weight; accuracy is not.

**C4 — one clause is wrong.** "32 implies 39 frames/beat = 0.78 s = ~77 bpm" is correct as a statement about the *corpus*, but wrong if read as the density the *loss* is balanced at. `dataset.py:473-502` uses a collar of 7 frames (`widen_target_mask=3`); `loss.py`'s `spread(targets, factor=2)` is `max_pool1d(..., 13)`, so the loss excludes **13** frames per beat, not 7. True negatives per beat in the loss = 39 − 13 = 26, so the shipped weight over-weights positives by 32/26 = **1.231**, a **+0.208 nat** bias (effective threshold 0.448, not 0.5). Write it as: corpus density 39 frames/beat (0.78 s, 77 bpm); the loss is balanced as if it were 45 frames/beat (0.90 s, 67 bpm); the 1.23x is exactly that gap. Magnitude: at the measured local slope (~0.53 sub-55 doubling per nat between pw8 and pw32), 0.208 nats accounts for ~0.11 of the 0.83 swing — a real bug, one sentence, not the mechanism.

**C5 — the doubling number should not be reported as an effect.** Wrong: "the sub-55 doubling fraction got WORSE (0.511 → 0.578)." Post-fix that pair is correct (shipped-old 0.511, wide-old 0.578) but per-track it is McNemar b=5, c=2, exact two-sided **p = 0.453** — 3 net tracks, one seed, n=45. Note also that "0.511 → 0.578" is a coincidence collision: pre-fix, 0.578 is the shipped-*per-item* cell and wide-old is 0.614. The pre-fix table implies a +0.103 penalty; the consistent post-fix penalty is **+0.067**.

**C5 — the claim is understated on continuity, not underpowered.** Missing: AMLt is significantly hurt. AMLt all −0.0205 (per-track p=0.005; run-level t=−5.42, p=0.032); **AMLt <55 −0.0464** (per-track p=0.026; run-level t=−7.38, **p=0.018**, seedwise −0.053/−0.034/−0.052); CMLt all −0.0156 (p=0.048). Corrected wording: "wider augmentation does not improve F on the slow band (Δ = −0.006, 95% CI [−0.017, +0.005]; MDE80 ≈ 0.008–0.015 F, so a small gain could have been missed) and it measurably degrades continuity: AMLt −0.021 overall and −0.046 below 55 bpm. The supporting evidence is the transition table, not the doubling fraction: wide pushes 12 tracks into DOUBLED (6 from GOOD, 6 from PHASE) against 6 recovered from HALVED; 41/217 tracks (19%) change category while the marginals barely move."

**C7 — arithmetic.** Wrong: ep99 = 115/30/45/26 (sums to 216). Right: **115/31/45/26**. ep4 (54/17/143/3) and ep24 (104/26/72/15) reproduce exactly. Also: there is no `anat_stock_ep99.json`; the "ep99" column is `anat_f0_shipped_s0.json` and should be labelled as that arm.

**C8 — arithmetic.** Wrong: GOOD at ep24 = 104 / 105 / 99. Right: **104 / 106 / 99**. Full rows: ph13_ep24 = 106/37/48/26, ph30_ep24 = 99/49/33/36.

**C8 — the central claim does not survive.** Wrong: "tracks land in PHASE, not GOOD." Of the 72 stock-DOUBLED tracks at γ=1.3, **14 go to GOOD and 14 go to PHASE** — an even split, and the 14 reaching GOOD are real repairs (dF +0.0745, dCMLt +0.2116). Net GOOD is flat because GOOD simultaneously **loses 15 tracks** (6→PHASE, 2→DOUBLED, **7→HALVED** at dF −0.067, dCMLt −0.178). At γ=3.0: 16 DOUBLED→GOOD against 24 GOOD lost, 15 to HALVED at dF −0.121. Right: "the phase loss does convert doubled tracks into correctly-tracked ones, but it over-suppresses on tracks that were already right, and the two effects cancel in the GOOD count. The failure mode is halving, not phase."

**C8 — the category table is not cut-robust and should not be the headline.** GOOD delta vs stock at F-cuts 0.30…0.70: γ=1.3 gives +14 +13 +8 +6 **+2** −4 −4 −3 +1; γ=3.0 gives +9 +8 +3 −5 **−5** −9 −6 −8 −4. The sign flips inside the plausible range. Under log-symmetric r windows γ=1.3's GOOD delta is +2 / −2 / +5 / +1 — noise. What is stable across every cut and window is DOUBLED falling 72→48→33 and HALVED rising 15→26→36. Lead with the paired continuity metrics instead: γ=1.3 dCMLt **+0.0363 ± 0.0084 (t=4.31)**, dAMLt +0.0197 ± 0.0081 (t=2.43), dF +0.0035 (n.s.); γ=3.0 dF −0.0148 (t=−2.22).

**C9 — GTZAN band sizes.** "the gain scaling with each corpus's doubling rate" is supported band-wise, but GTZAN's sub-55 band is **n = 6** (dbl 0.500 = 3/6). Per-band ΔF (delta 0→8): <55 n=6 +0.0425; 55–70 n=24 +0.0121; 70–100 n=68 +0.0127; 100–140 n=91 +0.0178; >140 n=61 +0.0018. Do not quote GTZAN's 0.500 next to SMC's 0.356 as comparable estimates. Also: `subdivision_oracle.py` computes its own ratios with the **pre-fix** convention (its baseline reproduces `anatomy_honest.json` 0.356/0.316 exactly), so its `dbl<55`/`half>100` columns must not be compared to any post-fix `anat_*` number. Its F/CMLt columns come from the standard evaluator and are fine.

**Hazard claim in the brief is false.** "All cache/anat_*.json should now be post-fix" is wrong. Seven files are PRE-fix (no `degenerate`/`span`/`n_est_whole`): `anat_pw8` (n=209), `anat_pw128`, `anat_cl005`, `anat_cl015`, `anat_cl050` (n=208), `anat_f0_wide_peritem_s0`, `anat_f0_shipped_peritem_s0` — plus `cache/anatomy_honest.json` (00:24). The fix is systematically directional, not noise: it removes 1.6–2.0 estimated beats per track (affecting 67–75% of tracks), moves 3–7 tracks out of DOUBLED and 3–4 into HALVED, and shifts every `logit_beat` mean by +0.12 to +0.22. `miss_frac` is the only column unchanged (0/217 tracks) and the only one safe to compare across the boundary.

**`MORNING_REPORT.md` §3, the 2×2 table.** The halved column reads 0.316 in all four cells because it was copy-pasted from the released-model baseline, not recomputed. Real pre-fix values: 0.316 / 0.316 / 0.316 / **0.342**; post-fix shipped-old **0.395**, wide-old **0.342**. **The pre-registered gate "halved shrinks by ≥0.10 → PARTIAL" was never actually evaluated.** Recompute all four cells or withdraw the verdict.

**`LOSS_DECISION_TREE.md` baseline** "doubled 0.356 / halved 0.316" derives from pre-fix `anatomy_honest.json` and is being used to judge post-fix arms. Regenerate it.

**Not a correction, but confirmed clean:** the leak-contaminated fold-0 released mean (all-SMC F 0.7120 vs honest fold-0 0.6510) appears in **zero** root markdown files. C9's GTZAN half uses `final0` on a corpus upstream holds out entirely — not contaminated by the `stage_wide_selection.py` round-robin, which only affects our own checkpoints.

---

## 2. WHAT IS SOLID

**S1 — Bidirectional metrical-level error is real, cross-corpus, and not an artefact of pooling, band sizes, or difficulty.** 1,501 fold-honest tracks, 12 datasets. Sub-55: 0.644 pooled / 0.701 macro tracked at a faster level; ≥140: 0.163 pooled / 0.224 macro halved. The AMLt−CMLt gap is U-shaped, minimum in 100–140 bpm (0.0282 pooled, 0.0264 macro). Equal-n deciles give a clean U with minimum at 115–134 bpm (gap 0.024). The gap minimum falls in 100–140 for 8 of 12 corpora; the 4 exceptions have no tracks below 70 bpm and cannot show the left arm. Dropping SMC *raises* the effect (sub-55 dbl 0.800, gap 0.364), so it is not a difficulty confound. **This is the paper's strongest claim.**

**S2 — Training does not reduce the metrical-level error fraction, and increases its absolute mass.** Relative gap 0.176 ± 0.022 (shipped) / 0.176 ± 0.013 (wide) across ep 9–99, fitted change −0.026 [−0.076, +0.024] and −0.008 [−0.039, +0.022], against F +0.114 and +0.123. Absolute gap grows 1.8–2.1x (0.052 → 0.094; 0.046 → 0.097, p=0.009). Both arms sit at 0.174 on the F-plateau. Relative gap is uncorrelated with F (r=−0.11). Needs a second seed per arm before the CIs are quotable.

**S3 — pos_weight causally sets the metrical level, monotonically, across the whole tempo range.** Sub-55 doubling 0.098 → 0.829 → 0.976 at pw 8/32/128 (209-track intersection, denominator 41). Monotone at every ratio threshold (r>1.25: 0.171/0.902/1.000; r>1.9: 0.049/0.659/0.927). Not confined to slow tracks: all-track median r 0.680 / 1.311 / 2.167; 55–70 doubling 0.000/0.446/0.893; >100 doubling 0.000/0.053/0.447. The three arms are otherwise identical (same 25 epochs, seed, fold, data, augmentation; only `pos_weight_override` differs), so the causal reading is clean. Fold-0-only subset agrees (0.111/0.889/1.000, n=9). Accuracy is *not* monotone: F 0.495/0.544/0.504.

**S4 — the loss's collar and the weight's collar disagree, by a factor of 1.23.** `dataset.py` assumes 7 excluded frames per beat; `loss.py` excludes 13. Corpus density 39 frames/beat (77 bpm); the loss is balanced as if 45 frames/beat (67 bpm). +0.208 nat positive bias, effective threshold 0.448. Accounts for ~0.11 of the 0.83 sub-55 doubling. Verified in code, unaffected by every data hazard. Overlap correction is negligible below ~230 bpm; `crop(factor=2)` costs 0.8% of negatives.

**S5 — wider tempo augmentation does not help and hurts continuity.** ΔF on <70 = −0.0062 [−0.0170, +0.0046], 3 seeds, p=0.262. AMLt all −0.0205 (run-level p=0.032); AMLt <55 −0.0464 (run-level p=0.018, seedwise spread only 0.019). 19% of tracks change category with near-identical marginals; wide pushes 12 into DOUBLED against 6 recovered from HALVED. Sub-55 doubling 0.511 → 0.578 post-fix, but that is 3 net tracks (p=0.45) and is evidence only in aggregate with the transition table.

**S6 — decode-time repairs all fail.** madmom DBN −0.051 overall (−0.142 below 55 bpm); Beat-Transformer-predicted-tempo DBN 0.627 → 0.565; tempo-banded peak-picking threshold, cross-fit, −0.0055. Extended: the C9 screen cannot be made non-oracle — deriving phase from the model's own decoded grid is an exact no-op (0.6449 at every delta, by construction), and from the half-rate self-grid it is destructive (−0.180 at delta=8).

**S7 — the phase-weighted loss buys continuity at flat F, and its cost is halving.** γ=1.3 vs stock at ep24: dCMLt **+0.0363 ± 0.0084 (t=4.31)**, dAMLt +0.0197 (t=2.43), dF +0.0035 (n.s.). γ=3.0 overshoots: dF −0.0148 (t=−2.22), dCMLt +0.0226 (n.s.). Median est/true ratio 1.236 → 1.044 → 1.000. DOUBLED collapses 72 → 48 → 33 while HALVED rises 15 → 26 → 36, stable across every F-cut and r-window tested. The 14 DOUBLED→GOOD conversions at γ=1.3 are real (dF +0.075, dCMLt +0.212); the 7 GOOD→HALVED losses are the price.

**S8 — the oracle subdivision screen bounds the level-fix at +0.030 F on SMC.** 0.6270 → 0.6573 at delta=8 (reproduced independently from `ts_tta_cache`). GTZAN 0.7868 → 0.7994 from a released checkpoint on a held-out corpus (clean). The gain rank-correlates with per-band doubling rate.

**S9 — TTA and a level fix mostly stack, and probability-space fusion is free.** TTA alone +0.0180; oracle screen alone +0.0303; both +0.0436 — **84% retention** (+0.0256 of +0.0303). Per-track gains correlate only r=0.289 (46 tracks helped by both, 41 TTA-only, 34 screen-only). Mechanism: TTA de-doubles only 10 of 23 doubled tracks, and 194 never-doubled tracks each gain +0.0127 — TTA is activation denoising; the loss targets level choice. 90% of TTA's gain is below 70 bpm (+0.0342 vs +0.0034 above). The 3-rate subset 0.85/1.0/1.2 is **LOFO-honest**: leave-one-fold-out selection picks it in all 8 folds, giving exactly 0.6449 — no selection bias. Probability-space fusion beats logit-space: 0.6449 → 0.6490 alone (p=0.26), +0.0081 at delta=8, best stacked cell **0.6811** (+0.0542 over base, p=1.3e-12). One line of code, never loses.

**S10 — PHASE tracks are not offset; they are phase-incoherent and low-evidence.** Circular resultant R of `fp_phase`: PHASE 0.198 median, only 12% with R>0.5 — the *least* coherent group (GOOD 0.357, HALVED 0.502). Pooled FP-phase histogram is flat, no bump at 0.5. `logit_beat` = −0.141 (vs GOOD +1.556), `miss_frac` 0.601 at r=1.205. **There is no offset to estimate on 88% of these tracks.**

**S11 — the ±60 ms shift-tolerance window is genuinely unsupervised and predictions fill it.** 209 tracks, 7,019 matched beats: pooled bias +2.4 ms, sd 31.5 ms; within ±20 ms only 0.466; per-track |bias| > 35 ms on **0%** of tracks; median per-track jitter 29.9 ms ≈ 0.036 IBI. Median recall 0.70 at ±70 ms vs 0.44 at ±35 ms. Predicted by the code: `look_at = cropped_targets + (1 − spread(targets, factor=2))` gives no gradient of either sign distinguishing a peak at the beat from one 60 ms away. corr(F, jitter) = −0.398.

**S12 — a period-only fix has a hard ceiling of +0.027 F on SMC.** The 71 level-wrong tracks have AMLt 0.362 (DOUBLED) / 0.368 (HALVED), not GOOD's 0.750 — so their phase is also wrong for ~60% of the duration. Only **12 of 217 tracks (5.5%)** are level-wrong with AMLt ≥ 0.7. Oracle level fix (every level-wrong track reaching its own AMLt): F 0.5919 → 0.6186 (+0.0267); CMLt bound +0.0935. If only the 12 clean-octave tracks became perfect: +0.0246 — nearly the whole gain lives in 12 tracks. Independently corroborated by the deployed track-level tempo head (oracle-bounded +0.0476 CMLt; measured optimum +0.0104, p=0.14, null).

**S13 — across all ten arms categorised, GOOD never exceeds 115 and GOOD's mean F never leaves 0.72–0.78.** Every intervention so far is a permutation of the non-GOOD tracks.

**S14 — SDEdit fidelity is quantified and 0.4/g2.0 is the right operating point.** Full grid from `sdedit_guidance_fidelity.json`: 0.4/g2.0 has median lag 2.9 ms, **zero** clips past 20 ms, salience 1.67 (vs 1.29–1.39 at noise 0.5), and the **lowest CMLt/F in the grid (0.576** vs real audio's 0.805) — maximally metrically ambiguous at nil timing risk. noise 0.5 buys drift and salience collapse without buying ambiguity.

---

## 3. THE PLAN, Aug 26 → Sep 16

Budget: ~21 days × 3 GPUs. The binding constraint is attention and analysis, not GPU-hours. **Total committed GPU-h below: ~68.** Everything else is CPU/inference.

Category definitions are frozen now, before any of this runs: GOOD = (0.7 < r ≤ 1.5) ∧ (F ≥ 0.5), r = n_est/n_true from post-fix `error_anatomy.py`. **Every criterion below is stated on a fold-rotated, honest 217-track GOOD count, paired by track, with exact two-sided McNemar on GOOD membership.** Power, measured from the stock→ph13 transition (n_disc = 32): the exact test needs **net +14 GOOD** for p < 0.05. That is the bar everywhere.

---

### H0 (Aug 26, today). Hazard cleanup. **0 GPU-h — inference only, on CPU/one idle GPU for ~30 min.**

Not an experiment; a precondition for every number in the paper.
- Re-run post-fix `scripts/sa3/error_anatomy.py` on pw8, pw128, cl005, cl015, cl050, and both `f0_*_peritem_s0` arms.
- Regenerate `cache/anatomy_honest.json` post-fix (this is the baseline `LOSS_DECISION_TREE.md` and the oracle script both rest on).
- Recompute all four cells of `MORNING_REPORT.md` §3 including the halved column, and record whether the pre-registered "halved shrinks ≥0.10" gate passes or fails.
- Add a schema assertion to `error_anatomy.py`'s consumers: refuse to load any file lacking `degenerate`/`span`/`n_est_whole`. This class of error must not recur before Sep 16.

**If it fails:** if post-fix pw8/pw128 do not preserve monotonicity, C3 — the paper's causal claim — is gone and the plan below must be rewritten. Probability low (the fix moves pw32 by −0.045 against a +0.88 slope), but it is checked first because everything depends on it.

---

### R1 (Aug 26, same day). Probability-space TTA fusion. **0 GPU-h — one line, recompute from `ts_tta_cache`.**

**Tests:** whether the banked 0.6449 should be 0.6490. **Design:** replace logit-mean with probability-mean in the 3-rate fusion. Already measured: 0.6449 → 0.6490 alone, +0.0081 at delta=8, best stacked cell 0.6811.
**Criterion:** none needed — it is a free reporting change with a measured non-negative effect at every delta. Adopt it and re-bank the anchor as **0.6490**.
**If it fails:** it cannot; the sweep already covers all 63 subsets and pmean ≥ logit-mean everywhere.

---

### R2 (Aug 27–28). E1: tolerance ablation, τ=1 vs τ=3. **CHEAP SCREEN. 2 GPU-h (one arm; τ=3 is the existing stock arm).**

**Tests:** whether the 30 ms localization jitter (S11) is loss-induced or is annotation jitter. This is the single cheapest experiment that can falsify an entire line of work.
**Design:** `ShiftTolerantBCELoss(tolerance=1)` is already parameterized; no new code. **Critical confound:** `pos_weight_i = (valid − beats·(2τ+1))/beats`, so shrinking τ *raises* the implicit weight — hold the effective weight fixed at 32 manually or this becomes a re-run of C3. Ship the unit test first: on a synthetic single-beat target, assert loss(peak at t) < loss(peak at t+2) for τ=1 and loss(peak at t) == loss(peak at t+2) for τ=3.
**Criterion:** GOOD ≥ 118 on fold 0's held-out set scaled to the 217 basis, i.e. **net +14 GOOD over the matched stock arm**, with offset sd dropping below 20 ms.
**If it fails** (sd drops, GOOD flat): the shift tolerance is correctly ignoring annotation jitter, localization is a dead end, and **R3 (E2, soft-target localization) is cancelled outright.** That is the point of running it — it buys a cancellation for 2 GPU-h.

---

### R3 (Aug 29–31). E3: grid-coherence loss. **COMMITMENT. 12 GPU-h (3 seeds × 2 arms).**

**Tests:** the only proposed mechanism that matches the measured PHASE signature. S10 shows PHASE tracks have R = 0.198 — scattered, not shifted. Nothing in a framewise BCE penalizes incoherence: scattered noise and a coherent wrong-level answer are indistinguishable to it.
**Design:** with predicted activation p and IBI-normalized phase φ(i) from the existing `_phase` helper (no gradient through it), z = Σ p_i·exp(2πi·φ(i)) / Σ p_i, L_coh = 1 − |z|, added to the stock forward. It is orthogonal to ph13's phase-0.5 penalty (L_coh supplies "any one phase at all"; ph13 supplies "which phase"), so run L_coh alone and L_coh + γ=1.3. Unit tests, all asserted before training: mass at φ=0 → 0; uniform → 1; split evenly at φ=0 and φ=0.5 → **1** (a doubled prediction scores as badly as noise — this means L_coh does *not* substitute for the count term, and must be stated); mass all at φ=0.5 → 0 (level-agnostic by design).
**Criterion:** **net +14 GOOD** over stock at matched epoch, fold-rotated, McNemar p < 0.05. Secondary guard, inherited from C8's failure mode: **ΔHALVED ≤ +5**. The γ=3.0 arm already shows what a halving-driven pseudo-gain looks like.
**If it fails** (R rises toward GOOD's 0.357 on PHASE tracks but PHASE count does not fall below 26): coherence is downstream of evidence, not causal. The remaining PHASE tracks are the ones with `logit_beat` ≈ −0.14 — an SNR problem. Report it as a measured bound and stop; do not iterate.

---

### R4 (Sep 1–3). Fine-tune the released checkpoints with the phase loss. **COMMITMENT, but cheap. ~6 GPU-h.**

**Tests:** whether S7's +0.036 CMLt survives on a full-corpus model, and — the real payoff — produces a number on the **same 0.6270 scale as `data/results.csv`**, escaping the 43%-corpus deltas-only caveat entirely.
**Design:** `third-party/beat_this/launch_scripts/train.py:143` already passes `ckpt_path=args.resume_checkpoint` to `trainer.fit`. Fine-tune each released fold-k checkpoint for a few epochs with `--loss phase_shift_tolerant_weighted_bce --phase-gamma 1.3` on whatever corpus is available; fold-k held out fold k, so honesty is preserved. **Verify before committing:** load one released checkpoint and confirm it ships with optimizer state and resume-compatible hyperparameters — the whole route dies if `ckpt_path` errors out. Far under one from-scratch arm. From-scratch arms stay in the paper as the mechanism ablation, licensing deltas only.
**Criterion:** on the 217 fold-honest tracks, **net +14 GOOD** over the released baseline, with F ≥ 0.6270 (no regression) and dCMLt ≥ +0.02.
**If it fails:** the phase loss is a small-corpus artefact. Report the from-scratch delta as the mechanism result and the released number as the headline unchanged. Still worth the 6 h — it is the only route to an absolute number.

---

### R5 (Sep 3, after R4). TTA on the fine-tuned checkpoint. **0 GPU-h beyond one re-cache (~1 GPU-h).**

**Tests:** the explicit falsification condition on S9. The 84% stacking figure comes from an *oracle* proxy applied to *released* activations; a real loss that changes activations everywhere could absorb TTA's denoising.
**Design:** rebuild the 3-rate TTA cache on the R4 checkpoint, fuse in probability space.
**Criterion:** TTA gain on the new checkpoint ≥ **+0.009 F** (half of the released model's +0.0180).
**If it fails** (< +0.009): the phase loss and TTA are redundant, the stacked headline collapses to whichever is larger, and S9's 84% must be withdrawn from the paper.

---

### R6 (Sep 4–6). Frozen-embedding local tempo head. **CHEAP SCREEN. ~4 GPU-h. No Beat This training at all.**

**Tests:** whether the encoder already carries local period, i.e. whether the level errors are recoverable *at all* before anyone pays for joint training.
**Design:** reuse §4.15 wholesale. `scripts/diagnostics/sandbox_octave_loss_v2.py` already *is* a frame-local tempo head — dilated conv over 512-d frozen embeddings, per-frame 60-bin beat-period grid (14…100 frames), masked. Inputs `/disk1/jaehoon/tempo_obs_cache/{emb,tgt,mask}.npy` (5,862 items, **no SMC**); `/disk1/jaehoon/smc_bt_embeddings/` has all 217 SMC tracks at `max_dev_vs_cache = 0.0`. Train `octbandlog, oct_w=50, band=3` and `base` on fold≠0, run over the 217 SMC embeddings, inject at β=0.2 into the beat-only class-corrected DBN2016 (min_bpm=30, obs_λ=16, λ=100) via the audited `eval_tempo_head_dbn.py` path, paired over 8 seeds. **Two gotchas:** `--device` defaults to `cuda:1` (the faulty PCIe link — override it), and `--stratified` must be passed or you get the degenerate 900-Ballroom split.
**Criterion:** of SMC's 71 level-wrong tracks, **≥ 25 convert to GOOD**. Baseline 0/71. Anything under ~12/71 is inside the S12 ceiling and means nothing.
**If it fails** (< 12/71): S12's AMLt-based ceiling is confirmed, **joint training is cancelled**, and §4.6's SMC head numbers (track acc 0.5161 vs octave-tolerant 0.7558) are the explanation. If it *succeeds* at ≥25, S12 is wrong — the DBN repairs phase as a side effect of correct period — and a joint-training arm becomes worth 2 h.
Also record: if the head's SMC octave-error rate does not fall ≥20% relative under `octbandlog`, §4.15 does not transfer off-corpus and the whole mechanism is a corpus artefact. Stop there.

---

### R7 (Sep 7). SDEdit pre-screen. **CHEAP SCREEN. 0 GPU-h beyond inference (~1 GPU-h).**

**Tests:** whether SDEdit touches the failure axis at all, before anyone spends 22 GPU-h.
**Design:** score the existing 998 `/disk1/jaehoon/sd_aug1000` clips and their 998 `/disk1/jaehoon/sd_control1000` counterparts with the released checkpoint. Compare per-clip **est/true beat-count ratio r**, not F. The grid already hints yes (CMLt/F 0.805 → 0.576).
**Criterion:** SDEdit clips must show a significantly higher fraction at r > 1.5 than their own controls.
**If it fails** (same r-distribution): SDEdit is orthogonal to metrical level, **R8 is cancelled and 22 GPU-h are saved.**

---

### R8 (Sep 8–11, conditional on R7). SDEdit augmentation under the fixed loss. **COMMITMENT. ~22 GPU-h (4 h generation + 17.6 h training), one overnight.**

**Tests:** whether generative augmentation adds metrical ambiguity that the phase loss can exploit. Runs only if R7 passes and R3 or R4 gave a live loss.
**Design, with three changes from Aug 24 — the earlier arms were inert by design.** The fine-tune arms at lr 1e-5 had r(pre,post) = 0.9776 at step 75 and were negative at every checkpoint in *both* arms; a converged checkpoint cannot express a 16.9% data change. Only the from-scratch pair was informative (F 0.5805, doubled 18.5% vs 0.5699, 33.3% — +0.0106 F, 5/27 vs 9/27, noise, but on the metrical axis).
1. **Rebuild ASAP-only at noise 0.4 / g=2.0** (S14). The 216 SMC-sourced clips of the 809 kept are why the store is pinned to `--eval-fold 0`; drop them and one store serves all 8 folds. Regenerate ~1000 asap clips (Aug 24: 998 clips in 59 min on 4 shards ≈ 4 GPU-h). **Requires adding a guidance flag to `sa3_sdedit_bulk.py`** — `grep` confirms it has none, so the entire 17,798-npy store is g=1.0. Re-stage sdctl from the same source windows (CPU, `build_sdedit_arms.sh` already does this).
2. **Phase-weighted loss, γ=1.3** — this is the "once the loss is fixed" condition. `sa3_finetune_vanilla.py` has no such flag; use `train.py`.
3. **From scratch, 1860 steps, fold-rotated over all 8 folds, both arms.** 599 steps/epoch at 3.84 it/s = 64.6 min/run; 16 runs = 17.6 GPU-h, ~6 h wall on 3 GPUs. This is what makes the 217-track GOOD count honest — `cache/run_phase5.sh` trains fold 0 on `bt_wide`, so 190 of 217 SMC tracks are in training, and a GOOD criterion on the remaining 27 has no power.
**Criterion:** sdaug GOOD − sdctl GOOD ≥ **+14**, McNemar p < 0.05. Secondary guard: **ΔPHASE ≤ ΔGOOD** — the phase loss alone traded 14 DOUBLED→GOOD against 14 DOUBLED→PHASE for net +2; an augmentation that just relabels the same trade is not a result.
**If it fails:** the route closes and the 22 h buys "we tried the strongest safe setting at full dose under the fixed loss" — a paragraph, not a paper. **The prior is against it**, and that should be on record: SDEdit's only manipulated axis is timbre/onset salience (1.78 → 1.67 at the recommended setting), and `percussion_bias_negatives` refuted salience as SMC's mechanism (drums on true beats 0.635 vs 0.655 baseline; salience-matched α=0.15 scored 0.875 vs classical's 0.668; corr(drum content, benefit) = +0.084, wrong sign; model is LTAS-indifferent). SDEdit preserves timing by construction, so it adds **zero** tempo or rubato coverage, and C1–C9 put the failure in metrical level, not timbre. Also abort if regenerated 0.4/g2.0 clips fail fidelity on a fresh 24-clip sample (any clip past 20 ms, or median lag > 5 ms) — inherited labels are the entire case for SDEdit over from-scratch generation.

---

### R9 (Sep 5–8, in parallel, CPU only). Second seed for C2. **2–4 GPU-h if checkpoints must be re-trained; 0 if existing ones are re-scored.**

**Tests:** the stated limitation on S2. Ten autocorrelated checkpoints from one seed per arm; the between-arm difference at matched epochs (0.018) is the size of the within-arm sd. A second seed per arm is required before the CI is quotable in a paper.
**Criterion:** none — this is a confidence-interval widening exercise, not a hypothesis test. Deliverable is a defensible CI on "the objective never reduces the metrical-level error fraction."
**If it fails** (second seed shows a real downward trend): S2 weakens to "no trend in one of two seeds" and the paper leads with the absolute-gap growth (1.8–2.1x, p=0.009) instead, which is the stronger statement anyway.

---

### E2 (soft-target localization) — **conditional, do not build yet.**
Only if R2 shows the jitter is loss-induced. Design if reached: μ_k = Σ_{i∈W} i·p_i / Σ_{i∈W} p_i over W = {t_k−3…t_k+3}, L_loc = (λ/K) Σ (μ_k − t_k)²/τ², λ ∈ [0.1, 0.3], with a guard that L_loc must not fire when Σp ≈ 0. It is tempo-invariant (frames, not phase) and cannot reintroduce a beat-rate prior since it is evaluated only at annotated beats. Unit tests including the known blind spot: two symmetric peaks at t_k±2 give L_loc = 0 — assert and report it, do not hide it. Falsified if R2 shows the jitter is annotation jitter (then L_loc fits the annotators), or if it drives sd down without lifting GOOD above 106.

**Do not build the alignment/offset head.** See §5.

---

## 4. THE HONEST CEILING

**The banked anchor moves to 0.6490** (R1, probability-space fusion, free).

**The measured ceilings, stacked:**

| component | measured | status |
|---|---|---|
| released baseline, 217 fold-honest | 0.6270 | banked |
| 3-rate TTA, prob-space | +0.0220 → 0.6490 | banked, LOFO-honest, free |
| **oracle** subdivision screen on top | +0.0256 → 0.6811 | **oracle, not realizable** |
| oracle level fix to each track's own AMLt | +0.0267 (from 0.5919 basis) | oracle |
| realized phase loss (γ=1.3), F | +0.0035, n.s. | measured |
| realized phase loss, CMLt | +0.0363, t=4.31 | measured |

**Is +0.03 over 0.6449 (i.e. F ≥ 0.675 on 217 fold-honest tracks) reachable? Yes, but only by re-baselining, not by a new mechanism.** R1 alone delivers +0.0040 for free. The remaining +0.026 must come from a *realized* level fix, and the best-measured realized level fix is the phase loss at **+0.0035 F, not significant**. The 0.6811 cell that would clear it uses an oracle screen with access to ground-truth phase; the two attempts to convert it to a decode-time repair returned an exact no-op and −0.180 respectively (S6). **My honest expectation: 0.649–0.660 by Sep 16, with the +0.03 claim reachable only if R4 lands and its gain stacks with TTA per R5 — two conditional events, neither better than even.**

**Is 0.70 on fold 0 reachable? No.** Fold 0's honest released F is 0.6510. Adding TTA's +0.022 gives ~0.673. Clearing 0.70 requires +0.049 realized on top, which exceeds the *entire oracle* level-fix ceiling of +0.027 (S12) and nearly exceeds the oracle screen's +0.030 (S8). It is arithmetically out of reach on the level axis alone. Anyone quoting a fold-0 number near 0.70 is looking at leak (the leak-contaminated all-SMC mean is 0.7120, and the gap to honest fold 0 is exactly 0.061 — the size of the leak).

**Why the ceiling is where it is.** The 71 level-wrong tracks have AMLt 0.362/0.368, not GOOD's 0.750 — they are not "right beats, wrong octave," they are wrong for ~60% of their duration. Only **12 of 217 tracks (5.5%)** are cleanly level-wrong-only, and if all 12 went to F=1.0 that is +0.0246. **Nearly the entire available gain lives in 12 tracks.** Across all ten arms categorised, GOOD never exceeds 115 and GOOD's mean F never leaves 0.72–0.78 — every intervention to date permutes the non-GOOD tracks rather than growing GOOD.

**What the paper should therefore rest on.** The contribution is the **mechanism**, not the number: S1 (bidirectional, cross-corpus, robust to every confound tested), S3 (pos_weight causally sets metrical level, monotone over a 0.88 range), S4 (the collar bug, 1.23x, verified in code), S2 (training does not reduce the error fraction and grows its absolute mass 1.8–2.1x), S8 (the oracle bounds the whole class of fixes at +0.030), S12 (the period-only bound at +0.027, and *why* — the AMLt evidence). A paper that reports "here is exactly how large this class of fix can be, and here is why nobody will get it by decoding" is publishable; a paper claiming +0.03 realized is not, on current evidence. Plan for the former and take the latter if R4/R5 deliver.

---

## 5. WHAT TO DROP

**Explicit alignment / per-excerpt offset regression head.** Killed by S10: PHASE tracks have circular resultant R = 0.198 (the *lowest* of any category — GOOD 0.357, HALVED 0.502), only 12% have R > 0.5, and their pooled FP-phase histogram is flat with no bump anywhere. A constant offset would give R near 1. There is no offset to estimate on 88% of these tracks; a regression head would fit noise. Reinforced by `logit_beat` = −0.141 on PHASE tracks vs GOOD's +1.556 — these are low-evidence tracks, not misaligned ones.

**Wider tempo augmentation (−60..+20%).** Killed by S5: ΔF on <70 = −0.006 [−0.017, +0.005] over 3 seeds, and AMLt significantly *hurt* — −0.021 overall (run-level p=0.032) and −0.046 below 55 bpm (run-level p=0.018, the most consistent effect measured). The transition table shows it pushes 12 tracks into DOUBLED against 6 recovered from HALVED: it shifts the population up a metrical level, the wrong direction for SMC.

**Decode-time repairs, all variants.** Killed by S6: madmom DBN −0.051 (−0.142 below 55 bpm); Beat-Transformer-predicted-tempo DBN 0.627 → 0.565; tempo-banded cross-fit peak-picking threshold −0.0055.

**Converting the C9 oracle screen into a self-supervised decode-time screen.** Killed by S6's extension: phase from the model's own decoded grid is an exact no-op by construction (0.6449 at every delta); from the half-rate self-grid it is −0.180 at delta=8. The screen's +0.030 is a ceiling, not a method.

**Per-track TTA rate selection, in every form tried.** Killed by S9: select-by-confidence 0.5903, select-by-consensus 0.6389, consensus-weighted fusion over 7 rates 0.6469 (+0.0019 vs fixed-3, p=0.68) — all at or below the fixed 3-rate 0.6449. Adding extreme rates hurts (r0.7 → 0.6416, r1.5 → 0.6398, all seven → 0.6383). The honest within-track ceiling for *any* rate selector is +0.0387, with only **31.3% best-rate agreement between halves of the same track** — weak signal, not worth a GPU-day. Revisit only if a selector predicting best rate from audio alone clears +0.02 held-out.

**pos_weight as a tuning knob for SMC.** Killed by S3's second half: pw8 achieves sub-55 doubling 0.098 by halving *everything* — 51.7% of all tracks and 34.1% of sub-55 tracks land in HALVED, and F drops to 0.4952 (vs 0.5436 at pw32), CMLt to 0.325. Metrical level is monotone in pos_weight; accuracy is not. The knob exists and is causal (that is the finding); turning it is not a fix.

**γ = 3.0 on the phase-weighted loss.** Killed by S7: dF −0.0148 (t=−2.22), 24 GOOD tracks lost of which 15 to HALVED at dF −0.121, against only 16 DOUBLED→GOOD. It overshoots into the halving failure mode. γ=1.3 is the operating point.

**Reproducing upstream's full training corpus from scratch to get an absolute number.** Not dead-measured but strictly dominated: `train.py:143` already accepts `ckpt_path`, so fine-tuning released fold-k checkpoints (R4) preserves fold honesty, keeps upstream's full-corpus training, and lands on the same 0.6270 scale — at a fraction of the cost. From-scratch arms remain the mechanism ablation and license deltas only.

**Fine-tuning a converged checkpoint at lr 1e-5 to test a data intervention.** Killed by R8's post-mortem: r(pre, post) = 0.9776 at step 75, and *both* arms negative at every checkpoint (sdaug −0.020/−0.029/−0.021; sdctl −0.008/−0.012/−0.007). The setup cannot express a 16.9% data change. Any future data-intervention test must be from scratch.

**SDEdit at noise 0.5.** Killed by S14: it buys drift (5.8 ms median lag, 4–8% of clips past 20 ms) and salience collapse (1.29–1.39 vs 1.67) without buying more metrical ambiguity than 0.4/g2.0, which has zero clips past 20 ms and the lowest CMLt/F in the grid.

**Salience / percussion-bias interventions generally.** Killed previously and unchanged: drums on true beats 0.635 vs 0.655 baseline; salience-matched mixes at α=0.15 scored 0.875 vs classical's 0.668; corr(drum content, benefit) = +0.084, wrong sign; the model is LTAS-indifferent. This is why R8 carries a genuine prior against it.

**Any comparison spanning the 09:45 Aug 26 `error_anatomy.py` boundary.** The fix is systematically directional, not noise: −1.6 to −2.0 estimated beats per track affecting 67–75% of tracks, 3–7 tracks out of DOUBLED, 3–4 into HALVED, `logit_beat` +0.12 to +0.22. `miss_frac` is the sole column unchanged (0/217) and the only one safe across the boundary. Enforce with the schema assertion in H0.