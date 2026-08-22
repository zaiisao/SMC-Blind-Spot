# Handover: turning *The SMC Blind Spot* into a solution paper

Written 2026-08-20 from the VBPM project. Everything below was read off disk in that
repo on that date; provenance is marked per claim.

## Why this document exists

*The SMC Blind Spot* (arXiv 2605.12287) was rejected from ISMIR 2026 for presenting
problems without solutions — the analysis was praised as thorough. The plan is to
resubmit with a solution attached: a trainable / better-specified beat-level DBN on
top of Beat This activations, targeting SMC.

Most of the machinery for that already exists in the **VBPM** project. Nothing needs
to be rebuilt from scratch. This file says what is there, what it measured, and what
is genuinely open.

## THE VBPM REPO IS THE CONTEXT SOURCE — GO READ IT

`/home/sogang/jaehoon/VBPM`

That repo carries roughly a year of beat/downbeat modelling work, most of it
diagnostic. **Whenever context is missing here, look there before re-deriving
anything.** In particular:

| what | where |
|---|---|
| the DBN ladder (R0/R1/R2/R3), SMC eval, caches | `rungs_lab/` |
| the Beat This end-to-end campaign + verdicts | `rungs_lab/experiments/bt_e2e/RESULTS.md` (304 lines, read it all) |
| SMC result JSONs | `rungs_lab/results_smc*.json`, `results_lamselect_smc.json` |
| madmom baseline | `baselines/madmom_dbn.py` |
| durable findings, retractions, verdicts | `/home/sogang/.claude/projects/-home-sogang-jaehoon-VBPM/memory/` |
| theory docs | `docs/`, and `git show 43ecf34:docs/ELBO_for_DBN.md` |

The memory directory is the highest-value entry point: it is an index of settled
verdicts *and* explicit retractions, which is what stops this project re-running
experiments that already came back null.

## What already exists in `rungs_lab/` (verified on disk 2026-08-20)

- **R0** — madmom as shipped.
- **R1** — madmom reimplemented in PyTorch. **Certified: `R1_matched_shipped`
  reproduces `R0_shipped` on 216 of 217 SMC songs.** This is the load-bearing fact —
  the reimplementation is faithful, so anything built on R1 inherits that.
- **R2** — unsupervised exact EM (Baum-Welch) over the transition. Verified three ways
  (adversarial review, blind black-box suite 16/16, Fisher-identity check: EM 40.27 vs
  250 Adam steps 39.05).
- **R3** — per-frame transition conditioning. Retired on effect size (~+0.001 F), not
  on failure.
- Harness: `eval_smc.py`, `eval_smc2.py`, `eval_stratified.py`, `certify.py`,
  cached BT activations and demix shards.

## THE ASSET: the integer-interval dithering artifact

This is the most publishable thing in the VBPM repo and it has **not been published**
(prior-art check against the Krebs and Böck PhD theses, plus a deep-research pass,
found the mechanism unanticipated; nearest neighbours are Krebs thesis §4.3.2, and
bang-bang PLL limit cycles in another field).

**Mechanism.** madmom's state space represents only *integer* frames-per-beat. A song
whose true interval is 21.4 frames cannot be represented, so the Viterbi path dithers
between 21 and 22, manufacturing fictitious tempo transitions **on metronomically
perfect input**. Maximum likelihood faithfully bills these to `transition_lambda`.

**Three independent confirmations** (scripts in `rungs_lab/experiments/bt_e2e/`):

1. *Isolation* (`synthetic_grid_probe.py`): synthetic exact beats.
   interval 21.0 → flat path → λ_MLE 133.4; interval 21.4 → dither → λ_MLE 39.7;
   real 300-song corpus → λ_MLE 40.27. **A perfect metronome reproduces the real-data
   λ to 1.5%.** The "generative λ ≈ 40" is the grid, not the music.
2. *Scaling law* (`train_r2_em_fps2.py`): 2× upsampled activations → λ 40.27 → 80.08.
   Predicted ratio 2.0, measured 1.988.
3. *Repair* (`mixture_kernel_probe.py`): two-component kernel
   `p(j|i) = w·Dither(|Δ|≤1) + (1−w)·exp(−λ|r−1|)`, both learned by the same exact EM.
   Real data → w=0.370, λ=93.1 — **agreeing with the hand-set 100 and the CRF's 98.6.
   Three estimators, one answer**, once the artifact is modelled.

**Consequence to carry:** a retraction lives with it. The earlier reading "generative
λ≈40 matches heavy-tailed tempo-increment statistics" is WITHDRAWN — the metronome
reproduces it with no musical timing variation at all. The tempo-increment heavy tails
(kurtosis ~13) are themselves suspect, being computed from frame-quantised annotation
intervals carrying the same dithering. **Re-derive on sub-frame-interpolated
annotations before reusing that statistic.**

## Numbers, and a harness warning

**Read this before merging any table.** The two projects have different harnesses and
their numbers are *not* directly comparable. Analyze-SMC uses `scripts/_harness.py`
with a 5 s trim; VBPM's `rungs_lab` uses its own. They happen to land close, which is
exactly how a mistake would slip through.

From **Analyze-SMC** (`FOLLOWUP_FINDINGS.md`, this repo):
- Beat This raw peak-pick **0.6270** (reproduces the paper)
- DBN λ=100 → 0.592, best fixed λ=5 → 0.603, **per-track optimal λ → 0.642 (oracle)**
- Best configuration found: TTA + GT-tempo + opt-λ = **0.681 F / 0.753 CMLt / 0.773 AMLt**
- Split: activation ≈78% of the gap, decoder ≈22%

From **VBPM** `rungs_lab/results_smc.json` (n=217) — **⚠ ALL THREE ROWS BELOW ARE BEAT
TRANSFORMER, NOT BEAT THIS.** They must never be merged with the Beat This block above.
See `PHASE0_RECERT.md` §3, which established this by direct measurement:
- `R0_shipped` 0.6266 F / 0.5369 CMLt / 0.7110 AMLt  ← Beat Transformer, 43.066 fps, madmom DBN
- `R1_obs6_shipped` 0.6389 F  ← Beat Transformer
- `R1_wide30_notempi` 0.6328 F / **0.5667 CMLt**  ← Beat Transformer

**The 0.6266 vs 0.6270 agreement is a COINCIDENCE between two different neural
frontends at two different frame rates under two different decoders.** Beat Transformer
activations peak-picked give 0.5055, not 0.6270. Under one shared harness, Beat This +
DBN gives `fixed100 F 0.5827 / CMLt 0.5091 / AMLt 0.6718` — not 0.6266.

**Rule: only `results_smc3.json`, and anything importing `scripts/_harness.py`, is
directly comparable to `FOLLOWUP_FINDINGS.md`.** `results_smc.json`,
`results_smc_diagnosis.json` and `results_lamselect_smc.json` are a separate
Beat-Transformer world and must be relabelled or re-run on Beat This before entering any
shared table.

Safe to merge, because it *is* Beat This under the Analyze-SMC harness
(`results_smc3.json`): per-song `smc_001` (48.2 BPM, max tempo change **1.57 BPM** — a
BPM difference over 8-beat windows, NOT a 1.57× ratio; the handover's original wording
was a unit error): peak-pick 0.5952 vs DBN fixed100 **0.4110**. For contrast, the same
song's Beat Transformer `R0_shipped` is F 0.506 / CMLt 0.000 / AMLt 1.000.

That last line is the paper's thesis in one row: **on the hard tracks the DBN actively
destroys information the activations already contained.**

## The mechanism the solution should claim (transferred from VBPM, 2026-08-20)

Measured today in VBPM on a Scriabin performance (asap468), and it generalises to the
madmom setting because the failure is the same kind:

- The per-bar tempo prior's scale is a **single global constant** fitted on corpus
  medians (`GAMMA = 0.0363`), which says a typical bar-to-bar tempo change is 1.037×.
- That song's **median** bar-to-bar change is 1.34×, p90 1.76×, max 3.32× — the median
  move is **8 scale units** into the tail.
- Cost to track the passage honestly: **71 nats over 19 bars, 3.7 nats/bar.**
- Inserting a phantom beat and holding tempo steady is *cheaper than telling the truth*.
  A phantom beat was visible in the trajectory plot at exactly the point where the
  performer stretched a bar to 2.3× median.

**The claim this supports:** a hand-set homogeneous tempo-transition prior is
mis-specified for expressive material, and a mis-specified prior is worse than no
prior — which is precisely why peak-picking beats the DBN on slow, rubato SMC tracks.
The fix is a prior whose *scale is heterogeneous*, not a better single constant.

## Plan

**Phase 0 — re-certify (half a day).** Re-run the SMC ladder; confirm R0 = R1 still
holds on current code. The VBPM record has an entire era of results invalidated by a
broken pipeline; do not skip this.

**Phase 1 — the blind spot, with a mechanism (2 days).** Per song, peak-pick vs DBN:
how many SMC tracks does the DBN *hurt*, and does that regress on tempo variability and
BPM? Upgrades the rejected paper's problem section from "SMC is hard" to "here is
exactly when and why the decoder backfires."

**Phase 2 — the fix (3–4 days).** In order of expected value:
1. **The dither-corrected mixture kernel** — already built, already beats the hand-set
   kernel bare (0.9193 vs 0.9100 on the 146-song fold), **never tested on SMC**.
2. **A heavy-tailed transition kernel** — the VBPM finding transferred.
3. **Heterogeneous scale** (per-song, or a mixture over scale marginalised exactly) —
   the strongest form of the argument.

**Phase 3 — the reviewer's objection (1 day).** Show the fix costs nothing on easy
corpora. A solution that fixes SMC and breaks ballroom is not a solution.

**Phase 4 — write.**

## Cautions — read these before promising anything

1. **The decoder is only ~22% of the gap.** Analyze-SMC already established that the
   activation function is the dominant bottleneck. All the rungs work targets the
   smaller share. Scope the claim accordingly; do not promise ≥0.70 from decoding alone.
2. **Per-track *oracle* λ only reaches 0.642.** Scale tuning is close to exhausted, so
   the win must come from the kernel's **shape** (dither component, heavy tail,
   heterogeneity), not from picking a better λ. Verify this across the full
   `results_lamselect_smc.json` in Phase 1 — it decides which Phase 2 candidate is worth
   the week.
3. **The ≥0.70 mission is a stretch.** The best number in any configuration so far is
   0.681, and it uses *ground-truth tempo*. A deployable solution is below that today.
4. **Remedies can be redundant rather than additive.** On the 146-song fold the mixture
   kernel's +0.0093 bare advantage vanished under deployment decode — peak-snap and the
   dither component repair the same symptom. Always report bare *and* deployed.
5. **F understates continuity damage by an order of magnitude.** Loose kernels break
   *bars*, not beats: λ=40.3 cost −0.001 F but −0.021 downbeat AMLt. Always report
   CMLt/AMLt alongside F.
6. **Methodological lessons paid for in VBPM, both on 2026-08-20:** conclusions drawn
   from short training runs reversed completely when run to convergence (40 epochs →
   1000 epochs turned five confident "this fails" claims into "this works"); and
   single-dataset checkpoint selection has repeatedly disagreed with held-out results.
   Multiple seeds, pooled selection, convergence checks.

## Contact points in VBPM worth copying over

`rungs_lab/` in full (ladder, harness, caches), `baselines/madmom_dbn.py`, and
`rungs_lab/experiments/bt_e2e/` (the artifact scripts: `synthetic_grid_probe.py`,
`mixture_kernel_probe.py`, `train_r2_em_fps2.py`, `oracle_ceiling.py`).
