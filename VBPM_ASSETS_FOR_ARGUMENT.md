# VBPM assets for the ICASSP 2027 argument

**Argument being supported.** Beat This! (Foscarin et al., ISMIR 2024) reaches SOTA F1 but
degrades on CMLt/AMLt. Its authors attribute this to a frame-wise loss that treats each beat
individually, and propose "new losses that enforce periodicity" as future work. We argue the
right answer is not a periodicity regulariser bolted onto a factorized loss, but an explicit
**probabilistic sequence model** — a neural/learned DBN with latent tempo/phase state, exact
marginalisation, and calibrated uncertainty.

**Sources mined.** `/home/sogang/.claude/projects/-home-sogang-jaehoon-VBPM/memory/` (index +
~170 memory files, read in full for the relevant subset); `/home/sogang/jaehoon/Analyze-SMC/vbpm/docs/`;
`/home/sogang/jaehoon/Analyze-SMC/vbpm/rungs_lab/` result JSONs;
`/home/sogang/jaehoon/Analyze-SMC/vbpm/rungs_lab/experiments/bt_e2e/`;
`/home/sogang/jaehoon/VBPM/.git`; `/home/sogang/jaehoon/Analyze-SMC/HANDOVER_FROM_VBPM.md`.
All reads were read-only.

**Epistemic labels used below** follow the record's own vocabulary: **VERIFIED/CERTIFIED**,
**PRELIMINARY**, **RETRACTED/WITHDRAWN**, **NULL**, **NOT TESTABLE**.

---

## 0. Two load-bearing caveats before any number is quoted

1. **The CMLt-is-AMLc bug.** `project_cmlt_is_amlc_bug` (found 2026-08-07, FIXED 2026-08-14 in
   commit `1bf2fbe`). `vbpm/scoring/evaluation.py` unpacked `mir_eval.beat.continuity` as
   `(_cc, _ac, cmlt, amlt)`; the true return order is `(CMLc, CMLt, AMLc, AMLt)`, so index 2 is
   **AMLc**, the octave-forgiving one. AMLt was right by accident. Severity measured: a perfectly
   double-time estimate scored **0.9917** under the old unpacking and **0.0000** correctly.
   *Every "CMLt" written by the `vbpm/` package before `1bf2fbe` is an AMLc.* True CMLt for
   baseline seed0 on the 250-window gtzan set is **0.600**, not the **0.744** that was reported.
   **However:** "the eleven call sites in `rungs_lab/` always unpacked indices 1 and 3 and were
   never affected, so SMC/DBN continuity numbers stand." **Every CMLt/AMLt number in Section 1
   below comes from `rungs_lab/` and is therefore clean.**

2. **The pre-2026-07-10 metric invalidation.** `project_metric_invalidation_audit`: all deployment
   F scores before 2026-07-10 were rendered through a broken pipeline (side-channel emission +
   naive filter). Same checkpoints scored 0.26–0.35 old vs 0.68–0.82 fixed; Spearman(old, fixed)
   = **+0.12, p=0.6, n=18** — i.e. zero rank correlation. Explicitly invalidated: the "~0.4
   deployment cap" (`project_deployment_cap`) and the joint-eval "CHART loses decisively"
   verdict (`project_joint_eval_verdict`). Do not cite either as settled.

---

## 1. Continuity evidence: where a probabilistic decoder improved CMLt/AMLt, and where F and continuity dissociate

### 1.1 THE CORE TABLE — SMC MIREX, n=217, `rungs_lab/results_smc3.json`

Computed fresh by me (macro-mean over the 217 songs, `mir_eval`, indices 1/3 — clean).
This is the certified SMC ladder behind `project_smc_certified_ladder`
("certified against the user's Analyze-SMC harness: peak-pick 0.6270 exact").

| decoder | F | CMLt | AMLt |
|---|---|---|---|
| Beat This peak-pick (no sequence model) | **0.6270** | 0.5138 | 0.6100 |
| madmom DBN, fixed λ=100 | 0.5827 | 0.5091 | **0.6718** |
| R2mix (unsupervised exact-EM mixture kernel) | 0.5878 | **0.5166** | 0.6680 |
| R3_wt (per-frame dither weight) | 0.5875 | 0.5174 | 0.6679 |
| R3_lamt (per-frame λ) | 0.6008 | 0.4960 | 0.5815 |
| ML per-song λ (label-free selection) | 0.6021 | 0.4886 | 0.5618 |
| ORACLE per-song λ | 0.6495 | 0.5801 | 0.6584 |

**Three readings, all directly on-argument:**

**(a) F is the wrong metric to judge the sequence model on.** The DBN at λ=100 *loses* to
peak-picking by **−0.0443 F** but *wins* by **+0.0618 AMLt**. Per-song counts I computed:
the DBN is worse than peak-pick on F for **122 of 217** songs, but *better* on AMLt for
**122 of 217** songs. Same n, opposite verdict, depending on which metric you look at.
If you evaluate a sequence model by F alone you conclude it hurts; by continuity you
conclude it helps. **This is the in-house measurement of exactly the failure of F that
Beat This!'s CMLt/AMLt degradation exposes.**

**(b) Restricted to the non-pathological BPM range (n=172, ≥55 BPM, i.e. outside the DBN's
own 55-BPM floor blind spot), the probabilistic decoder wins continuity outright:**

| decoder (n=172, bpm≥55) | F | CMLt | AMLt |
|---|---|---|---|
| peak-pick | **0.6506** | 0.5402 | 0.6250 |
| DBN fixed λ=100 | 0.6125 | 0.5526 | 0.6839 |
| R2mix (learned, unsupervised) | 0.6190 | **0.5642** | 0.6810 |
| ORACLE per-song λ | 0.6751 | **0.6082** | **0.6789** |

R2mix beats peak-pick by **+0.0240 CMLt** and **+0.0560 AMLt** while losing **−0.0316 F**.
Oracle-λ beats peak-pick on **all three** (+0.0245 F, +0.0680 CMLt, +0.0539 AMLt).

**(c) THE DISSOCIATION IN THE DANGEROUS DIRECTION — chasing F destroys continuity.**
Going from fixed λ=100 to per-song maximum-likelihood λ (label-free) *gains* **+0.0194 F**
while *losing* **−0.0205 CMLt** and **−0.1100 AMLt** (n=217). R3_lamt: **+0.0181 F**,
**−0.0131 CMLt**, **−0.0903 AMLt**. **An F-driven tuning of the transition kernel buys
0.02 F and pays 0.09–0.11 AMLt. F understates the continuity damage by ~5×.** This is
the single cleanest in-house demonstration that a scalar-tightness knob (the closest thing
in the record to a "periodicity strength" hyperparameter) trades continuity for F.

### 1.2 THE HANDOVER'S λ=40.3 CLAIM — located and confirmed, with a correction

`HANDOVER_FROM_VBPM.md` caution 5 states: *"F understates continuity damage by an order of
magnitude. Loose kernels break bars, not beats: λ=40.3 cost −0.001 F but −0.021 downbeat AMLt."*

The primary record is `project_dithering_artifact`, which cites `experiments/bt_e2e/RESULTS.md
@ d65e316` and says **"F-impact MEASURED, not asserted"**:

> deployed EM-λ 40.3 vs hand-set λ 100 costs beat F only **0.9494 vs 0.9504** but downbeat F
> **0.8924 vs 0.9057 (−0.013)** — damage concentrates on DOWNBEATS.

**The full six-metric source table, read off `RESULTS.md:209,212`** (fold-0 val, **n=146**,
ballroom/beatles/hainsworth/hjdb, deployed decode):

| | beat F | CMLt | AMLt | db F | db CMLt | db AMLt |
|---|---|---|---|---|---|---|
| deployed λ=100 (hand-set) | 0.9504 | 0.9191 | 0.9410 | 0.9057 | 0.8842 | 0.9366 |
| deployed em-single λ=40.3 | 0.9494 | 0.9144 | 0.9340 | 0.8924 | 0.8671 | 0.9154 |
| **Δ** | **−0.0010** | −0.0047 | −0.0070 | −0.0133 | −0.0171 | **−0.0212** |

**−0.0010 beat F against −0.0212 downbeat AMLt — a 21× understatement.** RESULTS.md:220-223
states it directly: *"F's −0.001/−0.005 understated the real cost by an order of magnitude. Loose
kernels break bars, not beats."*

**CORRECTIONS YOU MUST CARRY:** (i) the memory's downbeat *F* delta is **−0.0133**, and the
handover's "−0.021 downbeat AMLt" is the db AMLt column of this same table — both are correct,
they are different columns. (ii) `project_dithering_artifact` says explicitly: **"Those CMLt
columns are pre-`1bf2fbe` so actually AMLc."** The AMLt columns are unaffected (AMLt was right by
accident); **do not quote the CMLt columns of this table without re-scoring.** Dataset/n: fold-0
val of the 4-dataset CV pool, **n=146**.

**The same table's second row of the same shape** (`RESULTS.md:209,213`, learn-obs λ=30.9 /
obs=4 vs deployed λ=100, n=146): Δ beat F **−0.0047**, Δ CMLt −0.0157, Δ AMLt −0.0165,
Δ db F −0.0181, Δ db CMLt −0.0232, **Δ db AMLt −0.0277** — a **6×** understatement.

The mechanism the delta reflects is *generative vs discriminative λ*, not periodicity strength —
see `project_r1_5_em_rung` (**VERIFIED**, three-way taxonomy on one system, fold-0, n=146):

| λ source | λ | decode beat F |
|---|---|---|
| generative (exact Baum-Welch EM, fully unsupervised) | 40.27 | 0.9488 |
| discriminative (CRF, supervised) | 98.6 | 0.9506 |
| hand-set (madmom) | 100 | 0.9506 |

"Generative lands FAR from decode-optimal (but only −0.0018 F thanks to the flat F-vs-λ
plateau)." **The F-vs-λ plateau is itself the argument: F cannot discriminate between kernels
that differ 2.5× in tightness, while downbeat continuity can.**

### 1.2b THE LARGEST DISSOCIATION IN THE RECORD — and it is on Beat This activations, on SMC

`FOLLOWUP_FINDINGS.md:179-186` (this repo), **SMC n=217, Beat This activations, Analyze-SMC
harness** — i.e. *exactly* the setting of the paper:

| decoder on Beat This activations | F | CMLt | AMLt |
|---|---|---|---|
| peak-pick @0.5 | **0.627** | 0.514 | 0.610 |
| GT-tempo, DBN λ=100 | 0.618 | **0.700** | **0.729** |
| **Δ** | **−0.009** | **+0.186** | **+0.119** |

**−0.009 F buys +0.186 CMLt and +0.119 AMLt — a 20× ratio.** `FOLLOWUP_FINDINGS.md:74` names it:
*"the DBN (trades F for AMLt)."*

**This is the strongest single asset for the argument.** It says, on the paper's own dataset with
the paper's own activations: *if you rank decoders by F you discard the single largest continuity
gain available.* Beat This! degrades on CMLt/AMLt; a probabilistic decoder over a latent tempo
state recovers 0.186 CMLt at a cost of 0.009 F. The caveat to state plainly: **this row uses
ground-truth tempo**, so it is an oracle-conditioned upper bound on what tempo-state information
is worth, not a deployable system. It is the *headroom* argument, and it should be labelled as
such.

Compare the honest contrast in the same file (`FOLLOWUP_FINDINGS.md:69-76,104-110`), n=217 SMC:
**test-time augmentation improves all three at once** — F 0.627 → 0.645, CMLt 0.514 → 0.550,
AMLt 0.610 → 0.649 — and the file explicitly contrasts this with the DBN's trade. Use TTA as the
control that shows the trade is a property of the *decoder*, not of the metric.

### 1.2c The sign-flip: F goes UP while CMLt goes DOWN 10× as far

`RESULTS.md:287-288`, **n=146**, R4.5-v1 bare (tied-covariance Gaussian emission on rich
features) vs bare R1:

| | beat F | CMLt | AMLt | db F | db CMLt | db AMLt |
|---|---|---|---|---|---|---|
| bare R1 | 0.9114 | 0.9107 | 0.9325 | 0.8677 | 0.8831 | 0.9359 |
| v1 bare | 0.9146 | 0.8776 | 0.9354 | 0.8554 | 0.8490 | 0.9091 |
| **Δ** | **+0.0032** | **−0.0331** | +0.0029 | −0.0123 | −0.0341 | −0.0268 |

**Beat F improves by +0.0032 while CMLt collapses by −0.0331 — a sign flip, ~10× in magnitude.**
RESULTS.md:292-297 reads it correctly: *"the learned emission tracks continuously, just at the
wrong level more often"* — CMLt (correct metrical level) moves, AMLt (any level) does not.
**Had F been reported alone, v1 would have been recorded as a win.** The deployed version
(`RESULTS.md:284-285`) shows the same signature: Δ beat F −0.0112, Δ CMLt −0.0351,
**Δ AMLt +0.0010**.

*(CMLt-column caveat from §0 applies to both tables — re-score before publication.)*

### 1.3 Other F-vs-continuity dissociations in the record

- **SMC config sweep, n=217, `rungs_lab/results_smc_diagnosis.json`** (certificate: R0≡R1 on
  216/217 songs — the **CERTIFIED** load-bearing fact):

  | config | beat F | CMLt | AMLt |
  |---|---|---|---|
  | `R0_shipped` (madmom as shipped) | 0.6266 | 0.5369 | 0.7110 |
  | `R1_matched_shipped` (our PyTorch reimpl) | 0.6266 | 0.5369 | 0.7110 |
  | `R1_obs16_bare` | 0.5956 | 0.5296 | 0.6992 |
  | `R1_obs6_shipped` | **0.6389** ← best F | 0.5245 | 0.6907 |
  | `R1_obs6_thr02` | 0.4690 | 0.5010 | 0.6570 |
  | `R1_wide30_shipped` | 0.6277 | 0.5591 | 0.7012 |
  | `R1_wide30_notempi` | 0.6328 | **0.5667** ← best CMLt | 0.7079 |

  **The best-F configuration is not the best-CMLt configuration and is in fact −0.0124 CMLt /
  −0.0204 AMLt below the shipped baseline while being +0.0123 F above it.** Conversely,
  widening the tempo state space to 30 BPM (`wide30_notempi`) — a *state-space* change, i.e.
  precisely the "probabilistic model" lever and not a loss term — buys **+0.0298 CMLt** and
  **−0.0031 AMLt** for **+0.0062 F**. F would have called this change negligible; CMLt calls it
  the largest single win in the table.

- **Blind-spot stratum, `project_smc_certified_ladder` (VERIFIED on user's harness):**
  "fixed-100 damage in the bpm<55 stratum is **CMLt collapse (0.553→0.343)** with F less hit."
  My recomputation on the same file, n=45 songs below 55 BPM: peak-pick F 0.5365 / CMLt 0.4130
  / AMLt 0.5527 vs fixed-100 F 0.4690 / CMLt 0.3428 / AMLt 0.6255. The blind spot is a
  **continuity** collapse (−0.0702 CMLt) with an AMLt *gain* (+0.0728) — the DBN keeps a
  metrically-allowed grid while losing the correct one. That is the octave-error signature and
  it is invisible in AMLt.
- **BPM-extreme stratum, `rungs_lab/results_smc.json`, n=66 (≤p15 or ≥p85):** `R0_shipped`
  beat F 0.5263 / **CMLt 0.2477** / AMLt 0.6321 vs `bpm_mid` (n=151) F 0.6704 / **CMLt 0.6633**
  / AMLt 0.7456. F drops 0.144 between strata; **CMLt drops 0.416**. Nearly 3× the sensitivity.
- **Downbeat continuity is where structure wins, `project_threemetric_readout` (2026-07-12,
  327 fold-honest val songs, format F/CMLt/AMLt):**

  | | beat | downbeat |
  |---|---|---|
  | vanilla peak-pick | 0.833 / 0.670 / 0.752 | 0.723 / 0.541 / 0.638 |
  | structured filter (default) | 0.828 / 0.664 / 0.738 | 0.726 / 0.546 / 0.646 |
  | structured filter (winner read-out) | 0.830 / 0.664 / 0.745 | **0.732 / 0.555 / 0.659** |

  Verdict recorded: "on BT's sharpened act2 the filter **TIES** vanilla on beats … the filter
  **WINS downbeats + downbeat continuity across all three metrics** (bar-level structure = the
  structured model's real advantage)." **CAUTION: these CMLt figures are pre-`1bf2fbe`** — treat
  the CMLt column as AMLc. The AMLt column and the ordering stand.
- **Factor-B champion, fold-0 deployed val (`project_factor_b_campaign`, 2026-07-23):**
  downbeat **0.9082 / 0.8893 / 0.9417** (F/CMLt/AMLt) vs baseline **0.8987 / 0.8767 / 0.9291**.
  Recorded reading: **"CMLt gains more than F (whole-song rotation consistency = F↔CMLt
  alignment)"** — +0.0126 CMLt for +0.0095 F. Same caveat on the CMLt label (pre-`1bf2fbe`).
- **The single-song headline row (`rungs_lab/results_smc3.json`, `smc_001`, 48.15 BPM, max tempo
  change 1.57×):** peak-pick F 0.5946 / CMLt 0.3696 / AMLt 0.3696 vs DBN fixed-100 F 0.4107 /
  **CMLt 0.0000 / AMLt 0.0000**. The decoder does not merely score lower — it drives continuity
  to exactly zero on information the activations already contained.
- **Oracle-λ sweep vs peak-pick, SMC n=217** (`PHASE0_RECERT.md:305-312`, table at 245-253):
  Δ F **+0.0225**, Δ CMLt **+0.0663**, Δ AMLt **+0.0484**. That file's own words:
  *"worth +0.0225 F but +0.0663 CMLt and +0.0484 AMLt… Three times the effect in continuity…
  **a Phase-2 kernel should be graded on CMLt/AMLt first and F second.**"*
- **λ_oracle-stratified subsets** (`PHASE0_RECERT.md:276-284`, `results_smc3.json`):

  | subset | n | oracle-λ F/CMLt/AMLt | peak-pick F/CMLt/AMLt |
  |---|---|---|---|
  | λ_oracle == 1 | 73 | 0.6422 / 0.5267 / 0.6198 | 0.6518 / 0.5448 / 0.6482 |
  | λ_oracle ≤ 2 | 84 | 0.6403 / 0.5129 / 0.6079 | 0.6498 / 0.5280 / 0.6449 |
  | **λ_oracle ≥ 100** | **34** | **0.7028 / 0.7848 / 0.8295** | 0.6179 / 0.6136 / 0.6707 |

  On the 34 songs that genuinely want a stiff kernel the DBN wins **+0.085 F, +0.171 CMLt,
  +0.159 AMLt** — continuity gain is 2× the F gain. This is the per-cluster evaluation the
  framing discipline asks for, already computed.
- **Bar-period estimator with the last oracle removed** (`vbpm/docs/octave_error_audit.md:3-4`,
  commit `712adb9`, **gtzan n=993**, deterministic middle-45 s window): **F 0.752 → 0.737, true
  CMLt 0.844 → 0.710, est/ref 1.003 → 1.161.** Δ F −0.015 vs **Δ CMLt −0.134 — a 9×
  understatement** — and `est/ref` 1.161 names the cause (systematic octave halving).
  *(F 0.752 is a RETIRED number for citation — see §5.2 — but the CMLt/F ratio is a mechanism
  observation and survives.)*

**CONTROLS — the cases where the pattern does NOT appear. Report these; they are what makes the
pattern a finding rather than metric noise.**

- **Top-3 parity** (`RESULTS.md:209-211,218`): hand-set λ=100, CRF λ=98.6, and mixture
  (w=0.37, λ=93.1) agree within **±0.0004 on all six metrics.** Nothing hides in the continuity
  columns when the kernels genuinely agree.
- **Bare vs deployed decode** (`RESULTS.md:209,214,224-225`): the *inverse* pattern —
  Δ beat F **−0.0390** but Δ CMLt only **−0.0084** and Δ db CMLt **−0.0011**.
  *"The wrappers polish beat positions, they never fixed tracking."*
- **TTA on SMC** (`FOLLOWUP_FINDINGS.md`, n=217): improves all three at once (§1.2b).
- **R2mix and R3_wt on SMC** (n=217, `results_smc3.json`): move ≤0.008 on everything
  (0.5878/0.5166/0.6680 and 0.5875/0.5174/0.6679). **So the F-up/AMLt-down pattern is specific
  to the per-frame-λ and per-song-λ arms — the ones tuned toward F — not to the DBN family.**
  That is exactly the discrimination the paper needs.
- **Do NOT quote the oracle row of `results_lamselect_smc.json`:** **95 of 217 tracks (43.8%) pin
  λ_oracle at the grid floor of 20** — it is truncation-censored (`PHASE0_RECERT.md:231-241`).

### 1.4 The KVAE OOD result — a probabilistic filter helping *more* out of domain

`project_smc_mirex_ood_generalization` (2026-07-01/02, all four analyses independently re-run,
not taken from agent reports). Zero-shot: model trained only on ballroom/beatles/hainsworth/
rwc_popular; SMC and GTZAN never seen.

Full continuity table (**VERIFIED**, `continuity_metrics.py` rerun directly):

| | F | CMLc | CMLt | AMLc | AMLt |
|---|---|---|---|---|---|
| in-domain (`bt_val_rich`) | 0.895 | 0.846 | 0.875 | 0.875 | 0.906 |
| SMC (n=217) | 0.654 | 0.420 | 0.587 | 0.440 | 0.616 |

**Ablation that matters most for our argument:** a classifier-only baseline on the *same frozen
features with no filter* scores **SMC F=0.407, in-domain F=0.705**; the full KVAE (probabilistic
filter over a latent state) scores **SMC F=0.654, in-domain F=0.895**. **The filter adds +0.247 F
on SMC and only +0.190 in-domain — it contributes MORE out of domain, not less.** That directly
refutes "the sequence model is a crutch that only helps on easy in-domain data."

Failure signature: **CMLt−CMLc = 0.166 on SMC vs 0.030 in-domain** — "the filter gets the tempo
right but loses lock intermittently." Explicitly labelled in the record as **"best-supported
hypothesis, NOT PROVEN CAUSALLY"** (language was downgraded from "RESOLVED"/"CONFIRMED" on
2026-07-02 under `feedback_no_premature_causal_conclusions`). Three targeted interventions all
came back flat on the target metric — see §5.

---

## 2. What the VAE / neural-DBN / ELBO line actually produced

*(see also §2.5 for the docs/PDF findings)*

### 2.1 The source paper

`docs/ELBO_for_DBN.md` is **not** at `43ecf34` (that object does not exist in
`/home/sogang/jaehoon/VBPM/.git`). The correct recovery command is:

```
cd /home/sogang/jaehoon/VBPM && git show 60500e4:docs/ELBO_for_DBN.md
```

(commit `60500e4f400928d0b1a4c89b10f3109d3ed655f7`, "docs: save the source paper (ELBO for DBN)
as durable transcription", Jaehoon Ahn, Wed Jun 24 22:47:41 2026 +0900; 98 lines).

Content, "ELBO for DBN — Jaehoon Ahn (March 2026)". This is **theory only — it reports no
experiments.** Its own §1 states the thesis in exactly the terms our argument needs:

> "These models are DBNs; in the beat-tracking literature the multiple hidden variables (tempo,
> bar position, meter) are usually collapsed into one composite HMM state for exact Viterbi /
> forward-backward inference [Krebs 2015, Whiteley 2006]. **Our model retains the full DBN
> structure with three separate latent variables, each with its own distributional form, and
> uses variational inference rather than HMM decoding.**"

Spec: `z_t = [m_t, φ_t, φ̇_t]` (meter, bar phase, tempo). Tempo = **per-frame ungated Log-Normal
random walk** `log φ̇_t ~ N(log φ̇_{t−1}, σ²)` with σ **learned from audio** — a deliberate
departure from Krebs-2015/madmom's beat-gated `exp(−λ|φ̇_t/φ̇_{t−1} − 1|)`, and with **no
[min,max] tempo bound** (it cites Whiteley 2007's bounded RW and does not adopt it). Phase =
von Mises `φ_t ~ vM(φ_{t−1} + φ̇_{t−1}, κ)`, κ learned. Meter = full K×K transition from a
network `f^m_ψ(m_{t−1}, φ_t, φ_{t−1}, h)`, generalising madmom's fixed ε. Decoder = Bernoulli
`σ(NN_θ(z_t, h))` that reads `h`. Strict ELBO, β=1, single MC sample. Algorithm 2 = Best–Fisher
von Mises rejection sampler with implicit reparameterisation.

**This is the paper's own statement that a learned, audio-conditioned, heterogeneous-scale tempo
prior is the intended object.** It is directly the "not a periodicity penalty" position.

### 2.2 What was trained, and what it produced — the honest ladder

**VERIFIED POSITIVE — the chain posterior / neural DBN (`project_structured_q_verdict`, CLOSED
2026-08-18):**

> "`tchain` — chain over 96 phase bins, exact forward recursion, **tempo in the state** — scores
> gtzan **0.747/0.750/0.753/0.754** (four runs, two seeds, **n=993**), asap 0.745–0.782, rwc
> 0.797–0.839, ballroom ~0.97, beatles ~0.95; phase_err 0.28–0.31 rad. The amortized `interval`
> regression q, re-run on the same branch as the control, lands at the null floor (**0.074** vs
> nulls 0.074/0.075, phase_err 1.52) and is **RETIRED**."

**Single-variable A/B** (`schain`, 2026-08-18, `abl_chain.log` / `abl_frames.log`; **only
`chain_posterior` differs** — same prior, emission, trunk, potentials head, KL accounting, data,
epochs):

| val set | chain q | factorized q |
|---|---|---|
| ballroom | 0.973 | 0.000 |
| beatles | 0.943 | 0.000 |
| hainsworth | 0.794 | 0.000 |
| hjdb | 0.977 | 0.000 |
| rwc | 0.654 | 0.000 |

The factorized arm *freezes* — MAP bin-step 0 at **100% of frames**, prior-infeasible.
⚠ **The audit corrects the INTERPRETATION, not the numbers:** *"An optimization trap cannot be
attributed to a representational floor."* The correct reading is **"a frame-local kinetic trap"**
— the factorized arm lands *below its own mean-field optimum*. **State it as an optimisation
result, not a capacity result.** The 0.97-vs-0.000 numbers themselves stand.

**THIS IS THE SINGLE STRONGEST PIECE OF EVIDENCE IN THE WHOLE RECORD FOR OUR ARGUMENT**, and the
record's own naming note says so:

> "Naming: with exact recursion the variational gap is ~0, so this is a **neural DBN trained by
> near-exact marginal likelihood**."

> ⚠ **THIS NAMING CLAIM WAS SUBSEQUENTLY REFUTED — see §5.1b.** `audit_2026-08-19.md` (claim O)
> establishes that containment holds for the *non-amortized* family, but the **trained encoder
> reads x only, while the optimal potentials are functions of y**. So the bound is loose by a
> θ-dependent `E[KL(q(z|x) ‖ p(z|y,x))]`, and *"training tchain is maximum-likelihood training of
> a neural DBN"* is **UNSUPPORTED**. **The F numbers below stand. The zero-gap justification does
> not.** Call it "a chain/structured posterior with exact forward recursion," never "zero
> variational gap" or "exact maximum likelihood."

**The additional dataset breakdown** (from `docs/project_state.md`, same runs): gtzan **n=993
test-only, seeds 0/1 → F 0.750 / 0.753, CMLt 0.70, AMLt 0.85**; asap n=62 → 0.745–0.782;
rwc n=15 → 0.797–0.839; ballroom n=83 → 0.968–0.973; beatles n=23 → 0.943–0.962.
Phase error **0.28–0.31 rad** vs the retired amortized line's **1.52** (chance 1.571).
Head-to-head on the same time axis, n=60 gtzan (`quasi_svae_mechanism.md:6`, `verify_align.py`):
**tchain 0.498 vs Beat This 0.545** — *we do not beat Beat This and must not imply we do.*

**The competing explanation was tested and is insufficient — and this is precisely the
"periodicity regulariser vs probabilistic model" discriminator, measured:**

> "repairing the placement likelihood so evidence scales with N (`interval_kind: gauss_time`,
> the tutorial §7.5 Gaussian over beat times) **lifts continuity under a factorized q** — gtzan
> **CMLt 0.115 → 0.467, AMLt 0.184 → 0.556, est/ref 1.313 → 1.059** — but leaves **F at ~0.13
> against null 0.07**. `p(N|φ)` alone is exactly null. So **an under-specified likelihood costs
> continuity; the FAMILY decides whether the model works at all.**"

Read that as the paper argument: *a smoothness/periodicity-style repair to the factorized
objective moves CMLt by +0.352 and AMLt by +0.372 and still leaves the model at chance on F.
Changing the posterior family to a chain — i.e. actually being a sequence model — moves F from
0.074 (null) to 0.75.* **Same dataset (gtzan, n=993), same branch, same prior, same emission,
same encoder.**

Stated mechanism: "the independence cost of a factorized q over a near-deterministically coupled
phase chain is structural (KL(∏q_t ‖ p) carries a term equal to the posterior's **total
correlation**), θ-dependent, and its gradient **rewards decorrelation** — i.e. **it prices the
model toward a metronome**. A chain q with the prior's sparsity zeroes it by construction."

**PRELIMINARY — first corpus number for the continuous amortized VBPM
(`project_continuous_vbpm_corpus`, 2026-08-19, branch `archain-anchor`, commit `15b5afa`, 24
epochs, batch 16, ar_stride 8, seed 0, 30 s windows, rule-g deploy, gtzan held out entirely):**

    gtzan n=993   F 0.475   CMLt 0.616   AMLt 0.817   est/ref 1.218
                  emission-D 0.437   nulls 0.070 / 0.078

Reference points on the same split: the factorized q it replaced = **0.074 (= null)**; `tchain`
= **0.750**; madmom DBN on the same frozen frontend = 0.771 gtzan downbeat F **but on full songs,
so not comparable**. Explicitly labelled: "one seed, one protocol. Before any claim: seeds,
full-song eval, and the deployment read-out swap." **AMLt 0.817 vs F 0.475 with est/ref 1.218 —
the model tracks at an allowed metrical level 82% of the time and loses ~0.34 F to picking the
WRONG level.** Another clean F/continuity dissociation, this time in our own model.

**VERIFIED — exact marginalisation beats a learned point estimate (`project_anchor_k_v2_verdict`,
2026-08-08):** exact marginalisation of a categorical anchor in q (act-binned k-head), in-loop
protocol n=993: **rule-g F 0.752 vs baseline 0.468 / v1 0.397; AMLt 0.929.** Strict corrected
250-window protocol: F 0.302 → 0.562, **CMLt-TRUE 0.600 → 0.743**, deployed anchor 137 ms/36% →
27 ms/71%. Labelled "single seed, dev protocol; folds+seeds before claims."
**NOTE: `project_icassp_sprint_goal` (2026-08-15) RETIRES the 0.752 number** — see §5.

**VERIFIED — marginalisation beats argmax, mechanistically (`project_quasi_svae_mechanism`,
AUDITED 2026-08-19 by an 11-agent workflow, `docs/audit_2026-08-19.md`):** on the trained
`tchain`, **median 13% of posterior tempo mass is an octave-class away, 57% of songs carry >10%
competitor mass, 64% of songs flip between the frame-0 argmax and the smoothed argmax, and phase
marginals hold 43% of mass >60° off the mean** — "an ensemble of paths **no unimodal location-q
can represent**." And the decisive read-out contrast: the trained per-frame potentials are
statistically **uniform** (entropy 4.564 vs uniform 4.564, ±10% tilts, AUC 0.82) —
**peak-picked alone they score F 0.000; run through the exact forward-backward sweep they score
F 0.602 on the same songs.** *The encoder emits whispers; exact inference does the placement.*
That is the cleanest available statement that the win comes from marginalisation, not from a
better per-frame target.

Also: **quadrature, not discretisation.** Fourier-refining the trained potentials to 192/384 bins
moves the path 1e-4 rad, F unchanged at 0.602, logZ gap <1.4 nats. 96 bins is already the
continuum limit. Useful pre-emption of "your DBN is just a coarse grid."

**Exact EM on the real bar-pointer DBN (`project_r1_5_em_rung`, 2026-07-18, VERIFIED):** exact
Baum-Welch, E-step via the identity `d logZ / d logA_ij = E[#i→j]`, M-step = exact 1-D argmax;
fully unsupervised marginal `p(activations|λ)` with meters {3,4} marginalised by logsumexp over
per-meter forwards. Converged 100 → 51.5 → 43.5 → 41.3 → 40.6 → 40.4 → 40.3 → 40.28 → **40.27**
in ~5 iterations. **Fisher identity confirmed live:** SGD on the same marginal oscillates
39.0–41.6 around EM's 40.27 (5 EM iterations vs ~250 Adam steps).

**Certificates (CERTIFIED, load-bearing):** R0≡R1 re-earned twice — 30/30 event-identical
shipped **and** bare after two real bugs were found and fixed (`project_r0r1_recertification`,
2026-07-17), and 216/217 SMC songs (`results_smc_diagnosis.json`). "Known irreducible gap:
madmom computes observation densities in float32, ours float64 (~6e-8/frame)."
**Certificates do not transfer across regimes** — the original "25/25, 100% of frames" claim was
regime-specific (86 fps, clip, single-meter) and silently expired at 50 fps / squeeze / multi-meter.

### 2.3 What failed

- **`project_deployment_cap` (2026-06-25):** free-run deployment of the paper-faithful VAE is
  structurally capped ~0.40 beat F1; nothing exceeded it (widen, β<1, free-bits, KL-anneal,
  He-2019, overshoot, OU tempo, survival/renewal likelihood, audio-conditioned tempo mean,
  stop-grad distillation, free-run reconstruction, scheduled sampling, word-dropout, **and
  FIVO** — K=1≡ELBO verified, bound tightens with K but free-run stays ~0.37). Best ever 0.40.
  **⚠ This verdict is listed as INVALIDATED by `project_metric_invalidation_audit` (broken
  pipeline). Do not cite the 0.40 cap as settled.** The *structural insight* survives and is
  worth quoting: in the DBN, tempo is a **pure latent** and the observation is an audio-derived
  activation *generated from* the latent; the VAE inverted this (audio = decoder condition,
  beats = generated), so at deployment the prior's dynamics are **audio-blind**.
- **`project_vbpm_rebuild_verdict` (2026-07-27):** "amortization is the breaker; the objective is
  exonerated." Under an identical fixed-objective ELBO with a **free non-amortized q (SVI)**,
  PF-placed init is stable (corr 0.634→0.635, F 0.758→0.762; ELBO 990.6 vs surrender 2523.5).
  The **same objective** under **any** amortized q collapses from every init — **10/10 cells**
  (corr 0.30→0.002, teacher-forced F→0.000 within 50 steps). Analytic KL floor:
  `2log(2+r) − log(4r)` nats/frame vs a Cauchy RW, argmin at ρ_q = exp(−γ_p/2) = 0.999725,
  matching empirics to 4 dp.
- **`project_multimodality_gap` (2026-07-29/30) with its own 2026-07-30 CORRECTION:** the
  diagnosis ("unimodal L2 tempo head on an octave-multimodal target") was validated on a
  supervised probe (median tempo error **0.565% vs 16.6%**), and then **the fix was trained and
  FAILED inside the real ELBO**: shipped 0.3108 vs categorical 0.2479 (perm p=0.0514), CAT2
  0.2767 (p=0.0466, significantly worse). **"The pre-registered falsifier FIRED. Do NOT cite this
  memory as a validated fix."** Also recorded there: **IWAE/FIVO are DEAD** — 18 runs, ANOVA
  p=0.450; at T=1500 ESS = 1.08/32 so IWAE ≡ the ELBO of the luckiest sample.
- **`project_elbo_destroys_tempo` (2026-07-27):** tempogram-initialised log-tempo head reaches
  per-crop corr **0.57–0.64** at ELBO step 1 (teacher level 0.6335); ELBO training then destroys
  it: **0.6369 → 0.0909 (step 200) → 0.0265 (step 400)**, with level MAE degrading 10.8% → 37%,
  **worse than predicting the corpus mean (28.1%)** — an actively wrong tempo, not lazy collapse.
- **`project_octave_degenerate_likelihood` (2026-08-11, synthetic IDEAL features, oracle anchor,
  n=128/384/96):** the pointwise-Bernoulli likelihood is near-degenerate over integer multiples
  of the bar rate. Net preference for the true rate over 2×: **1.02 nats/song at pos_weight 1**
  (against recon ~209), 4.47 at pw5, 26.05 at pw30. Sub-multiples are catastrophically worse
  (−50 to −1417 nats), and the asymmetry holds exactly: 2×/3×/4× appear in 25%/14%/14–19% of
  per-song argmaxes, ÷2 and ÷3 **never** (0.0%). Its F/CMLt/AMLt table (96 songs, correctly
  unpacked continuity):

  | arm | F | CMLt | AMLt | est/ref |
  |---|---|---|---|---|
  | learned rate, deployed q(k|x), pw1 / pw5 | 0.037 / 0.039 | 0.000 | 0.000 | 0.24 |
  | null (true rate, random phase) | 0.082 | — | — | 1.00 |
  | ORACLE rate, deployed q(k|x), pw1 / pw5 | 0.157 / 0.154 | 0.140 / 0.154 | 0.198 / 0.195 | 1.34 |
  | ORACLE rate + CLOSED-FORM read-out | **0.499 / 0.513** | **0.995** | **0.995** | 1.01 |

  Two things to take: **(i)** a railed-rate model scores *below* a random-phase grid at the true
  tempo (0.037 vs 0.082) with CMLt/AMLt exactly 0; **(ii)** with the correct rate and a
  zero-parameter closed-form read-out, **CMLt and AMLt are 0.995** and only 70 ms phase precision
  limits F. **Getting the latent tempo right is what buys continuity**, and this was measured on
  noise-free perfectly periodic input, so it is a property of the objective, not the frontend.
  It carries an **attribution correction**: `project_r4_final_verdict`'s "all octave problems
  trace to BT activations" is overturned — on ideal input with an oracle anchor the degeneracy is
  still there, so **the likelihood owns it**.
- **`project_r4_5_domination` (2026-07-20):** unsupervised Gaussian emissions on rich features
  LOSE to the Böck plug-in (−0.011 beat F / −0.023 downbeat F) and the loss is
  **CMLt-concentrated (wrong metrical level)**. Cause verified as **emission domination** (the
  ASR acoustic-scale disease): 22,600-nat gaps, uniform-kernel decode ties the learned kernel,
  91% frame-argmax agreement, 96/256 dims variance-floored giving downbeat a +660-nat log-det
  bonus. Machinery verified exact (E-step identity 4.8e-6, M-step 1e-16, alignment 1.2e-7, blind
  suite 9/9). **Generative-Gaussian rich-emission route CLOSED.** Its **oracle ceiling** is worth
  keeping: **beats near ceiling (+0.009 headroom); downbeats WIDE OPEN (+0.048)** — "the
  project's best quantified in-domain opportunity."
- **`project_factor_b_campaign`:** R4 run 2, exact EM on a neural kernel + meter prior, +3,200
  nats to convergence, **deployed decode flat per-dataset** (all 0.9494/0.8940 vs init
  0.9506/0.8987). **"The transition side of the bar-pointer model is closed."**

### 2.4 What is NOT TESTABLE (do not call it negative)

`project_q_family_not_testable` (2026-08-15): the `mixture_q` / `phi0_diffusion` null is
**uninformative**, established by the experiment's own author under an adversarial workflow
(4 lenses, 32 candidates, 16 confirmed, **all 16 "uninformative"**). Reasons, measured:
(1) the corrector receives **exactly zero gradient** toward undisplaced phase — "a wire that is
cut"; (2) repair is actively **penalised** (displacement left alone −73.919 vs −74.097 repaired;
exhaustive scan of exactly-cancelling pulses best net −0.0146, still a loss); (3) the injected
corruption vM(17) has sd 0.24 rad = **64 ms, inside the 70 ms tolerance**; (4) **power to detect
+10 in-tolerance points was 8%**, MDE ~36–51 points.
**`project_paper_framing_discipline` is explicit: "The q-family ladder earns NOTHING and is a
liability if cited."**

---

## 3. Staleness of the end-to-end CRF negative result — VERDICT

**Short answer: the negative is NOT stale, it is WORSE than recorded — but it was measured in
the wrong regime, cannot be re-scored, and does not test our hypothesis.**

### 3(a) Are `vanilla_history.json` / `r2_history.json` the rematch? — YES, on content, not on git

**Git and mtimes are both useless here, and this must be stated honestly.** The entire `bt_e2e`
tree entered `/home/sogang/jaehoon/VBPM/.git` in a **single bulk commit**:

```
e7150aa6940bac2309d27bac2d3f01c817977715
Sun Aug 9 19:11:27 2026 +0900  Jaehoon Ahn
"labs: rungs_lab, tempo/octave audits, the isolation ladder, beat_this submodule"  (218 files)
```

`git log --all --follow` on `vanilla_history.json`, `r2_history.json`,
`vanilla_history_prelim.json`, `r2_history_prelim.json`, `train_bt.py`, `launch.sh`,
`final_eval.py`, `crf_baseline.py` and `RESULTS.md` returns **exactly that one commit and nothing
else** — no second commit anywhere in the 225-commit history touches `experiments/bt_e2e`. The
directory was imported three weeks after the experiments ran. File mtimes are one of exactly three
bulk-copy timestamps (2026-08-11 23:37:09, 2026-08-14 23:08:09, 2026-08-20 20:02:58) — `cp`/rsync
artifacts of the move into Analyze-SMC. **No run timestamp survives anywhere on disk.**

Identification therefore rests on **four independent content signatures, all four agreeing**:

| | `*_prelim.json` (first attempt, 2026-07-18) | `*.json` (the rematch) |
|---|---|---|
| vanilla epochs | **24** (00–23), opening loss **0.2541** — a *resumed* run | **30** (00–29), opening loss **0.4550** — fresh |
| vanilla lr | 1.00e-03 → **2.00e-04 at epoch 10** | **1.00e-03 for all 30 epochs** |
| r2 lr | 1.00e-03 throughout (never annealed) | 1.00e-03 throughout (**now symmetric**) |
| r2 learned λ | 80.00 → **14.18** (best **14.226956367492676**) | 88.61 → **105.51** (best **101.03919982910156**) |
| best beat F / db F | vanilla 0.9612674904550522 / 0.9395877314063663 @ep21; r2 0.9470625179951031 / 0.9199828335448514 @ep23 | vanilla **0.9657925959528175** / 0.9280965497485181 @ep**29**; r2 **0.9115376639931124** / 0.8573449021883721 @ep**19** |

**λ≈101 vs λ≈14.2 is decisive.** The ADDENDUM's fix #1 was to move the chassis to
`observation_lambda=6`; `train_bt.py:43,188-191` now hard-codes
`BT_SHIPPED_DECODE = dict(observation_lambda=6, num_tempi=None, threshold=0.2)` and threads it
into `CRFLearnedFactors`. Under obs=6 the CRF's MLE λ lands at ~101 — agreeing with madmom's
hand-set 100 and the CRF's 98.6. The `_prelim` file's 14.2 is exactly the number RESULTS.md's
*first* table calls "14.2 (learned)". Fix #2 (BCE-keyed scheduler) is present at
`train_bt.py:280-283` with the bug named in a comment; fix #3 (symmetric fresh 30 epochs) is
visible in the epoch counts and in `launch.sh` (`--epochs 30`, no `--init-from`). And
`train_bt.py:303` writes to `OUT_ROOT / f"{args.arm}_history.json"` — the *unsuffixed* names are
what the current fixed code produces; `_prelim` are hand-renamed archives.

**Result: vanilla 0.9658 vs r2 0.9115 beat F — a gap of 0.0543, up from 0.0142 in the first
attempt. Repairing the confounds made the e2e CRF arm lose HARDER, by ~3.8×.**

**NOT determinable:** the wall-clock dates of either run; whether the rematch used an identical
seed (`launch.sh` passes no `--seed`, so both arms default to `--seed 0` per `train_bt.py:176` —
consistent with "same seed" but not independently witnessed); whether any further run happened
between the rematch and the 2026-08-09 import.

**COMPARABILITY CAVEAT — do not quote 0.9658/0.9115 against RESULTS.md's table.**
`train_bt.py:178` defaults `--eval-songs 60` and `launch.sh` does not override it. RESULTS.md's
own Caveats section says: *"The 60-song training-time evals ran ~+0.01 high vs the full fold
(subset bias) — use this table, not the training logs."* So 0.9658 and 0.9115 are **60-song-subset
numbers, biased ~+0.01 high**, on a different footing from the 0.9506/0.9446 full-fold table. The
*direction* (vanilla wins by ~0.054) is robust; the magnitudes are not directly quotable.

### 3(b) Dataset / fold / n — and it is an EASY fold

From `train_bt.py:45,50,87` and the demix shard manifests (`demix_shard0..5.txt`, 1,305 lines):

- **Corpus:** 4 CV datasets — **ballroom 672, beatles 179, hainsworth 222, hjdb 232 = 1,305 songs.**
- **Split:** `EVAL_FOLD = 0`, Beat This official 8-fold CV, fold-honest.
  **n = 1,020 train / 146 val / 139 skipped** (no demix cache, or meter ∉ {3,4}, or the annotated
  path not representable on the tempo+meter grid). 1,020 + 146 + 139 = 1,305 exactly. The
  per-epoch `crops 1020` line in all four history files confirms the train count directly.
- **Crops:** beat-aligned, `MAX_CROP_FRAMES = 700` ≈ 16 s, batch 1.

**Hardness verdict: very easy.** Ballroom (672/1305 = **51%**) and hjdb (232, **18%**) are the two
most metronomic corpora in the standard suite — **69% of the training material is drum-machine-
regular dance music.** Beat F ≈ 0.95, with *bare* decode already at 0.911. **There is no SMC, no
gtzan, no asap, no rwc in this experiment.** Cross-reference §4.6: the rigid-grid oracle ceiling
on ballroom is **0.918** and on hjdb **0.999** — these are corpora where a constant-period grid is
already near-perfect, so a sequence model has essentially nothing to add.

**This is the single most important thing to say about the e2e negative: it was measured in
precisely the regime where the thesis predicts no gain. It does not refute the SMC hypothesis —
and it means the hypothesis has never been tested end-to-end on hard material.**

### 3(c) Has the code changed since? — No evidence of it, but git cannot prove it

`train_bt.py` as committed in `e7150aa` **contains all three ADDENDUM fixes** (obs=6 at line 43
and 191; BCE-keyed `scheduler.step(epoch_bce / max(n_crops,1))` at 280–283 with the bug named in
a comment; `--epochs 30` symmetric in `launch.sh`). So the committed code **is** the rematch
harness and the rematch histories are its output. There is **no evidence of any post-rematch
edit** to `train_bt.py`, `crf_baseline.py`, `final_eval.py` or `launch.sh` — but this is a weak
"no": a single-commit import cannot distinguish "unchanged" from "changed before import." The
only files edited after 2026-08-11 in this directory are `run_mixture_chain.sh`,
`run_r2_em_chain.sh` and `smc/eval_smc.py` (2026-08-20), and `PHASE0_RECERT.md:56-70` documents
those as **path repairs only** after the move.

**Two things that DO partially invalidate the surrounding tables:**

1. **The CMLt-is-AMLc bug applies to RESULTS.md.** The RESULTS.md tables are dated 2026-07-20,
   i.e. **before** the 2026-08-14 `1bf2fbe` fix. **The CMLt columns at `RESULTS.md:208-216` and
   `283-288` should be read as AMLc and need re-scoring before entering any paper.** The F and
   AMLt columns are unaffected. *(Note this sits in tension with `project_cmlt_is_amlc_bug`'s
   statement that "the eleven call sites in rungs_lab/ always unpacked indices 1 and 3 and were
   never affected." The two claims disagree about the bt_e2e tables specifically. **Resolve this
   by re-scoring before publication rather than picking a side.** The `results_smc*.json` files I
   aggregated in §1.1 are separate call sites and the memory's exoneration covers them.)*
2. **The checkpoints are gone.** `vanilla_best.pt`, `r2_best.pt`, `vanilla.log` and `r2.log` do
   not exist anywhere under Analyze-SMC or VBPM. **`final_eval.py` cannot be re-run on the
   rematch**, so it can never be put on the same footing as the 0.9506/0.9446 table without
   retraining from scratch (~40 min vanilla, ~3.5 h r2, from the epoch times: 73 s vs 420 s).

### 3(d) Is there a LATER e2e result? — NO

Every `*.json`, `*.log`, `*.md`, `*.csv`, `*.txt` under `rungs_lab/` was searched.

- All ~40 result JSONs (`results_smc*.json`, `results_r2*.json`, `results_r3_*.json`,
  `r4/results_*.json`, `mert/results_*.json`, `results_stratified.json`, `results_baseline.json`,
  `results_btminimal.json`, `results_campaign_r2mix.json`, `results_lamselect_smc.json`,
  `results_traces.json`) are **frozen-frontend decoder experiments** — they re-score existing
  activations under different decoders. **None trains a frontend through a structured decoder.**
- `rungs_lab/checkpoints/` holds only the 8 pretrained `bt_fold{0..7}_repacked.pt`.
  `rungs_lab/runs/` holds only `r2_seed0.pt`, `r2mix_seed0.pt`, `r3_lamt_seed0.pt`,
  `r3_wt_seed0.pt` — 880 B to 34 KB, i.e. **decoder parameters only**, not 37 MB frontends.
  **No end-to-end trained frontend survives anywhere.**
- RESULTS.md Part 2 (2026-07-20 onward) is all frozen-frontend generative work and explicitly
  demotes the CRF arm: *"The CRF-trained variant is NOT a rung (discriminative conditional
  objective, off-program): demoted to `experiments/bt_e2e/crf_baseline.py`, kept as the
  comparison estimator."*
- One later document covers this ground: `/home/sogang/jaehoon/Analyze-SMC/PRIOR_ART_NEURAL_DBN.md`
  carries a section **"ADDENDUM (verified 2026-08-20, main session): the unrecorded rematch was
  run, and it lost by more"** (line 345 onward), reaching the same 0.014 → 0.054 conclusion.
  **It is analysis of the same four files, not a new run.** Note that the same file's *body*
  (lines 209–217) still states the status as *"measured negative once, with an acknowledged
  confounded harness, rematch unreported"* — **that body text is stale relative to its own
  addendum. Two passages in one file make opposite claims about the same four files. Fix this
  before anyone cites it.**

**Conclusion: the rematch histories are the last word on Beat This end-to-end, and
`RESULTS.md` has never been updated. Part 1 still ends at "Rematch launched 2026-07-18 with all
three fixes" (line 88–89) while the rematch's own output has sat two directories away for a
month.**

### 3(e) RESULTS.md structure and content (305 lines, two dated parts)

**Part 1 — "Preliminary: vanilla Beat Transformer vs BT + R2 (end-to-end)", 2026-07-18
(lines 1–89).** Both arms train the official `Demixed_DilatedTransformerModel` **from scratch**,
fold-0, 1,020/146, ≤16 s beat-aligned crops, RAdam+Lookahead, lr 1e-3, clip 0.5, batch 1, 30
epochs. Arm delta is loss/decode only.

Table 1 — full fold-0 val, **n=146**, best checkpoints:

| activations | decode λ | beat F | downbeat F |
|---|---|---|---|
| vanilla | 100 | **0.9506** | **0.9053** |
| vanilla | 14.2 (learned) | 0.9467 | 0.8808 |
| r2 e2e | 100 | 0.9446 | 0.8899 |
| r2 e2e | 14.2 (learned) | 0.9374 | 0.8637 |
| r2 e2e | 14.2, threshold=0 | 0.9341 | 0.8600 |
| pretrained fold_0 (LEAKY reference) | 100 | 0.9622 | 0.9012 |

Failure archaeology (lines 41–50) — **reusable engineering facts:** Run 1 both arms NaN'd →
NaN-skip guard needed (skip non-finite loss AND non-finite grad norm, check weights per epoch).
Run 2 **pure CRF NLL on sigmoid outputs SATURATES within one epoch** (|logit| ≈ 55, 100% of
frames; the CRF gradient is flat under saturation, λ soaks all gradient, eval freezes
bit-identical) → **any future e2e-through-DP training needs a calibration anchor**; the fix was
hybrid CRF + BCE (BCE gradient is maximal at saturation).

Caveats (52–57): single fold, single seed, 30 epochs, hybrid weight 1:1 untuned, 16 s crops,
139 songs skipped, 60-song training-time evals ~+0.01 high.

**Follow-ups (59–66), all five STILL OPEN:** (1) decode-λ sweep on vanilla acts (is 100 optimal,
or 150–300?); (2) frozen-frontend pure R2; (3) full-song / 60 s crops for the CRF arm; (4) meter
set {2,3,4}; (5) **multi-seed + all 8 folds before any strong claim.**
**A sixth follow-up — run it on hard material — was never written down.**

ADDENDUM (68–89), the withdrawal, with its probe table on the full fold:

| | λ=100, obs=6 | λ=100, obs=16 |
|---|---|---|
| vanilla | **0.9506** | 0.9017 |
| r2 | **0.9446** | 0.8911 |
| r2 at λ=14 | 0.9374 | **0.9295** |

Under train-consistent obs=16, **λ=14 beats λ=100 (0.9295 vs 0.8911)** → the "likelihood-optimal
but F-suboptimal λ" reading is *"CONFOUNDED AND WITHDRAWN pending the obs=6-consistent rematch."*
Line 88–89: *"Rematch launched 2026-07-18 with all three fixes."* **The file never returns.**

**Part 2 — "2026-07-20: R2 promotion, the integer-interval dithering artifact, and the
mixture-kernel repair" (lines 93–305).** Key numeric blocks:

- **Dithering isolation:** interval 21.0 → λ_MLE **133.4**; 21.4 → **39.7**; 21.5 → **37.6**;
  real 300-song corpus → **40.27**. fps×2 scaling: 40.27 → **80.08** (ratio **1.988** vs
  predicted 2.0). Mixture repair: metronome w=0.51 λ→96; **real data w=0.370, λ=93.1**.
- **Bare Viterbi decode:** R1 shipped 0.9506 / R1 bare λ=100 **0.9100** / mixture bare **0.9193**
  (+0.0093).
- **Deployment decode:** R1 shipped 0.9506/0.9053, R1 λ=100 + wrapper 0.9504/0.9057,
  mixture + wrapper 0.9503/0.9053 → **"DEAD TIE."**
- **Oracle ceiling** (146 songs, 12 decode variants + half-bar flip): beat F shipped 0.9504 /
  oracle 0.9596 (**+0.009 headroom**); **downbeat F shipped 0.9057 / oracle 0.9533 (+0.048,
  WIDE OPEN)** — "the project's best quantified in-domain opportunity," still untouched.
- **Top-3 parity control:** hand-set λ=100, CRF λ=98.6 and mixture (w=0.37, λ=93.1) agree within
  **±0.0004 on all six metrics.**

**Stated open items in Part 2:** the +0.048 downbeat oracle prize; three named collectors —
(a) calibrated discriminative emission `p(class|f)/p(class)` (ASR hybrid, off-program),
(b) bar-phase disambiguation at decode (cheapest, targets the dominant error mode),
(c) R5 decoder-as-density (the on-program endgame). Transition axis "now closed at parity."
R4.5 generative-Gaussian route "CLOSED." Instrumentation lesson: *"log full-set or fixed-held-out
NLL, never fresh minibatches."* Re-derive the tempo-increment kurtosis on sub-frame-interpolated
annotations before reuse. Unchecked prior-art residue: madmom GitHub issues, Cemgil full texts,
non-English/gray literature.

### 3(f) STALENESS VERDICT, stated plainly

1. The e2e CRF negative is **current, not stale** — the confound-repaired rematch exists and the
   arm **lost by more** (gap 0.0142 → 0.0543).
2. It is **not re-scorable**: checkpoints and logs are destroyed; the surviving numbers are
   60-song-subset validation F that RESULTS.md itself tells you not to quote.
3. It is **not a test of our hypothesis**: 69% metronomic dance music, zero SMC, on corpora where
   a rigid grid already ceilings at 0.918–0.999.
4. It is **not a negative for a probabilistic sequence model** — it is a negative for **CRF-NLL
   as a training objective for a frontend**, which is a discriminative conditional objective that
   the record itself calls *"NOT a rung … off-program."* The generative rung on the same chassis
   (R2 exact EM) *wins* unsupervised: bare decode **0.9132 vs R1's 0.9033, +0.010** **[RETRACTED 2026-08-20: NOT a head-to-head. `results_r2_val.json` uses a different frontend (`bt_fold0_repacked.pt`) with peak-snap OFF (`correct=False`). At the deployment config it reverses: R2-generative(40.3) 0.9488 < R0/R1/CRF 0.9506, and em-single λ=40.3 loses to hand-set λ=100 on all six metrics. Do not write 'the generative rung wins'.]** on the same
   fold (`project_rungs_resurrection`, `results_r2_val.json`).
5. **What we may honestly say in the paper:** "training a SOTA frontend end-to-end through a
   CRF over the bar-pointer state space did not improve in-domain F on metronomic corpora
   (n=146, fold-0, single seed); we therefore keep the frontend frozen and place the
   contribution in the decoder/inference, where it is measurable on the material the failure
   analysis identified." **What we may NOT say:** that end-to-end structured training has been
   shown not to work, or that it has been tested on hard material. Follow-up #5
   (multi-seed, 8 folds) is unmet.

---

## 4. What separates "probabilistic model" from "periodicity regulariser"

This is the section the paper needs, so I have listed each item with what it discriminates.

### 4.0 A PRIOR OVER ABSOLUTE PERIOD WAS BUILT AND IT MONOTONICALLY HURTS

> ⚠️ **REVISED 2026-08-20 by forensic audit. Three corrections; read them before citing.**
> **(1)** The numbers below are anchored to the WRONG baseline row. 79.3% is the `acf-smallest`
> rule; the prior family's true w=0 is `acf-argmax` at **81.37%**. Corrected sweep:
> **81.37 → 77.54 → 73.01 → 69.59%**, halving 8.06 → 14.90%, doubling 4.73 → 7.55%.
> Against the published anchor the first step is not even significant (p = 0.129); against the
> correct one every step is (McNemar exact p = 5.3e-4, 3.4e-10, 4.5e-16; bootstrap 95% CI on the
> w=1 delta [−10.98, −5.84] pp). **The negative is real and stronger than recorded — but the
> numbers as printed below cannot be quoted.**
> **(2)** The doc's stated mechanism is FALSE. At a 2 s period the prior prefers the truth over
> its half by **1.231 nats**, and empirically repairs 29 of 83 halvings. The real mechanism is a
> closed-form threshold: the prior prefers the halved candidate above exp(μ + ln2/2) = **2.5877 s**
> and the doubled one below **1.2939 s**. Of the 130 songs it breaks, all 82 flipped-to-half lie
> above the first threshold and all 44 flipped-to-double below the second — 100%, no exceptions.
> In-band accuracy rises 87.9 → 94.3%; out-of-band collapses 65.9 → 23.0%. Bounded upside,
> unbounded downside.
> **(3) IT DOES NOT REACH FOSCARIN ET AL. Do not cite it against them.** Their `loss.py` contains
> only `MaskedBCELoss`, `ShiftTolerantBCELoss`, `SplittedShiftTolerantBCELoss` — the periodicity
> loss is prose, not code, so there is nothing to refute. Ours is a prior over the ABSOLUTE period;
> theirs would be a functional of the model's OWN OUTPUT SEQUENCE, which is scale-equivariant and
> therefore octave-BLIND by construction. **The two fail for opposite reasons.** Also: exact-period
> -at-±5% is not a beat-tracking metric, no F/CMLt/AMLt exists for any prior row, and AMLt is
> octave-invariant so the entire measured harm would be invisible in it. SMC is absent from the
> audit entirely, and the closest proxies disagree (gtzan classical +6.0 pp, asap −3.3 pp).
>
> **The defensible claim is narrower:** *a song-independent prior over the absolute tempo/period
> value cannot break the octave degeneracy — its best oracle-fit form (26 parameters fit directly
> on the gtzan test set) recovers 1.81 of the 15.11 pp separating w=0 from the 96.48% candidate
> ladder ceiling, and transfers +0.00 pp to held-out data — because it can only threshold on the
> period value, never on the evidence.* The slogan "no penalty on the marginal can fix a degeneracy
> in the relation" IS supported. "Therefore their periodicity loss will fail" is NOT.
>
> **To license the Foscarin citation** we must implement their loss ourselves (scale-equivariant
> autocorrelation-peak term added to `beat_this/model/loss.py`, weight swept over {0, 0.05, 0.2,
> 1.0}, fine-tuned from `final0`, scored through `scripts/_harness.py` on SMC and gtzan, 3 seeds).
> Estimated 8–16 GPU-hours. Pre-registered predictions: (i) AMLt−CMLt does not close, because a
> scale-equivariant loss cannot select an octave; (ii) F falls before CMLt rises.

`vbpm/docs/octave_error_audit.md:53-69`. A **train-fit log-normal prior over absolute bar period**
was added to the ACF bar-period rule and swept in weight. **gtzan exact-period accuracy, AS
PUBLISHED (superseded — see correction 1 above):**

| tempo-prior weight w | exact period correct | halving rate |
|---|---|---|
| 0 (shipped rule) | **79.3%** | **11.6%** |
| 0.5 | 77.5% | 10.2% |
| 1.0 | 73.0% | 13.2% |
| 2.0 | **69.6%** | **14.9%** |

The doc's own explanation, **now falsified** (see correction 2):

> *"The tempo prior is a clean negative and monotone in its weight: **halving a 2 s bar gives 1 s,
> which a prior fit on real bar periods still finds ordinary, so the prior cannot punish the
> actual error** — it only drags genuinely long bars toward the median."*

**Why this is the discriminator we need.** A periodicity/tempo prior is a *marginal* statement —
it scores how plausible a period is on its own. The octave error is a *relational* failure: the
wrong answer is exactly as marginally plausible as the right one. **No penalty on the marginal
can fix a degeneracy in the relation.** Only a model that carries the tempo as latent state and
scores the *joint* evidence-under-that-state — i.e. a probabilistic sequence model — can price
the difference. This is corroborated analytically by `project_octave_degenerate_likelihood`
(§2.3): under a pointwise emission the true rate beats 2× by **1.02 nats/song** against a recon
of ~209, and the measured asymmetry (×2 costs ~1 nat, ×0.5 costs ~30 — a **27× ratio**) survives
a **20× sharpness sweep and a 10× pos_weight sweep**.

**And the mirror-image control:** a *rigid* periodicity prior is not the answer either.
`session_2026-08-13_handover.md:190` — madmom's DBN on Beethoven rubato scores **0.437** against
**0.810** for Beat This peak-pick. `project_bt_aux_loss_plan` states the resulting design
constraint verbatim: *"the term must price DRIFT-WITH-CONTINUITY, not RIGIDITY. A naive coherence
penalty rebuilds the DBN failure — measured, BT peak-pick 0.810 vs BT+DBN 0.437 on Beethoven
rubato."*

**So both ends of the "periodicity regulariser" spectrum are pre-measured negatives in-house:**
a *soft learned* periodicity prior monotonically degrades period accuracy (79.3% → 69.6%), and a
*rigid* periodicity constraint halves F on rubato (0.810 → 0.437). **The claim "the answer is a
probabilistic model, not a periodicity regulariser" is not rhetorical for us — we have run the
regulariser, at both stiffnesses, and it lost both times.**

### 4.0b The source theory says the same thing, and says why

`NeuralDBN.pdf` (48 pp., "The Observation Distribution and the Posterior Distribution for the
Bar-Pointer Model, and the Derivation of the EM and Viterbi Algorithms," Moon-Ryul Jung, Sogang)
is **purely analytical — zero tables, zero figures, no corpus experiments, no F/CMLt/AMLt
anywhere**; its numbers are hand-worked toy checks (e.g. L=8, S=(3.0, 2.0, 0.5, −0.5, −1.5, −2.0,
−1.0, 0.2) → w*=3, λ*=8/3). Its central calibration argument is directly quotable:

> Böck's misplacement of the off-beat/near-beat count into the emission *"does no harm in the
> handcrafted-DBN Viterbi… **decoding asks only for the single best path, a per-frame score rather
> than a calibrated posterior**"* — but it *"would then report a **miscalibrated per-state
> likelihood**."*

**Viterbi/argmax is forgiving of miscalibration; an honest posterior is not.** That is the
theoretical statement of why "add a penalty and take the argmax" and "build a calibrated
probabilistic model" are different objects, and it is in the project's own source theory.

Also usable from it: the count ratio (L−w)/w = (λ−1) + {L/λ}/⌊L/λ⌋ and the argument that the
count belongs in the **prior**, not the emission; Krebs's tempo normaliser Z(u) is state-dependent
and **cannot** be dropped; EM for the two λs (emission width is an **integer count** → staircase
likelihood → chosen by **enumeration**; tempo rate λ_v is **continuous** → moment-matched
gradient); §6 "Version A" = input-conditioned λ = g_θ(x), λ_v = f_θ(x), a neurally-*calibrated*
DBN; §7 "Version B" = trained neural emission p_θ(b_k|z_k,x), a genuine neural DBN keeping
**exact** forward–backward over the discrete pointer grid — *"no encoder q_φ, no approximation,
and no ELBO,"* explicitly escaping the structured-VAE line's KL-collapse risk. And, in the
project's own theory document: **EM's objective *"is not the same objective as beat-tracking
F-measure; if the two disagree, the sweep against F-measure is the one that matches how the
system is judged."***

`VAEBPM_fin.pdf` (26 pp., "A Tutorial on Latent Variable Models for Beat Tracking: The
Bar-Pointer CVAE and Its Relationship to Standard Frameworks," 10 July 2026, marked *"Prepared
for Teaching Purposes"*) also has **zero tables, zero figures, no experiments, no datasets, no
metrics.** It is a tutorial: LVM recap, ELBO derivation, VAE, Sohn CVAE, §7 (audio-only encoder /
fixed physical prior — the architecture the project actually used), §9 Transformer bar-pointer
CVAE, §10 relation to the classical DBN (their CVAE differs in observation direction; dynamics =
Transformer rather than handcrafted; explicit p_θ(b|z); **inference amortized rather than
Viterbi**), §11 distinctions table, §12 misconceptions. §9.10–9.13 are **one-to-three-sentence
sketches** (§9.13 is literally *"Domain-adversarial training, test-time adaptation, masked ELBO
for heterogeneous annotations"*). **Note: `audit_2026-08-19.md:159` records that §11 retracts the
audio-only-encoder authorisation that §7 grants.** Do not cite either PDF for empirical support —
neither contains any.

### 4.1 The decisive one: a likelihood repair lifts continuity and still leaves F at chance

`project_structured_q_verdict`, gtzan **n=993**, single-variable, same prior/emission/encoder:

| | F | CMLt | AMLt | est/ref |
|---|---|---|---|---|
| factorized q, under-specified likelihood | ~0.074 (= null) | 0.115 | 0.184 | 1.313 |
| factorized q + `gauss_time` likelihood repair | **~0.13** (null 0.07) | **0.467** | **0.556** | **1.059** |
| chain q (neural DBN, exact forward) | **0.750** | — | — | — |

**A smoothness-style repair to a factorized objective is worth +0.352 CMLt / +0.372 AMLt and
essentially nothing on F. Making the model an actual sequence model is worth +0.68 F.** The
record's own summary: *"an under-specified likelihood costs continuity; the family decides
whether the model works at all."* If Beat This!'s authors add a periodicity term, this predicts
they will recover continuity metrics and not much else — which, framed correctly, is *not* a
refutation of them but a demonstration that continuity and correctness are separate axes and only
the sequence model buys both.

### 4.2 A per-frame *penalty* was built, tuned, and lost to a global constant — twice

`project_r3_perframe_lambda` (2026-07-18, fold-0, **n=146 val**) is the closest thing in the
record to "bolt an adaptive stiffness term onto the objective," and its full arc is a lesson:

1. Built: per-frame audio-conditioned λ_t = 100·exp(net), zero-init so λ_t starts at the global
   optimum; R1 certificate intact (2-D path byte-identical).
2. **First verdict: per-frame LOSES.** 0.8941 vs global λ=100's 0.9114 (−0.0172). Mean-pinned
   variants worse: base150/reg60 → **−0.050**; even reg=60 could not hold mean λ_t at 150 (it
   drifted to 551). Diagnosis at the time: "CRF-NLL is not F… the OBJECTIVE is the wall."
3. **RETRACTED (same day, adversarial review):** an off-by-one in `crf_nll`'s per-frame kernel
   indexing. Proof used: force a unique path so NLL must be 0 — the buggy index gave **NLL =
   −2.26 (impossible)**, the fixed index gives **exactly 0**. The global baseline was unaffected
   (net-init ⇒ frame-constant kernel ⇒ bug invisible), so **per-frame was trained on a corrupted
   objective while global was clean.**
4. **FLIPPED TO POSITIVE:** frozen R3 per-frame **beats** global λ, **0.9172 vs 0.9114 = +0.0058**
   (bare time-varying decode, fold-0, 146 val). λ_t healthy (mean ~200, std ~250, range
   [15,1500]); the phantom "over-stiffening" (buggy mean ~1000) was the bug.
5. **Caveats the record itself insists on:** "+0.0058 is over R1's MECHANISM (global λ, bare
   decode 0.9114), **NOT over deployed vanilla (0.9658, shipped decode)** — lower bar";
   "single fold/seed, 146 songs — margin could be noise."
6. **SMC (OOD) verdict:** the R3 net trained on CV data generalises to SMC (217 songs, median
   tempo-range **1.238** vs CV **1.035**) with a positive delta everywhere, concentrating mildly
   on tempo-varying songs (peak **+0.0108** at range >1.2) but breaking down at range >1.4
   (**+0.0074**). Absolute F ~0.41 vs 0.91 CV. Conclusion recorded: "R3 is a small correction on
   a much bigger problem it doesn't address… the SMC lever remains tempo COVERAGE."
7. **Handover says R3 was "retired on effect size (~+0.001 F), not on failure."**

**Use this as follows.** A per-frame adaptive-stiffness term — the most sophisticated form of the
"periodicity regulariser" idea anyone would actually build — is worth **+0.0058 F** over the
global-constant mechanism on n=146, **~+0.001 F** at deployment, and generalises OOD only in the
+0.006 to +0.011 range. Meanwhile changing the *posterior family* is worth **+0.68 F**. And the
same regulariser costs continuity: on SMC, `R3_lamt` gains **+0.0181 F** over fixed-100 and loses
**−0.0131 CMLt / −0.0903 AMLt** (§1.1c).

### 4.3 Marginalisation vs argmax — measured three ways

- **Near-uniform potentials, F 0.000 peak-picked vs F 0.602 through the exact sweep** on the same
  songs (`project_quasi_svae_mechanism`). Marginalisation *is* the algorithm.
- **64% of songs flip between the frame-0 argmax and the smoothed argmax**; median 13% of
  posterior tempo mass an octave class away; 57% of songs carry >10% competitor mass; phase
  marginals hold 43% of mass >60° off the mean. **No unimodal location-q can represent this.**
- **Exact marginalisation of a categorical anchor** (`project_anchor_k_v2_verdict`): in-loop
  gtzan F 0.468 → 0.752 (n=993), strict protocol CMLt-true 0.600 → 0.743, anchor error 137 ms/36%
  → 27 ms/71%. *(Number retired for citation — see §5.)*
- Counterexample to keep honest: **IWAE/FIVO are dead** at these horizons (18 runs, ANOVA
  p=0.450; ESS 1.08/32 at T=1500). "More Monte-Carlo" is not the mechanism; **exactness** is.
  This is a useful distinction to make in the paper — our claim is exact forward-backward /
  closed-form marginalisation, not particle methods.

### 4.4 Latent tempo state helping — direct evidence

- **`tchain` has "tempo in the state"** and that is the arm that works (§2.2).
- **Widening the tempo state space** (`R1_wide30_notempi`) is the largest CMLt gain in the SMC
  config table: **+0.0298 CMLt** (§1.3).
- **`project_r4_final_verdict` / `project_r4_supervised_verdict` (rungs_lab/r4/):** genuinely
  song-conditioned interpretable parameters — per-song prior TV 0.49 from the corpus mean, λ_t in
  54–137, **prior-vs-annotated-tempo Spearman 0.460 (run2b) / 0.63 (run1) unsupervised**,
  rising to **0.769** supervised and **0.797** with short-crop training. "**Unsupervised implicit-genre
  tempo priors EXIST.**" Deployed F is family-flat (0.9115 supervised / 0.9129 unsupervised /
  0.9132 R2mix) — i.e. **the latent tempo prior is real and readable but F cannot see it.**
- **`project_1overT_fix_breakthrough` (2026-08-09):** the initial-tempo prior can contribute at
  most log(36)/T nats/frame, so at T=1400 an oracle prior is worth 4e-4 (noise) — the head was
  untrainable for a *dimensional* reason. Training on 150-frame sub-crops: unsupervised octave
  log-odds **1.550 → 3.728** (78.8% → 92.5% correct) for BT, **0.925 → 4.005** (70.5% → 96.6%)
  for MERT, all **p < 1e-19**; both unsupervised short-crop arms **beat the supervised long-crop
  arm (2.617)**. **CAVEAT recorded: "deployed F not yet measured for short-crop arms."**
- **`project_octave_degenerate_likelihood` oracle-rate arm:** a correct rate unlocks the anchor
  posterior (Hq 0.998 → 0.42/0.47 — "the ONLY arm where the posterior concentrated"), **not the
  other way round.** And with the correct rate, CMLt/AMLt = 0.995.

### 4.5 Calibration and overconfidence

- **`reference_smc_blind_spot_paper`** (the user's own v1): models produce **"confident-but-wrong"**
  activations on the failure clusters, not merely noisy ones; the DBN's 55-BPM floor forces
  confident double-tempo predictions on **~21% of SMC tracks**.
- **`project_foldhonest_parity` (2026-07-12):** replacing Beat This's sharpened `act2` with an
  own BCE-trained `ActivationHead` as filter observation made the structured filter **beat
  same-evidence peak-pick for the first time: 0.917/0.922 vs 0.896 (both seeds, +0.021/+0.026).**
  Recorded thesis: **"honest BCE evidence is exploitable, pre-sharpened act2 is not"** — i.e. the
  *calibration* of the frontend, not its accuracy, is what a probabilistic decoder consumes.
  Same memory: **"calibration belongs in TRAINING not deployment"** (outlier-epsilon at deploy
  hurt monotonically).
- **`feedback_own_evidence_head`:** "BT's act2 carries their confident-but-wrong loss shaping;
  our own head trained on fold-honest data can be calibrated." A standing directive, not a
  one-off. **This is a direct, in-house, empirical statement that Beat This!'s frame-wise loss
  produces miscalibrated activations and that miscalibration is what breaks the sequence model.**
- **Nine measured instances of "off-manifold uncalibration"** (`project_factor_b_campaign`): the
  shared mechanism behind five refuted extensions. The sharpest: **metrical-level selection via
  classifier likelihood hurts everywhere** (val beat F 0.9506 → 0.9180; SMC zero-shot 0.483 →
  0.403) because **"cross-grid likelihood comparison is invalid — the classifier is only
  calibrated on correct-tempo grids."** A discriminative frame-wise model has no valid likelihood
  off its training manifold; a generative sequence model does by construction. **This is arguably
  the single most quotable calibration result for the argument.**
- **`project_smc_certified_ladder` ROBUST NEGATIVE, and it cuts the other way — quote it
  honestly:** per-song marginal likelihood prefers a *looser* λ than F does,
  **ρ(log λ_ML, log λ_OR) = −0.230, p=6e-4** on clean activations, and λ_ML floor-seeks to 1.
  "Evidence overshoots loose." Nuance recorded: the *oracle* λs are also loose (median 10) — SMC
  genuinely wants λ ≪ 100. So the likelihood is not perfectly task-aligned, and we must not claim
  it is.
- **`project_joint_eval_verdict` SNR sweep** (60-song subset, synthetic Gaussian noise on both
  rich input and act2): the structured model **crosses over** and beats peak-pick on beats and
  downbeats past ~1.5× std. Downbeat gap (peak − structured): +0.63 → +0.65 → +0.27 → **−0.15**
  → **−0.12** at noise 0/0.5/1/2/4. Mechanism: "the free-run prior asserts a periodic grid
  ~independent of the corrupted input → a **robustness floor** the discriminative head lacks."
  **Honest caveats the record insists on:** the crossover is "into MUTUAL MEDIOCRITY" (structured
  downbeat only 0.385 at the win); the noise is **synthetic feature noise**, not real degraded
  audio. Final disposition on record: *"structured generative post-processing is dominated by a
  strong neural frontend on clean data but crosses over to a graceful-degradation regulariser
  below an SNR threshold."*

### 4.6 The mis-specified-prior mechanism (the "heterogeneous scale" argument)

From `HANDOVER_FROM_VBPM.md`, measured 2026-08-20 in VBPM on a Scriabin performance (`asap468`):

- The per-bar tempo prior's scale is a **single global constant** fitted on corpus medians
  (`GAMMA = 0.0363`), i.e. a typical bar-to-bar tempo change of **1.037×**.
- That song's **median** bar-to-bar change is **1.34×**, p90 **1.76×**, max **3.32×** — the median
  move is **8 scale units into the tail**.
- Cost of tracking the passage honestly: **71 nats over 19 bars = 3.7 nats/bar.**
- **Inserting a phantom beat and holding tempo steady is cheaper than telling the truth.** A
  phantom beat was visible in the trajectory plot exactly where the performer stretched a bar to
  2.3× median.

**Claim it supports:** "a hand-set *homogeneous* tempo-transition prior is mis-specified for
expressive material, and a mis-specified prior is worse than no prior — which is precisely why
peak-picking beats the DBN on slow, rubato SMC tracks. **The fix is a prior whose scale is
heterogeneous, not a better single constant.**" This is *exactly* the periodicity-regulariser-vs-
probabilistic-model distinction: a periodicity penalty is a homogeneous scale by construction;
only a model with latent state can carry a per-song / per-bar scale and marginalise over it.

Corroborating: `project_crop_horizon_root_cause` — "short crops make the ELBO **PREFER a
metronome**; ~45 s+ separates truth from coast." And `project_rigid_grid_ceiling` (CORRECTED
2026-08-15, full corpus n=2893, annotations only, self-validated): the best-possible F of a
**constant-period** grid with oracle offset is

| | 45 s window | full song |
|---|---|---|
| gtzan | 0.902 | 0.902 |
| ballroom | 0.918 | 0.919 |
| hjdb | 0.999 | 0.999 |
| hainsworth | 0.737 | 0.685 |
| beatles | 0.713 | **0.349** |
| rwc | 0.565 | **0.258** |
| asap | 0.324 | **0.164** |
| POOLED | 0.781 | **0.716** |

**Protocol trap to state in the paper:** "we train and evaluate on 45 s excerpts, where the
constant-rate ceiling is 0.781 pooled — **the excerpt protocol STRUCTURALLY HIDES the phenomenon
the paper is about.**" And: **never evidence adaptive structure on gtzan/ballroom/hjdb**, where a
rigid grid already scores 0.90–0.999.

---

## 5. Retractions, nulls, and things we must NOT claim

### 5.1 Explicit retractions

| Withdrawn claim | What killed it | Source |
|---|---|---|
| **"Generative λ≈40 matches heavy-tailed tempo-increment statistics."** | A **synthetic metronome** reproduces it. `synthetic_grid_probe.py`: interval 21.0 → flat Viterbi path → λ_MLE **133.4**; interval **21.4 → dithering → λ_MLE 39.7**; real 300-song corpus → λ_MLE **40.27**. A perfect metronome reproduces the real-data λ to **1.5%**. Confirmed by a scaling law (2× upsampled activations → λ 40.27 → **80.08**, predicted ratio 2.0, measured **1.988**) and by repair (two-component `w·Dither(|Δ|≤1) + (1−w)·exp(−λ|r−1|)` kernel, same exact EM → **w=0.370, λ=93.1**, agreeing with hand-set 100 and CRF 98.6 — *three estimators, one answer*). | `project_dithering_artifact`, `HANDOVER_FROM_VBPM.md` |
| **The tempo-increment heavy-tail statistic itself (kurtosis ~13).** | "Suspect, being computed from **frame-quantised** annotation intervals carrying the same dithering. **Re-derive on sub-frame-interpolated annotations before reusing.**" | same |
| "The likelihood-optimal λ=14 is F-suboptimal." | Confounded by an `observation_lambda` train(16)/decode(6) mismatch: under train-consistent obs=16 decode, learned **λ=14 BEATS 100 (0.9295 vs 0.8911)**. | `project_bt_r2_e2e_preliminary` ADDENDUM |
| "Per-frame R3 loses to global λ (−0.017 / −0.050)." | Off-by-one in `crf_nll` kernel indexing; forced-unique-path NLL was **−2.26** (impossible), fixed index gives **exactly 0**. Flipped to **+0.0058** for per-frame. | `project_r3_perframe_lambda` |
| "The categorical q(l_t) over integer lags fixes the multimodality gap." | Trained and **failed** inside the real ELBO, 8 seeds/arm: shipped 0.3108 vs categorical 0.2479 (p=0.0514), CAT2 0.2767 (p=0.0466, worse). **"Do NOT cite this memory as a validated fix."** | `project_multimodality_gap` CORRECTION |
| "All octave problems trace to BT activations." | On **ideal synthetic** features with an oracle anchor the octave degeneracy is still present ⇒ **the likelihood owns it**. | `project_octave_degenerate_likelihood` |
| "MERT wins over BT features / the layer race." | The layer race was a **dataset leak**; the MERT F difference is a **tie, not a win**; the octave win was a **genre shortcut that fails on SMC**. The n=250 hainsworth advantage (0.816 vs 0.772) is **GONE at n=920** (0.792, range [0.72–0.88] vs BT 0.831); the single-seed fused 0.9658/0.9200 was **"a lucky draw"**, honest 5-seed fused mean **0.9548**. | `project_mert_audit_retractions`, `project_mert_conditioning_verdict`, `project_factor_b_campaign` |
| "The ELBO structurally prefers a diffuse posterior (312 vs 6.9 nats)." | Arithmetic reproduces but the pair is unfair and it reads κ_p off the very checkpoint it explains. Honest residual claim: **the emission carries only ~7 nats of phase information per 30 s window, ~190× short of what tolerance-level sharpening needs.** | `docs/audit_2026-08-19.md` via `project_quasi_svae_mechanism` |
| "Truth is the objective argmax by 2.0 nats." | **REFUTED as stated** — the scoring function omitted the rate prior and refit the rate per candidate. Truth and the model's path are **TIED**. | same |
| "κ=383 is 100–1000× too loose." | Direction only. The Wing–Kristofferson estimator is rejected on 30–95% of every human corpus and the 4 ms figure comes from two machine-quantised datasets. **κ is UNMEASURABLE from this corpus** and inconsequential in every sweep. | same |
| "A dead-drift / constant-rate class cannot reach 0.8 on gtzan (ceiling 0.639)." | **FALSE — it reaches 0.902.** The 0.639 was a **blues-only subsample** (gtzan's catalog is genre-ordered, so "first 60 songs" = 60 blues tracks) **plus a median rather than least-squares estimator**. | `project_rigid_grid_ceiling` |
| Eight further claims withdrawn 2026-08-13 (per-frame rate trains worse; "every arm sweeps 1.5× too fast"; "the model is half-bar locked"; a 2.5 s bar-length ceiling; the evidence-head precondition; "the KL blocks the fit"; "alignment recon rewards a blurrier anchor"; "a_t can only learn a sinusoid"). | Each died to a control that should have been run first — unnormalised divisor, wrong printed reference band, flat histogram, tanh rate bound, non-reproduction under adversarial re-derivation, seed spread, coin-flip sign across 20 songs, weight sharing. | `project_2026_08_13_retractions` |

### 5.1b THE FORMAL CLAIMS LEDGER — `vbpm/docs/audit_2026-08-19.md`

This file is a **formal claims ledger** with an explicit *"MUST BE RETRACTED FROM
docs/quasi_svae_mechanism.md"* list at lines 115–124. **It is the most important single document
for not embarrassing ourselves, because it retracts the mechanism story of the very arm that
produces our best numbers.** Everything below is retracted or downgraded:

| Retracted claim | What the audit found |
|---|---|
| **"Zero variational gap / training tchain is maximum-likelihood training of a neural DBN"** (Part I.1) | **REFUTED for the object actually trained** (claim O). Containment holds for the *non-amortized* family; the trained encoder **reads x only** while optimal potentials are functions of **y**. *"I.3 refutes I.1."* The bound is loose by a θ-dependent `E[KL(q(z|x)‖p(z|y,x))]`. |
| **"Three independent structural reasons" location-families fail** | **OVERSTATED** — II.1/II.2/II.3 are **one scalar KL decomposed three ways**, not three independent mechanisms. |
| **"The 0.973-vs-0.000 A/B shows a representational floor"** | **REFUTED** — the factorized arm has bin-step 0 at 100% of frames = a **kinetic trap**. *"An optimization trap cannot be attributed to a representational floor."* Numbers stand, interpretation does not. |
| **archain ladder step 7: "MONOTONE convergence to truth, ratio 1.06→1.02, med|err| 499→129 ms"** | **REFUTED as stated — spliced from two different logs.** The quoted run goes ratio **2.01 / 612 ms at ep250** and never recovers (ep399: 615 ms, **CMLt 0.000**). Tail-mean F **0.418 ± 0.213**. *"No run in the directory earns the word 'converges.'"* |
| **The archain capstone: "F 1.000 / 100% in-tol / 38 ms at ep375"** | **REFUTED as a conclusion.** Tail F values at ep≥150: 0.583, 0.583, 0.417, 0.833, 0.750, 0.167, 0.154, 0.083, 1.000, 0.667 — **mean 0.491, sd 0.299. The 1.000 is the max of 17 draws.** |
| **"The ELBO defects to the double octave; the defection is the OBJECTIVE's, not the family's"** + the +0.04 / +2.12 nat pricings | **REFUTED** by sibling run `ov_stride8` (ratio 1.00 → 1.97 → 1.98 → 1.00/28 ms/F 1.000 → 1.00/13 ms/F 1.000): *"a **limit cycle of the single-sample von Mises gradient**, not a stationary preference."* |
| **archain ladder steps 1–4 as a mechanism ordering** | Differ by **less than the noise floor**; one song, one seed. |
| **"κ_physical is 100–1000× too loose" + the 4 ms Wing–Kristofferson jitter constant** | *"RETRACT … entirely"* (claim D). The estimator is at its floor on **30.6% / 0.5% / 84.0% / 50.5% / 78.4% / 95.3%** of ballroom/hjdb/beatles/hainsworth/rwc/asap. Pooled median κ **4.71e13**, pooled median jitter **0.0 ms**. |
| **"Tightening kappa fixes placement, 60% → 100% in-tolerance"** | **REFUTED** — the two logs are **bit-identical to ep200** and diverge on nondeterminism alone. |
| **"The rubato sweep tests rubato"** | **REFUTED** — selection ranked *whole-song* CV while training on a *30 s window* (asap169 1.103 → 0.083; asap456 0.980 → 0.011). **Only asap343 is genuinely rubato in-window.** |
| **"Slow songs fail"** | **REFUTED as a single phenomenon** (asap254 is an annotation anomaly). |
| **"Truth is the objective argmax by 2.0 nats"** | **OVERSTATED** — truth and the model's path are **TIED**, 0.1 nats inside an 8–12 nat epoch-to-epoch recon wander. |
| **"The ELBO structurally prefers a diffuse posterior, 312 vs 6.9 nats"** | Arithmetic reproduces but the pair is unfair and it reads κ_p off the checkpoint it explains. Honest residual: **the emission carries ~7 nats of phase information per 30 s window, ~190× short** of what tolerance-level sharpening needs. |
| **Sampled-vs-deployed overconfidence (sampled F 0.164, max 0.467, P(F>0.5)=0% vs deployed 0.733)** | **UNVERIFIABLE AND CONFOUNDED** — all 64 draws share one stochastic φ₀; an identically-configured replicate gave 0.600 at ep399. Honest statement retained: *"the mean path scores 0.6–0.93 depending on checkpoint and nondeterminism; draws score ~0.16."* **Do NOT use this as our calibration evidence.** |

**Audit NULL results (claim K):** κ 3000 vs 30000; `cau3k` vs `cau30k` (max gap **0.056** against
sds up to **0.157**); the `ts_` sigma sweep; stride 8 (**0.466 ± 0.401**) vs stride 25
(**0.314 ± 0.077**). *"The only supra-floor between-arm effect is base vs tempered-walk and it is
**NEGATIVE**."*

**⚠ THE PROJECT-WIDE NOISE FLOOR — apply this to every single-song/single-seed number in this
document.** Within-run tail sd of a final-epoch F snapshot is **~0.16 overall and ~0.30 on
"locked" songs**; adjacent-checkpoint |ΔF| across **73 logs** is **median 0.076, mean 0.142,
p90 0.344, max 1.000**. And `archain_lab.py:53` hard-codes `torch.manual_seed(0)` —
**no archain result on record is multi-seed.**

**Also confirmed absent from the record:** `p(N|φ)` (the count term) is **parked deliberately,
not rejected — UNTESTED** (`count_term_and_bernoulli.md`). Two real holes in the ELBO derivation
were found by line-by-line audit: a **Dirac q(φ₁)** and the **missing p(N|φ)**; fixing p(N|φ)
alone **changes nothing** (identical to control on every metric). Meter as a first latent is
**measured dead**: corpora are 85–100% m=4 with **KL ≈ 0.02**, and meter identifiability from
timing alone is ≈ chance.

### 5.2 Numbers explicitly RETIRED for citation (even though they were real)

`project_icassp_sprint_goal` (2026-08-15, user's own instruction):

> "**0.733 IS RETIRED.** The user identifies it as pre-phase-VAE (Bernoulli-era) work. Do not
> cite it, do not re-certify it, do not use it as the baseline. … **Same retirement applies to
> anchor_k v2's 0.752.** … keep the MECHANISM lessons, drop the numbers."

So: **do not put gtzan 0.733 (rate-grid) or gtzan 0.752 (anchor_k v2) in the paper as results.**
The *mechanisms* (search-beats-regression; exact marginalisation of a categorical anchor) remain
citable. Note this creates a known hole the record flags: "the 'amortization gap' framing
(interval corpus F 0.16 vs search 0.733) loses its denominator, so the mechanism story needs a
contrast measured INSIDE the current line" — the `tchain` 0.750 vs factorized 0.074 A/B (§2.2)
is that replacement contrast.

### 5.3 Nulls that must not be resurrected

- **Q_reg sweep (Kalman process-noise floor) — RULED OUT.** 0.001 → 0.01 → 0.05 → 0.1: the
  in-domain↔SMC gap stays flat then widens (0.241 → 0.240 → 0.250 → 0.257) while **both**
  in-domain and SMC F decline monotonically (0.895 → 0.877; 0.654 → 0.620). mat_Q's trace barely
  moves (4.7–5.1) — the learned covariance compensates. **"Do not re-try without a new hypothesis."**
- **Local/piecewise rubato augmentation — NEGATIVE.** in-domain 0.895/0.845 → 0.879/0.841;
  SMC 0.654 → 0.640; **SMC CMLt−CMLc gap (the actual target) 0.166 → 0.162, flat.**
- **Adaptive per-frame Kalman noise — NEGATIVE on target, mechanistically informative.**
  in-domain 0.895/0.845 → 0.861/0.844; SMC 0.654 → 0.630; SMC CMLt−CMLc **0.166 → 0.159**.
  Diagnostic: scale_Q is genuinely adaptive (within-song CV 0.18–0.19) but **scale_R collapsed
  toward its floor** (per-song means [0.005, 0.024] SMC vs [0.006, 0.014] in-domain).
  **Three levers tried, three negatives on the target metric.**
- **Beat This tempo augmentation with pitch co-adjust — already tried, did not help**
  (`project_smc_07_mission`).
- **Hazard-vs-volatility conditioning — NULL under BOTH objectives**, at 700 and 1400 frames;
  horizon-doubling refuted as the fix. Recorded cause: the information is **absent from BT
  features at this horizon**.
- **IWAE / FIVO — DEAD** (18 runs, ANOVA p=0.450; ESS 1.08/32 at T=1500).
- **Defect-tolerant bar Viterbi — worse at every penalty**; "grid insertions/deletions
  self-compensate, rigid rotation is the right prior."
- **Evidence injection (classifier probs replacing the downbeat channel, joint re-decode): 0.8396
  with CMLt collapse.**
- **The transition side of the bar-pointer model is CLOSED** (R4 run 2: +3,200 nats to
  convergence, deployed decode flat).
- **θ-as-latent — CLOSED**, right-shaped gradient but 6000× too weak. "Do not retry."
- **PLL read-out — NULL** on clean gtzan after controls.
- **Dispersion term — NULL/NEGATIVE** (10-nat demand loses to a 90-nat walk price).
- **Uniform-φ₀ — NEGATIVE**: κ_θ collapses, in-tolerance 54% → 4%, **while the ELBO ties or
  improves**. (A nice standalone datum: the ELBO is not monotone in the task metric.)
- **Adaptive learned proposal scales — dead 4/4 checkpoints, CMLt 0.000.**

### 5.4 Things that are NOT negative and must not be described as such

- **The q-family ladder is NOT TESTABLE, not negative** (§2.4). Cite it and a reviewer can
  destroy it.
- **`project_probe_training_disconnect`:** two probes passed while the training A/B was null.
  **Distrust probe→training inference** — do not promote any probe result to a training claim.

### 5.5 Standing methodological constraints from the record

- **`feedback_track_cmlt_amlt` (user directive, 2026-08-04):** every beat/downbeat result must
  report CMLt and AMLt alongside F; **AMLt must NEVER be optimised for.** Oracle calibration on
  record: a half-bar offset scores **F 0.000 / AMLt 0.903**; double or half rate scores
  **F 0.68 / AMLt 1.000**; a quarter-bar offset scores ~0.02 on everything. So *F low + AMLt
  high* = a genuine metrical mistake (right rate, wrong phase); *both low* = noise. Report the
  **estimate/reference count ratio** with them.
- **`HANDOVER_FROM_VBPM.md` caution 1:** on SMC the **decoder is only ~22% of the gap** (activation
  ≈78%). Scope the claim; do not promise ≥0.70 from decoding alone.
- **Caution 2:** per-track **oracle** λ only reaches 0.642 — "scale tuning is close to exhausted,
  so the win must come from the kernel's **shape**, not from picking a better λ."
- **Caution 4:** remedies can be **redundant rather than additive** — the mixture kernel's
  +0.0093 bare advantage **vanished under deployment decode** (peak-snap and the dither component
  repair the same symptom). **Always report bare AND deployed.**
- **Harness warning:** Analyze-SMC uses `scripts/_harness.py` with a 5 s trim; VBPM's `rungs_lab`
  uses its own. "They happen to land close, which is exactly how a mistake would slip through."
  Do not merge tables across the two without recomputing.
- **Convergence:** "conclusions drawn from short training runs reversed completely when run to
  convergence (40 epochs → 1000 epochs turned five confident 'this fails' claims into 'this
  works')." And single-dataset checkpoint selection has repeatedly disagreed with held-out results.
- **Seed noise floors:** pooled seed sd 0.082 on the phasevae metric (MDE 0.209 at n=3, 0.115 at
  n=8); one hainsworth song = 0.04 downbeat F, so single-seed hainsworth deltas <0.08 are not
  conclusions. Plus the archain floor in §5.1b (adjacent-checkpoint p90 |ΔF| = 0.344).
- **`feedback_track_cmlt_amlt` again, as a hard rule:** never make AMLt a training target. The
  emission `a + b·cos(φ)` is maximally penalised at a half-bar offset — exactly where AMLt scores
  0.903 — so training toward AMLt inverts the emission's ordering.
- **Confidence read-outs are invalid at stride > 1** (audit S9): the per-frame κ returned under
  stride is the broadcast stride κ, and within a stride the path is deterministic. *"Any
  'confidence' read off it at stride>1 is meaningless."* Do not build a calibration claim on
  strided runs.
- **The evidence resultant is self-consistency, not truth:** it stays 0.78–0.87 even when half the
  downbeats are wrong.
- **The ELBO does not rank placement** — three recorded ranking inversions. 1.3 nats spans
  in-tolerance 100% vs 54% (ballroom_0: ep400 = 100% at recon −40.40; ep599 = 54% at recon
  −39.08 — **the best recon of the run is the worst placement**). 12/12 runs × 2500 epochs touch
  ≥83% in-tolerance and then wander off to 9–92%.

---

## 6. Bottom line

### 6.1 The three strongest assets

1. **SMC, n=217, Beat This activations, the paper's own harness — the DBN trades 0.009 F for
   0.186 CMLt and 0.119 AMLt** (`FOLLOWUP_FINDINGS.md:179-186`). Twenty-to-one. Rank decoders by
   F and you throw away the largest continuity gain available. This is the F-vs-continuity
   dissociation measured on exactly the system and dataset the paper is about. *(Uses GT tempo —
   label it a headroom result.)* Supporting rows on the same n=217: per-song oracle λ beats
   peak-pick **+0.0225 F / +0.0663 CMLt / +0.0484 AMLt**, and on the 34 songs that want a stiff
   kernel it is **+0.085 F / +0.171 CMLt / +0.159 AMLt**.
2. **A periodicity prior was built and it monotonically hurts** (`octave_error_audit.md:53-69`):
   gtzan exact-period accuracy **79.3% → 77.5% → 73.0% → 69.6%** as the fitted log-normal tempo
   prior's weight goes 0 → 0.5 → 1.0 → 2.0, with halving rising **11.6% → 14.9%**. Reason:
   *"halving a 2 s bar gives 1 s, which a prior fit on real bar periods still finds ordinary, so
   the prior cannot punish the actual error."* Paired with the rigid-end control (madmom DBN
   **0.437** vs peak-pick **0.810** on Beethoven rubato), **both ends of the periodicity-penalty
   spectrum are pre-measured negatives in-house.** No other group can say that.
3. **Marginalisation, not the per-frame target, is what does the work.** Same trained potentials,
   two read-outs: **peak-picked → F 0.000; through exact forward–backward → F 0.602**
   (100 gtzan songs, 150k frames). The potentials are statistically **uniform** (per-frame entropy
   **4.564 nats vs log 96 = 4.564**, tilts ≤ ±10%, AUC 0.823). And the posterior is genuinely
   multimodal: **median 13% of tempo mass an octave class away, 57% of songs with >10% competitor
   mass, 64% of songs where the smoothed decision differs from the frame-0 argmax, 43% of phase
   mass >60° off the circular mean.** *"An ensemble of coherent paths, not a tube around one"* —
   which no unimodal or factorized posterior can represent, and which no per-frame penalty can
   create.

   **Runner-up, and the cleanest single discriminator if you need one sentence:** repairing the
   *likelihood* under a factorized q takes gtzan **CMLt 0.115 → 0.467 and AMLt 0.184 → 0.556**
   while **F stays at null (~0.13 vs null 0.07)**; changing the *posterior family* to a chain
   takes F to **0.750**. Same dataset (n=993), same prior, same emission, same encoder.
   *A smoothness/periodicity repair buys continuity metrics and not correctness; the sequence
   model buys both.*

### 6.2 Staleness verdict on the e2e negative — one paragraph

**Current, not stale, and worse than recorded — but off-target.** `vanilla_history.json` and
`r2_history.json` **are** the confound-repaired rematch (identified by four agreeing content
signatures, decisively the learned λ moving 14.23 → 101.04 under the obs=6 chassis fix; git
cannot help, because the whole `bt_e2e` tree entered the repo in one bulk commit `e7150aa` on
2026-08-09, and every mtime is a copy artifact). The repaired rematch made the e2e CRF arm lose
**harder**: vanilla **0.9658** vs r2 **0.9115**, a gap of **0.0543** against **0.0142** before.
No later end-to-end result exists anywhere — every surviving checkpoint under `rungs_lab/runs/` is
880 B–34 KB of decoder parameters, and `RESULTS.md` still ends Part 1 at *"Rematch launched."*
**But three things blunt it:** the numbers are 60-song-subset validation F that RESULTS.md itself
says runs ~+0.01 high and must not be quoted against its own table; the checkpoints and logs are
destroyed so it can never be re-scored without retraining; and it was run on **1,020 train / 146
val songs that are 69% ballroom + hjdb**, corpora where a constant-period grid already ceilings at
**0.918–0.999** and where the thesis predicts no gain. **It is a negative for CRF-NLL as a
frontend training objective — which the record itself calls "NOT a rung … off-program" — not a
negative for a probabilistic sequence model, and the generative rung on the same chassis (R2 exact
EM) wins unsupervised on the same fold (0.9132 vs 0.9033, +0.010).** **[RETRACTED 2026-08-20: NOT a head-to-head. `results_r2_val.json` uses a different frontend (`bt_fold0_repacked.pt`) with peak-snap OFF (`correct=False`). At the deployment config it reverses: R2-generative(40.3) 0.9488 < R0/R1/CRF 0.9506, and em-single λ=40.3 loses to hand-set λ=100 on all six metrics. Do not write 'the generative rung wins'.]** Follow-up #5 (multi-seed,
all 8 folds) is still unmet, and "run it on hard material" was never written down at all.

### 6.3 What we must NOT claim — the short list

1. **Not** that the tempo-increment law is heavy-tailed and justifies a loose kernel. The
   "generative λ≈40 matches heavy-tailed tempo statistics" reading is **WITHDRAWN** — a synthetic
   metronome at interval 21.4 reproduces λ=39.7 against the real corpus's 40.27, to 1.5%, with no
   musical timing variation at all. The kurtosis-13 statistic is itself computed from
   frame-quantised annotations carrying the same artifact. **Re-derive on sub-frame-interpolated
   annotations before reusing it.**
2. **Not** "zero variational gap" / "exact maximum-likelihood training of a neural DBN" for
   `tchain`. REFUTED for the trained object (the encoder reads x, the optimal potentials are
   functions of y). Say "a structured chain posterior with exact forward recursion."
3. **Not** that the chain-vs-factorized 0.97-vs-0.000 A/B demonstrates a *representational* floor.
   It is a **kinetic trap** — an optimisation result. The numbers stand, the interpretation does not.
4. **Not** gtzan **0.733** (rate-grid) or **0.752** (anchor_k v2) as results — both explicitly
   **RETIRED by the user** on 2026-08-15. Keep the mechanisms, drop the numbers.
5. **Not** the archain ladder's "monotone convergence to truth," the capstone **F 1.000**, the
   "objective defects to 2×" nat pricings, the 4 ms annotation-jitter constant, "κ is 100–1000×
   too loose," "tightening κ fixes placement," "the rubato sweep tests rubato," or "slow songs
   fail" — all formally retracted in `audit_2026-08-19.md`.
6. **Not** the q-family ladder as evidence against enriched posteriors. It is **NOT TESTABLE**
   (zero gradient to the corrector, repair penalised, corruption inside the 70 ms tolerance,
   8% power). `project_paper_framing_discipline`: *"earns NOTHING and is a liability if cited."*
7. **Not** the categorical q(l_t) as a validated fix (trained, and the pre-registered falsifier
   fired), **not** MERT's advantage (gone at n=920; the 0.9658 was a lucky draw, honest 5-seed
   mean 0.9548), **not** the gtzan 0.639 rigid-grid ceiling (it is 0.902 — the 0.639 was 60 blues
   tracks and a median estimator), **not** the ~0.40 deployment cap or the joint-eval verdict
   (both rendered through the pre-2026-07-10 broken pipeline, Spearman(old, fixed) = +0.12, p=0.6).
8. **Not** that the likelihood is task-aligned. The **ROBUST NEGATIVE** stands: per-song marginal
   likelihood prefers a looser λ than F does, **ρ(log λ_ML, log λ_OR) = −0.230, p=6e-4**, and
   λ_ML floor-seeks to 1. And an F-driven per-song λ gains **+0.019 F** while losing **−0.110
   AMLt** on SMC (n=217) — which cuts *for* our "F is the wrong metric" argument but *against*
   any claim that maximising the model's own likelihood is automatically right.
9. **Not** that structured inference beats peak-picking on SMC. It never has, deployably:
   **0.589 < 0.620**. The honest targets on record are beating 0.620 deployably, or realising the
   oracle headroom (0.654 σ-oracle / 0.703 TTA) without oracle selection. Best number in any
   configuration so far is **0.681**, and it uses ground-truth tempo.
10. **Not** that the decoder is where the SMC gap lives. Analyze-SMC already established
    **activation ≈78% / decoder ≈22%**. Scope the claim; do not promise ≥0.70 from decoding alone.
11. **Not** anything evidenced on gtzan, ballroom or hjdb about adaptive structure — a rigid grid
    scores **0.902 / 0.918 / 0.999** there. And note the protocol trap: **45 s excerpts
    structurally hide the phenomenon** (pooled constant-rate ceiling 0.781 at 45 s vs 0.716
    full-song; beatles 0.713 → 0.349, rwc 0.565 → 0.258, asap 0.324 → 0.164). **Evaluate
    full-song.**
12. **Not** any single-song / single-seed archain number. The adjacent-checkpoint |ΔF| p90 is
    **0.344**, and `archain_lab.py:53` hard-codes seed 0 — nothing on that branch is multi-seed.
