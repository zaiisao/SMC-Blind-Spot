# Specification: a tempo observation model for the bar-pointer DBN

Status of every claim below is marked **[PROVEN]** (derived, and where possible verified
numerically), **[MEASURED]** (empirical, with n and p), or **[CONJECTURE]** (stated so it can be
attacked). Two further tags are used where independent validation forced them: **[UNVERIFIED]** (a
number quoted here that could not be reproduced or whose source script could not be located) and
**[RETRACTED]** (a claim withdrawn, with its reason recorded in §8 — nothing is deleted silently).
Nothing is asserted that is not in one of those states. Where theory and measurement disagree, that
disagreement is recorded rather than resolved by preference.

Every empirical claim carries `n` and `p`. **No metric mean appears without naming the decoder that
produced it** (peak-pick vs DBN, joint vs beat-only state space) — these have reversed signs before.

*Last audited against independent re-runs: 2026-08-20. Sections 2, 3, 5 and the gates in §6 were
materially weakened; see §8.*

**ADVERSARIAL AUDIT 2026-08-22.** Six negatives were attacked for bugs and three positives
independently re-derived from source. Outcome, stated bluntly:

- **ONE NEGATIVE FLIPPED.** The octave-aware tempo loss did **not** fail; v1's penalty acted on one
  grid bin while the metric scores a ±3.12-bin band, so it was a correctly measured null of a loss
  that did nothing. Fixed, it **passes the unmodified pre-registered bar on both splits, 8/8 seeds,
  p=0.008** (new **§4.15**). This is the route §4.7 calls "the only route upward", and it was closed
  off for a day by an artefact.
- **Five negatives survived**, but two only after real bugs were found and fixed (the `MARG_FIX`
  prior-scale bug in §4.11; four defects in the contrastive scorer in §5.2). One (§4.6/§4.7
  injection) is now **proven bug-free** to 16 significant digits.
- **Three published positives are PARTIAL, not confirmed.** §4.8's "ranked first" is **false** (it
  ranks second), its n=187 cross-system row is **unreproducible**, and its headline statistic is
  **circular** (de-circularised: OR 3.49, p=0.043, not OR 8.87, p=1e−4). §4.9's break-even of 24.4%
  is a **β=5 artefact** — at the operating β=0.2 it is 76.2% — so gate **G1 is retired**. §4.9's
  "consistency check" cites its own script as independent corroboration; the true independent figure
  is +0.1684, not +0.1733. §4.6's band row is frame accuracy mislabelled, its residual counts are
  wrong, and "bias eliminated" holds only for the median.
- **§4.10's "40% overlap" is withdrawn as unsupported**: the interaction CI spans additive to fully
  redundant and the experiment is ~7× underpowered (n≈1602 needed).

Every correction below is tagged in place with its date; nothing was overwritten silently.

---

## 1. The model as it stands

State at frame `t`: `z_t = (φ_t, φ̇_t)`, with

- `φ_t` — position within the beat (or bar), discretised
- `φ̇_t = i` — tempo, stored as an integer **beat period in frames**

The state space is Krebs's (2015) bar-pointer construction: a tempo of `i` frames owns exactly `i`
states, so the pointer advances **+1 state per frame** at every tempo. This is what makes the
construction tempo-invariant by design.

The joint factorises as

```
p(z_{1:T}, y_{1:T})  =  p(z_1) · ∏_t p(z_t | z_{t-1}) · ∏_t p(y_t | z_t)
```

with the two conditionals as shipped:

**Transition.** Deterministic advance within a beat; at beat boundaries only,

```
p(φ̇_t = j | φ̇_{t-1} = i)  ∝  exp(−λ · |j/i − 1|),   row-normalised,
                              zeroed where the value ≤ np.spacing(1)
```

**Emission** (Böck 2016), three mutually exclusive classes assigned *geometrically*:

```
class(φ) = BEAT      if (φ mod 1) < 1/obs_λ
         = DOWNBEAT  if φ < 1/obs_λ
         = NO_BEAT   otherwise

p(y_t | z_t) = p(y_t | class(φ_t))
```

**Known misuse at `beats_per_bar=1` (fixed) [PROVEN].** `bar_pointer/state_space.py:94–95` assigns
`BEAT` then overwrites with `DOWNBEAT` where `state_positions < 1/obs_λ`. This is *correct madmom
behaviour for `beats_per_bar>1`* (bar position 0 is a downbeat), but it is **degenerate when the
joint class assignment is reused at `beats_per_bar=1`**: all positions lie in [0,1), so every `BEAT`
is overwritten. Verified: `np.bincount(position_classes) = [2710, 0, 211]` — zero BEAT states.
The beat-only decoder used throughout this spec (`scripts/exp_oracle_tempo_beatonly.py:
make_beatonly_dbn`) relabels DOWNBEAT→BEAT (counts `[2710, 211, 0]`) and feeds a zero downbeat
channel, which makes the decorrelation `max(beat−0, floor)` a no-op and yields the correct 2-class
Böck beat-only emission. The unreachable DOWNBEAT column is `log(0) = −inf` but is only ever gathered
via `lcd[:, s2c]`, so no NaN propagates. **Nothing else in the repo is affected.**

*Null reproduction*: the corrected beat-only decoder at β=0 reproduces madmom's
`DBNBeatTrackingProcessor(min_bpm=30)` **to within +0.0035 F** (0.6141 vs 0.6106, Wilcoxon p=0.024,
178/217 tracks bit-identical); CMLt +0.0013 (p=0.58) and AMLt +0.0003 (p=0.97) are not significant.
It is *not* bit-identical, and should not be described as such.

*Operational note*: `scripts/exp_oracle_tempo_beatonly.py` hardcodes `DEV='cuda:3'`; re-runs must
edit that line.

---

## 2. The defect, stated formally

### Proposition 1 (the emission is tempo-blind) **[PROVEN]**

`class(φ)` is a function of `state_positions` alone. `state_interval_frames` — the tempo — appears
nowhere in it. Therefore

```
p(y_t | z_t) = p(y_t | φ_t)     for all t,   independent of φ̇_t.
```

*Verified in code*: the class map is built from `position_classes`, indexed only by position.

### Proposition 2 (class occupancy is tempo-invariant *in the continuum limit only*) **[PROVEN, weakened]**

At tempo `i` the beat region *ideally* spans `i/obs_λ` frames per beat, and a `T`-frame window
contains `T/i` beats, giving `#BEAT = T/obs_λ` independent of `i`. **In the shipped code this holds
only in the continuum limit.** The region is the integer set `{k : k/i < 1/obs_λ}`, i.e.

```
#BEAT frames = (T/i) · ceil(i/obs_λ)      — NOT independent of i
```

*Verified numerically*: the `ceil` form matches the code exactly for all 149 tempi at fps=86.13 and
all 87 tempi at fps=50 (`num_tempi` None and 60). At `obs_λ=16`, fps=50, the beat-class frame
fraction ranges **0.0625–0.1176** (a 1.88× spread), not the constant 1/16. Running the R2 test on a
constant activation (act=0.5, T=3000 frames = 60 s) gives a cross-tempo accumulated-evidence spread
of **448 nats/60 s**, not 0, favouring `i=17` (176 BPM) over `i=16` (188 BPM). By `obs_λ`:
4 → 145.4, 8 → 300.5, 16 → 448.0, 32 → 413.9 nats/60 s. The sign of the bias flips when the beat
activation falls below `1/obs_λ`.

The fractional rule is therefore ~3.7× *better* than the fixed-W D=15 alternative (1671 nats), not
infinitely better. **[UNVERIFIED]** the 1671 and 5134 nat figures for the fixed-W variants are cited
from an unlocated script and were not reproduced.

### Corollary 1 (tempo has no *explicit* likelihood term) **[PROVEN for the explicit emission factor only]**

Combining 1 and 2: there is no factor written in the code of the form `p(observation | φ̇)`. Tempo is
identified indirectly, never explicitly observed.

**The stronger reading — "tempo evidence spread is zero" — is false.** Validation found three
implicit tempo-dependent channels, all previously unrecorded:

1. **`ceil` quantisation of the beat region** (Prop 2): a genuine tempo-dependent likelihood factor
   worth **448 nats/60 s** on constant input at `obs_λ=16`.
2. **Initial distribution** (`r1_2016_dbn.py:101–105`): uniform over *states*, and tempo `i` owns `i`
   states, so the implied tempo marginal is `p(φ̇ = i) ∝ i` — a `log(i_max/i_min) = 1.97` nat prior
   favouring slow tempi. Inert for Viterbi (a shared constant that cancels in path comparison), but
   **live for `forward_log_likelihood`, which is the R2+ training objective.**
3. **Row normalisation of the transition** (`state_space.tempo_transition_probabilities`): makes
   `log p(stay at i | i)` tempo-dependent even though `exp(−λ|1−1|)=1` — range −0.7341..−0.0008 at
   λ=100 and −2.6358..−0.5823 at λ=11.45. Paid `T/i` times, so accumulated over 60 s the spread is
   **22.8 nats at λ=100** (favours 214 BPM) and **125.1 nats at λ=11.45** (favours 30 BPM) — the
   preferred tempo *reverses sign with λ*.

So "tempo enters **only** through the transition prior" is false as written.

### Proposition 3 (tempo evidence is weak, not absent) **[MEASURED, weakened]**

Corollary 1 does not make octave hypotheses *equally* likely — halving the tempo places beat regions
on alternate activation peaks, which costs likelihood. On **`smc_001` specifically**, beat-only
state space (beats_per_bar=1, class-corrected), min_bpm=30, obs_λ=16, **λ=100**, Beat This
activations at FPS=50, `path_score` (not peak-pick, not joint):

```
truth   −5832.2      double tempo  −5925.6  (Δ = −93.4)
drift   −5807.7      phase shift   −6061.6
```

These four numbers reproduce exactly. **But the generalisation does not hold.** Across tracks
(λ=100, finite-score subset): truth beats double on 85.7% (n=119, p=5.3e−16), beats phase shift on
84.2% (n=133, p=2.1e−18), and beats drift on **83.5%** of draws (n=266 pooled, mean +274.3 nats,
median +173.6, p=1.4e−28). At λ=11.45 (no −inf, full n=216): drift0 truth-wins 0.866 (p=4.2e−27),
drift1 0.815 (p=3.3e−23).

So: **a drifted path outscores the truth on only 12–17% of drift draws (27.7% of tracks for at least
one of two draws); `smc_001` is in that minority and is not representative.** The earlier blanket
statement "drift outscores the truth" is retracted (see §8).

Two further findings, both previously unreported:

- At λ=100, **half**-tempo is statistically indistinguishable from truth (n=93, mean +10.0, median
  −23.0, truth-wins 0.430, Wilcoxon p=0.712, binomial p=0.213). The octave asymmetry holds for
  *doubling* only. At λ=11.45 half is separated (mean +143.4, truth-wins 0.740, p=5.2e−12).
- At λ=100 the **ground-truth path itself scores −inf on 80/216 SMC tracks (37%)**: real tempo change
  exceeds the `np.spacing(1)`-truncated transition support. All λ=100 statistics above are
  survivorship-conditioned on the surviving steady-tempo tracks.

**The specification's central claim, restated to what survives**: the model can express tempo but has
no *explicit* likelihood term for it, so the evidence separating metrical levels is thin (≈93 nats
per track against doubling on `smc_001`; **zero separation against halving at λ=100**) — but it is
not so thin that drift routinely wins.

*Reproduction*: these numbers exist nowhere in the repo outside this document; the validation
harness was reconstructed as `/tmp/prop3_verify.py` (outputs `/tmp/prop3_lam100.json`,
`/tmp/prop3_lam11.45.json`). **TODO: commit it.**

---

## 3. Predictions the defect makes, and their status

All decoder details are stated per row, per the standing honesty rule (peak-pick vs DBN, joint vs
beat-only state space — these have reversed signs before).

| # | Prediction from Corollary 1 | Status |
|---|---|---|
| P1 | Errors should concentrate at metrical levels (octave), not at random tempi | **[MEASURED, weakened]** the failure mode *is* a per-track binary octave switch, not continuous drift — but it is graded, not a cliff. Peak-pick th=0.5, no DBN, n=40 fold-honest SMC tracks, librosa phase vocoder; median δ(s) with octave-failure count (\|δ\|>0.5): s=0.50 +0.757 (23/40) \| 0.70 +0.029 (13/40, p=1.3e−4) \| 0.85 +0.012 (6/40, **n.s. p=0.13**) \| 1.00 0.000 (0/40, *true by construction — self-reference, not a data point*) \| 1.20 −0.018 (9/40, p=0.022) \| 1.50 −0.411 (21/40; **threshold-unstable**, −0.058 to −0.411 over th 0.3–0.7 — quote the 21/40 fraction, not −0.411) \| 2.00 −0.966 (28/40, p=1.7e−7). |
| P2 | Supplying `p(obs \| φ̇)` should produce a large gain, mostly in *continuity* not F | **[MEASURED]** oracle: ΔCMLt **+0.2221** (p=5.5e−21), ΔAMLt +0.1181 (p=3.6e−15), ΔF only +0.0615 (p=1.0e−11), n=217, beat-only PyTorch DBN2016 (class-corrected), β=5, min_bpm=30, obs_λ=16. |
| P3 | Tuning λ — a prior on a variable with no likelihood — should be near-worthless | **[MEASURED]** three methods converge (MLE 11.45, contrastive 9.02, hard-mined 7.55). n=87 held-out SMC, beat-only PyTorch DBN Viterbi, paired vs shipped λ=100: mined 7.55 → F +0.0167 (p=0.020), AMLt −0.0440 (p=0.034), CMLt −0.0018 (p=0.54); synth 9.02 → F +0.0142 (p=0.026), AMLt −0.0407 (p=0.058); MLE 11.45 → F +0.0146 (p=0.011), AMLt **−0.0350 (p=0.154, n.s.)**. The "≈+0.017 F / −0.044 AMLt" figure was the *mined* arm only. |
| P4 | The gain should survive a *noisy* tempo observation, since it only needs to break ties | **[RETRACTED as an accuracy-threshold claim — see §8.]** What survives: a **single constant tempo per track** (median IBI, zero time resolution) still buys ΔF **+0.0343 (p=3.3e−04)** and ΔCMLt **+0.1684 (p=5.4e−12)**, i.e. 76% of the CMLt gain, n=217, same beat-only decoder, at β=5. *(Corrected 2026-08-22: the previously quoted +0.0390/+0.1733 came from `exp_octave_degradation.py`'s 0% arm, which uses an unweighted `median(diff(gt))`; the independent duration-weighted computation gives the figures above. At β=0.2 the same constant oracle buys only +0.0599 — see §4.9.)* |
| P5 | Removing unevaluated bar states should help on a beat-only corpus | **[MEASURED, mechanism corrected — see below]** beat-only vs joint madmom DBN, n=217, Beat This activations, min_bpm=30, λ=100, 50 fps: F +0.0255 (p=0.001), CMLt +0.0570 (p=0.0014), AMLt +0.0250 (p=0.0075), CMLc +0.0437 (p=0.015), AMLc +0.0197 (**p=0.11, n.s.**). |

**P5, mechanism.** The correct statement is *not* "the joint beat+downbeat state space hurts on SMC".
It is: **injecting Beat This's downbeat activations into the DBN observation model hurts**. Control:
keeping the bar state space (`beats_per_bar=[3,4]`) intact but making the downbeat channel
uninformative (`dp := bp/4`) matches beat-only within noise — F +0.0006 (p=0.59), CMLt +0.0006
(p=0.60), AMLt +0.0012 (p=0.47), n=217. **The bar-level state space itself costs essentially
nothing; 100% of the loss comes from the real downbeat probabilities.** This therefore does *not*
reproduce Böck & Davies' "joint modelling" explanation. Diagnostic: `dp > bp` on 19.0% of all frames
and **28.0% of beat-peak frames** (`bp>0.5`), so the `max(bp−dp, ε)` beat likelihood is truncated at
many true beats — a calibration artefact of Beat This's independent sigmoid heads.

**That artefact is NOT the mechanism [MEASURED 2026-08-21, `scripts/exp_conditional_downbeat.py`].**
A conditional parameterisation `p(beat)·p(downbeat|beat)`, which clips the ratio to 1 instead of
flooring the beat class to ε (and so preserves total beat mass where `dp > bp`), is
**indistinguishable** from the shipped additive form: CMLt −0.0007 (p=0.99), F −0.0039 (p=0.34),
n=217. The ε-flooring is real but inert.

**The loss is the downbeat CONTENT, in either parameterisation.** With the bar state space intact but
the channel made uninformative (`dp := bp/4`), the joint decoder recovers to F 0.6100 / CMLt 0.5665 —
matching beat-only (0.6141 / 0.5685). Beat This's real downbeat predictions are actively harmful on
SMC however they enter the emission.

**The harm is SOURCE-SPECIFIC, not a property of downbeat evidence [MEASURED 2026-08-21].** Swapping
in **Beat Transformer's** downbeat channel (different architecture, independently calibrated;
resampled from BT_FPS to 50 fps) makes the joint decoder indistinguishable from beat-only:

| arm | F | CMLt | AMLt | vs beat-only |
|---|---|---|---|---|
| beatonly | 0.6141 | 0.5685 | 0.6988 | — |
| joint + Beat This downbeats | 0.5886 | 0.5149 | 0.6741 | CMLt **−0.0536** (p=0.0021) |
| joint + Beat Transformer downbeats | 0.6110 | 0.5710 | 0.7079 | CMLt +0.0025 (**p=0.64, n.s.**) |

So "downbeat observations hurt on SMC" is **false as a general claim**. Beat This's independently
trained sigmoid heads are miscalibrated on this material; another model's downbeats are harmless
(AMLt even trends up, +0.0091, p=0.15).

*Consequence for the octave problem*: bar-level states are the only part of the model representing
metrical hierarchy, and hierarchy is what could disambiguate metrical level. That route is **OPEN**:
the bar state space is now available at essentially zero cost with a properly calibrated downbeat
source. It does not help yet (+0.0025, n.s.), but "free" is a different starting point from "−0.054".
Note SMC carries **no downbeat annotations** (verified: all 217 files single-column), so downbeat
*accuracy* cannot be diagnosed here — only its downstream effect on beat CMLt, which is what the
table above measures.

*Correction history*: this section previously read "Route closed", generalising from the Beat This
result to downbeat evidence in general. That was an interpretation outrunning its measurement.

**METER-SET SWEEP [MEASURED 2026-08-22, `scripts/exp_meter_sets.py`].** 12 joint arms (meter sets
`[3,4]` / `[2,3,4]` / `[2,3,4,5,6,7]` / `[2,3,4,6,8,9,12]` × downbeat sources Beat This / Beat
Transformer / uninformative `bp/4`), n=217, all vs the beat-only decoder:

1. **No meter set beats beat-only.** Best arm `[2,3,4]`/btrans, dCMLt +0.0047 (p=0.58). The
   "madmom's `[3,4]` is too narrow" hypothesis is refuted: the `uninf` control sits at −0.002 to
   −0.004 (p≈1) at EVERY width, so the bar state space is inert regardless of how many meters it
   represents. This closes joint decoding as an improvement path with a 12-arm sweep rather than a
   single comparison.
2. **The Beat This downbeat harm is mostly a METER-RESTRICTION interaction, refining this section's
   mechanism a second time.** dCMLt −0.0583 (p=0.00093) under `[3,4]` shrinks to **−0.0099 (p=0.27)**
   the moment duple meter is allowed. Mechanism: the downbeat head over-fires (`dp > bp` on 28.0% of
   beat-peak frames ≈ "every other beat looks like a downbeat"); a `[3,4]`-only decoder has no
   hypothesis fitting that pattern, while meter 2 absorbs it. So the correct statement is now:
   *Beat This's downbeat channel hurts a `[3,4]` decoder; with duple meter available the harm is
   ~6× smaller and non-significant — though still never a gain.* (Read across arms; a formal
   difference-of-differences test was not run.)
3. **Compound meters actively hurt with calibrated downbeats**: `[2,3,4,6,8,9,12]`/btrans is the one
   significantly negative non-bthis arm, dCMLt −0.0058 (p=0.034) — likelihood spent on meter
   hypotheses the corpus does not support.

Disclosures on P5: (a) the two arms use madmom's *differing* defaults (`DBNBeatTrackingProcessor`
`num_tempi=None, threshold=0` vs `DBNDownBeatTrackingProcessor` `num_tempi=60, threshold=0.05`);
equalising them changes the gap by ≤0.004 and if anything **favours joint** (matched: F +0.0268,
p=1.6e−4; CMLt +0.0581, p=9.7e−4). (b) "beats on all five metrics" is a **mean-level** claim; only
3/5 metrics reach p<0.01 at min_bpm=30. (c) Only one activation source (Beat This) was tested; it may
not transfer to Beat Transformer, whose heads are calibrated differently.

**Reference means** (n=217, Beat This cached activations, mir_eval 5 s trim, ±70 ms), F/CMLt/AMLt:
peak-pick (max-pool width 7, th=0.5, no state space) .6270/.5138/.6100; joint_min30
.5851/.5102/.6735; beatonly_min30 .6106/.5672/.6985; joint_min55 .5757/.4738/.6540; beatonly_min55
.5943/.5028/.6675. beatonly_min30 vs peak-pick: F **−0.0163** (p=0.0089), CMLt +0.0534 (p=6.9e−4),
AMLt +0.0885 (p=2.4e−11) — the DBN *costs* F and buys continuity.

P1, P2, P3, P5 are mutually consistent and were measured independently; P4 no longer supports what it
was written to support. P3 is the one that reframes the project: **λ is not the lever, and no amount
of estimator sophistication will make it one.**

---

## 4. The proposed extension

Add the missing factor. The observation model becomes

```
log p(y_t | z_t)  =  log p(y_t | class(φ_t))  +  β · [ log q_θ(φ̇_t | h_t) − log p(φ̇_t) ]
```

- `h_t` — the frame's 512-d Beat This embedding
- `q_θ(· | h_t)` — a learned distribution over the DBN's 60-point tempo grid
- subtracting `log p(φ̇)` makes the term a **scaled likelihood** (ASR-hybrid), not a posterior

### Requirement R1 (no double counting) **[PROVEN, and previously violated]**

The transition model already encodes a tempo prior. Adding a *posterior* would count it twice. The
likelihood-ratio form is therefore mandatory. Related failure already made and fixed: a head trained
with weights `∝ prior^(−1/2)` is *already* prior-corrected, and dividing by the label prior again
injected **5.47 nats per tempo (~11,000 nats/track)**. The correct denominator is the head's own
marginal.

### Requirement R2 (tempo-neutrality must be preserved) **[PROVEN necessary, threshold corrected]**

Any added term must not covertly reintroduce a tempo bias, or it will look good frame-wise and decode
badly — the failure mode of two earlier attempts. **The test cannot be "spread = 0 nats", because the
shipped emission itself does not pass it** (Proposition 2: 448 nats/60 s at `obs_λ=16` from `ceil`
quantisation). Corrected test: **the accumulated-evidence spread across tempi for a constant input
must not exceed the 448-nat/60 s baseline of the shipped emission**, and its sign must not align with
the existing biases catalogued in Corollary 1.

Note also that R2+ objectives built on `forward_log_likelihood` inherit the uniform-over-states
initial distribution, i.e. a live `p(φ̇ = i) ∝ i` prior worth 1.97 nats (Corollary 1, channel 2).
This must be divided out or the tempo grid re-weighted.

### Requirement R3 (β must not be a swept hyperparameter) **[DESIGN]**

Sweeping β would repeat the λ mistake. β is learned by the conditional/contrastive objective in §5.

### Requirement R4 (soft targets) **[DESIGN, from Böck & Davies 2020]**

The DBN consumes a distribution; training `q_θ` against hard classes and reading it softly is a
mismatch. Targets are a smoothed interpolated histogram over inter-beat intervals with widening
(±2 BPM at weights 0.5/0.25). **Formerly justified by P4; that justification is retracted (§8)** —
the surviving support is weaker: a *constant* per-track tempo still buys 78% of the CMLt gain
(ΔCMLt +0.1684, p=5.4e−12, n=217, β=5 — corrected 2026-08-22 from +0.1733, §4.9), so temporal precision is not where the value is. Whether
*near-misses in tempo value* carry the value is **[CONJECTURE]**, untested.

---

### Requirement R5 (the tempo term must be counted ONCE PER BEAT) **[SATISFIED — the shipped `/i` form is exactly the per-beat form]**

The HMM factorisation `∏_t p(y_t | z_t)` assumes observations are conditionally independent given the
states. `h_t` comes from a transformer with self-attention over the whole 1500-frame chunk, so every
embedding already aggregates the entire window — `h_t` is local in *index*, not in *information*.
Adding `β·log q_θ(φ̇|h_t)` at every frame with unit weight would therefore count the same evidence
`T` times (T ≈ 2000 on SMC). The requirement is that it be counted **once per beat**.

**This is already satisfied, exactly.** A tempo of `i` frames owns exactly `i` states, so each beat
contains exactly `i` frames, and the shipped `/i` division gives

```
Σ_{t ∈ beat} β·ratio(t,i)/i  =  β·(1/i)·Σ_{t ∈ beat} ratio(t,i)  =  β·mean_beat(ratio)
```

i.e. the per-frame form is **identically** one beat's worth of evidence per beat, using the
beat-averaged `q`. Not an approximation — an algebraic identity, given the bar-pointer property that
a tempo of `i` frames owns `i` states.

**Boundary-only application would be slightly WORSE**, not better: it point-samples `q` at one frame,
where the current form averages over the beat. The head's per-frame argmax changes on ~6.8% of
frames, so averaging is the lower-variance estimator.

*Correction history*: this requirement was first recorded as "PROVEN necessary, currently VIOLATED",
and Q10 was raised on the theory that T-fold over-counting might be inflating the +0.0494/−0.3621
split of §4.6. Both were wrong — the algebra above was not done before writing them. No measured
result changes; the §4.6 and §4.7 numbers were never affected by this.

### Requirement R6 (path scores are NOT comparable across tempo hypotheses) **[PROVEN, measured]**

Any procedure that compares `max_z log p(y, z)` between decodes at different tempi is invalid in the
shipped model, because Corollary 1's implicit channels are uncorrected and tempo-dependent:

| channel | magnitude | direction |
|---|---|---|
| `ceil` quantisation of the beat region (Prop 2) | **448 nats/60 s** at obs_λ=16 | favours fast (i=17 ≈ 176 BPM) |
| transition row-normalisation | 22.8 nats at λ=100 (125.1 at λ=11.45) | favours fast at λ=100, **slow** at λ=11.45 |
| initial distribution `p(φ̇=i) ∝ i` | 1.97 nats | favours slow |

The two dominant terms both favour fast tempi and are ~450 nats — the same order as the score gaps
that decide real hypotheses. **Measured consequence** (§4.6 / §8 R-11): choosing the octave by total
Viterbi path score selects the doubled-tempo hypothesis on **150/217 tracks** and destroys tracks the
head had already got right (`smc_001` CMLt 1.000 → 0.000). Cross-tempo model comparison requires
subtracting these terms first. **This is a prerequisite for any octave-arbitration scheme.**

---

---

## 4.6 Measured behaviour of the learned `q_θ` **[MEASURED, 2026-08-21]**

`scripts/train_tempo_observation.py` — dilated-conv head over frozen 512-d Beat This embeddings,
60-point tempo grid, soft targets per R4, trained on 5138 corpus tracks (2244 unstretched + 2894
phase-vocoder stretched at s ∈ {1.0,1.4,1.7,2.0,2.4}), **14.55 M training frames** (16.73 M is the
whole cache, corrected 2026-08-22). SMC never trained on — **[VERIFIED 2026-08-22 by audit]**: the
cache `meta.json` holds 2535 unique ids across {gtzan, ballroom, asap, hainsworth, beatles}, zero
containing "smc"; the md5-split keeps all stretched copies of a track on one side; and the SMC
embeddings are fold-honest (each track encoded by the Beat This fold that held it out).

**Tempo estimation on SMC (n=217).** This is the covariate-shift result and it is large:

| | previous head | this head |
|---|---|---|
| frame accuracy (±10%) | 0.122 **[UNVERIFIED — no surviving artefact]** | **0.4609** (macro; micro 0.4613) |
| median log₂ bias | −0.633 (below 55 BPM) | **0.000** |
| track accuracy | — | 0.5161 |
| octave-tolerant track accuracy | — | 0.7557 |

By band — **these are per-track FRAME accuracies, not track accuracies** (mislabelled until
2026-08-22): <55 BPM **0.5609** (n=49, bias +0.023) | 55–70 0.3866 (n=58) | 70–90 0.4589 (n=52) |
90–120 0.5192 (n=44) | >120 0.2424 (n=14, bias **−0.977**). Band *track* accuracies differ by up to
7 points: 0.6327 / 0.4483 / 0.5000 / 0.5682 / 0.2857. **The sub-55 BPM band — the entire subject of
the blind spot — is now the best-performing band.** The cost fell on >120 BPM, which SMC barely
contains (n=14) and which was not checked on a fast corpus.

**"The systematic slow-tempo bias is eliminated" — WEAKENED 2026-08-22.** The *median* log₂ bias is
0.000 and the old −0.633 sub-55 bias is genuinely gone, but the residual error mass still leans
slow: mean log₂ bias −0.113 (Wilcoxon vs 0 p=0.081, n.s.), half-tempo errors outnumber double
**38:14** (binomial p=0.0012), and |bias|>0.5 tails are 55 slow vs 32 fast (p=0.018). Correct
statement: *median* bias zero, slow band solved, residual tails still slow-leaning.

Residual error structure (n=217, recomputed 2026-08-22 at ±10%): **112 correct** (= 0.5161×217, the
previously printed 113 contradicted the spec's own track accuracy), **38 at exactly ½×**, 14 at 2×,
7 at 3×, 4 at ⅓×, 10 at 3/2×, 6 at 2/3×, 26 other. The failure is now *predicting half*, mirroring
the old failure of predicting double.

*Aggregation robustness [MEASURED 2026-08-22]:* track accuracy under six aggregations —
median-of-argmax 0.5161, mean-posterior argmax 0.5069, log-mean-posterior 0.5115, modal bin 0.4977,
arithmetic/geometric mean of argmax intervals 0.4608. Median-of-argmax is best but not anomalously
so; the earlier "24.9% vs 32.3% octave error on the same head" discrepancy is an
aggregation-and-object mismatch (see §4.11), not a coding bug.

**Beat tracking on SMC (the decisive test).** Injected per-frame (R5-violating) at β into the
class-corrected beat-only DBN2016, min_bpm=30, obs_λ=16, λ=100, n=217, vs the tempo-blind β=0 null:

| β | F | CMLt | ΔCMLt vs null | p |
|---|---|---|---|---|
| 0.0 | 0.6141 | 0.5685 | — | — |
| 1.0 | 0.5961 | 0.5202 | −0.0483 | 0.0064 |
| 5.0 | 0.5795 | 0.4715 | **−0.0971** | 4.4e−05 |

**Pooled, the learned head HURTS, monotonically in β.** For scale: a *constant per-track median IBI*
buys ΔCMLt **+0.1684** (β=5; only +0.0599 at β=0.2, §4.9), and the oracle buys +0.2221. The head is far below the trivial baseline.

**But the pooled number is two opposite effects cancelling** (β=5, split by head correctness):

| group | n | CMLt β=0 | CMLt β=5 | Δ | p |
|---|---|---|---|---|---|
| head tempo **correct** | 113 | 0.7594 | **0.8088** | **+0.0494** | 0.024 |
| head predicts **half** | 40 | 0.4469 | 0.0848 | −0.3621 | 2.1e−05 |
| head predicts **double** | 14 | 0.3827 | **0.0000** | −0.3827 | 0.018 |
| other wrong | 50 | 0.2864 | 0.1504 | −0.1360 | 0.0011 |

**This is the central empirical finding of the extension.** On the 52% of tracks where the head has
the right tempo, supplying the missing likelihood raises CMLt to 0.8088 — *above* the oracle's pooled
0.7906 — with no oracle information. The mechanism specified in §4 works. What destroys it is a
**confidently wrong octave**, which drags the decoder to the wrong metrical level where CMLt scores
0 by definition.

**Consequence for gating.** Accuracy is the wrong control variable; **octave correctness** is. A
symmetric-jitter error model cannot express this, which is exactly why the retracted σ-noise study
(§8 R-1) was so misleading: it modelled tempo error as small noise that averages out, when the error
that matters is a discrete factor of two that averages to nothing and destroys the decode.

**ADVERSARIAL AUDIT OF THE INJECTION PATH [PROVEN CORRECT, 2026-08-22].**
`scripts/audit_head_inject_sanity.py` pushes a *synthetic oracle* distribution (log-Gaussian,
SHARP=0.15, window-relative frames) through the **exact** code path of `eval_tempo_head_dbn.py`
(same `tmap` gather, same `/bf` per-beat normalisation, same log-ratio, same Viterbi, same
`state_path_to_events`). Result n=217: F 0.6755669584737327, CMLt 0.7906024625736449 — **bit-identical
to 16 significant digits** with the published oracle ceiling from `exp_oracle_tempo_beatonly.py`.
Therefore frame alignment, the `tmap`/`bf` normalisation, the grid order and the readout contain **no
bug**; the negative in §4.7 is not an implementation artefact. Denominator choice was also measured
as a separate arm (head marginal vs uniform): max effect **0.0062 CMLt**, and the head's own marginal
is the *better* denominator at low β. Alignment is guaranteed by construction
(`/disk1/jaehoon/smc_bt_embeddings/_report.json`: 217/217 tracks, `max_dev_vs_cache = 0.0`).

**THE CEILING ON THIS HEAD, decomposed [MEASURED 2026-08-22].** Splitting the oracle's +0.2221 by the
head's tempo verdict: on the **114 head-correct tracks the oracle buys only +0.0907**; 80% of the
oracle's total gain is carried by the 103 tracks where the head is *wrong*, because those are the
low-baseline tracks with headroom. Hence for **any** β, denominator, smoothing or gate, the total
gain extractable from this head is bounded by (114/217)×0.0907 = **+0.0476 CMLt**. The intuition
"the machinery works when tempo is right, so a bug must be hiding a large win" is **false** — there
is only +0.09 to win, on half the corpus.

### 4.7 The β sweep, and why re-weighting cannot rescue the head **[MEASURED, 2026-08-21]**

Predicted from magnitudes: the head's log-ratio range is ~12.7 nats and `/i` makes each beat carry
~`β`·12.7 nats, so a 40-beat track holds ~`β`·500 nats against the emission's ~93-nat total octave
preference. At β=5 the tempo term is **~27× stronger** than the evidence that could overrule a wrong
octave — the head does not participate in the octave decision, it dictates it. Commensurate weighting
is β ≈ 0.1–0.2.

Measured (n=217, beat-only, class-corrected, per-frame injection), ΔCMLt vs the β=0 null:

| β | 0.05 | 0.10 | **0.20** | 0.30 | 0.50 | 1.0 | 5.0 |
|---|---|---|---|---|---|---|---|
| ΔCMLt | −0.0029 | −0.0008 | **+0.0104** | +0.0061 | −0.0095 | −0.0483 | −0.0971 |
| p | 0.65 | 0.81 | **0.14** | 0.28 | 0.58 | 0.0064 | 4.4e−05 |

The predicted **shape** is confirmed — monotone-negative becomes an inverted U peaking exactly in the
commensurate range — so the magnitude account of the catastrophe is correct. **But the peak is
+0.0104 at p=0.14: null.** Optimally weighted, the head contributes nothing pooled.

**Conclusion, and it constrains what is worth trying next.** At low β the gain on correct-head tracks
and the catastrophe on octave-wrong tracks both shrink toward zero together; re-weighting trades one
against the other and cannot manufacture a net win. The ceiling on this head is set by the **size of
the correct group (113/217)**. Therefore: learning β (G3) is not worth doing at this stage — its
optimum is ~0.2 and worth ~+0.01 — and the only route upward is moving tracks out of the ½×/2×/3×
groups, i.e. **G1 at source (octave-aware training)**.

*Reproducibility note (2026-08-22):* the on-disk `/tmp/overnight/tempo_head_dbn.json` records
β=0.2 ΔCMLt **+0.0054**; a fresh re-run of the identical arm gives **+0.0104** (the number in the
table above). Sign and significance unchanged; one of the two is stale. Cite the fresh run.

**GATING [MEASURED 2026-08-22].** An *oracle* gate — inject at β=0.2 only where the head's tempo is
correct — gives **+0.0181 CMLt, p=0.013, n=217** (significant, unlike the ungated +0.0104). A
*realistic* confidence gate cannot reach it: per-track confidence predicts head correctness at
AUC **0.683** (−entropy 0.686, max-p 0.686), and the best threshold buys **+0.0089, p=0.138, n.s.**
So a small real win exists behind a gate we do not have, capped at +0.0476 (above).

---

## 4.8 Metrical level and the oracle headroom **[MEASURED, 2026-08-21; overclaim corrected same day]**

The oracle tempo channel is worth ΔCMLt +0.2221 (n=217, §4.6). That number has been used all along to
size the tempo-head work. **Most of it is not reachable from audio.**

**Ambiguity defined without reference to any head**: a track is metrically ambiguous if the
*tempo-blind* β=0 decoder shows a large AMLt−CMLt gap, i.e. it tracks a *valid* metrical level that
is not the annotated one. AMLt forgives double/half/off-beat; CMLt does not.

| threshold | n | oracle ΔCMLt (ambiguous) | oracle ΔCMLt (rest) | share of total gain |
|---|---|---|---|---|
| gap > 0.15 | 45 | +0.6270 | +0.1161 | **58.5%** |
| gap > 0.20 | 43 | +0.6215 | +0.1234 | **55.5%** |
| gap > 0.30 | 36 | +0.6243 | +0.1421 | 46.6% |

At gap>0.2 those 43 tracks (20% of SMC) have null **CMLt 0.0273 / AMLt 0.6509** — the decoder is
locked to an alternative level and scores ~zero against the annotation. The oracle lifts them to
CMLt 0.6489 (p=7.7e−08) purely by being told which level the annotator chose.

**Therefore the headroom splits by KIND of correction:**
- **+0.1234 CMLt** on 174 tracks — refining tempo *within* the annotated metrical level
- **+0.62 CMLt** on 43 tracks — correcting the metrical *level* itself

**That split is about the kind of correction, NOT about reachability.** A separate test was needed to
ask whether the level is recoverable from audio at all, and it does not support the strong reading.

*Proxy validity [MEASURED 2026-08-22].* Scoring each track's β=0 decode against all five `mir_eval`
reference variations shows the ambiguous group is **90.7% octave displacement** (16 double, 13
half-odd, 10 half-even) and **9.3% pure off-beat** (4 tracks: smc_018, smc_153, smc_274, smc_285).
The AMLt−CMLt gap is therefore a good but not pure metrical-*level* proxy; the 9% contamination
belongs in the text. They are also **not merely hard**: null AMLt is 0.6509 vs 0.7106 for the rest,
i.e. the decoder locks a stable periodicity, at the wrong level.

**Cross-system test** — *provenance corrected 2026-08-22*. The published "n=187 tracks with both
activation sources" is **not reproducible**: all 217 SMC tracks have a Beat Transformer cache, so
the real filter was a peak-picking validity condition, and no script reproduces it. Reproducible
restatement (peak-pick each system on its **native** frame rate — Beat This 50 fps, Beat Transformer
43.066 fps — threshold 0.5, ≥3 peaks in both, tempo = 60/median IBI, agreement within 8%):
**n=193 (38 ambiguous / 155 rest)**, inter-system agreement **39.5% vs 41.9%** (OR 0.90, p=0.86),
agree-with-each-other-but-not-the-annotation **26.3% vs 3.9%** (OR 8.87, p=1.0e−4); robust over
thresholds 0.3–0.6 and tolerances 5–15%. The frame-rate objection fails (both systems' peaks are in
seconds; the 20 vs 23 ms grid is three orders of magnitude below an octave).

**CIRCULARITY, previously unflagged.** The ambiguous group is *defined* by Beat This disagreeing with
the annotation on level, and one conjunct of the 26.3% statistic re-uses Beat This vs annotation
("Beat This alone disagrees" is 81.6% vs 32.3%, OR 9.30 — built in). The de-circularised statistic is
P(Beat Transformer agrees with Beat This | Beat This disagrees with the annotation): **32.3% (10/31)
vs 12.0% (6/50), OR 3.49, Fisher p=0.043**. The finding **survives at roughly half the odds ratio and
one-thousandth the significance** of the headline. Quote the conditional, not the conjunction.

*Superseded original table* (n=187; Beat This and Beat Transformer
are different architectures with different training, peak-picked, no DBN). If the annotated level
were merely one convention among perceptually available ones, independent systems should converge on
the *same* alternative:

| | ambiguous (n=36) | rest (n=151) |
|---|---|---|
| the two systems agree with **each other** | 41.7% | 42.4% |
| agree with each other **but not with the annotation** | **27.8%** | **4.0%** |

Inter-system agreement is **identical** across groups, so the ambiguous tracks are not generally
"tracks where models agree and the annotator differs". Only **27.8%** of them show two systems
converging on a non-annotated level — a genuine ~7× enrichment over the 4.0% baseline, but a
minority. On the other ~58% the two systems disagree with **each other**: those are simply hard.

**Corrected arithmetic.** The ambiguous group carries 55% of the oracle gain; ~28% of that group
shows cross-system agreement on an alternative level. Metrical *convention* therefore plausibly
accounts for **~15% of the oracle headroom, not "more than half."**

*Retracted wording (same day):* "more than half the oracle headroom is metrical-level convention, not
audio content." Wrong by ~4x. The error was inferring unreachability from *our* model picking a
different level — a better model may well recover the annotated level, and on most of these tracks
two models cannot even agree with each other.

**Independent corroboration from the dataset's own annotators.** SMC ships per-track difficulty tags
(`SMC_MIREX_Tags/*.tag`) naming why each excerpt was hard. The automatically-flagged ambiguous group
is enriched for the annotators' own **"ambiguity of metrical level"** tag: 16.3% vs 4.0%, odds ratio
**4.64, Fisher p=0.0087**. Note "slow tempo" runs the *other* way (20.9% vs 35.6%, OR 0.48) — this is
a distinct axis from the tempo blind spot, not a restatement of it.

*Limits, stated because this claim is load-bearing:* (a) **CORRECTED 2026-08-22 — "ranked first" is
FALSE; it ranks SECOND.** With a tag parser that normalises three SMC file defects (parenthesised
tags, annotator codes fused to the tag line such as "a1slow tempo", alias variants), **20** tags with
n≥5 were tested. First is **"beat phase ambiguity"**: 7/43 vs 4/174, OR **8.26**, Fisher **p=0.0014**
— the *only* tag surviving Bonferroni (p_bonf=0.027). "Ambiguity of metrical level" is second:
Bonferroni p=0.173 at k=20 (not ≈0.13 at k=15), Benjamini–Hochberg q=0.087 (survives FDR 10%, not
Bonferroni). Both are level/phase-ambiguity tags so the *a priori direction* holds, but the predicted
tag was not the top hit and the "ranked first" argument that licensed calling this corroborative is
withdrawn. Restricting the group to the 39 octave tracks sharpens it: 7/39 vs 7/174, OR 5.22,
p=0.0053. This remains **corroborative, not confirmatory** — a pre-registered replication is needed. (b) Coverage is
modest: 7/43 vs 7/174, so the tag is enriched in the group but explains only a minority of it.
(c) Annotator *confidence* does **not** separate the groups (1.91 vs 1.84, p=0.75) — the annotators
were not less sure overall, they specifically flagged metrical level on a subset. (d) SMC ships one
annotation per track; we do **not** have competing label sets, so this is not a measurement of
inter-annotator disagreement.

**Why this reframes the project.** It explains, consistently and after the fact, three separate
failures: why `q_θ`'s octave errors cluster on exactly these tracks; why its register confidence has
AUROC **0.499** (nothing to be uncertain about — it is confidently right about a *different* level);
and why every octave-repair scheme failed (R-10, R-11). ~~We were chasing a ceiling that is ~55%
annotator convention.~~ **[RETRACTED 2026-08-22 — internal contradiction.]** ~55% is the share of the
oracle gain carried by the ambiguous group, **not** the share attributable to convention; the
section's own "Corrected arithmetic" paragraph already replaced it with **~15%**. The correct closing
sentence: *we were chasing a ceiling of which roughly 15% is annotator convention.*

*Scope limit:* every figure in §4.8 is on the **beat-only, class-corrected DBN2016** (min_bpm=30,
obs_λ=16, λ=100). The joint (3,4) decoder has a different oracle ceiling (+0.2398 CMLt), so this
split must never be quoted alongside joint-decoder rows.

*Provenance debt:* no committed script reproduces the tag or cross-system tables; both had to be
re-implemented during the audit. `scripts/analyze_tags.py` silently drops parenthesised and
code-fused tag lines. Commit the real scripts before citing.

---

## 4.9 The octave-error tolerance curve — the gate G1 should always have been **[MEASURED, 2026-08-21]**

`scripts/exp_octave_degradation.py`. Start from the oracle tempo (constant per track, median IBI) and
corrupt a fraction of TRACKS with a whole-track octave flip (x2 or /2). Sweep the fraction. This
replaces the retracted sigma-noise study (§8 R-1), whose i.i.d. per-frame jitter was averaged out by
the per-beat normalisation and therefore measured smoothing rather than degradation.

n=217, beat-only class-corrected DBN2016, min_bpm=30, obs_λ=16, λ=100, β=5, vs the tempo-blind null:

| octave error | F | CMLt | AMLt | ΔCMLt | p |
|---|---|---|---|---|---|
| — (no tempo) | 0.6141 | 0.5685 | 0.6988 | — | — |
| 0.0% | 0.6532 | 0.7418 | 0.7728 | **+0.1733** | 1.7e−12 |
| 11.1% | 0.6337 | 0.6637 | 0.7631 | +0.0952 | 3.2e−04 |
| 23.0% | 0.6073 | 0.5790 | 0.7480 | +0.0105 | 0.66 |
| 36.4% | 0.5750 | 0.4742 | 0.7230 | −0.0943 | 0.010 |
| 49.3% | 0.5452 | 0.3759 | 0.6971 | −0.1927 | 4.8e−07 |
| 100.0% | 0.4308 | 0.0301 | 0.5949 | −0.5384 | 1.1e−27 |

**BREAK-EVEN OCTAVE-ERROR RATE = 24.4% — BUT ONLY AT β=5. [CORRECTED 2026-08-22]**

The break-even is a property of the **injection strength**, not of the estimator. Re-running the
identical script at **β=0.2** — the regime the learned head is actually injected in — n=217, same
RNG, same nesting:

| realised octave error | CMLt | ΔCMLt | p |
|---|---|---|---|
| 0.0% | 0.6284 | +0.0599 | 2.1e−06 |
| 23.0% | 0.6120 | **+0.0435** | 0.0021 |
| 49.3% | 0.5856 | +0.0171 | 0.25 |
| 100.0% | 0.5452 | −0.0233 | 0.091 |

**Break-even moves 24.4% → 76.2%; the slope flattens from −0.718 to ≈ −0.083 (8.6×).** At β=0.2 an
oracle-with-flips estimator at the head's 24.9% octave-error rate still buys +0.0435 CMLt (p=0.002),
and even a 100%-wrong tempo costs only −0.023 (n.s.). Every sentence deriving *"the head sits exactly
at break-even, which is why every pooled result is null"* is **invalid at the operating β**. §4.11
retracted this gate qualitatively; this measurement shows the retraction did not go far enough — the
curve is not even the right *shape* at the operating point. Never quote "24.4%" without "at β=5".

*Grid bug, found/fixed/immaterial:* 45/217 (20.7%) flipped tempi fall outside the DBN grid
[13.95, 100] frames, so the "octave flip" is really a clamp to 30/215 BPM. A grid-safe variant gives
break-even **23.60%** instead of 24.4%; all other arms within 0.008 CMLt.

*Injection-strength incommensurability:* the Gaussian (SHARP=0.15) carries ~48 nats of log-ratio
spread against a **uniform** reference, while the head carries ~13 nats against its **own marginal**;
combined with β 5 vs 0.2 the curve delivers ≈**90×** more tempo evidence per beat than the head runs
it was used to explain.

Our learned head sits at 54/217 = 24.9% by one aggregation and 32.3% by another (§4.11); the object
the DBN actually consumes scores 34.6%.

**This explains every pooled null measured today.** The head is not broken and the injection is not
mis-specified: the head is at exactly the octave-error rate where a tempo channel is worth zero.
§4.7's peak of +0.0104 (p=0.14) is the curve evaluated at our operating point.

**Actionable slope**: ≈ **−0.71 ΔCMLt per unit octave-error rate**, i.e. **+0.007 CMLt per percentage
point of octave-error reduction**. To match the beat-only decoder win (+0.0570) requires octave error
≤ ~17%; to reach +0.10 requires ~11%.

~~*Consistency check*: the 0% arm (+0.1733) reproduces the independently computed constant-median-IBI
figure exactly, from a different script.~~ **[RETRACTED 2026-08-22 — FALSE.]** The genuinely
independent script (`scripts/diag_tempo_head_bracket.py`, arm `oraclemed5`, same decoder, β=5, n=217)
gives ΔCMLt **+0.1684** (p=5.4e−12) / ΔF **+0.0343** (p=3.3e−04), not +0.1733 / +0.0390. The
difference is `np.median(np.diff(gt))` (unweighted per-beat) vs the duration-weighted median of the
frame-wise IBI series. **The +0.1733 quoted at P4, §4.9, §4.10 and gate G1b is this script's own 0%
arm cited as if it corroborated itself.** Use +0.1684 when an independent figure is wanted.

*Housekeeping:* `exp_octave_degradation.py` hard-codes its output path and does not tag by β
(the β=5 result is preserved at `/tmp/overnight/octave_degradation_beta5.json`), and its docstring
mislabels its own starting point as the +0.2221 time-varying oracle when it is the constant
median-IBI oracle (0.7418).

*AMLt barely moves* across the sweep (0.7728 → 0.5949 while CMLt goes 0.7418 → 0.0301), confirming
the curve isolates metrical-level correctness — AMLt forgives octave errors by construction.

~~**Supersedes G1.** The gate is not an accuracy threshold and never was: it is `octave-error rate < 24.4%`, with the payoff linear in the margin below it.~~ **[RETRACTED 2026-08-22 — G1 is RETIRED, not replaced.]** The break-even is a property of β, not of the estimator, and §4.11 shows the curve's family (a clean constant oracle with whole-track flips) is not the family a real head belongs to. The replacement screen is **G1c** (§4.11): accuracy on the 82 tracks where the tempo-blind DBN fails.

---

## 4.10 TTA and the tempo head are REDUNDANT, not additive **[MEASURED, 2026-08-21]**

`scripts/exp_tta_plus_tempo.py`. All four rows recomputed in ONE setup — beat-only class-corrected
DBN2016, min_bpm=30, obs_λ=16, λ=100, β=0.2, TTA rates {0.85, 1.0, 1.2}, n=217, paired.

| arm | F | CMLt | AMLt | CMLc | AMLc |
|---|---|---|---|---|---|
| null | 0.6141 | 0.5685 | 0.6988 | 0.4612 | 0.5587 |
| tta | 0.6246 | 0.5808 | 0.7200 | 0.4786 | 0.5876 |
| tempo | 0.6156 | 0.5789 | 0.6983 | 0.4716 | 0.5616 |
| tta+tempo | 0.6243 | 0.5820 | 0.7276 | 0.4829 | 0.5992 |

Paired vs null: tta CMLt +0.0123 (**p=0.46**), AMLt +0.0212 (p=0.044), AMLc +0.0289 (p=0.0017);
tempo CMLt +0.0104 (p=0.14); **tta+tempo CMLt +0.0135 (p=0.74)**, AMLt +0.0288 (p=0.014),
AMLc **+0.0405 (p=0.00044)**.

~~**ADDITIVITY: expected +0.0227 if independent, actual +0.0135 → 40% overlap.**~~
**[WEAKENED TO UNSUPPORTED, 2026-08-22.]** The "40% overlap" was reported without a test. The correct
statistic is the paired interaction (tta+tempo − tta) − (tempo − null): CMLt **−0.0092, 95% CI
[−0.0268, +0.0084], t p=0.30**. The CI contains both 0 (perfectly additive) and −0.0104 (complete
redundancy). Interaction sd = 0.1315, so 80% power to detect an overlap this size needs **n ≈ 1602
tracks**; SMC has 217 — the experiment is ~7× underpowered on the metric the claim is made on.
Worse, the sign is **metric-selective**: on AMLt the interaction is **+0.0081** and on AMLc
**+0.0087**, i.e. **super-additive** (tta+tempo AMLc +0.0405 exceeds the +0.0318 sum of parts). And
it rests on an untested β: re-running at β ∈ {0.1, 0.2, 0.4, 0.8} flips the script's own verdict
ADDITIVE / REDUNDANT / REDUNDANT / ADDITIVE. **Do not claim redundancy in the paper.** What is safe:
report the measured combination, never the sum.

**TTA's headline shrinks on the correct decoder.** The verified +0.0366 CMLt figure was
**peak-picked, no DBN**; on beat-only it is +0.0123 and **not significant**. Same redundancy one
level up: the beat-only decoder already recovers much of what TTA recovered. This is why the figures
had to be recomputed rather than combined across setups.

**This half is now near-proven [MEASURED 2026-08-22, `scripts/audit_tta_ratesweep.py`, n=217, both
decoders, 9 rate subsets, one script].** Peak-pick @0.5: null CMLt 0.5138 → best subset
{0.85,1.0,1.2} 0.5504, ΔCMLt **+0.0366 (p=0.0029)**, ΔF +0.0180 (p=0.00023) — an exact reproduction
of the published headline, which independently validates the cache, the un-warping and the null.
Beat-only DBN: null 0.5685 → same subset 0.5808, **+0.0123 (p=0.46)**. The **decoder change alone
moves the null by +0.0547 CMLt, more than TTA's entire peak-pick CMLt gain**. TTA's peak-pick gain is
continuity repair (CMLc +0.0491, AMLc +0.0594); the DBN transition model already supplies continuity,
so the shrinkage is mechanistically forced. Exhaustion over subsets confirms no rescue: best-of-9 on
the DBN is {0.85, 1.2} at +0.0147 (p=0.25); all-7 −0.0541; fast{1,1.2,1.5,2} −0.1002 (p=0.00098);
widest{0.5,1,2} −0.1110 (p=2.9e−05). **No code bug**: un-warping verified, `expit(cache r1.0)` is
bit-identical to the project-wide null, and the rate subset used was already optimal.

*Untested-until-now and mildly positive:* dropping r=1.0 entirely ({0.85, 1.2}) is the **best** DBN
subset (+0.0147 CMLt, +0.0115 F) and ties the best on peak-pick (+0.0186 F, p=0.011) — the
unstretched pass contributes nothing. Does not flip the null.

**What the combination does buy**: continuity at whatever metrical level is chosen — AMLt +0.0288
(p=0.014), AMLc +0.0405 (p=0.00044) — but **not CMLt**. ~~Exactly as §4.9 predicts: at the 24.4% break-even, nothing moves CMLt until octave error falls.~~ **[EXPLANATION RETRACTED 2026-08-22:** the 24.4% break-even is a β=5 artefact and does not apply at the β=0.2 used here (§4.9). The CMLt null stands; the *reason* offered for it does not.**]** Note also that the AMLt/AMLc gains here are **super-additive**, which contradicts the redundancy story one paragraph above — see the additivity correction.

**End-to-end, honestly attributed.** Against the decoder a practitioner would use today (madmom joint
DBN, CMLt 0.5102) the full stack reaches 0.5820, **+0.0718 CMLt** — but ~80% of that is the decoder
fix (§3/P5), not TTA and not the tempo head.

---

## 4.11 The octave-error rate is NOT a sufficient gate either **[MEASURED 2026-08-22 — this weakens §4.9]**

**Step 0 of the head-improvement plan** (`scripts/exp_head_acf_blend.py`,
`scripts/eval_tempo_head_dbn.py` with `ACF_W`). 37 of the head's 103 failures are tracks where the
annotated tempo IS the autocorrelation peak of the model's own beat activation (group A), so blending
`log q <- (1-w) log q_head + w log q_acf` should recover them. It does, by the argmax metric:

| arm | octave-error rate (argmax) | correct |
|---|---|---|
| head alone | 32.3% | 51.6% |
| ACF alone (τ=10, **no training at all**) | 24.0% | 64.5% |
| blend w=0.8, τ=10 | **20.7%** | 66.4% |

37 tracks recovered, 5 broken — a 9:1 trade, consistent across τ ∈ {10, 30, 60}.

**And it bought nothing in the decoder.** Injected into the beat-only DBN, n=217, best over the β
sweep: head alone **+0.0104** (p=0.14); blend w=0.5 **+0.0046** (p=0.21); blend w=0.8 **+0.0058**
(p=0.83). The blend is if anything *worse* than the unblended head.

*Flattening was checked and does not explain it*: blending reduces the log-ratio range 13.09 → 9.27
nats (−29%, equivalent to scaling β by ~0.7), which the β sweep covers.

**A REAL BUG WAS FOUND HERE [CONFIRMED, FIXED, RE-RUN 2026-08-22 — found independently by two
auditors].** `scripts/eval_tempo_head_dbn.py` ≈line 156 computed `ratio = lq − logmarg` using the
head's **full** marginal, but under the blend only (1−`ACF_W`) of the blended log-density carries the
head's prior. The residual −`ACF_W`·log p_head is a spurious per-tempo tilt of 0.8×3.872 = **3.10
nats** favouring slow tempi — ~31% of the injected signal was prior-inversion artefact. Measured
consequence: it collapsed the true-vs-half-tempo margin 1.714 → 0.582 nats/beat (Δ −0.788, p=0.0029)
and the object the DBN actually received had octave error **37.3% — WORSE than the head's 34.6%**,
not the 20.7% this section's table advertises. **So this section's premise as written was factually
wrong: the argmax improvement never reached the decoder.** Fixed as `MARG_FIX`
(`logmarg_eff = (1−ACF_W)·logmarg`); the injected object is then **24.0% octave error / 64.5%
correct** and the margin is restored to 1.71 nats/beat.

**And after the fix the negative still stands.** Blend w=0.8, τ=10, n=217, ΔCMLt vs the β=0 null:
β=0.05 +0.0000 (p=0.65) | 0.1 **+0.0028 (p=0.21)** | 0.2 −0.0035 | 0.3 −0.0049 | 0.5 +0.0018 |
**5.0 −0.0448 (p=6.8e−05)**. The 13-point-better argmax arm decodes **worse** at the strength where
effects are rankable. A separate re-run with the marginal fix and an extended β grid (to 1.2) gives
best +0.0028 (p=0.21) and F *significantly negative* at β=0.2/0.3 (−0.0046 p=0.024 / −0.0056
p=0.0077). An oracle-path **point-tempo** injection of the blended estimate (SHARP=0.15, uniform
prior — the apples-to-apples test against §4.9 that had never been run,
`scripts/exp_acf_blend_pointtempo.py`) also fails at every strength: β=0.5 −0.0029, β=1 −0.0041,
β=2 −0.0047, β=5 −0.0098 (F −0.0128, p=0.02). **The fix made the evidence strictly better and the
decoding no better.** That makes §4.11 stronger, not weaker.

**WHY, and this is the useful part [MEASURED 2026-08-22].** The blend is genuinely better evidence by
a proper scoring rule — mean per-frame log q(true bin) −4.111 → **−2.537 nats**, p=1.3e−31, n=217 —
so "the blend is just worse evidence" is refuted. It fails because the beat-activation ACF is
**redundant with the DBN's own evidence**: both read the same Beat This activation. On tracks where
the tempo-blind DBN already succeeds (CMLt≥0.5, n=135) the blend's tempo is correct **93.3%** of the
time; on tracks where it fails (n=82) only **22.0%**. Fisher **p=2.13e−28, OR 0.02**. Maximum
recoverable is 18/217 = +0.083 even at perfect exploitation, against 64/217×0.36 = −0.106 of damage
from confidently wrong octaves — expected net negative, matching the observed −0.0098 at β=5.

**MEASUREMENT-OBJECT MISMATCH, corpus-wide.** Every octave-error rate quoted in §4.6/§4.9/§4.11 is
computed on `q_θ`; the decoder consumes `q_θ / p_head`. Same head, same tracks: **32.3% (q) vs 34.6%
(injected)**, and §4.9's "our head sits at 54/217 = 24.9%" is a *third* object — 24.9% could not be
reproduced by any aggregation tried (per-frame argmax + median gives 32.3%).

**DATA-LOSS BUG:** `eval_tempo_head_dbn.py` wrote all three `ACF_W` arms to the same
`/tmp/overnight/tempo_head_dbn.json` (`ACF_W` absent from the filename), so the w=0.0 and w=0.5
per-track results were silently overwritten by w=0.8. Same family as the `%g` → `r1` bug. Now
parameterised by `TAG`.

**STANDING TRAP for every planned fusion experiment:** any fusion of `q_θ` with another tempo source
(ACF, DeepRhythm, TTA, a second head) **must rescale the `log p_head` subtraction by the head's blend
weight**, or it silently injects up to 3.9 nats of slow-tempo bias — larger than the octave evidence
being fused.

**CONSEQUENCE — §4.9's break-even does not gate the head.** The 24.4% figure was measured by taking
the ORACLE tempo (a clean correct constant per track) and flipping a fraction of tracks by an octave.
That characterises *a perfect estimator with octave errors*. Our head emits a broad, per-frame
distribution; fixing its argmax does not turn it into a flipped oracle. The curve is a correct
measurement of its own family and was over-generalised to gate a different object.

**Two argmax proxies have now failed the same way:**
- frame accuracy — 12.2% → 46.1% with bias eliminated, decoding got *worse* (§4.6)
- octave-error rate — 32.3% → 20.7% *on `q`* (but 34.6% → 37.3% **on the object injected**, and
  24.0% after the `MARG_FIX`), decoding did not move either way (this section)

Both summarise by argmax something the DBN consumes as a **distribution**. **No argmax statistic of
`q_θ` has been shown to predict decoding gain.** Until one is, the only trustworthy gate is the
end-to-end paired CMLt on the beat-only decoder — expensive (~7 min per configuration) but the only
measurement that has not misled.

*Also notable*: a closed-form autocorrelation with **no training** (24.0%) matches the trained head's
comparable rate. The head is not beating a textbook periodicity estimate at picking metrical level.

**THE NEGATIVE IS ABOUT THE SOURCE, NOT THE MECHANISM — scope corrected 2026-08-22.** The earlier
conclusion "abandon the head-improvement plan" over-generalises. The injection mechanism is
**proven sound** (§4.6 audit: the oracle buys +0.2221 through this exact code path, bit-identical).
What fails is that an ACF of the Beat This beat activation is statistically redundant with the
decoder's own evidence. The untested and unrefuted branch is a periodicity source **statistically
independent** of that activation — spectral-flux or mel-band onset-envelope ACF, DeepRhythm (results
already at `/tmp/overnight/deeprhythm_smc.json`), or an out-of-domain head. **Screening criterion for
any candidate [NEW GATE, cheap]:** not overall octave-error rate, but accuracy restricted to the
**82 tracks where the tempo-blind DBN scores CMLt<0.5**. The ACF scores **22.0%** there; the bound
above predicts anything below ~50% on that subset will lose. Run this screen before any retraining.

---

## 4.12 The decoder envelope: 83% of the F gap is in the ACTIVATIONS **[MEASURED 2026-08-22]**

`scripts/exp_decoder_ceiling.py`, n=217 SMC, Beat This activations, mir_eval trim, F-measure ±70 ms.
Each decoder is granted **per-track ORACLE hyperparameters** (threshold, transition_lambda) — a
deliberately generous bound no deployable system can reach.

| | F |
|---|---|
| raw peak-pick @0.5 (no oracle) | 0.6270 |
| A — peak-pick, per-track oracle threshold | 0.6696 |
| B — wide DBN [30,215], per-track oracle λ ∈ {1..500} | 0.6525 |
| **C — per-track best of {A, B} = the decoder envelope** | **0.6908** |

**Decomposition of the gap to F = 1.0, from the deployed 0.6270:**

| component | F | share |
|---|---|---|
| reachable by ANY decoder on these activations | +0.0638 | **17%** |
| not reachable by any decoder — the activations themselves | +0.3092 | **83%** |

**Consequences.**
1. The paper's current "F=0.673 upper bound on any decoder" is **understated**; the true envelope is
   **0.6908**, because per-track *choice between decoders* beats either alone. Correct this.
2. Every decoder-side intervention in this project competes inside a **0.064 F box**. Measured
   effects of ±0.01–0.03 (beat-only decoding, TTA, the tempo head, β, λ) are exactly the scale that
   box permits. This is not a failure to find the solution; it is a measurement of where a solution
   cannot be.
3. **Tempo is very nearly the whole decoder-side opportunity.** The tempo oracle reaches F 0.6756,
   i.e. **76% of the entire remaining decoder headroom** (0.0486 of 0.0638). Nothing else on the
   decoder side is worth comparable amounts.
4. Note peak-picking (0.6270) **beats** the beat-only DBN (0.6141) on F while losing on continuity —
   consistent with the F-vs-continuity trade recorded throughout this document.

*Caveats*: the envelope is Beat This activations only — a different frontend has a different ceiling,
and that is the axis the 83% lives on. The per-track oracle hyperparameter selection makes C an
upper bound, not an achievable target.

---

## 4.13 SMC annotation quality is NOT the limiting factor **[MEASURED 2026-08-22, annotations only]**

Model: observed onset `t_k = s_k + e_k` with `e_k` i.i.d. timing noise, variance σ². Then
`IBI_k = t_{k+1} − t_k` carries `(e_{k+1} − e_k)`, so consecutive IBIs share `−e_k` and the lag-1
autocovariance absorbs `−σ²`. Pure jitter with locally constant tempo drives the lag-1
autocorrelation toward **−0.5**; smooth rubato drives it **positive**. The sign therefore separates
annotation noise from genuine expressive timing, using no model at all.

Measured, n=217 (tracks with ≥12 beats):

| statistic | value |
|---|---|
| lag-1 autocorrelation of IBI | median **+0.205**, mean +0.197 |
| rubato-dominated (r₁ > 0) | **71.9%** of tracks |
| jitter-dominated (r₁ < 0) | 28.1% (strongly, r₁ < −0.35: 6.0%) |
| implied jitter σ (n=61 with γ₁<0) | median **15.6 ms**, 90th pct 46.8 ms |
| IBI coefficient of variation | median **0.091** |

**Against F's ±70 ms window**: σ = 15.6 ms puts ~**0.00%** of beats outside tolerance; even the 90th
percentile (46.8 ms) implies an annotation-imposed F ceiling of ~0.865, and that applies to ~10% of
tracks. Annotation jitter cannot account for more than a small slice of §4.12's 0.309 activation-
limited gap.

**Consequences.**
1. **The misannotation hypothesis is bounded and small.** With §4.8's metrical-level disagreement
   (~10 tracks, ≈ +0.03 F if all were wrong), annotation issues total well under 10% of the F gap.
2. **λ = 11.45 is REAL RUBATO, not annotation noise.** It was previously flagged that the MLE's high
   tempo variability might be jitter; the positive autocorrelation refutes that. madmom's shipped
   λ=100 genuinely is mis-specified relative to how SMC's tempo moves — and §5.1's explanation
   stands: the transition term regularises rather than models, so matching reality still hurts.
3. **IBI CoV = 0.091** — 9% beat-to-beat tempo variation — is a model-free one-number
   characterisation of why SMC is hard, derived from annotations alone.

---

## 4.14 The CRF decoder: eight defects fixed, then a structural degeneracy **[MEASURED 2026-08-22]**

`scripts/train_crf_decoder.py`, `scripts/gate_crf_plumbing.py`. A log-linear (CRF) model on the
beat-only bar-pointer state space whose path score is a weighted sum of feature functions
(Böck emission, learned tempo head `q_θ`, activation ACF, tempo transition), with the weights learned
by conditional ML against competing paths. Motivation: every combination weight in this project has
been set by an independent grid search (β=0.2, ACF_W=0.8, ensemble w=0.6, Krebs's λ=100); a CRF makes
them parameters of ONE objective. Prior art is explicit and must be cited — Durand & Essid ISMIR 2016
name joint NN+CRF training as future work; the idea is not ours (see PRIOR_ART_NEURAL_DBN.md).

**Eight defects found and fixed**, each masked by the previous one:

| # | defect | how it showed |
|---|---|---|
| 1 | control decoded with the LEARNED λ | "null" read F 0.6215 not 0.6141 |
| 2 | objective separable, no direction gradient | cos(w,w*) *degraded* 0.947→0.941 while ‖w‖ inflated |
| 3 | mining compared STATE-PATH identity | 250/250 corpus tracks "wrong"; correct decodes fed back as negatives |
| 4 | numerator was a hand-built path | positive lost under every weighting → w collapsed |
| 5 | L2 REWARDED the collapse | w_trans went negative; loss oscillated 1.15 ↔ 36.6 |
| 6 | mining reference spanned the whole track | 70% of ballroom/GTZAN mined as wrong (implausible) |
| 7 | beat filter used the wrong offset (`bf<T` for `ff≤bf<ff+T`) | mask and positive built from the wrong beats |
| 8 | numerator mask ±40 ms vs the metric's ±70 ms | numerator excluded paths that would score a perfect F |

**Verification gates now pass.** G1: at `w=[1,0,0,1]` the CRF reproduces the beat-only DBN
**bit-exactly** — identical Viterbi paths on 40/40 tracks, ΔF 0.00000, kernel agreement 7.1e-15.
G3 (recovery of a known weight vector): cosine **0.9807** under unit-norm projection (0.9711 under
L2). Mining rate after fix 6/7: **54/250 = 21.6%** at init, consistent with Beat This + DBN on those
corpora (previously 175/250).

**THE REMAINING PROBLEM IS STRUCTURAL, NOT A BUG.** The loss is the gap between the CONSTRAINED
maximum (numerator, re-decoded under the current weights) and the FREE maximum (denominator).
Anything that makes those two decodes AGREE reduces the loss — and the cheapest way to make them
agree is to weaken the structure that distinguishes them. Measured trajectory: corpus wrong-decode
count 54 → 58 → 119 → 250 of 250 while the loss fell.

The degenerate direction is reachable through **two redundant parameters**:
- λ, which slid 100 → 18.36 (toward the generative MLE region, as in §5.1/P3);
- `w_trans`, which does the same thing by rescaling. Freezing λ closed the first route and the
  optimiser immediately took the second: with λ pinned at 100 and ‖w‖ projected to 2.0, the em:trans
  ratio still rotated 1.00 → 1.64 (`w=[+1.708, −0.029, −0.003, +1.040]`), i.e. the transition prior
  weakened again, and decoding degraded identically.

Note the unit-norm projection *does* work for its purpose — it removes the collapse in MAGNITUDE
(‖w‖ held exactly at 2.0). The residual failure is a ROTATION, which no norm constraint can prevent.

**THE CORRECT FORMULATION is forward-SUM MMI, not max-vs-max**: numerator = log-sum-exp over all
paths consistent with the reference, denominator = log-sum-exp over all paths. Flattening then raises
both sums together and buys nothing, so the degenerate direction disappears. `mbr_decode.py`'s
`PosteriorDecoder` already implements forward–backward (gates passed at 1e-11), so the machinery
exists — but replacing Viterbi maxima with lattice sums is a rebuild of the objective, not a patch.

**Status: NOT REFUTED, NOT DEMONSTRATED.** No trustworthy CRF result exists. The implementation is
verified; the objective is not yet admissible. Any number produced by the max-vs-max version should
be disregarded.

*Also recorded*: the CRF's first (pre-fix) run independently drove `w_tempo` to −0.010, and the
post-fix runs hold it at −0.029 to −0.047 — a fourth independent line of evidence, arrived at by a
different route, that the learned tempo head carries no usable weight (cf. §4.6, §4.7, §4.11).

---

## 4.15 Octave-aware tempo loss — **THE ONE NEGATIVE THAT FLIPPED** **[MEASURED 2026-08-22]**

*Correction history:* `scripts/sandbox_octave_loss.py` (v1) reported that both octave-aware loss
variants **failed** the pre-registered bar (oct_err down ≥20% relative **AND** acc10 down ≤0.01),
which is the result that closed off "G1 at source (octave-aware training)" — the route §4.7 names as
**the only route upward**. **That failure was an artefact.** Re-run as
`scripts/sandbox_octave_loss_v2.py`, the `oct` variant **PASSES** the unmodified bar on both splits.

**Four defects in v1, in order of decisiveness:**

- **DECISIVE — penalty/metric support mismatch.** The `oct` penalty pushed mass off exactly **one**
  grid bin (the nearest ×2 / ÷2 partner). The grid step is 0.0481 log₂ while the oct_err **metric**
  counts a band of ±0.15 log₂ = **±3.12 bins**. Mass displaced by one bin still scores as an octave
  error, so the loss was a near no-op *with respect to the metric it was judged on*. Proof, mean `q`
  mass inside the metric's band: base 0.1469 → `oct` 0.1473 (w=1) / 0.1428 (w=5) — **unmoved** — vs
  0.0968 for the band-penalised version at w=50. Fixed as `octband` (full ±3-bin window) and
  `octbandlog` (`−log1p(−mass)`, so the gradient does not vanish as mass shrinks).
- **`oct_w` hard-coded at 1.0 and never swept** — ~20–100× too small. Dose–response is monotone and
  the bar is only cleared at w≥20 (split A) / w≥50 (split B).
- **Degenerate train/val split.** `meta` is ordered by dataset, so v1's `[:train_tracks]` /
  `[:val_tracks]` slices took **900 Ballroom-only** training tracks and **160 GTZAN-only** validation
  tracks — Ballroom being the most tempo-unambiguous corpus in existence, a pathological population
  for studying octave errors. Fixed with a fixed-seed shuffle. This *changed the answer*: w=20 passes
  on the original split, not on the stratified one.
- **Underpowered and undertrained, and compared UNPAIRED.** 2 seeds, 1200 steps, unpaired seed means,
  against a 0.024 absolute bar with base seed SD acc10 0.037 / oct_err 0.013. Fixed: **8 paired
  seeds, 3000 steps**, exact two-sided sign test. Effects that were p=1.000 unpaired are p=0.008
  paired.

*Ruled out, verified line by line:* the `Factorised` forward returning unnormalised logits is
intended (both loss and eval apply `log_softmax` over the grid); the penalty `gather` indexes the
tempo dimension correctly. Minor: the (chroma, register) map collides on 3 adjacent bin pairs, all
inside the ±10% tolerance, so it cannot affect acc10 or oct_err.

**Results (8 paired seeds, 3000 steps, corpus fold 0 only; SMC untouched):**

| split | variant | w | acc10 | oct_err | Δacc10 | oct rel | k/8 | p_sign | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A | base | — | 0.5545 | 0.1399 | — | — | — | — | — |
| A | `oct` (v1) | 1 | 0.5705 | 0.1425 | +0.0160 | +2.3% | 2/8 | 0.289 | FAIL |
| A | `octband` | 20 | 0.6133 | 0.1052 | +0.0588 | **−24.0%** | 8/8 | 0.008 | **PASS** |
| A | `octbandlog` | 50 | 0.6074 | 0.0950 | +0.0529 | **−30.4%** | 8/8 | 0.008 | **PASS** |
| A | `fact` | 1 | 0.6213 | 0.1724 | +0.0668 | +24.6% | 0/8 | 0.008 | FAIL (worse) |
| B | base | — | 0.5184 | 0.1159 | — | — | — | — | — |
| B | `octband` | 50 | 0.5960 | 0.0867 | +0.0775 | **−23.6%** | 8/8 | 0.008 | **PASS** |
| B | `octbandlog` | 50 | 0.6032 | 0.0857 | +0.0847 | **−24.5%** | 8/8 | 0.008 | **PASS** |
| B | `fact` | 5 | 0.7328 | 0.1579 | +0.2144 | +39.1% | 0/8 | 0.008 | FAIL (worse) |

(Split A = v1's original slices, 900 Ballroom train / 160 GTZAN val; split B = stratified, 1500/400
mixed. Full grid incl. w ∈ {1,5,20,100} in `/tmp/overnight/v2_w*.log`, `/tmp/overnight/v3_strat_w*.log`.)

**RECOMMENDED PROMOTION: `octbandlog`, `oct_w=50`, `band=3`** — the only variant passing on **both**
splits with acc10 **up**. Criterion (2) never binds anywhere: acc10 rises for every variant tried, so
the entire v1 decision rested on criterion (1), which is precisely the quantity v1's single-bin
penalty could not move by construction. **v1 was not noise around a null — it was a correctly
measured null of a loss that did nothing.**

**`fact` SURVIVES as a negative and is now stronger:** significantly *worse* on oct_err (0/8 seeds
better on both splits, p=0.008) despite a huge acc10 gain (+0.114 to +0.214). Mechanism visible in
the diagnostics: it raises octave-*tolerant* accuracy the most (0.8906 vs base 0.6366 on split B)
while raising octave-band mass (0.1649 vs 0.1316) — it learns tempo **modulo octave** beautifully and
then picks the register worse. **Do not promote `fact`.** This is exactly the failure the
pre-registration was written to catch.

**TRANSFER TO SMC — [UNVERIFIED], and the distinction matters.**
*Measured:* the sandbox population is **easier** than SMC. Base octave headroom (octave-tolerant acc
minus acc10) is 0.136 (A) / 0.118 (B) at frame acc10 0.5545 / 0.5184; the deployed head on SMC (§4.6)
has track acc 0.5161 vs octave-tolerant 0.7558 — a gap of **0.2396**, 1.8–2.0× more octave ambiguity
at a *lower* frame accuracy (0.4609). So the sandbox **understates** the available headroom and these
verdicts are conservative.
*Inferred, NOT measured:* applying §4.9's slope of +0.007 CMLt per percentage point to a 25–30%
relative cut of SMC's 24.9% track octave-error rate lands at ~17.4–18.7%, worth roughly **+0.043 to
+0.053 CMLt**. **Treat this chain as [CONJECTURE]**: it assumes a frame-level relative reduction on
corpus transfers to a track-level reduction on SMC (untested), *and* it uses the §4.9 slope, which
§4.9/§4.11 now show is a β=5 artefact that does not gate a real head. Settle it only by a full
retrain at `oct_w=50` plus an SMC evaluation, quoting the decoder explicitly (beat-only
class-corrected DBN2016, min_bpm=30, obs_λ=16, λ=100, β≈0.2).

*Second, smaller issue with the bar itself (NOT changed):* oct_err is an **unconditional** rate, so a
variant that converts gross errors into correct predictions is charged for octave errors that were
always going to survive. Split B `octband` w=20 shows it: acc10 +0.1227 with oct_err −16.0% FAILED,
while the *share* of remaining error that is octave-type was essentially flat (0.2572 → 0.2645). The
bar was left untouched and it did not matter — w=50/100 clears it outright — but if a future variant
fails only criterion (1) while raising acc10 by >0.10, interrogate the bar, not the variant.

---

## 5. Training objectives, and why each is or is not admissible

### 5.1 Generative MLE — **NOT ADOPTED [MEASURED, weakened from "REJECTED [PROVEN + MEASURED]"]**

Fitting λ to observed tempo changes gives **λ = 11.45** (10,267 transitions; exponential family fits
well, ratios 0.84–0.95). **[UNVERIFIED]** — the 11.45 value and the 10,267-transition count were
produced by a script not re-run during validation.

Measured cost, n=87 held-out SMC, beat-only PyTorch DBN Viterbi, obs_λ=16, min_bpm=30, Beat This
activations, mir_eval 5 s trim, paired vs shipped λ=100: **ΔAMLt −0.0350, p=0.154 — not
significant** (ΔF +0.0146, p=0.011). The previously stated "−0.084 AMLt" is not reproduced; the
largest AMLt cost observed on any arm is the *mined* arm's −0.0440 (p=0.034).

The structural argument stands but is now weaker in one leg: in decoding the transition term acts as
a **regulariser**, not a density, and matching reality makes decoding more permissive. **However,
Proposition 3 no longer supports "drift outscores truth" as a general fact** (truth wins 83.5% of
drift comparisons, n=266, p=1.4e−28), so that leg is retracted. What remains: Krebs set λ by **grid
search on Cemgil accuracy**, never by ML, and rejected MLE explicitly; and the measured MLE arm buys
F while costing AMLt (n.s.). **Status is therefore [MEASURED, directionally] rather than
[PROVEN + MEASURED]** — MLE is not *demonstrated* harmful, it is demonstrated not helpful.

### 5.2 Conditional (discriminative) training — **ADMISSIBLE [BUILT]**

```
L = − log  exp(score(z_true)) / Σ_k exp(score(z_k))
```

With the full path set this is exact MMI; with sampled negatives it is NCE for the same quantity.
It normalises against **competing decodings** rather than against reality, which is the correct
denominator for a decoder.

**Status [MEASURED]:** implemented end-to-end; held-out numbers reproduce exactly (split
train 130 / held-out 87, ids bit-identical). With synthetic negatives it recovered λ≈9.02; with
hard-mined negatives λ≈7.55. Both reproduce the P3 trade-off. **This does not falsify the
objective** — it confirms P3, since only λ was ever being optimised. `β` was never trained, because
`q_θ` does not yet exist (confirmed: `path_score(..., None, None)` at lines 282–283, `--init-beta`
default 0.0, `beta.grad is None` after a real backward pass while `raw_lam.grad = 2.13`; the module
docstring's "transition_lambda **and beta** are learned" is **false and must be corrected in the
script**).

The mining count is **115/129 at λ=100** in reproduction; the previously stated **118/129 is
[UNVERIFIED]** (the run log was not kept).

**Four implementation defects found in `scripts/train_contrastive_dbn.py`. None reverses the
conclusion, but all must be disclosed:**

- **(b) invalid paths [CONFIRMED].** `beats_to_state_path` does not build a valid bar-pointer path:
  mean **95.8 illegal pointer moves** and 139.7 held (repeated-state) frames per track, on **100% of
  129 tracks**. Under the model's own `dense_transition` these have log-probability −inf, so the
  scored quantity is not a model path score. *Untested* whether a valid construction moves the
  optimum.
- **(c) missing boundary term [CONFIRMED].** `path_score` applies `log_T` only where the tempo index
  *changes* (37.2/track) instead of at every beat boundary (89.4/track). The omitted self-transition
  `log_T[v,v]` is −0.11 nats at λ=100 vs −2.06 at 7.55, i.e. ~52 missing λ-dependent terms per track.
  **Not fatal**: re-sweeping with it fixed moves the optimum from ~7.55 to **~5**, i.e. *further*
  from 100, never toward it (corrected objective: λ=3 −51.65, λ=5 −51.23, λ=7.55 −51.78,
  λ=11.45 −52.87, λ=20 −56.59; n=128 tracks finite at every λ). The docstring's "within-beat advance
  is zero-cost" is true; "only tempo-change boundaries carry transition mass" is false.
- **(e) λ-dependent data dropping.** The −inf comes from the hard threshold mask in
  `log_tempo_transition`: at λ=100, 2272/3600 tempo pairs are forbidden and **57/129 tracks (~44%)
  are non-finite and silently skipped** — exactly the tracks with real tempo variation. At λ=20 only
  1 track; at λ=9, zero. The objective is thus a *different function* at each point on the
  trajectory. Biases early gradients only, but must be stated.
- **saturation.** The synthetic-negative log-posterior can saturate to exactly 0.0 with gradient
  ~1e−142 (one track's positive path has zero tempo-index changes, hence zero gradient), so λ motion
  in the synthetic arm is driven by a minority of tracks — an additional reason **λ=9.02 is weakly
  determined**.

The **gradient itself is correct [PROVEN]**: finite differences vs autograd on
`log_tempo_transition` w.r.t. log-λ agree to 8 decimals at λ=100 (−20.69923597) and λ=9
(0.91405482), with the fd recomputing the mask, so the `detach` is verified inert.

**ALL FOUR DEFECTS NOW FIXED AND RE-RUN [MEASURED 2026-08-22, `scripts/train_contrastive_dbn_fixed.py`].**
No impossibility proof exists and the a-priori argument ran the *other* way — with the missing
self-transition restored, a constant-tempo positive collects ~89 terms of `log_T[v,v] = −log Z(λ)`,
which increases monotonically toward 0 as λ grows, so a correctly scored constant-tempo positive is
**rewarded** by high λ. The buggy scorer's low-λ bias was structural. It was therefore a live
candidate for a false negative. It is not one.

Fixes: (b) `beats_to_state_path` replaced with a block-wise resync-greedy construction (a tempo of
*i* frames owns exactly *i* states, pointer +1/frame, wrap only at block end) — **0 illegal moves
over 780 paths**, down from 95.8/track on 100% of tracks; positive and negative paths now also cover
the same frame span, so their emission sums are commensurate. (c) `log_T` now applied at **every**
beat boundary (~89/track, was 37.2), including the high-λ-rewarding self-transition. (e) the madmom
hard mask is gone from the training objective (reproduced exactly: with the mask, 93/130 tracks
non-finite at λ=100 and 130/130 at λ=1000; without it, 0/130 at every λ from 3 to 1000). Plus **β is
now actually trained**, jointly with λ, against the existing head.

Corrected objective sweep (mean contrastive NLL, n=130 train, **0 dropped at every point**): λ=3
47.72 | **5 46.69** | 7.55 47.31 | 9.02 48.07 | 11.45 49.47 | 20 57.30 | 100 177.61 | 1000 2469.81.
Clean unimodal minimum at λ≈5, monotone and steep away from 100. Gradient training from λ=100
converges to **λ = 5.085** (was 7.55 mined / 9.02 synthetic under the buggy scorer).

Held-out n=87, beat-only DBN Viterbi, paired Wilcoxon vs shipped λ=100:

| arm | F | CMLt | AMLt | AMLc |
|---|---|---|---|---|
| shipped λ=100 | 0.6155 | 0.5891 | 0.7050 | 0.5622 |
| learned λ=5.09 | 0.6322 | 0.5848 | 0.6596 | 0.4997 |
| MLE λ=11.45 | 0.6301 | 0.5903 | 0.6699 | 0.5129 |

learned − shipped: F **+0.0167 (p=0.020)**, CMLt −0.0043 (p=0.61), AMLt **−0.0454 (p=0.029)**, AMLc
−0.0625 (p=0.0086). **The three bugs moved λ (7.55 → 5.085) but reproduce the SAME trade-off to
within 0.0014 AMLt.** P3 stands.

**β arm:** joint learning gives λ→3.772, **β→−0.199** — the objective wants a *negative* weight on the
tempo head. Held-out: learned+β F 0.6202 / CMLt 0.5335 / AMLt 0.6173; shipped100+β F 0.6037 / CMLt
0.5562 / AMLt 0.6795 — both worse than their β=0 counterparts on every metric. Learning β does not
rescue the objective. (Caveat: β was fit on the SMC train split only, not on corpus data as
`scripts/decode_tempo_observation.py` does; whether that flips its sign is **[UNVERIFIED]**.)

**Supported conclusion, and no more:** the contrastive optimum is **λ ≈ 5** with the scorer fixed
(5–9 across scorer variants) — nowhere near the shipped 100, and every value in that range trades F
for AMLt. λ is not the decoding bottleneck.

**The real target this exposes [MEASURED].** The corrected objective ranks λ=5 best while decoded
AMLt ranks λ=100 best, **on the same tracks**: the contrastive loss is *anti-correlated* with the
metric that matters. That mismatch, not the value of λ, is the measurable contribution available
here. A likely cause, and it is testable: the negatives are near-degenerate with the positive's own
quantisation noise — drift negatives use sd = 2% of the IBI while the 60-tempo log grid spacing is
~4%, and the GT positive path must itself snap every inter-beat interval to that grid, so the
positive already carries spurious tempo-index changes of the same magnitude as the negatives it must
beat. The objective prefers a permissive λ because it is being asked to separate two hypotheses that
differ by **less than the state space's own resolution**. `num_tempi=None` (every integer interval),
or drift negatives an order of magnitude larger, would make the discrimination real — **[UNVERIFIED,
untested]**.

### 5.3 Equivariance regularisation — **ADMISSIBLE, UNPROVEN [CONJECTURE]**

`f(stretch(x,s)) = f(x) − log₂ s` needs no labels, so it can use unlabelled audio. Rationale: it
would make behaviour on the well-populated 70–100 BPM region *determine* behaviour at 40 BPM.

**Honest status — corrected.** The previous entry ("two runs disagreed, differing only in an
anchoring probability; run-to-run variance exceeds the effect") is **wrong on its facts** and is
retracted (§8). What validation established:

- **Both cited runs (`eq_lam0`, `eqA_lam0`) ran with `lam_eq = 0.0`.** In
  `scripts/train_equivariance.py` the loss is `(sup + args.lam_eq * eq) / args.accum`, so the
  equivariance term contributed **zero gradient** (`out_b` computed and discarded). Neither run is an
  equivariance-training run; they are plain supervised fine-tunes on phase-vocoded wide-range
  stretched audio.
- Their divergence is **mechanistically explained, not noise**: without identity anchoring, s=1 has
  measure zero, so 100% of supervised frames are vocoder output and the model learns to follow the
  vocoder consistently (defect 0.062) while shifting off the real-audio domain (peak ΔF −0.092);
  with `p_identity=0.3` it sees a 30/70 domain mixture and becomes inconsistent (defect 0.290, worst
  at s=0.5: +0.848). This is a controlled domain-shift effect.
- **"Differing only in `--p-identity`" is false/[UNVERIFIED]**: `eq_lam0/fold0.json` records *no*
  `p_identity` key, i.e. it predates the flag and came from a different version of the script.
- `draw()` is **not** degenerate: identity is drawn per view and the guard
  `if abs(log2(s_b/s_a)) < 0.05: redraw s_b` prevents view collapse. One benign asymmetry: only view
  A carries the supervised loss.

**The one genuine equivariance run is `eqA_lam10`** (`lam_eq=10`, `p_identity=0.3`, fold 0, seed 0,
~420 used steps), and it is **negative on both readouts [MEASURED]**: equivariance defect
mean\|δ\| 0.279 → **0.304 (WORSE)**; SMC ΔF **−0.1098 peak-pick** (n=27, Wilcoxon p=1.8e−04) and
**−0.1227 DBN** (p=2.1e−04).

**Supported claim [MEASURED]:** *supervised wide-range vocoder fine-tuning significantly HURTS SMC F
under both decoders (−0.092 to −0.174, n=27, all p<1.4e−03), and the single equivariance-loss arm
neither reduced the defect nor recovered F.* **Unsupported:** any claim that equivariance training
reduces the defect. Status remains **[CONJECTURE]**, now with one negative datum against it.

**Decoder disclosure:** the DBN numbers in this subsection come from `make_dbn(min_bpm=55,
transition_lambda=100)` in the **JOINT** beat+downbeat state space via `dbn_beats(bp, dp, dbn)` —
*not* the beat-only state space used elsewhere in this spec, which has reversed signs before. Any
comparison table must state this.

**Caveat on the primary readout:** mean\|δ\| is a mean over per-`s` **medians** of a peak-picked
log₂ median-IBI, with **no n and no p**, and it can improve while beat tracking collapses (`eq_lam0`
reaches δ=−0.197 at s=2.0 while losing 0.128 F under the DBN). It must not be quoted as a
significance-bearing statistic without a per-track paired test.

**[UNVERIFIED] possible vanishing gradient through `soft_log_period`.** The `lam_eq=10` eq-loss
trace (0.72, 0.44, 0.36, 0.30, 0.27, …, 0.31) is nearly step-for-step identical to the `lam_eq=0`
trace (0.73, 0.47, 0.35, 0.35, 0.34, …, 0.43) — a weight-10 term did not make the quantity it
optimises fall any faster. Suspect `tau=30` softmax saturation over the ACF plus
`clip_grad_norm_=0.5`. **This must be checked (log the eq-only grad norm; sweep `tau`) before
concluding equivariance training cannot work** — per the standing rule, a negative result requires a
code-error hunt first.

**Scope:** single fold (fold 0), n=27 SMC tracks for F, 20 tracks for the defect table, one seed. No
replication.

**Known threat:** a ratio objective can be satisfied by reading vocoder artefacts instead of tempo.
Mandatory control: train on beat-destroyed audio; the objective must **fail** there.

---

## 6. Evaluation protocol

- **Decoder**: beat-only state space (P5), `min_bpm = 30` — 21% of SMC lies below 55 BPM, and
  excluding it collapses those tracks' CMLt to 0.0528.
- **Primary metric**: CMLt. Justified by P2 — F understates this intervention roughly fourfold,
  because F factorises over beats within ±70 ms while tempo is a sequence-level property.
- **Report both decoders** where a claim could reverse: peak-picking and DBN disagreed in *sign* on
  the fine-tuning result (−0.033 vs +0.003), which is how a calibration artefact was once
  misreported as model damage. A second instance found during validation: the s=1.50 equivariance
  median moves from −0.411 to −0.058 across peak-pick thresholds 0.3/0.5/0.7 (see §8/R-9).
  **Every reported mean must name its decoder — peak-pick vs DBN, and joint vs beat-only state
  space.** Note the DBN *costs* F relative to peak-picking on SMC (−0.0163, p=0.0089, n=217) and buys
  continuity, so F-only comparisons across decoders are misleading.
- **Fold honesty**: every track encoded and evaluated with the checkpoint that held it out.
- **Splits**: `q_θ` trains on corpus audio only; SMC is never trained on. Any parameter fitted on
  SMC (e.g. β) must be fitted on a train split and reported on a held-out split, **and labelled as
  SMC-tuned** — this is precisely the reviewers' objection and must not be papered over.

### Gates before anything ships

| Gate | Threshold | Rationale |
|---|---|---|
| ~~**G1**~~ **RETIRED 2026-08-22** *(weakened by §4.11, killed by §4.9's β-sweep — the 24.4% break-even is a β=5 artefact; at the operating β=0.2 it is 76.2% and the slope is 8.6× flatter. Do not gate anything with it, and do not quote "+0.007 CMLt per percentage point")* | ~~octave-error rate < 24.4%~~ | §4.9, §4.11 |
| **G1c** *(new, replaces G1 — cheap subset screen)* | any candidate tempo source must be **≥50% correct on the 82 SMC tracks where the tempo-blind beat-only DBN scores CMLt<0.5**. The beat-activation ACF scores 22.0% there (OR 0.02 vs the DBN's own successes, p=2.1e−28) and loses. Overall octave-error rate is **not** a valid screen. | §4.11 |
| **G1b** *(oracle-reference floor)* | pooled ΔCMLt should be read against the **constant per-track median-IBI oracle**: **+0.1684** (p=5.4e−12, n=217, independent script `diag_tempo_head_bracket.py`); the previously quoted +0.1733 is `exp_octave_degradation.py`'s own 0% arm, not an independent figure (§4.9). NOTE: that baseline is computed from ground-truth IBI, so it is a *ceiling reference*, not a test-time competitor. It captures **76%** of the full oracle's CMLt gain, which says the value is in getting the tempo CONSTANT right, not in time resolution. | sizes the target and tells us where the headroom is |
| **G2** | tempo-neutrality spread ≤ **448 nats/60 s** (the shipped emission's own baseline), not 0 | R2, Prop 2 |
| **G3** | β learned, not swept | R3 / P3 |
| **G4** | held-out CMLt gain, beat-only decoder, n=217, decoder named | §6 |
| ~~G5~~ | ~~applied at beat boundaries~~ — **RETIRED**: the `/i` form already is the per-beat form (R5) | — |
| **G6** *(new)* | any cross-tempo score comparison subtracts the R6 normalisation terms first | R6 — violating it produced the worst result measured (R-11) |

## 7. Open questions, honestly held

1. **Can `q_θ`'s octave errors be fixed?** — *superseding the retracted "30% accuracy" target.*
   Accuracy is now 46.1% with **zero bias** and the slow band solved (§4.6); the entire remaining
   problem is 54/217 tracks at a wrong metrical level. **Route (a) is no longer untested and no
   longer a failure**: an octave-aware training loss (`octbandlog`, `oct_w=50`, band=3) cuts corpus
   octave error 24–30% relative with acc10 *up*, 8/8 paired seeds, p=0.008 (**§4.15**). Whether that
   transfers to SMC and to CMLt is **[UNVERIFIED]** and is the single highest-value open experiment.
   Route (b), R6-corrected octave arbitration at decode time, remains untested (§8 R-11 shows the
   evidence is there). Note the ceiling constraint from §4.6: gating/re-weighting *this* head cannot
   exceed **+0.0476 CMLt** — improvement must happen at source.
   Three independent estimators fail identically on slow SMC (Beat This
   head bias −0.633, DeepRhythm −1.165, MERT zero-shot −0.948) — all read slow material as too
   fast, all near-unbiased at 70–90 BPM. Stretch-augmented training data is the current bet.
2. **Is the encoder capped?** Untrained embedding readout showed an octave collapse at s=2.0
   (−1.046), but that probe compared a *trained* decoder against an *untrained* readout and is not
   decisive. Needs a trained linear probe on frozen embeddings.
3. **Does equivariance add anything over plain augmentation?** Unresolved; see 5.3. Blocked on the
   `soft_log_period` vanishing-gradient check.
4. **Does any of this generalise off SMC?** Untested. λ and β fitted on SMC measure SMC-tuned
   decoding. A general claim requires fitting on the Beat This corpora and testing SMC untouched.
5. **What accuracy must `q_θ` reach?** **Open, and no longer even bounded** — see §8/P4. Needs an
   oracle-degradation experiment with **temporally correlated** noise (per-track bias, octave flips),
   not i.i.d. per-frame noise.
6. **Do the three implicit tempo channels in Corollary 1 matter in practice?** The `ceil`
   quantisation (448 nats/60 s) and the row-normalisation term (22.8–125.1 nats/60 s, sign-flipping
   with λ) were measured on constant input only. Their effect on real decoding is **untested**, and
   the λ-dependent sign reversal of the third channel means λ tuning and tempo bias are confounded.
7. **Does the P5 result survive a properly parameterised downbeat channel?** The loss is from
   `max(bp−dp, ε)` truncating 28.0% of beat-peak frames, a Beat This calibration artefact. A
   conditional `p(beat)·p(down|beat)` parameterisation is **untested** and might recover the joint
   arm entirely.
8. **Does P5 hold for other activation sources?** Only Beat This tested. Beat Transformer's heads are
   calibrated differently.
9. **Does a valid bar-pointer path construction change the contrastive optimum?** `beats_to_state_path`
   produces −inf-probability paths on 100% of tracks (§5.2b); untested.

---

10. ~~**Does the per-beat form (R5) change the sign?**~~ — **CLOSED.** The `/i` form is
    algebraically identical to per-beat application with beat-averaged `q` (R5). No over-counting
    exists, so it cannot be inflating the §4.6 split.
11. **Does subtracting the R6 terms make cross-tempo comparison valid?** All three are analytically
    computable from the state space. If it makes octave arbitration work, it is simultaneously a fix
    to the 2016 module and the missing piece of the extension. (Q6 asks whether the channels matter;
    this asks whether *correcting* them recovers a usable arbiter — R-11 shows the evidence is there
    under a systematic offset.)
12. **Does the >120 BPM regression generalise?** Measured only on SMC's 14 fast tracks. Requires
    ballroom/GTZAN before claiming the head is a general improvement — or an explicit statement that
    it is a slow-tempo-specialised component. Given the target is the slow blind spot, a stated
    trade may be acceptable; an unstated one is not.
13. **Is the stretched training data teaching slowness or the phase vocoder?** The mandatory
    beat-destroyed control (`--shuffle-audio`, implemented, never run) is outstanding. The
    equivariance fine-tune improved on *stretched*-slow while degrading on *genuinely*-slow — the
    signature of learning the stretch artefact. `q_θ` is trained on the same vocoder output, so the
    same doubt applies to §4.6.

---

## 8. Retracted claims

Claims removed from the body above, kept here with the reason, per the document's honesty convention.

### R-1. "Supplying a *noisy* tempo observation at 30.9% accuracy gives ΔCMLt +0.2234, indistinguishable from the perfect oracle" (was **P4**) — **RETRACTED as an accuracy-threshold claim**

The numbers are real but the experiment does not measure what it was read as measuring. The σ arms
apply **i.i.d. per-frame** lognormal noise (`noisy = true·exp(N(0,σ))`, redrawn every frame), and the
injection divides by beat length, so a beat's total tempo evidence is the *arithmetic mean over its
frames* of `−0.5((log g − log noisy_t)/0.15)²`. That expands exactly to
`−0.5((log g − mean_t log noisy_t)/0.15)²` minus a `g`-independent constant: **the decoder sees a
Gaussian of unchanged width 0.15 centred on the beat-mean of `log(noisy)`. The frame noise is exactly
averaged away.**

SMC beat lengths at FPS=50: median **36.6 frames** (p10 21.4, p90 60.1). Effective per-beat log error
SD is `σ/√L` = 0.0198 (σ=0.12) and 0.0413 (σ=0.25), versus the `log(1.1) = 0.0953` accuracy
tolerance — so **100% and 97.1% of beats land inside 10%**. The quoted "57.4% / 30.9% accuracy" is a
**per-frame** statistic and overstates the degradation by roughly `√36.6 ≈ 6×` in log-error.

The σ arms are therefore near-identical to the perfect oracle **by construction**, and the fact that
they score marginally *higher* (+0.2234 vs +0.2221) is a tell that the noise acts as **smoothing, not
degradation**. A real tempo head has **temporally correlated** errors (octave errors, whole-track
bias) that do not average out, so this experiment says nothing about the accuracy a learned head must
reach. Presenting it as "94–100% of the gain survives" invites reviewer attack.

**Replacement [MEASURED]:** a single **constant** tempo per track (median IBI, zero time resolution)
still buys ΔF +0.0343 (p=3.3e−04) and ΔCMLt +0.1684 (p=5.4e−12), i.e. 76% of the CMLt gain, n=217,
beat-only PyTorch DBN. That is a legitimate robustness statement about *temporal resolution*, not
about *accuracy*.

*Related, and NOT retracted*: a phase-leakage attack on the oracle was **negative** — displacing the
oracle step boundaries by a random sub-beat offset retains 96%/98% of the gain (ΔF +0.0588
p=2.2e−10, ΔCMLt +0.2169 p=5.5e−21), so the gain is driven by tempo magnitude, not by phase
information leaking through the step edges. The β=5 per-beat scaling was also verified correct: the
division by `state_interval_frames` means each beat contributes exactly one beat's worth of tempo
evidence regardless of tempo, so slow and fast hypotheses are not differentially rewarded, and the
log-softmax-normalised ratio to uniform means the transition model's tempo prior is not double
counted.

### R-2. "A drifted path outscores the truth; the prior is not tight enough to protect the true path from tempo wander" (was **Prop 3**) — **RETRACTED as a general claim**

True on `smc_001` (+24.5 nats) but false in general: truth beats drift on **83.5% of draws** (n=266
pooled, mean +274.3 nats, p=1.4e−28); drift wins on only 12–17% of draws (27.7% of tracks for at
least one of two draws). `smc_001` was a one-track selection and was load-bearing for §5.1's MLE
rejection argument, which has been correspondingly weakened.

### R-3. "The joint beat+downbeat *state space* hurts on SMC, reproducing Böck & Davies' joint-modelling explanation" (was **P5**'s mechanism) — **RETRACTED**

The bar-level state space is **free**: with `beats_per_bar=[3,4]` intact and the downbeat channel
made uninformative (`dp := bp/4`), the joint arm matches beat-only within noise (ΔF +0.0006 p=0.59,
ΔCMLt +0.0006 p=0.60, n=217). 100% of the loss comes from injecting Beat This's **real downbeat
activations**. The measured gap is unchanged; only the mechanism claim is withdrawn.

### R-4. "Two equivariance fine-tuning runs differing only in `--p-identity` gave opposite conclusions; run-to-run variance exceeds the effect" (was **§5.3**) — **RETRACTED**

Both runs had `lam_eq = 0.0` — the equivariance loss was multiplied by zero, so neither trained it.
They are supervised wide-range vocoder fine-tunes, and their divergence is a controlled domain-shift
effect (100%-vocoder vs 70/30 mixed supervision), not variance. They also did *not* differ only in
`--p-identity`: `eq_lam0` predates the flag. The genuine `lam_eq=10` arm was ignored by the original
claim and is cleanly negative. See §5.3.

### R-5. "Accumulated-evidence spread across tempi for the fractional rule is 0.0 nats" (was **Prop 2 / R2 / G2**) — **RETRACTED**

`ceil(i/obs_λ)` quantisation gives **448 nats/60 s** on constant input at `obs_λ=16`, violating the
spec's own R2 zero-nat test for the shipped emission. The fractional rule is ~3.7× better than
fixed-W D=15, not infinitely better. The 1671 and 5134 nat comparison figures remain **[UNVERIFIED]**
— no script producing them could be located.

### R-6. "Generative MLE costs −0.084 AMLt" (was **§5.1**) — **RETRACTED (numeric)**

Reproduction gives **−0.0350 AMLt, p=0.154 (not significant)**, n=87 held-out, beat-only PyTorch DBN.
The −0.044 AMLt figure belongs to the *hard-mined* λ=7.55 arm only and was incorrectly applied to all
three arms.

### R-7. "Hard mining found 118/129 tracks decoding wrongly" (was **§5.2**) — **[UNVERIFIED]**

Reproduction at λ=100 gives **115/129**. The 129 denominator is confirmed; 118 could not be verified
because the run log was not saved. Do not cite 118.

### R-8. "`β` is learned by gradient alongside `transition_lambda`" (docstring of `scripts/train_contrastive_dbn.py`, line 31) — **RETRACTED (false)**

`beta.grad is None` after a real backward pass. Only λ was ever optimised. **The script's docstring
must be corrected.**

### R-9. "The flat equivariance region [0.7, 1.2] coincides with the shipped augmentation range, and Beat This collapses outside it" (was **P1**) — **RETRACTED as stated**

Wrong on both ends: **s=0.70 lies *outside* the shipped range** `[0.833, 1.250]`, so it cannot
coincide with it, and both s=0.70 (p=1.3e−4) and s=1.20 (p=0.022) are significantly non-equivariant
by Wilcoxon signed-rank, n=40. **The only s tested that is statistically indistinguishable from
perfect equivariance is s=0.85** (p=0.13). The medians are insensitive to a 20–30% octave-failure
minority.

"Collapses outside it" overstates a **graded** degradation: octave-failure counts rise smoothly
(6 at 0.85 → 9 at 1.2 → 21 at 1.5 → 28 at 2.0) with no cliff at the augmentation boundary. The
inside/outside contrast is real but is a shift in proportion — 15/80 failures at s∈{0.85, 1.2} vs
34/80 at s∈{0.7, 1.5}, **Fisher p=0.0019**.

The median also conceals strong **bimodality**: at s=2.0, 7/40 tracks are still perfectly equivariant
(\|δ\|<0.1) while 19/40 sit at exactly one octave and 4/40 are worse than −1.4 (down to −1.807);
std=0.502, IQR [−1.026, −0.273]. So "−0.966 is one octave" is true of the **median only**, and the
defect is a **per-track binary octave switch whose mixing fraction varies with s**, not a continuous
magnitude that grows to 1.0. Finally, δ(1.00)=0.000 is **trivially true by construction**
(s=1.0 is the self-reference) and is not an observed data point supporting flatness.

*Not retracted*: the shipped augmentation range itself is **confirmed** —
`beat_this/launch_scripts/train.py:44` sets `tempo = {min: −20, max: 20, stride: 4}` in **percent
speed**, and `preprocess_audio.py:371` applies `stretch_factor = 1 + stretch/100` where >1 means
faster/shorter, so duration factors are `s ∈ [1/1.20, 1/0.80] = [0.833, 1.250]`. The reciprocal
mapping in the original claim was right. The librosa sense is also correct
(`time_stretch(y, rate=1/s)`; s=2.0 → duration ×2.000 exactly), so there is no sign error, and all
seven medians reproduce digit-for-digit.

*Scope limit*: δ is self-referenced to the model's own unstretched output, so it measures
**equivariance only** and is silent on absolute tempo correctness; and the 40 tracks are the *first*
40 of the fold-honest SMC list, not a random sample.

---

### R-10. "Octave-smearing `q_θ` will keep the +0.0494 on correct tracks while removing the −0.36 on octave-error tracks" — **REFUTED (my hypothesis, tested and wrong)**

Proposed 2026-08-21 as the fix for the octave catastrophe: spread `q_θ` across ×2 and ÷2 partners so
the head constrains tempo *modulo octave* and the emission picks the octave. Tested at three weights,
n=217, β sweep, beat-only decoder. ΔCMLt at β=5: **−0.1080** (w=0.3), **−0.1085** (w=0.6),
**−0.1289** (w=1.0) — all *worse* than unsmeared (−0.0971), monotonically worse in the smear weight.
At low β it decays to neutral (β=0.5, w=1.0: −0.0005, p=0.81) rather than turning positive.

**Why it was wrong, structurally:** handing the octave decision to the emission is *precisely what
the β=0 null already does*, so smearing cannot beat the null's own octave discrimination — while it
destroys the commitment that produced the +0.0494. The reasoning error was proposing a "fix" whose
ceiling is the baseline it was meant to improve on.

### R-11. "Choosing the octave by total Viterbi path score is a stronger arbiter than the emission's local preference" — **REFUTED (my hypothesis, and it violated R6)**

Decode under `h ∈ {0.5, 1, 2}` with `q_θ` fully committed, pick the highest total path score. Result,
n=217, β=5: **h=0.5 chosen on 150/217 tracks**; CMLt 0.3221 vs null 0.5685, **ΔCMLt −0.2464**
(p=5.1e−12) — the worst arm measured. It flipped tracks the head had already got right
(`smc_001` head 46.9 vs true 48.2 BPM, CMLt 1.000 → 0.000).

**Why it was wrong:** it compares likelihoods across tempo hypotheses in a model whose tempo-dependent
normalisation is uncorrected — a direct violation of **R6**, which was already recorded in the spec
when the experiment was proposed. The two dominant implicit channels (~448 + ~23 nats, both favouring
fast) are the same order as the score gaps being compared.

**What survives, and is worth pursuing:** the arbiter is *not* noise. On tracks where the head
predicted half tempo it correctly repaired them — `smc_093` CMLt 0.000 → 0.842, `smc_272`
0.000 → 0.934, `smc_060` 0.610 → 0.732. It carries genuine octave evidence sitting on a systematic
offset. **Correcting the R6 normalisation terms before comparing is the open experiment**, not a
refutation of octave arbitration as such.

---

### R-12. "Octave-aware tempo losses (`oct`, `fact`) fail the pre-registered bar" (was `scripts/sandbox_octave_loss.py`) — **RETRACTED for `oct` (FALSE NEGATIVE); UPHELD and STRENGTHENED for `fact`**

Recorded 2026-08-22. The `oct` penalty pushed probability off a **single** grid bin (step 0.0481
log₂) while the oct_err metric scores a **±3.12-bin** band, so it moved octave-band mass by 0.4%
(0.1469 → 0.1473) — a null of a loss that did nothing, not evidence against octave-aware training.
Compounded by `oct_w` hard-coded 20–100× too small, a dataset-ordered split that trained on
Ballroom-only and validated on GTZAN-only, and 2 unpaired seeds at 1200 steps against a 0.024 bar.
With a band penalty, a swept weight, a stratified split and 8 paired seeds at 3000 steps, the bar is
**passed on both splits with acc10 up, 8/8 seeds, p=0.008** (§4.15). Cost of the error: this is the
route §4.7 identifies as the only one upward, and it was recorded as closed. `fact` is now a
*stronger* negative — significantly worse on oct_err, 0/8 seeds, p=0.008.

### R-13. "§4.9's break-even octave-error rate of 24.4% gates `q_θ`" — **RETRACTED (β-artefact)**

Recorded 2026-08-22. §4.11 had already weakened this qualitatively. Re-running
`exp_octave_degradation.py` at the operating β=0.2 moves the break-even to **76.2%** and flattens the
slope 8.6× (−0.718 → −0.083). The gate is a property of injection strength, not of the estimator.
Gate **G1 is retired** and replaced by **G1c** (accuracy on the 82 tracks where the tempo-blind DBN
scores CMLt<0.5). The derived figure "+0.007 CMLt per percentage point" must not be quoted.

### R-14. "TTA and the tempo head are 40% redundant" (was §4.10) — **RETRACTED as unsupported**

Recorded 2026-08-22. Reported without a test. The paired interaction is CMLt −0.0092, 95% CI
[−0.0268, +0.0084], p=0.30 — spanning perfectly additive to fully redundant; n≈1602 is needed for
80% power and SMC has 217. On AMLt/AMLc the interaction is *positive* (super-additive), and the
verdict flips with β. The CMLt null for TTA on the beat-only DBN is unaffected and is confirmed by
exhaustion over 9 rate subsets.

### R-15. "'Ambiguity of metrical level' ranked first among the annotator tags" (was §4.8) — **RETRACTED (false)**

Recorded 2026-08-22. It ranks **second**. "Beat phase ambiguity" ranks first (7/43 vs 4/174, OR 8.26,
p=0.0014) and is the only tag of 20 surviving Bonferroni. The enrichment itself survives at FDR 10%
(q=0.087) but not Bonferroni (p=0.173, k=20). The a-priori *direction* holds; the "ranked first"
argument that licensed calling the result corroborative does not. Also retracted in §4.8: the
unreproducible n=187 cross-system row, and the closing "~55% annotator convention" sentence that
contradicted the same section's own corrected arithmetic (~15%).

## 9. Prior art this must cite

- **Krebs et al. 2015** — bar-pointer state space; λ by grid search on Cemgil accuracy, not ML.
- **Böck 2016** — the three-class geometric emission analysed in §1.
- **Böck & Davies 2020** — joint downbeat decoding hurts on SMC (source of P5; **note: our P5
  control shows the loss is from the downbeat *observations*, not from joint state-space decoding,
  so we do not in fact reproduce their "joint modelling" explanation — see §3 and §8/R-3**); tempo predicted
  jointly with beats; hop-size augmentation; and the extrapolation-outside-tempo-support argument
  (they targeted *fast*/HJDB, we target *slow*/SMC). Also, on SMC: *"may require the reformulation
  of the inference techniques used to recover the final outputs, rather than intervention at the
  point of training the networks."*
- **Sá Pinto 2023** — already published the tempo-range and decoder-oracle results; our novelty must
  be the mechanism, not the observation.
- **Quinton, ISMIR 2022** — equivariant self-supervision for tempo. Verify before writing §5.3.
