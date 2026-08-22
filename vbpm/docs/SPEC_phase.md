# VBPM — Stage P specification (`z = r`, phase only)

Companion to `docs/SPEC.md` (Stage 0, normative) and `docs/SPEC_stage1.md` (the full
`[m, φ]` target). Stage P is a rung **below** Stage 1: it isolates bar phase by holding
meter fixed. Where this document and `SPEC.md` disagree, §11 says so explicitly; nothing
here silently supersedes anything there.

## 0. Sources, and how much each is trusted

| source | status |
|---|---|
| `ELBO_for_DBN` (the paper) | target model. `git show 43ecf34:docs/ELBO_for_DBN.md`. `φ ∈ [0,2π)` is **bar** phase; a wrap is a downbeat. |
| `docs/SPEC.md` | normative for Stage 0. §12's expansion contract binds this document. |
| `docs/SPEC_stage1.md` | the stage above. Stage P must extend into it, not be a detour. |
| `VAEBPM_fin.pdf` | held by the user. §8.1.5/§8.3 read-out alternatives **unread**. Ask; do not reconstruct. |
| measurements in §10 | clean `vbpm/` pipeline, 2026-07-30 or later. Campaign-era numbers not admitted. |

---

## 1. Why Stage P exists

Stage 0 chose meter as the first latent. That choice has three consequences, all measured,
and together they say phase should have come first:

**Meter is nearly constant in the data; phase is uniform by construction.** `gtzan` is 93%
`m=4`, `hjdb` 100%, `beatles` 85%, `hainsworth` 90%. A latent that is 93% one value carries
almost no entropy, and Stage-0 runs measured it dead (KL ≈ 0.02). Bar phase has **full
entropy on every corpus** — a crop may begin anywhere in the bar, so `r` is uniform over
`{0..m−1}` whatever the meter distribution is. Phase is the variable with something to learn
on data we actually have.

**The meter-first choice built a dataset in which phase cannot be studied.** If the latent is
one `m` per crop, cutting crops at bar boundaries is the natural design — it makes `m_true`
well-defined as a median over complete bars. That is `make_crops`. And it sets `r = 0` in
~99% of crops (§10.1). So phase has been handed to the model for free since the beginning,
and no amount of work on meter would have surfaced that.

**Phase is the task.** `φ = 0` *is* the downbeat. Stage 0 predicts how many beats are in a
bar and never predicts which beats are downbeats, so it has never been able to be *wrong*
about placement. Stage P produces a deployable placement read-out.

Stage P does **not** answer "is the VAE encoding anything?" — it is exactly enumerable, like
Stage 0. It is the cheap, high-information rung that makes that question askable: it produces
the un-aligned dataset every higher rung needs, and it settles whether `h` carries phase at
all before weeks are spent on continuous `φ`.

| rung | latent | inference | answers |
|---|---|---|---|
| 0 | `m` | exact enumeration, K=3 | can the machinery be made to work? |
| **P1** | `r` (one per crop) | exact enumeration, R=4 | **does `h` carry phase at all?** |
| **P2** | `r_i` per beat, with slip | exact forward–backward | does a time index earn its cost? |
| 1 | `(m, r)` / `(m, φ)` | forward–backward / sampling | `SPEC_stage1.md` |

---

## 2. Scope

**In.** Bar phase as the only latent. Un-aligned crops. A downbeat-**placement** read-out and
metric. Exact enumeration (P1) and exact forward–backward (P2).

**Out.** Meter as a latent (held fixed, see below). Tempo. Beat finding. Continuous `φ`.
Sampling. All are higher rungs.

**The beat grid is given**, as at Stage 0.

### 2.1 Meter is held fixed — a declared, scoped violation of C3

`SPEC.md` C3 says `m` is never hardcoded to 4 **as policy**, because v1 hardcoded `M=4` in
read-out, emission and sawtooth and made meter causally inert for weeks. Stage P fixes
`m = 4`. This is permitted **only** under all four of the following, and an implementation
that drops any one of them is in breach of C3:

1. **Declared, not implicit.** `m` is a named constant of the stage, never inlined.
2. **The corpus is restricted to `m_true == 4` crops.** The model is then not knowingly wrong
   on the data it sees. Measured: this retains **9,932/18,902 = 52.5%** of crops, and
   **93% / 100% / 85%** of `gtzan` / `hjdb` / `beatles` (§10.2). An earlier draft said "~64%
   and all of" those corpora; both were wrong, and this is condition 2 of the C3 escape, so
   the escape rested on a wrong number. It still holds at 52.5%, with a smaller corpus.
3. **`m` is a parameter of every function that uses it**, so P2 → Stage 1 restores it as a
   latent without rewriting anything around it (`SPEC.md` §12).
4. **An exit criterion exists**: Stage P is complete when §8.3's criteria are graded, and
   meter returns as a latent at Stage 1 regardless of the outcome.

---

## 3. Conventions

`SPEC.md` C1–C6 are inherited and binding. Added:

| # | convention | why |
|---|---|---|
| P1 | Phase is **bar** phase and a wrap **is** a downbeat. Never beat phase. | A prior implementation silently used beat phase, making the meter latent inert. |
| P2 | Crops are **not** bar-aligned. Any path assuming `y[0] == 1` is wrong. | §10.1: ~99% of Stage-0 crops start on a downbeat. |
| P3 | The read-out reports **placement** — which beats are downbeats — not a class label alone. | Stage 0 could not be wrong about placement because it never predicted it. |
| P4 | Every reported figure carries **raw accuracy and the confusion matrix**; per-class recall from `n < 50` carries an interval. | §10.4: a degenerate predictor scored 0.422 balanced with 0.239 raw. |
| P5 | Any claimed effect states the **run-to-run margin** it clears. | §10.3: identical re-runs move ±0.08. |
| P6 | `r_true` is derived from annotations and is **scoring-only**; it never reaches the deployable path. | C2, restated for the new label. |

---

## 4. The model

### 4.1 Latent

**P1 — static.** One offset per crop:

    r ∈ {0, …, m−1},   m = 4   (§2.1)

`r` is the **index of the first downbeat within the crop**: beat `i` is a downbeat iff
`i ≡ r (mod m)`. This matches `emission_counts`' `slots = arange(r, n, m)` and §6.2's
`r_true`. `R = m = 4` states.

> An earlier draft called `r` "the bar pointer at the crop's first beat", which is the
> NEGATION of the formula beside it — a pointer `r` at beat 0 makes beat `i` a downbeat when
> `i ≡ −r (mod m)`. The two agree only at `r = 0`, i.e. on ~99% of Stage-0 crops (§10.1),
> which is exactly why the contradiction survived being written down. The formula is
> authoritative; the prose was wrong.

**P2 — per beat, with slip.** `r_i` for `i = 1..n`:

    r_i = (r_{i−1} + 1) mod m     with probability 1 − ε_hold − ε_skip
    r_i = r_{i−1}                 with probability ε_hold   (a bar is locally long)
    r_i = (r_{i−1} + 2) mod m     with probability ε_skip   (a beat is dropped/pickup)

**Two free parameters, not one**, and normalisation is asserted. An earlier draft wrote
`1 − ε` on the advance while also naming `ε_hold` and `ε_skip`, which sums to 1 only under an
unstated `ε = ε_hold + ε_skip`.

> **P2 is not optional dressing, and P1 alone does not motivate it.** With a *deterministic*
> advance and fixed `m`, `r_i` is a deterministic function of `r_1`: the sequence carries no
> information beyond the single offset, forward–backward provably reduces to a 4-way argmax,
> and a time index buys **nothing**. Slip is what makes the recursion non-trivial. An
> implementation of P2 without slip has implemented P1 with extra steps, and a test must
> assert the reduction (§9).

> **P2's loss is currently unwritable and P2 is therefore BLOCKED.** §4.5 defines an encoder
> for the static P1 latent only; no encoder over the chain `r_{1:n}` is specified anywhere, so
> there is no P2 objective for `φ` to receive gradient from. P2 exact *inference* is
> specifiable and testable today; P2 *training* is not. Specify the chain encoder before
> implementing P2.

`ε_hold, ε_skip` are learned, and expected to be small. They are the model's statement about
annotation noise, pickup bars, and mid-crop meter change.

### 4.2 Observation

Unchanged from `SPEC.md` §4.2: `y ∈ {0,1}^n`, per-**beat** downbeat indicator on the given
grid.

### 4.3 Emission

Unchanged from Stage 0, with `r` promoted from marginalised nuisance to latent:

    p_θ(y_i | r_i) = Bern(y_i ; sigmoid(α))   if r_i == 0
                   = Bern(y_i ; sigmoid(β))   otherwise

**θ = {α, β}, the same two scalars as Stage 0, deliberately frozen in form.** The emission
must not gain capacity at this stage: any change in performance is then attributable to the
latent and to `ψ`, not to a richer decoder.

**The emission does not see `h`** (`SPEC.md` §4.3): `b ⊥ h | z`, and giving the decoder `h`
opens a route to fit `y` directly and leave the latent unused.

> **`SPEC.md` §4.3's marginalisation consequences do not survive here and must not be
> inherited.** Cyclic-shift invariance and "downbeat phase is unobservable" are consequences
> of marginalising `r` uniformly. Once `r` is the latent, phase **is** observable and
> off-by-one is **not** equivalent. See §11 A3 — this changes the meaning of a currently
> passing frozen test.

### 4.4 Conditional prior — `p_ψ(r | h)` ← the deployable path

    p_ψ(r | h) = softmax_r [ logits_ψ(h) ],   logits_ψ(h) ∈ R^R

Reads `h` only (C2). **This is the component Stage P exists to test**, because of a property
that can be checked before any training happens:

> **Shift-invariant heads are EXPECTED to score near chance. This is an empirical
> expectation, not a theorem, and an earlier draft wrongly asserted it as proof.** Two
> reasons it is not a proof: `AutocorrHead`'s masked FFT autocorrelation is *linear*
> (zero-padded, normalised by `T − lag`), so boundary effects leak position at ~0.2% of logit
> scale; and more fundamentally, a different `r` means a **different audio window**, not a
> cyclic shift of one signal, so shift-invariance is not even the governing property. Measure
> it; do not assert it, and do not write a test that asserts exact invariance.
>
> | head | phase-capable? | mechanism |
> |---|---|---|
> | `mean_max` | **no** | mean and max over `t` are permutation-invariant |
> | `peak_summary` | **no** | all ten dims are peak rates, ratios, and autocorrelation — shift-invariant |
> | `AutocorrHead` | **no** | projects, autocorrelates, pools over the lag axis; absolute position does not survive |
> | `TransformerPrior` | **yes** | positional encoding + query token |
>
> `AutocorrHead` is the load-bearing case: it is our best in-fold *meter* arm (0.561 ALL-CV)
> and it is provably blind to phase. **If it scores above chance, something is leaking**, and
> that must be diagnosed before any other Stage-P number is believed.

`ψ` supplies a **per-beat audio potential** `ψ(r_i | h_i)`; at P1 the crop's logits are that
potential summed along each offset comb. Two consequences, both load-bearing:

- the P2 → P1 reduction at `ε = 0` then holds **in `ψ` as well as the likelihood**, so "P2
  reduces to P1" is a claim about the whole model rather than the emission alone;
- the read-out is shift-**equivariant**, which §8.4's control requires. A corollary: **`ψ`
  must carry no per-offset bias** — §1 says `r` is uniform by construction, so a bias models
  nothing and breaks equivariance.

The reducer is deleted at P2 structurally. At P1 it is deleted empirically, by the table
above — and note that `logits_ψ(h)` at P1 IS crop-level, so an earlier draft's claim of "no
pooled crop-level summary anywhere" was false at P1.

### 4.5 Encoder — `q_φ(r | h, y)`

Structured, following `SPEC.md` §4.5: built in the same shape as the exact posterior so any
gap is a statement about the encoder rather than about what it can see.

    q_φ(r|h,y) = softmax_r [ logits_ψ(h) + c · log p_θ(y|r) ]

`c` learnable; at `c = 1` this is the exact posterior. Training only.

### 4.6 Objective

    ELBO(h,y) = E_q[ log p_θ(y|r) ] − KL( q_φ(r|h,y) ‖ p_ψ(r|h) )

**P1**: exact sum over `R = 4` terms. No sampling, no seed sensitivity, bit-reproducible.
**P2**: exact forward–backward over `R` states, `O(n·R²)`.

Identities that must hold and are testable (`SPEC.md` §4.6):
- `ELBO ≤ log p(y|h)` for any `q`; equality **iff** `q` is the exact posterior.
- Slack `= KL(q ‖ p(r|y,h))` — the **reverse** KL.
- **P2 with `ε = 0` reduces exactly to P1** (§4.1).
- **P1 with `ψ` uniform and `r` marginalised reduces exactly to Stage 0's emission**, to
  numerical tolerance. This is the formal statement that Stage P *extends* Stage 0.

### 4.7 Parameter sets

| set | object | reads | deployable? |
|---|---|---|---|
| **θ** | emission — `{α, β}` | `r` | yes |
| **ψ** | prior `p_ψ(r\|h)` (+ slip `ε` at P2) | `h` | **yes — the deployable path** |
| **φ** | encoder `q_φ(r\|h,y)` — `{c}` | `h`, `y` | no |

**Every parameter must receive gradient, asserted by reading gradients**, not by checking the
optimiser. Measured precedent: 26 tensors / 45.3% zero-gradient in one configuration, 41 /
50.88% in another, with `ψ` in the optimiser both times. A brand-new implementation
reproduced this failure mode within an hour on 2026-08-03 (zero-initialised output layers
made `d(loss)/d(body)` exactly zero at step 0).

---

## 5. Training

- **P1 is deterministic for the closed-form/linear path only.** The attention-based head —
  the one P-2 is measured on — is NOT reproducible run to run (±0.08, §10.3). An earlier
  draft claimed P1 is deterministic outright. Assert on a distribution over draws, never a
  draw (`SPEC.md` §9).
- **Crop length must be checked, not assumed.** Measured: at 256-frame crops the ELBO
  *prefers* a metronome (truth-coast +12.3 at 256/512 against −46.6 at 1500). A phase model
  trained on crops that are too short will correctly converge to something useless.
- **Free-bits is not permitted** as a fix for a dead latent. It is a one-way door and it
  zeroes prior-side gradients below the floor. If `r` is dead, find out why.

---

## 6. Data

### 6.1 Un-aligned crops — mandatory

`vbpm/data.py:make_crops` cuts at bar boundaries, so `r = 0` in ~99% of crops (§10.1).
Required: crop starts sampled at arbitrary **beat** offsets, uniform over the bar, **and
each crop must contain exactly `n = CROP_BARS · m` beats, i.e. `n ≡ 0 (mod m)`.**

> **The whole-bar requirement is not tidiness, it is what makes chance equal `1/m`.** The
> number of downbeat slots is `len(range(r, n, m))`, which DEPENDS on `r` whenever
> `n % m ≠ 0`: verified, `n=13 → [4,3,3,3]`, `n=14 → [4,4,3,3]`, `n=15 → [4,4,4,3]`. In every
> such case a purely shift-invariant COUNT summary identifies `r` at **0.500**, twice chance,
> with no leak at all — and P-0 would fire on clean data. Assert `n ≡ 0 (mod m)` at the point
> crops are cut, and derive chance from the count partition rather than hardcoding `1/m`.

`vbpm/data.py` already has `extract_crops_unaligned`, added 2026-08-03 on branch
`worktree-agent-ab514a763c8bd8872` (commit `a1aa115`) and **purely additive** — the existing
functions are byte-identical, so Stage 0 is unaffected. It is unmerged and unverified against
this document; treat it as a candidate, audit it, do not assume it.

**This is the only change that makes phase learnable rather than vacuous.** Everything else
here is inert without it.

> Expect scores to go **down** relative to Stage 0: un-aligned crops are a harder task than
> the one every Stage-0 number was measured on. A Stage-P number is not comparable to a
> Stage-0 number, which is why §8.2 requires the Stage-0 baseline re-run on un-aligned crops.

### 6.2 `r_true`, and the crops where it does not exist

    r_true = index of the first downbeat within the crop  ∈ {0, …, m−1}

Scoring-only (P6). **`r_true` is well-defined only when the meter is constant across the
crop.** Measured (§10.1): on ~2% of crops no single offset fits — the meter varies within the
crop. Those crops must be **excluded from scoring and counted as an exclusion**, never
assigned a fictional label. At P2 they become the interesting cases (slip), not the excluded
ones.

### 6.3 `h`

Frozen pretrained features. C6 stands: do not collapse to `[T,2]` as policy.

> **Rich `[T,512]` features are not comparable across frontend checkpoints** (§10.5). Stay
> within one checkpoint, use the semantically pinned 2-dim path, or align explicitly — else
> the number measures feature alignment as much as phase.

### 6.4 Splits

Fold-honest, pooled out-of-fold, `gtzan` test-only, **per-dataset reporting mandatory**
(§10.4).

### 6.5 Synthetic bench

Stage P needs its own, with **known phase**: crops of known `r_true`, including (for P2)
slip and a mid-crop meter change. **A phase model that cannot recover known phase from clean
synthetic input is broken, and this must be established before any real-data number exists.**
The synthetic bench cannot substitute for real-data controls — it is balanced by
construction and the real corpus is not.

---

## 7. Deployment

Only `h` is available. Predict `r̂ = argmax_r p_ψ(r|h)` (P1) or Viterbi over `r_{1:n}` (P2),
then:

    predicted downbeats = { beat i : i ≡ r̂ (mod m) }        (P1)
                        = { beat i : r̂_i == 0 }             (P2)

> ⚠️ Tutorial §8.1.5/§8.3 read-out alternatives A–D are **unread** (`SPEC.md` §7). Ask the
> user for `VAEBPM_fin.pdf` before choosing a read-out; do not reconstruct from notes.

---

## 8. Evaluation

### 8.1 Metrics

- **Downbeat placement F on the given beat grid** — primary. The grid is given, so a
  predicted downbeat is correct iff it lands on the annotated downbeat beat. No tolerance
  window.
- **Offset accuracy** — `r̂ == r_true`. Chance is `1/m = 0.25` **only when `n ≡ 0 (mod m)`**
  (§6.1); otherwise it is up to 0.50. Chance must be **derived from the count partition**, in
  code, never hardcoded.
- **NLL of `y`** under the deployable path.
- Raw accuracy and confusion matrix beside every figure (P4), per dataset.

### 8.2 Baselines — all of them, or the result is uninterpretable

1. **Uniform-`r` null** — Stage 0's marginalised emission. Beating this is the minimum claim
   that phase is being used at all.
2. **Stage 0 re-run on un-aligned crops** — the only honest predecessor.
3. **Peak-picking on `h`** — the non-latent baseline, and the one that is genuinely
   phase-capable, since a peak has a position.
4. **Majority-`r`** — collapse detector.

### 8.3 Pre-registered criteria

Fixed **before any number exists**. Margins must clear the ±0.08 run-to-run band (P5) and be
reported per dataset.

| # | claim | criterion |
|---|---|---|
| **P-0** | the leak detector | **`AutocorrHead` and `peak_summary` score at the DERIVED chance level (§8.1) on offset accuracy, within a 95% Wilson interval.** Materially above ⇒ a leak; stop and diagnose. Expectation, not proof (§4.4) |
| P-1 | phase is recoverable at all | on the synthetic bench with known phase, `TransformerPrior` offset accuracy ≥ 0.95 |
| P-2 | `h` carries phase | on real data, `TransformerPrior` beats chance by ≥ 0.10 pooled and on ≥ 4 of 6 CV corpora |
| P-3 | the model beats the null | placement F beats the **majority-`r`** and **random-`r`** nulls by ≥ 0.10. (An earlier draft named the uniform-`r` null, which by construction emits NO placement and cannot be scored on F.) |
| P-4 | ~~the latent is used~~ | **WITHDRAWN — incoherent as written.** At Stage P the prior IS the deployable path (C2), so posterior→prior degradation measures the train/deploy gap, where a large value is the failure mode, not evidence of use. Needs redefinition before it can be graded; do not implement it. |
| P-5 | the time index earns its cost (P2 only) | P2 beats P1 on placement F, **and** fitted `ε > 0` with a likelihood-ratio margin |

**P-0 is the one to run first.** It costs nothing, it is falsifiable, and every other number
is worthless if it fails.

### 8.4 Required controls

- **Leakage** — deployable path never reads `y` (C2), asserted.
- **Shuffled-`r_true` null**, scored **held out**.
- **Shift consistency** — cyclically shift a crop's beat window by one beat; `r̂` must move by
  exactly one. A model that ignores the shift is not reading phase.
- **Gradient audit** — read gradients for every parameter (§4.7).
- **Degenerate-predictor check** — raw accuracy and confusion beside every balanced figure.

---

## 9. Testing

`tests/` is **frozen** and is Stage 0's. It must not be edited. Stage-P properties that
conflict with Stage-0 properties are stage-scoped, not violations — notably
`EQUIVALENT = {"downbeat_off_by_one"}` in `tests/mutants.py`, which `test_equivalent_mutant_survives`
requires to **survive**. At Stage P that mutant must be **killed** (§4.3). Stage P therefore
gets its own suite in a new directory, with its own registry.

The method is `SPEC.md` §9's, unchanged: properties parameterised over an implementation, run
against **an oracle that must pass**, against **corruptions that must be caught**, and against
**provably-equivalent corruptions that must survive**. "Passes ⟹ proper" cannot come from more
assertions; it comes from fixing a set of wrongnesses and proving each is caught.

**Build the oracle before the model** — a non-trained reference producing correct values for
every module given oracle inputs, as `tests/subject.py` does for Stage 0.

Required unit checks before any model claim:
- P2 with `ε = 0` reduces to P1, exactly.
- P1 with uniform `ψ` and `r` marginalised reproduces `Stage0.emission_logp_all`, to `<1e-9`.
- Forward–backward against brute-force enumeration for `n ≤ 8`, exactly.
- `ELBO ≤ log p(y|h)` numerically, on the same model.
- Shift consistency (§8.4) on synthetic crops with known `r_true`.

Candidate **equivalent** mutants that must survive: a consistent relabelling of `r` states
together with the emission; a global phase offset of exactly one bar. Candidate mutants that
must be **killed**: off-by-one in `r`; `argmax` over the emission instead of the posterior;
ignoring `h`; a shift-invariant summary substituted for `ψ`.

---

## 10. Findings that constrain Stage P

**§10.1 — crops are bar-aligned; `r_true` is ill-defined on ~2%.** Over all 18,902 Stage-0
crops: `r = 0` fits perfectly on **18,530 (98.03%)** and is the best-fitting offset on
**18,758 (99.24%)**. The residual ~2% are crops whose meter varies internally, so no single
offset fits. (An agent reported 18,902/18,902 = 100% on 2026-08-03; that was re-measured and
is wrong — it conflates `y[0] == 1` with "a constant-`m` offset fits".)

**§10.2 — the `m = 4` restriction.** Class shares: `asap` 35/31/34%, `rwc` 45/26/29%,
`ballroom` 0/30/70%, `beatles` 7/8/85%, `hainsworth` 3/7/90%, `gtzan` 1/6/93%, `hjdb` 0/0/100%.
Restricting to `m_true == 4` retains **9,932 of 18,902 crops = 52.5%** (`asap` 34%, `rwc`
29%, `ballroom` 70%, `beatles` 85%, `hainsworth` 90%, `gtzan` 93%, `hjdb` 100%).

**§10.3 — run-to-run nondeterminism is ±0.08.** Identical code and seed: `linear` reproduces
exactly; FFT and attention arms do not (CUDA reductions). Effects below this are noise.

**§10.4 — balanced accuracy is treacherous.** `gtzan` is 19/120/1838 crops for `m` = 2/3/4;
a degenerate always-2 predictor scored **0.422 balanced with 0.239 raw**, carried by 18/19 on
a 19-crop class. `hjdb` is 100% `m=4`, so any accuracy there is vacuous.

**§10.5 — rich features are checkpoint-fragile.** Under a checkpoint swap on identical songs
with an identical trained model: `tf512` −0.188, `autocorr` −0.162, `tf512n` −0.205, while the
pinned 2-channel `linear` **+0.168**. Caveat: the swapped checkpoint had trained on those
songs, so the leak biases in the swap's favour and is the same order as the drops. Supported
claim: *512-dim readers are checkpoint-fragile, 2-channel readers are not.*

**§10.6 — the downstream stack is not the bottleneck.** Synthetic-`h` control: `asap` 0.988,
ALL-CV 0.991. Read with care — the bumps sit at the annotations that define the labels, so it
proves the crop/label/prior/fit chain is sound and **cannot** separate "the frontend is deaf"
from "the annotation's metrical level differs from the audio's".

**§10.7 — `asap` fails under two independent frontends.** Trained `asap`-only, `[T,2]` arms.
Beat This: linear 0.335, autocorr 0.468, tf2 0.533. Beat Transformer (which never trained on
`asap`): linear 0.335, autocorr 0.430, tf2 0.489. The reducer sits at the 0.333 floor for
**both**; trained heads reach 0.43–0.53. So the information is in the activations and the
hand-built summary cannot see it.

**§10.8 — short crops make a metronome optimal.** Truth-coast +12.3 at 256/512-frame crops
against −46.6 at 1500.

**§10.9 — a Stage-0 baseline is invalidated.** `experiments/stage0_downbeat_decode.py`'s
`grid-oracle` arm was scored on data where `r = 0` is always correct. It is not a phase result.

---

## 11. Amendments to `docs/SPEC.md` this stage requires

Proposals. This document does **not** edit `SPEC.md` or `tests/`.

| # | clause | required change |
|---|---|---|
| A1 | §4.3 marginalisation consequences | scope to Stage 0: cyclic-shift invariance and unobservable downbeat phase follow from marginalising `r`, and both end when `r` becomes the latent. |
| A2 | `tests/mutants.py` `EQUIVALENT = {"downbeat_off_by_one"}` | **ACCEPTED 2026-08-03 (user).** Stage-scoped: the mutant is equivalent at Stage 0 and must be **killed** at Stage P. `tests/` stays frozen and unedited; Stage P carries its own registry. The scoping note is applied at `SPEC.md` §4.3. |
| A3 | §4.4 the reducer | Stage-0 scaffolding with a scheduled deletion, not a component with a future. §4.4 here gives the falsifiable reason. |
| A4 | C3 (`m` never hardcoded) | permit a **declared, scoped, corpus-restricted, parameterised** fixed `m` for a staging rung, under §2.1's four conditions. |
| A5 | §6 crops | record that Stage-0 crops are bar-aligned (§10.1) — a property of the Stage-0 dataset, not of the task. |
| A6 | §2 "beat tracking arrives with `φ` at Stage 1" | inconsistent with §1's ladder (tempo is Stage 2); the beat grid stays given through Stage 1. |

`SPEC.md` §12's expansion contract is satisfied: `m` remains a parameter everywhere (§2.1),
`z` gains a time index at P2, and the θ/ψ/φ split, C1–C6, the deployable-path rule and the
evaluation controls all survive.

---

## 12. Open questions

Blocking:

1. ~~**A2** — does the frozen suite's equivalent-mutant requirement get stage-scoped?~~
   **RESOLVED 2026-08-03**: yes, stage-scoped. See `SPEC.md` §4.3.
2. **Read-out choice** — tutorial §8.1.5/§8.3 unread (§7).

Not blocking:

3. Whether `extract_crops_unaligned` (unmerged, §6.1) is correct. Audit, do not assume.
4. Whether one offset per crop is the right granularity, or whether the ~2% ill-defined crops
   (§6.2) argue for going straight to P2.
5. Whether `asap`'s residual failure is frontend quality or metrical-level disagreement
   (§10.6, §10.7). Unresolved, and it bounds what any rung can achieve on the largest corpus.
