# VBPM — Stage 1 specification (`z_t = [m_t, φ_t]`)

Companion to `docs/SPEC.md`, which specifies Stage 0 and remains normative for everything
this document does not override. Where the two disagree, the disagreement is called out
explicitly in §12; nothing here silently supersedes anything there.

## 0. Sources, and how much each is trusted

| source | status |
|---|---|
| `ELBO_for_DBN` (the paper) | the target model. §2–§6 define `z_t = [m_t, φ_t, φ̇_t]`. Recoverable via `git show 43ecf34:docs/ELBO_for_DBN.md`. |
| `docs/SPEC.md` | normative for Stage 0. §11 states the target, §12 the expansion contract this document must satisfy. |
| `VAEBPM_fin.pdf` (professor's tutorial) | held by the user, not by this document. §8.1.5/§8.3 read-out alternatives A–D are **still unread** and are Stage-1 material (`SPEC.md` §7). Ask before reconstructing them. |
| measurements in §10 | run in the clean `vbpm/` pipeline on or after 2026-07-30. Campaign-era numbers are **not** admitted. |

---

## 1. Why Stage 1 exists

Stage 0 answered "can the machinery around one discrete latent be made to work?" It cannot
answer the question the project actually cares about, and the reason is structural, not a
matter of tuning:

**Stage 0 does no variational inference.** Its `q` is `softmax(prior + c · emission)`, and
the exact posterior is `softmax(log prior + emission)`. At `c = 1` the two coincide: the
bound is tight, the KL is zero, and the ELBO is computed by exact enumeration over `K = 3`.
`SPEC.md` §4.6 states this as a defining property. So every failure mode the VAE literature
is about — posterior collapse, the amortisation gap, the KL trade-off — is **not load-bearing
at Stage 0**, and Stage 0 passing tells us nothing about whether they are handled.

**Stage 0's latent is nearly observed.** The emission reads `y`, the per-beat downbeat
indicator. Knowing which beats are downbeats all but determines beats-per-bar. The model is
not discovering latent structure; it is fitting a map to something the training signal
already hands it.

**Stage 0's deployed object is a classifier.** `predict` takes `argmax` of the prior logits;
the emission never runs at deployment. And that classifier is currently beaten on held-out
transfer by a ten-dimensional hand-built summary feeding a linear layer (§10.1).

Stage 1 is the first point where the answer to "is the ELBO objective modelling anything of
value?" becomes measurable. `SPEC.md` §12 says so in its own words: *"Stage 1 requires
sampling and reparameterisation."*

The staging discipline, restated:

| stage | latent | what it adds |
|---|---|---|
| 0 | `m` | discrete latent, conditional prior, encoder, exact ELBO — `SPEC.md` |
| **1A** | `+ r_{1:n}` | time-indexed latent, bar-pointer recursion, **exact** forward–backward. The reducer dies here. **This document.** |
| **1B** | `+ φ_{1:T}` | continuous circular latent, von Mises, reparameterised sampling, approximate posterior. **This document.** |
| 2 | `+ φ̇` | tempo latent; the beat grid stops being given |

**1A and 1B are one deliverable, not two options.** This is the load-bearing design decision
of this document and §11 defends it: 1A is the exact-inference reference, 1B is the VAE, and
the *gap between them* is the only way to separate "the model is wrong" from "the inference
is wrong" with the model class held fixed. Building 1B alone produces a number nobody can
interpret. This project has already spent months failing to separate those two things.

---

## 2. Scope of Stage 1

**In.** A latent with a time index. Bar phase, discrete (1A) and continuous (1B). A
transition kernel with meter gated on bar crossing. Exact forward–backward (1A).
Reparameterised sampling and an approximate posterior (1B). A downbeat-**placement**
read-out and metric. Un-aligned crops (§6.1) — mandatory.

**Out, deliberately.** Tempo `φ̇` as a latent. Beat *finding*. Both are Stage 2.

**The beat grid is still given.** Stage 1 is handed beat times, exactly as Stage 0 is. The
frame-level phase *rate* is therefore determined by the given grid and `m`; what is latent is
the phase *offset* and its drift. Beat tracking needs `φ̇` and is Stage 2.

> **This contradicts `SPEC.md` §2**, which says *"Beat tracking arrives with `φ` at Stage 1."*
> That sentence is inconsistent with `SPEC.md` §1's own ladder, which puts the tempo walk and
> phase advance at Stage 2 — you cannot track beats without a tempo. One of the two has to
> give. This document resolves it by keeping the grid given at Stage 1; see §12 for the
> amendment that has to be accepted or rejected before implementation starts.

**The reducer is deleted, not improved.** `vbpm/reducers.py` exists for exactly one reason:
`h` is `[T,D]` with variable `T`, the Stage-0 prior needs a fixed-size input, and `s(h)`
bridges them. Its own docstring concedes it is not trained (*"s(h) is a per-crop constant"*).
Once evidence enters per frame through a recursion there is nothing to summarise. An
implementation that keeps a pooled crop-level summary anywhere on the generative path has not
done Stage 1.

---

## 3. Conventions

`SPEC.md` §3's C1–C6 are inherited unchanged and remain binding. C3 in particular gets
sharper here: with `φ` present, beats-per-bar is what converts phase to downbeats, so a
hardcoded 4 is not a shortcut but a deletion of the latent.

Added, and binding:

| # | convention | why |
|---|---|---|
| C7 | Phase is **bar** phase: `φ ∈ [0, 2π)`, and a wrap **is** a downbeat. Never beat phase. | `ELBO_for_DBN` §11 is explicit and a prior implementation silently used beat phase, making the meter latent inert. |
| C8 | Crops are **not** bar-aligned (§6.1). Any code path that assumes `y[0] == 1` is wrong. | Measured: 3,971/4,000 Stage-0 crops start on a downbeat, so phase is currently handed to the model for free. |
| C9 | A phase read-out reports **placement** — which beats are downbeats — not only `m`. | Stage 0 could not be wrong about placement because it never predicted it. Adding a latent without a read-out that can expose it repeats the Stage-0 defect. |
| C10 | Every reported figure carries **raw accuracy and the confusion matrix** beside any balanced figure, and per-class recall from a class with `n < 50` carries an interval. | Measured: a degenerate always-2 predictor scored 0.422 balanced with 0.239 raw on gtzan, carried by an 18/19 recall on a 19-crop class (§10.3). |
| C11 | Any claimed effect must clear the **run-to-run** margin, stated with the claim. | Measured: identical re-runs of the FFT/attention arms move balanced accuracy by ~±0.08 (§10.2). |

---

## 4. The model

### 4.1 Latent

**Stage 1A — discrete, on the beat grid.** For a crop of `n` beats, indexed `i = 1..n`:

    z_i = (m_i, r_i),   m_i ∈ values = (2,3,4),   r_i ∈ {0, …, m_i − 1}

`r_i` is the bar pointer: which beat of the bar beat `i` is. The state space is the valid
pairs, `Σ_m m = 9` states for `values = (2,3,4)`.

**Stage 1B — continuous, on frames.** For a crop of `T` frames, `t = 1..T`:

    z_t = (m_t, φ_t),   φ_t ∈ [0, 2π)   BAR phase (C7)

`φ` is the continuous relaxation of `r`: `r` is what `φ` reads out to at beat times. What
`φ` buys over `r` is **drift within a crop** — expressive timing, pickups, a bar that is
locally long — which `r`'s deterministic advance cannot express.

> **Identifiability warning, stated up front.** `φ` is observed only through which *beats*
> carry downbeats (§4.3), so its continuous part is weakly identified: many `φ` trajectories
> produce the same `y`. Expect the posterior over `φ` to be broad between beats and sharp at
> them. An implementation that reports a confident `φ` between beats is reporting its prior.
> This is the most likely place for Stage 1 to look like it works when it does not, and §8.4
> exists to catch it.

### 4.2 Observation

Unchanged from `SPEC.md` §4.2: `y ∈ {0,1}^n`, per-**beat** downbeat indicator on the given
grid, `y_i = 1 ⟺ beat i is a downbeat`.

> Not the paper's `b`. `ELBO_for_DBN` §5.4's `b_t ∈ {0,1}` is a per-**frame beat** indicator.
> Ours is a per-**beat downbeat** indicator. `SPEC.md` §11 already records this departure;
> it persists at Stage 1 because the beat grid is still given (§2). It closes at Stage 2.

### 4.3 Emission

**1A.** Given the pointer, the downbeat is determined, so the emission is Stage 0's with `r`
promoted from marginalised nuisance to state:

    p_θ(y_i | r_i) = Bern(y_i ; sigmoid(α))   if r_i == 0
                   = Bern(y_i ; sigmoid(β))   otherwise

**Two learnable scalars, θ = {α, β}** — the same two as Stage 0. This is deliberate: the
emission is held **fixed** across Stage 0 → 1A → 1B so that any change in performance is
attributable to the latent and its inference, not to a richer decoder.

**1B.** Identical, with `r` read out from phase at beat times:

    r_t = 0  ⟺  φ wrapped at t

**The emission does not see `h`.** `SPEC.md` §4.3's argument carries unchanged: `b ⊥ h | z`,
and giving the decoder `h` opens a route to fit `y` directly and leave the latent unused. At
Stage 1 this matters more, not less — a time-indexed latent gives a decoder-`h` shortcut more
places to hide.

The marginalisation consequences in `SPEC.md` §4.3 **no longer hold and must not be
inherited**: once `r` is a state rather than a uniformly-marginalised nuisance, `log p(y|·)`
is *not* invariant to cyclic shift, and downbeat phase *is* observable. That is the point of
the stage. Any test asserting shift-invariance is a Stage-0 test and must not be run here.

### 4.4 Transition — `p_ψ(z_t | z_{t−1}, h)`

**1A, the bar pointer.** Deterministic advance, meter gated on bar crossing:

    r_i = (r_{i−1} + 1) mod m_{i−1}
    crossing_i  ⟺  r_i == 0
    p(m_i | m_{i−1}, crossing_i) = δ(m_i = m_{i−1})          if not crossing_i
                                 = Categorical(A[m_{i−1}])    if crossing_i

`A` is a learned `K×K` row-stochastic matrix. **The gate is mandatory, not an optimisation.**
Redrawing `m` freely per step is a measured failure mode: an i.i.d. draw at every bar costs
**−1.07 nats/bar against −0.045** for a transition that persists, a factor of 24
(`SPEC.md` §10.4). This is the paper's own semantics (`ELBO_for_DBN` §3), expressed on the
beat grid instead of on `φ`.

**1B, the phase advance.** Von Mises around the advance the given grid implies:

    φ_t = φ_{t−1} + Δ_t + ε,   ε ~ vonMises(0, κ),   Δ_t = 2π / (m · IBI_t · fps)

`Δ_t` comes from the **given** beat grid (§2), so at Stage 1 the phase *rate* is observed and
only the offset and its drift are latent. `κ` is learned; it is the model's statement about
how much timing slack it allows. Meter is gated on the wrap `φ_{t−1} + Δ_t ≥ 2π`, matching
`ELBO_for_DBN` §11 exactly.

**Where `h` enters.** `h` conditions the **transition**, per frame. It does not enter the
emission (§4.3) and there is no pooled summary anywhere (§2). This is the structural change
that kills the reducer: evidence is consumed `h_t` at a time by the recursion.

> **Prior art to respect, not repeat.** A previous implementation made the prior mean
> audio-blind, which produced an open-loop metronome at deployment — shifting the ground-truth
> beats by 0.5 s moved the inferred phase by 0.0099 rad. If `h` does not reach the transition,
> Stage 1 will reproduce that. Assert it by reading gradients (§4.7), not by inspection.

### 4.5 Encoder

**1A has no encoder, and that is the point.** Forward–backward computes the exact posterior
over `(m, r)` sequences at `O(n · S²)` for `S = 9` states. There is no `q`, no bound, no gap.
1A's objective is the exact log-likelihood.

**1B — `q_φ(z_{1:T} | h, y)`.** Structured, following `SPEC.md` §4.5's reasoning: build `q`
in the same shape as the exact posterior so any gap is a statement about the encoder, not
about what the encoder can see. Concretely, a filtering posterior
`q(z_t | z_{t−1}, h_{1:T}, y)` factorised over `t`, reading `h` bidirectionally.

**Reparameterised sampling.** Von Mises via the standard rejection sampler with a
reparameterised acceptance path.

> **A verified trap.** A prior implementation's Best–Fisher von Mises sampler was wrong:
> `E[cos]` came out a constant ~0.8 instead of tracking `A(κ)`. Every strict-ELBO number
> taken through it was untrustworthy. **The sampler must be tested against `scipy.stats.vonmises`
> before any model result is believed** — moments *and* tail, across `κ ∈ [0.1, 100]`.

### 4.6 Objective

**1A.**

    L(h, y) = log p(y | h) = logsumexp over state sequences, by forward–backward

Exact. No sampling, no bound, no seed sensitivity. Trained by direct maximisation (or EM;
the Fisher identity makes the gradients agree).

**1B.**

    ELBO(h,y) = E_q[ Σ_t log p_θ(y | z_t) ] − KL( q_φ(z_{1:T}|h,y) ‖ p_ψ(z_{1:T}|h) )

Categorical `m` is enumerated (`K = 3`; a relaxation is strictly worse, `SPEC.md` §4.6).
Circular `φ` is sampled with reparameterisation — enumeration ends here, which is precisely
why this stage can answer the question in §1.

**The identity that makes the pair worth building:**

    ELBO_1B(h,y)  ≤  log p(y|h)  =  L_1A(h,y)      when 1A and 1B share a generative model

The gap is the variational + amortisation cost, **measured in nats, on the same data, with
the model class held fixed.** No other configuration in this project's history can produce
that number.

> This requires 1A and 1B to be the *same generative model* up to the discretisation of `φ`.
> They are not identical — 1A's advance is deterministic, 1B's is von Mises — so the identity
> holds in the limit `κ → ∞`. **Report the discretisation check** (1B with large `κ` against
> 1A) or the gap is not interpretable.

### 4.7 Parameter sets

| set | object | reads | deployable? |
|---|---|---|---|
| **θ** | emission — `{α, β}` | `z` | yes |
| **ψ** | transition `p_ψ(z_t\|z_{t−1},h)`, incl. `A`, `κ` | `z_{t−1}`, `h` | **yes — the deployable path** |
| **φ** | encoder `q_φ` (1B only) | `h`, `y` | no |

**Every parameter must receive gradient, asserted by reading gradients.** Measured precedent:
in one configuration **26 tensors / 45.3%** of parameters had exactly zero gradient because
the learned prior was never wired into the path that ran; in another, **41 tensors / 50.88%**.
Both times ψ *was* in the optimiser. Checking the optimiser proves nothing.

---

## 5. Training

Inherits `SPEC.md` §5 except where sampling changes it.

- **1A**: deterministic given data and initialisation, like Stage 0. Bit-reproducible.
- **1B**: stochastic. Report seed, and report results over ≥3 seeds with spread, never one
  draw (C11).
- **Crop length.** Measured: at 256-frame crops the ELBO *prefers* a metronome — truth-coast
  scored +12.3 at 256/512 frames against −46.6 at 1500. Short crops make a phase-free
  solution optimal, so a phase model trained on them will correctly converge to something
  useless. **Crops must be long enough that phase drift is visible**; 8 bars is the Stage-0
  unit and its frame length must be checked against this, not assumed.
- **Free-bits is not permitted as the fix for a dead latent.** It is a one-way door and it
  zeroes prior-side gradients when the KL is under the floor. If `φ` is dead, find out why.

---

## 6. Data

### 6.1 Crops must not be bar-aligned — mandatory

`vbpm/data.py:make_crops` cuts crops at bar boundaries. **Measured: 3,971 of 4,000 crops have
bar offset 0.** Phase is therefore handed to the model for free, and a phase latent trained on
this data would learn "always 0" and score well.

Required change: crop starts sampled at arbitrary **beat** offsets, uniformly over the bar.
`m_true` derivation is unaffected (it is a median over complete bars). A new per-crop
`r_true` is derived for **scoring only** and never reaches the deployable path (C2).

This is the single change that makes phase learnable rather than vacuous. Everything else in
this document is inert without it.

> **Consequence to expect and not to hide.** Un-aligned crops make the task *harder* than the
> one every Stage-0 number was measured on. Scores should go **down**. A Stage-1 number is
> not comparable to a Stage-0 number unless the Stage-0 baseline is re-run on un-aligned
> crops, and §8.2 requires exactly that.

### 6.2 `h`, and a measured constraint on it

`h` is frozen pretrained features (`SPEC.md` §6.1). C6 stands: do not collapse to `[T,2]` as
policy.

> **Rich `[T,512]` features are not comparable across frontend checkpoints.** Measured
> (§10.4): under a checkpoint swap on identical songs with an identical trained model, all
> three 512-dim readers lose 0.16–0.21 balanced accuracy while the 2-channel reader *gains*
> 0.168. Any Stage-1 design that trains on fold-checkpoint features and deploys on `final0`
> features is measuring feature alignment as much as meter. Either stay within one
> checkpoint, use the semantically pinned 2-dim path, or align explicitly.

### 6.3 Splits

Unchanged: fold-honest, pooled out-of-fold, `gtzan` test-only. Per-dataset reporting is
mandatory, never pooled-only — corpora differ enormously (§10.3).

### 6.4 Synthetic bench

`tests/synth_bench.py` is Stage 0's. Stage 1 needs its own, and it must include **known
phase**: crops with a known `r_true`/`φ_true` trajectory, including drift and a mid-crop
meter change. **A phase model that cannot recover known phase from clean synthetic input is
broken, and this must be established before any real-data number is produced.**

> The synthetic bench cannot substitute for real-data controls (`SPEC.md` §8): it is balanced
> by construction and the real corpus is 19:1.

---

## 7. Deployment

At deployment only `h` is available.

- **1A**: Viterbi (or posterior-max) over `(m, r)` given `h` → predicted downbeat placement.
- **1B**: the filtering/smoothing posterior mean of `φ`, wrapped → placement.

Both must run **without `y`** (C2). The read-out reports placement and meter (C9):

    predicted downbeats = { beat i : r̂_i == 0 }

> **The training/inference gap is structural** (`SPEC.md` §7) and is not fixed by fitting ψ
> harder. At Stage 1 there is an additional, measured trap: a prior whose transition mean does
> not read `h` deploys as an open-loop metronome that ignores the audio entirely. §8.4's
> perturbation control exists to detect exactly that.
>
> ⚠️ Tutorial §8.1.5/§8.3 read-out alternatives A–D remain **unread** (`SPEC.md` §7). They are
> Stage-1 material. Ask the user for `VAEBPM_fin.pdf` before choosing a read-out; do not
> reconstruct them from notes.

---

## 8. Evaluation

### 8.1 Metrics

Stage 1 predicts placement, so it is scored as placement, not only as classification:

- **Downbeat F-measure on the given beat grid** — the primary metric. A predicted downbeat is
  correct iff it lands on the annotated downbeat beat. No tolerance window is needed: the grid
  is given, so this is exact.
- **Meter balanced accuracy**, as Stage 0, with raw accuracy and confusion beside it (C10),
  per dataset.
- **Phase read-out error** — circular distance between predicted and true bar phase.
- **NLL of `y`** under the deployable path.

### 8.2 Baselines — all four, or the result is uninterpretable

1. **Stage 0 re-run on un-aligned crops.** The only honest predecessor (§6.1).
2. **Uniform-phase null** — the model with `r` marginalised uniformly, i.e. Stage 0's
   emission. Beating this is the minimum claim that phase is being used.
3. **Peak-picking on `h`** — the non-latent baseline.
4. **1A**, as the exact-inference ceiling, when evaluating 1B.

### 8.3 Pre-registered success criteria

Fixed **before any number exists**. All margins must clear the ±0.08 run-to-run band (C11)
and be reported per dataset.

| # | claim | criterion |
|---|---|---|
| S1 | phase is recoverable at all | on the synthetic bench with known phase, 1A placement F ≥ 0.95 |
| S2 | phase is learnable from real audio | 1A downbeat-placement F beats the uniform-phase null by ≥ 0.10 pooled, and on ≥ 4 of 6 CV corpora |
| S3 | the time index earns its cost | 1A beats Stage-0-on-un-aligned-crops on meter balanced accuracy, or matches it while adding placement |
| S4 | **the VAE encodes something** | 1B's latent is used: replacing the posterior sample with a prior sample degrades placement F by ≥ 0.10. A model that does not degrade has a dead latent, whatever its ELBO says |
| S5 | the inference is sound | the 1B→1A gap in nats is reported, with the `κ → ∞` discretisation check |

**S4 is the criterion this stage exists for.** S1–S3 can all pass with a latent that is
decoration. S4 cannot.

### 8.4 Required controls, each of which has caught a real error here

- **Leakage** — deployable path never reads `y` (C2), asserted, not assumed.
- **Shuffled-label null**, scored **held out**, not on the training split.
- **Audio-blindness probe** — shift the ground-truth beats by ~0.5 s and confirm the inferred
  phase moves accordingly. Measured precedent: a prior implementation moved by 0.0099 rad,
  i.e. it was deploying a metronome. This control is mandatory.
- **Gradient audit** — read gradients for every parameter (§4.7).
- **Prior-sample ablation** — S4's mechanism, run as a control on every reported model.
- **Degenerate-predictor check** — raw accuracy and confusion beside every balanced figure
  (C10).

---

## 9. Testing

`tests/` is **frozen** and is Stage 0's. It must not be edited to accommodate this stage;
Stage-0 properties that Stage 1 legitimately violates (notably §4.3's cyclic-shift invariance)
are Stage-0 properties, and a Stage-1 implementation is not required to pass them.

Stage 1 gets its own suite in a new directory, built on the same principle: a known-correct
reference that **must pass**, plus a fixed set of corruptions each of which **must be caught**,
plus at least one provably-equivalent corruption that must **survive**. Candidate equivalences
to protect: a global phase offset of exactly one bar; relabelling meter states consistently.

Required unit-level checks before any model claim:

- von Mises sampler against `scipy.stats.vonmises`, moments and tail, `κ ∈ [0.1, 100]` (§4.5).
- Forward–backward against brute-force enumeration on short sequences (`n ≤ 8`), exactly.
- The gate: `m` provably cannot change except at a crossing.
- ELBO ≤ exact log-likelihood, on the same model, numerically.

---

## 10. Findings that constrain Stage 1

Measurements, not opinions. All 2026-08-03 in the clean pipeline unless noted.

**§10.1 — the reducer is still the transfer champion.** 18,902 crops, fold-honest, ALL-CV /
gtzan: hand-built 10-dim peak summary + linear **0.512 / 0.623**; AutocorrHead on `[T,512]`
0.561 / 0.357; TransformerPrior `[T,512]` 0.438 / 0.298; `[T,2]` 0.490 / 0.469; normalised
`[T,512]` 0.521 / 0.422. No trained head beats the reducer on transfer.

**§10.2 — run-to-run nondeterminism is ±0.08.** Identical code, identical seed: `linear`
reproduces exactly; FFT and attention arms do not (CUDA reductions). gtzan transfer moved
0.275→0.357 and 0.217→0.298 between identical runs. Any effect below this is noise.

**§10.3 — balanced accuracy is treacherous on skewed corpora.** gtzan is 19/120/1838 crops
for `m` = 2/3/4. A degenerate always-2 predictor scored **0.422 balanced with 0.239 raw**,
carried by 18/19 on the 19-crop class; SE on that recall is ~0.11. `hjdb` is 100% `m=4`, so
any accuracy there is vacuous. `asap` is near-uniform (35/31/34%) and is the only corpus where
balanced accuracy is straightforwardly meaningful.

**§10.4 — rich features are checkpoint-fragile.** Identical songs, identical trained model,
only the producing checkpoint swapped: `tf512` −0.188, `autocorr` −0.162, `tf512n` −0.205,
while the pinned 2-channel `linear` **+0.168**. Caveat recorded honestly: the swapped
checkpoint had trained on those songs, so the leak biases in the swap's favour and is the same
order as the drops. The supported claim is *512-dim readers are checkpoint-fragile, 2-channel
readers are not*; isolating a basis rotation needs a fold-to-fold swap where both checkpoints
held the song out. **Not yet run.**

**§10.5 — crops are bar-aligned.** 3,971/4,000 have `r = 0`. See §6.1. This is the finding
that makes §6.1 mandatory.

**§10.6 — the downstream stack is not the bottleneck.** Synthetic-`h` control (Gaussian bumps
at annotated beat/downbeat times, everything else real): `asap` **0.988**, ALL-CV **0.991**,
on the same 9,742 / 16,925 crops. Read with care — the bumps are placed at the annotations
that define the labels, so this proves the crop/label/prior/fit chain is sound and **cannot**
distinguish "the frontend is deaf" from "the annotation's metrical level differs from what the
audio expresses".

**§10.7 — `asap` fails under two independent frontends.** Trained `asap`-only, same folds,
`[T,2]` arms. Beat This: linear 0.335, autocorr 0.468, tf2 0.533. Beat Transformer (whose
training folds exclude `asap` entirely): linear 0.335, autocorr 0.430. The peak-summary
reducer sits at the 0.333 floor for **both** frontends while trained heads reach 0.43–0.53.
So the information is present in the activations and the hand-built summary cannot see it.
Note for any future use: gtzan and RWC *are* in Beat Transformer's training data.

**§10.8 — short crops make a metronome optimal.** Truth-coast scored +12.3 at 256/512-frame
crops against −46.6 at 1500. See §5.

**§10.9 — meter switches are rare and persistent.** Per-bar switch rate 0.00523; free redraw
costs −1.07 nats/bar against −0.045 for a persistent transition (`SPEC.md` §10.4). This is why
§4.4's gate is mandatory.

---

## 11. Why 1A and 1B are one deliverable

1A alone would be a good beat-tracking model and would answer none of §1's question: exact
forward–backward over a discrete state space is exact inference, so the variational machinery
is still not load-bearing. It would repeat Stage 0's mistake at higher cost.

1B alone produces an uninterpretable number. If placement F comes out at 0.6, that is
consistent with "the generative model is wrong" and with "the model is right and the
approximate posterior is failing", and nothing in the run distinguishes them.

Together they do. 1A gives `log p(y|h)` exactly; 1B gives the ELBO on the same data and the
same generative structure; the difference is the variational cost in nats, and the placement-F
difference is that cost in the units we care about. That is the measurement this project has
never made, and it is the reason to build Stage 1 at all.

Build order: **1A first** (it is the reference and the cheaper build), synthetic validation
before real data (§6.4), then 1B against it.

---

## 12. Amendments to `docs/SPEC.md` this stage requires

`SPEC.md` is normative and `tests/` is frozen. This document does **not** edit either. The
following are proposals, to be accepted or rejected by the user before implementation.

| # | clause | required change |
|---|---|---|
| A1 | §4.1 "One `m` per crop" | amend: `m` gains a time index at Stage 1, piecewise-constant and gated on bar crossing, as §4.1/§4.4 here. `SPEC.md` §12 already anticipates this. |
| A2 | §2 "Beat tracking arrives with `φ` at Stage 1" | **inconsistent with §1's own ladder** (tempo is Stage 2). Amend to: the beat grid remains given at Stage 1; beat finding arrives with `φ̇` at Stage 2. |
| A3 | §4.3 marginalisation consequences | mark as Stage-0-only: cyclic-shift invariance and unobservable downbeat phase are consequences of marginalising `r`, and both end when `r` becomes a state. |
| A4 | §4.4 the reducer | mark as Stage-0 scaffolding with a scheduled deletion at Stage 1, not a component with a future. |
| A5 | §4.6 exact enumeration | scope to Stage 0 explicitly; Stage 1B samples, as §12 already says it must. |
| A6 | §6 crops | record that Stage-0 crops are bar-aligned (§10.5) and that this is a property of the Stage-0 dataset, not of the task. |

`SPEC.md` §12's expansion contract is **satisfied** by this design: `z` becomes continuous
(1B), gains a time index including for `m`, the ELBO stops being exactly enumerable, and the
θ/ψ/φ split, C1–C6, the deployable-path rule and the evaluation controls all survive.

---

## 13. Open questions

Blocking implementation:

1. **A2** — does the beat grid stay given at Stage 1? This document assumes yes and the whole
   of §4.4's `Δ_t` depends on it.
2. **Read-out choice** — tutorial §8.1.5/§8.3 alternatives A–D are unread (§7). Needs
   `VAEBPM_fin.pdf`.

Not blocking:

3. Frame-rate for 1B. `φ` is identified only at beats (§4.1), so a frame-level `φ` may be
   mostly prior between beats. A beat-indexed 1B is a cheaper alternative worth measuring
   against the frame-indexed one.
4. Whether `A` (the `K×K` meter transition) is identifiable at all on 8-bar crops, given a
   0.00523 per-bar switch rate (§10.9). A sticky two-parameter kernel may be all the data
   supports.
5. Whether `asap`'s failure is frontend deafness or metrical-level disagreement (§10.6,
   §10.7). Unresolved, and it bounds what Stage 1 can achieve on the largest corpus.
