# VBPM — Stage 0 specification (`z = m`)

**Status:** DRAFT, 2026-07-30. Replaces `SPEC_v2_stage0.md`, which had the right *scope* but
untrusted content.

## 0. Sources, and how much each is trusted

| source | status | covers |
|---|---|---|
| `ELBO_for_DBN` (Jaehoon Ahn, March 2026) — `git show 43ecf34:docs/ELBO_for_DBN.md` | **Authoritative.** The user's own paper. | the full model (§11), from which Stage 0 is carved |
| `VAEBPM_fin.pdf` (professor's tutorial, 2026-07-10) | **Authoritative, NOT IN REPO.** Only 41 lines of working notes exist. | §7 — currently a stub. Every §7 citation is therefore **second-hand**. |
| Repo code (`frontends/`, `data/`, `tests/`) | Descriptive: what exists, not what is correct. | §6 |
| Campaign measurements | Findings with provenance; cited where used. | §10 |
| `SPEC_v2_stage0.md` | **Not trusted.** Do not cite. | — |

Rule: **every claim is traceable to one of the first four rows.** Where something is
undecided this document says so rather than guessing.

---

## 1. Why Stage 0 exists

The full model (§11) has three coupled latents: meter `m`, bar phase `φ`, tempo `φ̇`. Built
all at once, a failure anywhere is indistinguishable from a failure anywhere else — which is
the history of this project. Fourteen parallel implementations live under `vbpm_campaign/`
and none is authoritative.

**So `φ` and `φ̇` are deliberately removed.** Stage 0 keeps exactly one latent, `z = m`, and
adds the others only once the machinery around this one is proven. The point is to make
"where is it broken?" an answerable question.

The staging discipline:

| stage | latent | what it adds |
|---|---|---|
| **0** | `m` | discrete latent, conditional prior, encoder, exact ELBO — **this document** |
| 1 | `+ φ` | continuous circular latent, von Mises, reparameterised sampling, time recursion |
| 2 | `+ φ̇` | the dynamics: tempo walk, phase advance, bar crossing |

Stage 0 must be built so that Stage 1 **extends** it rather than replaces it (§12).

---

## 2. Scope of Stage 0

**In.** One discrete latent `m ∈ values` (beats per bar). Emission `p_θ(y|m)`, conditional
prior `p_ψ(m|h)`, encoder `q_φ(m|h,y)`, an exact-enumeration ELBO, and the deployable path.

**Out, and deliberately so.** Phase, tempo, any time recursion, any sampling, any dynamics.
`z` has no time index at Stage 0: **one `m` per crop**, not per frame.

**The beat grid is given.** Stage 0 does not find beats. It is handed a beat grid and models
*which of those beats are downbeats*. Beat tracking arrives with `φ` at Stage 1.

> This is what makes `m` identifiable at Stage 0. Given only beat *times*, beats-per-bar is
> near-unrecoverable (§10.1). Given a downbeat-bearing observation on a beat grid, it is
> recoverable by construction.

---

## 3. Conventions

Binding. These are the seams this project has actually broken.

| # | convention | why |
|---|---|---|
| C1 | `m` is a **count** (beats per bar), never an index. Conversion is explicit and one-way. | v1 produced a count, consumed it as an index, and merged bpb=3 with bpb=4 into one class of 139 songs — for weeks. Neither function was individually wrong; the seam was. |
| C2 | Anything on the deployable path reads `h` and **never** `y`. | A `predict(h)` signature does not prove annotation-freedom: a `y` stashed during training and reused later is the realistic leak. |
| C3 | `m` is never hardcoded to 4 as policy. Beats-per-bar factors read the latent. | Standing directive. v1 hardcoded `M=4` in read-out, emission and sawtooth, making meter causally inert. |
| C4 | One module owns `fps`; nothing else divides by it. | Frame/second confusions are silent and survive unit tests. |
| C5 | Functions returning log-probabilities are suffixed `_logp`; all logs natural. | Two Appendix A entries **violate** this and are grandfathered: `predict(h)` and `exact_posterior(h,y)` both return `[K]` log-probs without the suffix. Do not add more. |
| C6 | The frontend is not collapsed to `[T,2]`. | Standing directive (§6.1). |

---

## 4. The model

Tutorial **§9/§12 Sohn-standard**: three parameter sets, not two.

### 4.1 Latent

    m ∈ values,  values = (2, 3, 4)      a COUNT of beats per bar (C1)

`K = |values|`. One `m` per crop — **a deliberate simplification with a measured price, not
a necessity.**

In the full model `m` *does* carry a time index: `m_{1:T}`. But it is **not free per-frame**.
`ELBO_for_DBN` §3 gates the transition on bar crossing — `p_ψ(m_t | m_{t-1}, φ_t, φ_{t-1})`
fires only when `φ_{t-1} + φ̇_{t-1} ≥ 2π`, and between crossings `m` is *copied*. So `m_t` is
piecewise-constant, changing only at bar boundaries.

The paper's *particular* gate is a phase condition, so that exact transition cannot be ported
to a model without `φ`. **But that does not make a time-indexed `m` impossible at Stage 0**,
and an earlier draft of this section wrongly claimed it did. Bar boundaries are locatable here
without `φ`: `(m, r)` already determines where downbeats fall, so one could write
`p(m_i | m_{i-1})` gated on "beat `i` is a bar boundary under `(m_{i-1}, r)`" and let `m` vary
across bars within a crop. That is a well-defined segmental model.

The real reasons for one `m` per crop are:

1. **It is what keeps §4.6 true.** With a single `m` the ELBO is an exact sum over `K = 3`
   terms — no sampling, no seed, bit-reproducible. That is Stage 0's defining property. With
   `m` per bar you enumerate `K^B` sequences or run forward–backward at `O(B·K²)`: tractable,
   but a *different algorithm*, and §4.6's guarantee would no longer hold.
2. **§10.4 says it buys almost nothing here.** The per-bar switch rate is **0.00523** — over
   an 8-bar crop, an expected 0.04 switches. It would add a sequence model to capture an event
   occurring in roughly one crop in twenty-five.

The price is real and measured: Stage 0 cannot represent within-song meter change below crop
granularity (§10.9's 4.1% of songs). `m` regains its time index at Stage 1 (§12).

> The gating is load-bearing in the other direction too: redrawing `m` **without
> persistence** is a measured failure mode, not a harmless generalisation — an i.i.d. draw at
> every bar costs **−1.07 nats/bar against −0.045** for a transition that persists, a factor
> of **24** (§10.4, re-verified 2026-07-31). "Meter evolves freely over `t`" is not the target
> and never was.

> The gating is load-bearing in the other direction too: redrawing `m` **without
> persistence** is a measured failure mode, not a harmless generalisation — an i.i.d. draw at
> every bar costs **−1.07 nats/bar against −0.045** for a transition that persists, a factor
> of **24** (§10.4, re-verified 2026-07-31). "Meter evolves freely over `t`" is not the target
> and never was.

### 4.2 Observation

    y ∈ {0,1}^n     per-BEAT downbeat indicator on a given beat grid of n beats
    y_i = 1  ⟺  beat i is a downbeat

### 4.3 Emission — `p_θ(y | m)`

    p_θ(y | m) = (1/m) Σ_{r=0}^{m-1} Π_i Bern(y_i ; π_i^{(r)})

    π_i^{(r)} = sigmoid(α)  if (i − r) mod m == 0     (beat i is a downbeat)
              = sigmoid(β)  otherwise

**Two learnable scalars, θ = {α, β}.** The bar offset `r` is **marginalised uniformly, not
maximised** — the crop may start anywhere within a bar and no offset is privileged.

**The emission does not see `h`, and this is settled** (user, 2026-07-30). The reason is a
conditional independence, not a safety margin: `h` carries no information about `b` beyond
what `z` already carries, so `b ⊥ h | z`. Adding `h` would not improve the generative story —
it would only give the decoder a route to fit `b` directly and leave the latent unused — the
collapse §10.10 describes.

> This **overrules `ELBO_for_DBN` §5.4** ("Decoder reads h"), verified verbatim against the
> paper. Recorded as a deliberate, reasoned departure, not an oversight.
>
> **It rests on the argument alone.** An earlier draft said "the design argument and the
> measurement agree, which is why it is treated as closed" — but §10.10, the measurement in
> question, has since been shown to describe code that was not running (the primary
> implementation defaulted to `latent_only=True`, and the prose asserting otherwise was a
> stale copy from a sibling module). One argument, no corroboration. The conditional
> independence `b ⊥ h | z` is sound and the decision stands on it; do not cite §10.10 as
> support.

Consequences that follow from marginalising, and are therefore *design properties* rather
than accidents:

- `p_θ(·|m)` is a proper distribution over `y` (the `1/m` is load-bearing; dropping it biases
  toward large `m`).
- When `m | n`, `log p_θ(y|m)` is invariant to **any** cyclic shift of `y`.
- Downbeat *phase* is unobservable: an implementation that places the downbeat one beat late
  is provably equivalent. Tests must not reject it.

> **These three consequences are STAGE-SCOPED, and the third one inverts later.** All of them
> follow from marginalising `r` — the proof for the third is that the offset-`r` mask
> `{i : (i−r) mod m == 0}` shifted by one is `{i : (i−(r−1)) mod m == 0}`, and as `r` ranges
> over `0..m−1` so does `(r−1) mod m`, so the two enumerate the SAME SET of masks in a
> different order under a symmetric logsumexp. The moment `r` stops being marginalised and
> becomes the latent — Stage P (`docs/SPEC_phase.md`), Stage 1 (`docs/SPEC_stage1.md`) —
> the reordering is no longer invisible: it IS the answer, relabelled. Downbeat phase becomes
> observable, cyclic-shift invariance ends, and `downbeat_off_by_one` must be **killed**
> rather than survive.
>
> `tests/mutants.py`'s `EQUIVALENT = {"downbeat_off_by_one"}` and
> `test_equivalent_mutant_survives` are therefore Stage-0 statements, correct here and not
> portable. `tests/` stays frozen and unedited; a later stage carries its own registry in
> which that same corruption is a must-kill. Recorded 2026-08-03 (user decision) so that
> "the suite is green" keeps a defined meaning across stages.
>
> Note the registry already forbids the neighbouring assumption: `offset_fixed_at_zero`
> ("bar offset never marginalised: assumes crops start on a bar") is a **must-kill** mutant.
> The Stage-0 *dataset* satisfies that assumption in ~98% of crops (`SPEC_phase.md` §10.1),
> which is exactly the gap this scoping note exists to keep visible.

### 4.4 Conditional prior — `p_ψ(m | h)`   ← the deployable path

    p_ψ(m | h) = softmax_k [ logits_ψ(h) ]

Reads `h` only. This is the path that survives to deployment; everything else is training
scaffolding.

**Reducing `h`.** `h` is `[T, D]`; the prior needs a fixed-size input. The Stage-0 **default**
is the smallest thing that can work:

    s(h) = concat[ mean_t h,  max_t h ]           ∈ R^{2D}
    logits_ψ(h) = W · s(h) + b                    W ∈ R^{K×2D},  b ∈ R^{K}

Mean **and** max, because beats-per-bar lives in the *rate* of downbeat peaks and a mean alone
discards peakiness. Linear, because Stage 0's job is to expose breakage, not to win.

> This is a **default, not a finding.** It is the one Stage-0 component chosen for simplicity
> rather than derived, and it is the most likely thing to be wrong. It must therefore be
> **swappable behind one function** — `reduce(h) -> R^{2D}` and `logits_psi(s) -> R^K` are
> separate, replaceable pieces, so a pooling change is not a rewrite. Alternatives (a small
> temporal encoder; a hand-built autocorrelation summary) are an experimental axis, not a
> revision of this spec.

> **Do not copy the oracle's reducer.** `tests/subject.py` scores `h` with a hand-built
> beat/downbeat peak-count ratio. That exists so the oracle can be trusted without training
> anything; it is a stand-in, not a recommendation.
>
> An earlier draft of this section asserted "the tests are behavioural, so mean⊕max passes
> them too". That was written without running it and it was **false**: mean⊕max failed
> `check_shuffled_labels_destroy_deployable_accuracy`. The reducer was subsequently
> exonerated — under a true class-derangement null it drops to 0.111–0.222, *below* chance,
> so it was only ever exploiting the 25% of labels the old shuffle failed to break, and the
> same check failed the **oracle** on 15 of 31 seeds. The test was at fault and is fixed.
> The episode is left here because "asserted, not run" is the failure mode this document
> exists to prevent.

### 4.5 Encoder — `q_φ(m | h, y)`

Reads `h` **and** `y`. Training only. Returns a normalised log-distribution over the `K`
values, ordered by `values`.

**Structured, not free-form.** `q` is built as the prior's logits plus a `y`-dependent term:

    q_φ(m|h,y) = softmax_k [ logits_ψ(h)  +  g_φ(y) ]

This is not cosmetic. The exact posterior *is* `softmax[logits_ψ(h) + log p_θ(y|m)]` (§4.6),
so writing `q` in the same shape makes the amortization gap a statement about `g_φ` alone —
which is the whole point of Stage 0. A free-form `q(h,y)` conflates "the encoder is
mis-parameterised" with "the encoder cannot see the prior", and this project has already spent
weeks failing to separate those two.

**One encoder.** The implementation exposes a single `q_φ`, matching the paper (§5.1 gives one
posterior network) and matching deployment. `g_φ(y) = c · log p_θ(y|m)` with `c` a learnable
scalar is the Stage-0 default: it can represent the exact posterior at `c = 1`, so any gap it
shows is a fitting result rather than a capacity limit.

> **Not a spec concept: `capacity`.** An earlier draft mandated three interchangeable encoders
> (`"exact"` / `"full"` / `"coarse"`) so the amortization gap could be dialled. That is a
> *test fixture*, not a model — your paper has one encoder and so does a deployed VBPM — and
> it entered this document from `tests/subject.py` rather than from any source in §0.
> Removed 2026-07-30 (user).
>
> It was also inoperative. `"coarse"` only means anything if its summary of `y` is genuinely
> lossy, but on the synthetic bench `y.mean() = 1/m` **exactly** and `m = n/y.sum()`, so every
> summary is sufficient and the measured gap is negligible.
> A draft quoted **2.6e-4 nats**; that figure reproduces at no configuration tried — the
> observed range is 1.6e-5 to 4.29 depending almost entirely on the optimiser, not the
> encoder. The *argument* (every summary of a clean periodic `y` is sufficient for `m`) is
> what carries this; the number was a single unlabelled draw and is withdrawn. The suite keeps `capacity` as
> an optional affordance for the oracle; two checks that genuinely need a zero-gap `q` skip
> loudly against an implementation that does not offer one, rather than failing it.

### 4.6 Objective

    ELBO(h, y) = E_q[ log p_θ(y|m) ] − KL( q_φ(m|h,y) ‖ p_ψ(m|h) )

Both terms are **exact sums over K ≤ 4 terms**. There is no sampling at Stage 0, therefore no
gradient variance and no seed sensitivity. An implementation that Monte-Carlo-estimates this
is defective. (An earlier draft cited §10.5 here; that finding is now classed as a
hypothesis, and the argument stands without it -- enumerating three terms exactly is
strictly better than sampling them.)

> **This departs from the paper, deliberately.** `ELBO_for_DBN` §5.1 specifies *"Gumbel-Softmax
> sampling, τ annealed"* for the meter, and §5.5 a single-MC-sample loss. Stage 0 enumerates
> instead. With `K = 3` a relaxation is strictly worse — it is biased, needs a temperature
> schedule, and buys gradient variance for nothing, since the sum it approximates is three
> terms long at `K = 3`. The paper's choice is the right one at Stage 1, where `φ` is continuous and
> enumeration ends; it is not right here. Recorded as a departure, like §4.3, rather than
> left implicit.

**Stage 0 is deterministic.** Given the data and the initialisation, nothing in the model is
random. The consequence for testing is that any remaining variability is *sampling* noise —
which songs were drawn — and a check that compares one draw against a fixed threshold is
measuring luck. Assert on a distribution over draws, not a draw (§9).

Identities that must hold, and are testable:

- `ELBO ≤ log p(y|h)` for **any** `q`.
- Equality **iff** `q` is the exact posterior `p(m|y,h) ∝ p_θ(y|m) p_ψ(m|h)`.
- Slack `= KL(q ‖ p(m|y,h))` — the **reverse** KL. (The forward `KL(p‖q)` is a different
  number and is not the bound's slack.)

### 4.7 Parameter sets

| set | object | reads | deployable? |
|---|---|---|---|
| **θ** | emission `p_θ(y\|m)` — `{α, β}` | `m` | yes |
| **ψ** | prior `p_ψ(m\|h)` | `h` | **yes — the deployable path** |
| **φ** | encoder `q_φ(m\|h,y)` | `h`, `y` | no |

**Every parameter must receive gradient.** In one measured configuration **41 tensors /
50.88%** of parameters had exactly zero gradient (§10.2). An implementation asserts this; it
does not assume it, and it asserts it by **reading gradients** — not by checking that ψ is in
the optimiser, which in the measured case it *was*. This is why `named_params()` is part of
the required interface (Appendix A) rather than a debugging convenience.

---

## 5. Training

Maximise the mean ELBO over crops by gradient ascent on **θ, ψ, φ jointly**. Exact
enumeration, so: no sampling, no annealing, no seed. Two evaluations of the same crop must
agree bit-for-bit, and two fits from different seeds must land on identical parameters.

**A crop is a contiguous range of beats**, not of frames. Stage 0 has no time recursion, so
"crop" means only "how much `y` one `m` is asked to explain". `n = len(y)` must cover at least
`MIN_BARS` bars at the largest legal `m` — with `values = (2,3,4)` and `MIN_BARS = 3`, that is
**`n ≥ 12` beats**. Longer is strictly better for identifiability and costs nothing at Stage 0
(the emission is `O(n·m)`). §10.6's short-crop pathology is a *time-recursion* effect and does
not apply here; do not import a frame-count crop length from Stage-1 code.

> **A song is not automatically one crop, and treating it as one is a bug.** Crops are the
> unit precisely so that `m` can differ *between* them — which is the only resolution at which
> Stage 0 can represent meter change at all, `m_t` being unavailable without `φ` (§4.1).
>
> The sharper cost is to the **labels**, not the model. §6.2 derives `m_true` as the median
> beat-count over complete bars. On a song that is half 3/4 and half 4/4 that median describes
> neither half, so a whole-song crop does not merely lose resolution — it **fabricates a
> label**. §10.9 measures 4.1% of songs with more than one recurring bar length, and they are
> concentrated in `asap`, the corpus carrying most of the meter signal (§6.3). Crop, or those
> songs are silently mislabelled.

`h` is sliced to the frames spanned by those beats, so `T` tracks `n`.

**Defaults**, so two independent implementations are comparable: Adam, **`lr = 0.5`, 500
steps**, **full batch** (every crop each step), `float64`. Any parameter that is tied (e.g.
`β := α` in a meter-free null) is handed to the optimiser **once** — passing the same tensor
twice silently doubles its learning rate.

> **These were `lr = 0.05` / 400 steps and that underfits.** Measured on the synthetic bench,
> mean balanced accuracy over 5 seeds, oracle vs a §4.4-shaped mean⊕max prior:
>
> | steps/lr | oracle | §4.4 prior |
> |---|---|---|
> | 250 / 0.1 | 1.000 | **0.656** |
> | 500 / 0.5 | 1.000 | 0.956 |
> | 800 / 0.3 | 1.000 | 1.000 |
>
> The oracle converges almost immediately because its hand-built per-class features already
> contain the answer; a prior that must *learn* does not. A budget tuned against one
> implementation silently becomes part of the contract — and this one would have made a
> perfectly capable §4.4 prior look broken. These remain working values, not tuned optima:
> a difference between two runs should be a difference in the *model*, so if you change them,
> change them for both sides of the comparison.

---

## 6. Data

### 6.1 `h` — the deployable evidence

`h` is **features extracted from a pretrained model** — a beat tracker (Beat This, Beat
Transformer) or a music foundation model (MERT). A contract, not a vendor.

Two admissible depths (user, 2026-07-30):

| depth | shape | notes |
|---|---|---|
| **rich** | `[T, ~512]` | the representation immediately **before** the two-channel compressor |
| **compressed** | `[T, 2]` | the beat / downbeat activation pair |

Both are legitimate inputs; which to use is an experimental variable, not a fixed decision.
C6 constrains only the *frontend* — VBPM must not itself collapse a rich representation to
`[T,2]` and call it the frontend's output.

The frontend is **frozen**; VBPM trains no part of it. VBPM must not reuse the frontend's own
beat activations as its evidence head — it owns an evidence head per frontend, trained
fold-honest.

`frontends/beat_this.py` and `frontends/beat_transformer.py` exist; MERT does not yet.

### 6.2 `y` and labels

`y` from downbeat annotations on the beat grid. Beat/downbeat annotations are **trusted**
(user, 2026-07-30). Annotation files are `[time, beat_in_bar]` pairs with `beat_in_bar == 1`
marking a downbeat.

**Downbeats are a subset of beats.** `y_i = 1` iff beat `i` carries an annotated downbeat,
matched within `DOWNBEAT_TOL_S = 0.02 s`. A downbeat that matches no beat is a data error, not
a new beat — surface it, do not silently insert one.

**`m_true`** = the median, over complete bars, of the number of beats in `[d_k, d_{k+1})`;
rounded **half up**; requiring at least `MIN_BARS = 3` complete bars, otherwise the crop is
rejected and carries no label.

> The half-up tie-break on an even bar count is **a choice, not a derivation** — a song
> alternating 3 and 4 has median 3.5. It is pinned so two implementations agree, and it is
> only reachable on the 4.1% of songs in §10.9. Do not treat it as meaningful.

### 6.3 Splits and corpora

**Train and evaluate on the corpora already in the repo** (user, 2026-07-30) — the catalog in
`data/songs.py`, no new acquisition for Stage 0. Every corpus is used; none is dropped for
being 4/4-only, because a 4/4 corpus still tests calibration and still supplies the class
imbalance the model has to survive.

**What is actually usable, measured 2026-07-30 after wiring `rwc` and `asap`:**

| dataset | m=2 | m=3 | m=4 | n | folds? |
|---|---|---|---|---|---|
| **asap** | **141** | 99 | 223 | 463 | yes |
| ballroom | 0 | 171 | 501 | 672 | yes |
| beatles | 9 | 11 | 159 | 179 | yes |
| gtzan | 7 | 54 | 930 | 991 | **test-only** |
| hainsworth | 6 | 11 | 204 | 221 | yes |
| hjdb | 0 | 0 | 232 | 232 | yes |
| **rwc** | **43** | 29 | 47 | 119 | yes |
| **total** | **206** | **375** | **2,296** | **2,877** | |

`asap` and `rwc` together supply **184 of 206** bpb=2 songs. Both were reporting **zero usable
songs** until 2026-07-30 — not because the audio was missing but because their filenames do
not match their annotation stems (`rwc` names audio by sequential piece number against
disc/track annotations; `asap` nests audio by composer/piece/performer). The catalog reported
absence, so the absence looked like a fact about the data. It was a fact about the matcher.

> **Both mappings are verified, not assumed.** `corr(last beat time, audio duration)` = 0.9999
> for both, with **zero** songs whose last beat falls past the end of the audio; shuffled
> controls collapse to |corr| < 0.01. `rwc`'s pairing is *positional*, so `data/songs.py`
> refuses to pair a subset whose counts differ rather than risk a silent misalignment.
> `asap`'s is a structural name match, and 4 annotations are **dropped** rather than
> fuzzy-matched to a `no_repeat` variant — a repeat and a no-repeat performance are different
> lengths, and that is precisely how a corpus gets silently misaligned.

**Meter results are read per-dataset, never pooled.** `hjdb` is 232/232 fours and `ballroom`
has **no bpb=2 at all**; a headline number pooled over them is dominated by strata carrying no
signal. Report `asap`, `rwc` and `ballroom` separately — and note that `asap` and `rwc` are
exactly the classical/rubato material §10.1 identifies as the only place meter is recoverable
from timing at all.

Beat This 8-fold splits are **structurally enforced**; no bypass. Corpora carry an alignment
statistic before ingestion — one self-sourced corpus was misaligned in ~75% of crops and had
to be quarantined.

### 6.4 Synthetic bench

**Purpose: the simplest data imaginable.** This is a sanity check that the model can learn
meter at all — not a benchmark. It stays noise-free and exactly periodic. Do not "harden" it;
a hard case belongs on real annotations.

**Easy is fine. Bypassable is not.** A bench certifying "the model learns meter" must not be
solvable by any mechanism other than the one under test. Three requirements, each of which
closes a measured leak:

| requirement | the leak it closes |
|---|---|
| the **beat** period drawn independently of `m` | `beat_period = bar_s / m` is an identity, so exactly one of the two can be `m`-independent. Holding `bar_s` independent — the earlier choice — leaves `corr(m, beat_rate) = +0.80` and the meter readable straight off tempo. Holding the beat period independent pushes the correlation onto `bar_s`, which is observable *only* by finding downbeats. It is also physically honest: real music holds beat rate in a human range and lets bar duration vary with meter. |
| `n_bars` derived per song from a target **beat count**, not held fixed | held fixed, `n_beats = n_bars · m` gives `corr(m, n_beats) = 1.0000` and a **downbeat-blind** peak-counter scores balanced accuracy **1.000** — the emission never has to work |
| every song distinct | a previous bench emitted 160 bit-identical songs, so effective `n` was 1 |

Measured after both fixes at **seed 3**, `n = 36` songs: `corr(m, n_beats) = +0.04`,
`corr(m, beat_rate) = −0.09`, `corr(m, duration) = +0.07`; downbeat-blind nearest-centroid
classifiers score 0.361 / 0.333 / 0.417 against chance 0.333. `corr(m, bar_s) = +0.85`,
which is correct and must stay.

> **These are one draw, and the seed is named because it matters.** Over seeds 0–7 the
> downbeat-blind scores range 0.306–0.528 — the bench is not bypassable *on average*, but no
> single seed's numbers are a property of the bench. Reproducing a different value is not a
> bug. (§4.6 states the rule; this section previously broke it.)

**What this bench cannot do.** On clean periodic `y`, `y.mean() = 1/m` and `m = n/y.sum()`
exactly — every summary statistic is sufficient. So the **amortization gap is ≈0 here by
construction**, and that is the correct answer, not a defect. A deliberately lossy encoder
therefore cannot demonstrate a gap on this bench either, and no test should claim it does. Measuring the gap
— the thing Stage 0 exists to isolate — needs real annotations. **Currently nothing measures
it** (§13).

---

## 7. Deployment

At deployment only `h` is available: predict `argmax_k p_ψ(m|h)`.

**The training-inference gap is genuinely present** (tutorial Misconception 6) and must not
be assumed away. Its structure (§6.8.6–6.8.7): `h`-only inference at best matches the
**aggregated posterior** — the mixture over `y` of per-instance posteriors — which is
"definitionally broader" than any single posterior. The gap is **structural** and **not
fixable by fitting ψ harder**.

The strongest correct statement about ψ: at convergence of two-step generalized EB (§6.8.5),
ψ converges to that aggregated posterior. That is what to assert; not that the gap is zero.

> ### ⚠️ Incomplete — requires `VAEBPM_fin.pdf`
> Tutorial §8.1.5/§8.3 define the deployment read-out alternatives A–D; §6.8.8 (Sohn hybrid /
> GSNN) and §6.8.11 (physical-prior anchoring) are named remedies. **I have 41 lines of notes,
> not the source, and will not reconstruct these from them.**

---

## 8. Evaluation

Stage 0 predicts a class, so it is scored as classification:

- **Balanced accuracy** (mean of per-class recall), not raw. An always-4 predictor scores
  exactly **1/K = 0.333 balanced** whatever the skew, which is the entire reason for the
  metric.

  > **Derive the raw baseline; never write it as a constant.** It depends on which subset is
  > scored and it has been wrong in this document twice. Its value today: **0.7981** over the
  > 2,877 usable songs, **0.8032** over the 1,886 CV-eligible ones (gtzan is test-only and
  > skews more 4/4). Earlier drafts said 0.76, then 0.870, then 0.8827 — each correct for a
  > corpus that had changed underneath it. Compute it from the songs actually being scored.

- **Pool out-of-fold predictions; do not average per-fold scores.** Score each song with the
  checkpoint that held *it* out, collect all predictions, compute balanced accuracy **once**
  over the pool. Fold-honesty is preserved — every prediction is out-of-fold — while the
  minority class stays estimable. Per fold there are only ~25 bpb=2 songs, giving a standard
  error of ~0.10 on that recall; a per-fold balanced accuracy is noise, and averaging noisy
  per-fold numbers discards the pooling that makes the estimate work.

  Pooled over the 1,886 CV-eligible songs (2:199, 3:321, 4:1366), the standard error of
  balanced accuracy is **0.0157**. So chance is 0.333 and **a result must clear ≈0.38 to be
  3σ above a constant predictor.** Below that, report it as indistinguishable from collapse.

- **Distinct predicted classes** — collapse detector.
- **NLL of the true class** under the deployable path.
- **Confusion matrix.**

Required controls, each of which has caught a real error here:

- **Leakage.** Deployable scores come from a path that never saw `y` (C2).
- **Held-out controls.** A shuffled-label null scored on the *training* split is not a
  control: the prior memorises which label each feature value was dealt. Score it held out.
- **Baselines.** Majority class, and peak-counting on `h`.

> **The synthetic bench cannot stand in for any of this.** It is balanced by construction, so
> raw and balanced accuracy coincide there and an always-4 predictor scores 0.333 either way.
> The real corpus is **19:1** — a completely different regime, and the one where a degenerate
> predictor looks good. Every control in this section therefore has to run on real data;
> `tests` currently exercises none of them (`nll_true`, `confusion` and `majority_predict`
> exist in the reference but no property calls them). Passing the synthetic suite says
> nothing about class-imbalance behaviour.

---

## 9. Testing

`tests/` is the executable half of this document, and is already Stage 0.

Properties are parameterised over an implementation and run against three things: a
known-correct reference (**must pass**), 16 named corruptions (**15 must be caught, 1 is
provably equivalent and must SURVIVE**), and
the real package (the verdict). "Passes ⟹ proper" cannot come from more assertions — the set
of wrong programs isn't enumerable — so it comes from fixing a set of wrongnesses and proving
each is caught. Provably-equivalent corruptions (e.g. downbeat off-by-one, §4.3) must
**survive**; killing one means the suite over-specifies. See `tests/README.md`.

The suite is therefore the acceptance criterion, and **Appendix A is the interface it binds
to**. Definition of done for Stage 0: `pytest tests --impl=vbpm` green, with
`_vbpm_factory()` a thin adapter — if the adapter has to do real work, the implementation has
drifted from Appendix A and that is the finding.

---

## 10. Findings that constrain Stage 0

Measurements, not opinions. Recorded so they are not re-run.

**10.1 · Meter is near-unrecoverable from beat timing alone.** Beats-per-bar from inter-beat
intervals reaches balanced accuracy 0.512 vs 0.333 chance, and **exactly** chance on
beatles/gtzan/hainsworth. Real recovery only on `asap` (0.741 vs 0.333 chance, classical
rubato) and `ballroom` (0.632 — but ballroom has only **two** classes, so its chance is
**0.500**, making that a +0.132 margin, far weaker than `asap`'s +0.408). **This is why §4.2's observation is a downbeat indicator, not a beat
indicator.** A beat-only Bernoulli emission makes the meter latent *faithfully* vacuous — and
that is a model fact, not a prior to be swapped.

**10.2 · Parameters silently die.** Reproduced 2026-07-30: 26 tensors / **45.26%** at exactly
zero gradient — and that figure *understates* it. The true count is **41 tensors / 193,719
params / 50.88%**. Over half the network never trained.

**The mechanism is not what was recorded.** ψ was *not* "never registered". It **is**
registered and **is** handed to the optimiser via `model.parameters()`; it receives nothing
because the rollout never calls it — the learned prior was replaced by fixed constants and a
registered **buffer**, and buffers are not parameters. So the bug is invisible to the obvious
check ("is it in the optimiser?") and visible only by reading gradients. Read gradients
regularly; do not wait for output metrics to look wrong.

**10.3 · The count/index seam.** v1's shipped bug (C1). Both functions passed their own tests.
Testing with a **disjoint vocabulary** — no integer is both a legal meter and a legal index —
stops the confusion landing on a plausible neighbour.

**10.4 · Meter is genuinely sticky, and persistence is worth 24×.** Re-verified 2026-07-31 by
re-running `vbpm_campaign/alternatives/alt_meter_model.py` over 2,778 songs / 153,293 bars:
per-bar switch rate **0.00523**, songs with ≥1 switch **141/2778 (5.1%)**. Held-out
log-likelihood per bar, over models of the meter sequence:

| model | LL/bar |
|---|---|
| full K×K Markov | **−0.0452** (best on held-out LL, AIC and BIC) |
| sticky Markov | −0.0452 |
| hierarchical Dirichlet-multinomial | −0.0663 |
| constant-per-song + blip | −0.1346 |
| **i.i.d. Categorical per bar** | **−1.0727** |

Drawing `m` afresh each bar — no persistence — is **24× worse** than a transition that
persists. This is the empirical case for the paper's bar-gated, copy-between transition
(§4.1), and against any "meter is just a per-step categorical" shortcut. Note the measurement
is per **bar**; a per-*frame* ungated redraw is strictly worse still, but that is inference,
not something measured here.

Constant-`m`-per-crop (§4.1) is therefore well-supported — but it means Stage 0 **cannot**
model within-song meter change, which is real (~1,963 Lakh files, and §10.9's 4.1%).

The same run re-confirms §10.1's *mechanism*: normalised IBI is flat across beat-in-bar
(means 1.146 / 1.117 / 1.044 / 1.049 at positions 1–4), so a **beat-only** emission leaves the
meter latent with no likelihood gradient — `KL → 0`, posterior = prior. That is why §4.2's
observation is a downbeat indicator.

**10.5 · Seed noise has already invalidated results here.** 60% of seed variance was float
noise, making 17 A/B results vacuous. Stage 0's exact enumeration removes the excuse.

**10.6 · Crop length is a modelling decision.** At 256–512 frames the ELBO *prefers* a
metronome; only at ~1500 does it prefer truth. Several "training failures" were correct
convergence to the objective's real optimum at short crops. Stage 0 has no time recursion, so
this bites at Stage 1 — recorded here so it is not rediscovered.

**10.7 · `values = (2,3,4)` is validated by the annotations, and 2-vs-4 is identifiable.**
Measured over all 4,737 annotated songs with usable beat-in-bar counters, by taking the modal
number of beats between consecutive downbeats:

| bpb | songs | share |
|---|---|---|
| 2 | 212 | 4.5% |
| 3 | 397 | 8.4% |
| **4** | **4,080** | **86.1%** |
| 5–12 | 47 | 1.0% |
| 1 | 1 | 0.0% |

`{2,3,4}` covers **99.0%** of the corpus, so the vocabulary is right. The class balance is
severe (4 outnumbers 2 by 19:1), which is why §8 mandates balanced accuracy.

Identifiability was checked, not assumed: with a saturated emission on a downbeat indicator,
`argmax_m p_θ(y|m)` recovers the true period for m = 2, 3, 4, separating by **≥ 17 nats**.
The 4-vs-2 ambiguity I had worried about does not arise — it only exists given beat *times*
without downbeat labels. Residual structure remains (for period-4 `y`, `m=2` at −19.9 scores
better than `m=3` at −31.2, since 2 divides 4), but it is not close.

**10.8 · Not every dataset can test meter.** `candombe`, `filosax`, `guitarset`, `hjdb` are
**100% 4/4** and carry zero meter signal. `gtzan` is 93.7% 4/4. Meter discrimination lives
almost entirely in `asap` (2:144, 3:99, 4:224 — classical, the only corpus with a
near-balanced spread), `ballroom` (3:173, 4:512) and `rwc` (2:43, 3:29, 4:146). A meter result
reported on gtzan alone is close to meaningless; **`asap` carries 68% of all bpb=2 songs**.

**10.9 · Within-song meter change is real but rare.** 196 of 4,737 songs (**4.1%**) show more
than one recurring bar length. Stage 0's one-`m`-per-crop assumption (§4.1) is therefore sound
for ~96% of songs at song level, and better at crop level — but it is an assumption with a
measured cost, not a free simplification.

**10.10 · A likelihood that sees `h` leaves the latent unused — ARGUED, NOT RELIABLY
MEASURED.** The claim is that an `h`-conditioned emission fits `b` from `h` alone and the
latent goes dead (`KL ≈ 0`). It is consistent with the all-latents-dead outcome that recurs
here, but it is the **weakest citation in §10** and it is demoted on purpose:

- The 2026-07-30 forensic audit found `vbpm/model.py` defaults `latent_only=True`, so the
  primary implementation never ran the `h`-conditioned emission at all;
- `vbpm/evaluate.py`'s collapse diagnostic describes a decoder that "reads h (§5.4)" and
  "rides the audio" — a near-verbatim copy of `faithful/evaluate.py` left in place after the
  model beneath it changed. The prose describes code that was not running.

So the code this finding describes may not be the code that produced it. **§4.3 does not
depend on this.** Its argument is the conditional independence `b ⊥ h | z`, which stands on
its own; §4.3's claim that "the design argument and the measurement agree" should be read as
one argument plus one unverified corroboration, not two independent legs.

Numbering note: §4.3 previously cited §10.7 for this, which was wrong.

**Provenance warning for this whole section.** §10.1, §10.4, §10.7, §10.8 and §10.9 are
derived from **annotations alone** — no model, no training run — and are safe. §10.2 and
§10.3 are weaker than "reproduced": §10.2's *arithmetic* is corroborated against a live model
instantiation, but no audit script survives, so **which** tensors were dead cannot be
re-derived; §10.3 is a historical bug report, not a measurement. (§10.2's figure is an
*under*-statement; see there). §10.5, §10.6 and §10.10 are **inherited from campaign runs
whose instrumentation is now known to be broken**: the metric those runs selected checkpoints
against was a ground-truth-meter × constant-rate metronome (phase advance std 1.2e-7). Treat
them as hypotheses, not measurements.

---

## 11. Where this is going

The target (`ELBO_for_DBN` §2–§6) is `z_t = [m_t, φ_t, φ̇_t]` with
`p_ψ(z_t|z_{t-1}) = p_ψ(m_t|m_{t-1},φ_t,φ_{t-1}) · p_ψ(φ_t|φ_{t-1},φ̇_{t-1}) · p_ψ(φ̇_t|φ̇_{t-1})`:
a Log-Normal tempo walk, a von Mises phase advance, and a meter transition **gated on bar
crossing** (`φ_{t-1} + φ̇_{t-1} ≥ 2π`). `φ ∈ [0,2π)` is **bar** phase; a wrap is a downbeat.

**Stage 0 is the paper's `t = 1` meter factor**, not an invention and not a mutilation of the
`t ≥ 2` one. `ELBO_for_DBN` §5.1 gives, for the meter Categorical:

> prior `t=1`: `f^init_ψ(h)`; `t≥2`: `f^m_ψ(m_{t-1}, φ_t, φ_{t-1}, h)`. posterior `t=1`:
> `f^m_φ(b,h)`; `t≥2`: `f^m_φ(b, z_{t-1}, h)`. KL = `Σ_k π^q_k log(π^q_k/π^p_k)`.

An `h`-only prior, a `(b, h)` posterior, and `KL(q‖p)`. So §4.4 and §4.5 match the paper's
`t=1` factor **in shape**, and there is no conditioning to strip at `t = 1` — it has none.

> **This correspondence is partial, and an earlier draft overstated it.** It claimed Stage 0
> "IS the paper's `t=1` meter factor … that is §4.4, §4.5 and §4.6 **exactly**", and that "the
> only departure is inference". Both are wrong:
>
> - **The paper's `b` is not this document's `y`.** Paper §5.4: `b_t ∈ {0,1}` is a per-frame
>   **beat** indicator over `T` frames. §4.2 here: `y ∈ {0,1}^n` is a per-**beat** **downbeat**
>   indicator on a given beat grid. Different support, different semantics.
> - **The emission is not in the paper at all.** `p_θ(y|m)` with a uniformly marginalised bar
>   offset appears nowhere in `ELBO_for_DBN`; §4.3 already says it *overrules* §5.4.
>
> So there are at least **two** departures — the emission (§4.3) and the inference (§4.6) —
> and the emission is the larger. Corrected 2026-07-30 after the claim failed a fact-check.
> It is left visible because "this is exactly the paper's term" is the kind of claim that,
> believed, would stop an implementer treating the emission as the deliberate choice it is.

Nothing here should contradict the target **except where a departure is declared** (§4.3,
§4.6).

---

## 12. The expansion contract

Stage 1 must **extend** Stage 0, not replace it. Concretely, Stage 0 code must not assume:

- that `z` is discrete (Stage 1 adds a continuous circular latent);
- that `z` has no time index (Stage 1 adds `t`) — **including `m`**. `m` is constant per crop
  at Stage 0 only because its transition is gated on a bar crossing that `φ` defines (§4.1).
  When `φ` arrives, `m` becomes `m_{1:T}`: piecewise-constant, copied between crossings,
  redrawn at them. Code that stores `m` as one value per crop must be able to become one
  value per frame without the emission, prior or encoder being rewritten around it;
- that the ELBO is exactly enumerable (Stage 1 requires sampling and reparameterisation);
- that `K` is small, or that the posterior is a `K`-vector.

The pieces that must survive unchanged: the θ/ψ/φ split (§4.7), the deployable-path rule
(C2), the count/index discipline (C1), and the evaluation controls (§8).

---

## 13. Open questions

Resolved 2026-07-30 — kept visible so the reasoning is not relitigated:

- ~~What is `h`?~~ → §6.1. Pretrained features, rich `[T,~512]` or compressed `[T,2]`.
- ~~Does the emission see `h`?~~ → **No** (§4.3). `b ⊥ h | z`; overrules paper §5.4.
- ~~Is `{2,3,4}` right, and is 2-vs-4 identifiable?~~ → **Yes to both** (§10.7). 99.0% coverage,
  ≥17 nats separation.

Closed 2026-07-30 by **deciding**, not by measuring — flagged as such so they can be reopened:

- `h`-reduction for `p_ψ` → §4.4, mean⊕max + linear, **required to be swappable**.
- Encoder form → §4.5, structured `logits_ψ(h) + g_φ(y)`, **one** encoder.
- Crop → §5, a beat range, `n ≥ 12`.
- `m_true` → §6.2, median over complete bars, half-up, `MIN_BARS = 3`.
- Optimiser → §5, Adam / 0.05 / 400 / full batch / float64.
- Datasets → §6.3, the ones in the repo, reported per-dataset.

Still open, and **not blocking implementation**:

1. **Which of the 14 `vbpm_campaign/` implementations, if any, survives as the basis?** The
   working assumption is none: Stage 0 is small enough to write from this document.
2. **Rich `[T,~512]` vs compressed `[T,2]` `h`** (§6.1) — an experiment, and the reason §4.4's
   reducer must not assume `D`.
3. Tutorial §8.1.5/§8.3 read-out alternatives A–D (§7) — Stage-1 material.

---

## Appendix A — the interface (normative)

`tests/` is written against a concrete surface. An implementation that is correct but
differently shaped cannot be scored, so this appendix is binding. The reference oracle in
`tests/subject.py` implements exactly this and can be read as the executable copy.

Construction: `Stage0(values=(2,3,4))`. All log-probabilities natural (C5), all returns
`float64` tensors of shape `[K]` ordered by `values`, all gradients live.

**`capacity` is deliberately absent** — see §4.5. The suite may pass a `capacity=` hint; an
implementation is free to ignore it, and the two checks that require a zero-gap `q` then skip
rather than fail.

| method | reads | returns | must satisfy |
|---|---|---|---|
| `to_idx(m) -> int` | a **count** | position in `values` | raises on an illegal count; never the identity (C1) |
| `to_value(k) -> int` | an index | the count | `to_value(to_idx(m)) == m` |
| `emission_logp_all(y)` | `y` | `[K]` `log p_θ(y\|m)` | offset marginalised (§4.3); normalises over `y` |
| `prior_logp(h)` | `h` | `[K]` `log p_ψ(m\|h)` | sums to 1; **must not accept `y`** |
| `predict(h)` | `h` | `[K]` | the deployed path; `= prior_logp` by default (C2) |
| `q_logp(h, y)` | `h`, `y` | `[K]` `log q_φ` | sums to 1; depends on **both** arguments |
| `exact_posterior(h, y)` | `h`, `y` | `[K]` | `∝ emission_logp_all + prior_logp` |
| `log_evidence(h, y)` | `h`, `y` | scalar | `logsumexp(emission_logp_all + prior_logp)` |
| `elbo(h, y)` | `h`, `y` | **scalar** | `E_q[log p_θ(y\|m)] − KL(q ‖ p_ψ)` |
| `fit(songs, steps, lr, seed)` | list of `{h, y}` | `self` | maximises mean `elbo`; §5 defaults |
| `named_params()` | — | `{name: tensor}` | **every** trainable tensor, θ ∪ ψ ∪ φ (§10.2) |
| `param_groups()` | — | `{group: {name: tensor}}` | the §4.7 split: keys `theta`, `psi`, `phi` |

**`param_groups()` draws the spec/implementation line.** θ is spec-mandated — §4.3 fixes it as
two named scalars `{α, β}` — so anything may address `alpha` and `beta` by name. **ψ and φ have
no mandated parameterisation**: §4.4 explicitly requires the reducer be swappable, and §4.5's
`g_φ` is an implementation choice. So nothing outside the implementation may name a ψ or φ
tensor or assume its shape; reach them by group and by `p.shape`.

> This is not a style rule. `tests/subject.py` stores ψ as a `[3]` weight vector over three
> hand-built per-class features; §4.4 specifies `W ∈ R^{K×2D}` = `[3,4]`. Both are called
> `w_prior`, and `copy_` from one into the other **raises** — so a test that hardcoded the
> oracle's shape did not merely fail a spec-conformant implementation, it crashed it. The
> suite bound to one implementation's field layout while claiming to bind to this appendix.
> Fixed 2026-07-30; recorded so it is not reintroduced.

> **`elbo()` used to return a dict** of `{elbo, recon, kl, q_logp, prior_logp, lik}`, justified
> here as "the suite checks the decomposition, not just the total." That justification was
> false: **no check ever read `recon`, `kl`, `lik` or `prior_logp`.** The KL direction is
> pinned behaviourally instead — by the bound-tightness identity and the reverse-KL slack
> identity (§4.6) — and `q_logp` is already its own method. Five unread keys were contract for
> free. Scalar since 2026-07-30 (user).

`named_params()` is not introspection sugar. It is how "50.88% of parameters had exactly zero
gradient" (§10.2) becomes a test instead of a post-mortem, so it must enumerate the true
optimiser set — including tied parameters, once each.

A song is `{"h": [T,D], "y": [n] of {0,1}, "m_true": int}`.

Wiring: fill in `_vbpm_factory()` in `tests/conftest.py` to return
`(values=None) -> Stage0`, then run `pytest tests --impl=vbpm`. It currently
fails on purpose with this list.
