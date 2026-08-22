# Related work the ISMIR 2026 reviewers asked for

Compiled 2026-08-20 in response to the ISMIR 2026 reviews of *The SMC Blind Spot*
(paper 53; strong reject / weak reject / weak reject). Reviewer 2 was the only one
to name specific missing citations, but they are load-bearing: R2 marked
"discusses, cites and compares with all relevant related work" as **Disagree**,
and that is the one hard, cheap-to-fix defect in the whole review set.

## 1. Sá Pinto (2023) — the direct-overlap thesis  ✅ READ (253 pp., `644518.pdf`)

**Towards Human-in-the-Loop Computational Rhythm Analysis in Challenging Musical
Conditions.** António Sá Pinto. Doctoral dissertation, Faculdade de Engenharia da
Universidade do Porto (Digital Media), defended September 2023. Co-supervised by
Rui Penha (ESMAE) and Gilberto Bernardes (FEUP).
Handle: <https://repositorio-aberto.up.pt/handle/10216/153038>

R2's words: it "certainly overlaps in several aspects — the use of different
parameterisations of the DBN, tempo limits, estimating upper bounds on the SMC
dataset (and other datasets)."

PDF obtained by the user (`644518.pdf` in this repo; text at `/tmp/sapinto.txt`).
R2 was right, and the overlap is more specific than their comment suggested.
**Read §3.1, Table 3.1, §3.3.2 / Table 3.2, and §5.4 before revising the paper.**

### 1a. §3.1 — the "expressive parameterisation" IS our tempo-range/λ result

Table 3.1, p. 60, on the Böck et al. (2016) RNN + madmom DBN:

| parameter | default | "expressive" |
|---|---|---|
| minimum tempo (BPM) | 55 | **35** |
| maximum tempo (BPM) | 215 | **135** |
| transition-λ | 100 | **10** |

Chosen by "a grid search across these three parameters on a subset of musical
examples... for which the (then) state-of-the-art RNN is deemed to perform poorly,
i.e., by having an information gain lower than 1.5 bits," then confirmed by informal
subjective assessment. The motivation is identical to ours: "a shift towards a slower
range of allowed tempi (following evidence about the greater difficulty of tapping to
slower pieces of music), together with a lower value for the transition-λ."

**This directly pre-empts our widen-the-tempo-range and lower-λ findings.** Our
numbers (min_bpm 55→30, best fixed λ = 5) are the same move, one notch further. We
cannot present that as a novel finding. We must cite Table 3.1 and state the delta.

They also explicitly decline to generalise: "their goal was to find an optimal value
across a wide range of musical examples. Here, our focus is on highly expressive
music, therefore we do not need to pursue a more general solution."

### 1b. §3.3.2 / Table 3.2 — they already computed a per-excerpt decoder-choice oracle

Same SMC (217 excerpts, 40 s), Davies & Böck (2014) metrics:

| | F | CMLc | CMLt | AMLc | AMLt |
|---|---|---|---|---|---|
| Default (Böck et al. 2016a) | 0.563 | 0.350 | 0.472 | 0.459 | 0.629 |
| Expressive (35/135/λ=10) | 0.540 | 0.306 | 0.410 | 0.427 | 0.565 |
| **Optimal per-excerpt choice of the two** | **0.624** | **0.456** | **0.611** | **0.545** | **0.703** |

Two things matter here.

1. **The expressive parameterisation LOSES on SMC overall** (0.540 vs 0.563 F, and on
   every other metric). Their own reading: "the SMC dataset is not entirely composed
   of highly expressive musical material." This is independent confirmation of our
   central tension — no single DBN setting wins on SMC — from a different activation
   model. **Good for us:** it makes the "no single setting" claim model-independent
   rather than a Beat This quirk.
2. **The "optimal choice" row is a per-excerpt oracle decoder-selection envelope** —
   structurally the same experiment as `docs/findings/FOLLOWUP_FINDINGS.md` Blind spot 1 (peak-pick
   oracle-threshold 0.670, wide DBN oracle-λ 0.653, per-track best-of-two 0.691).
   **Our decoder-envelope framing is therefore not novel in form.** It is novel in
   substrate (Beat This / Beat Transformer vs the 2016 RNN) and, critically, we can
   explain *why* the envelope exists, which they do not attempt.

Note their caveat, which we should adopt: they warn that large apparent gains "might
be attributed to straightforward corrections at the metrical level," citing a case
where fixing a 180°-phased-out track yields "a (potentially misleading) improvement
of 100 p.p." This is R2's F ≥ 0.8 / metrical-level objection, already articulated in
the prior work.

### 1c. §5.4 — their SMC upper bound

Best configurations combine finetuning with "the tempo range parameterisation of the
DBN," giving SMC mean improvement of **4.0 to 9.1 percentage points** over the
state of the art, and reducing required user correction operations by up to
two-thirds. A greedy per-file search over configurations gives "an average increase
of **23.1%**, relative to the baseline performance" — this is the "theoretical upper
bound" and the ">20%" figure in the abstract. It is an upper bound on
**user-assisted** improvement (finetuning on 12.5 % of each file), not an autonomous
one.

**Constituent papers (these are the thesis chapters, and are reachable):**

| year | title | venue |
|---|---|---|
| 2019 | Towards user-informed beat tracking of musical audio | CMMR 2019, 577–588 |
| 2019 | Tapping Along to the Difficult Ones: Leveraging User-Input for Beat Tracking in Highly Expressive Musical Content | CMMR (Springer LNCS 12631, ch. 5) |
| 2020 | Shift If You Can: Counting and Visualising Correction Operations for Beat Tracking Evaluation | arXiv / ISMIR LBD |
| 2021 | User-Driven Fine-Tuning for Beat Tracking | *Electronics* 10(13):1518 — Pinto, Böck, Cardoso, Davies |
| 2023 | Bridging the Rhythmic Gap: A User-Centric Approach to Beat Tracking in Challenging Music Signals | CMMR 2023 |

The 2019 CMMR paper is the one that most directly pre-empts us: it
"re-parameterises an existing deep learning based approach to enable it to more
reliably track highly expressive music," explicitly motivated by "the perceptual
difficulty of tapping the beat... associated with the musical properties of
expressive timing and slow tempo." That is our §on widening the DBN tempo range,
arrived at from the same premise. **We must cite it and state the delta.**

The 2021 *Electronics* paper (open access, read) is targeted fine-tuning of a
state-of-the-art network from "a very limited temporal region of annotated beat
locations," plus "a new annotation-correction approach for evaluation."

### Why this thesis is an opportunity, not just a citation debt

Sá Pinto's answer to "SMC is hard" is **put a human in the loop** — user taps, user
corrections, per-excerpt fine-tuning. Our answer is **fix the decoder's
mis-specification so no human is needed**. That is a clean, honest contrast and it
sharpens exactly the axis all three reviewers said was missing. It also converts
R2's "why not just fine-tune a model for this dataset?" from an unanswered objection
into a positioned one: fine-tuning and user-input are the established line of
attack; ours is the first that leaves the system fully automatic.

## 2. Schreiber, Urbano & Müller (2020) — global tempo on SMC is not well-posed

**Music Tempo Estimation: Are We Done Yet?** TISMIR 3(1):111–125, DOI
10.5334/tismir.43. Open ePrint at audiolabs-erlangen.de (read).

Datasets analysed include SMC alongside ISMIR04, Hainsworth, GTzan, Ballroom, RWC,
GiantSteps. The relevant findings: a **single global tempo is not meaningful** for
many SMC recordings; SMC's tempo annotations carry reliability problems; and the
paper effectively recommends against SMC for global-tempo benchmarking. It also
argues Accuracy1/Accuracy2 are too strict to capture real tempo behaviour.

**Impact on us — this cuts both ways and we should say so ourselves before a
reviewer does.** R2 flagged it against our lines 75–79. Our GT-tempo oracle
experiments *derive a global tempo from the annotations and hand it to the DBN*.
Schreiber et al. say that quantity is ill-defined for much of SMC. So:

- It **weakens** the oracle-tempo framing (already the reviewers' least-liked part,
  and R1 noted the oracle-tempo experiments "show little to no improvement" anyway).
- It **strengthens** the new thesis. If no single global tempo exists for this
  material, then a decoder built on a *homogeneous* tempo-transition prior is
  mis-specified by construction — which is precisely the claim the VBPM
  heterogeneous-scale work is set up to make. Schreiber becomes supporting
  evidence for the fix rather than an objection to the analysis.

Cite it in the motivation, not the errata.

## 3. Böck, Krebs & Widmer (2014) — the dataset-specific-model precedent

**A Multi-model Approach to Beat Tracking Considering Heterogeneous Music Styles.**
ISMIR 2014, Taipei, pp. 603–608. Zenodo DOI 10.5281/zenodo.1415240.

R2 raised this behind the question "if this content is special, why not retrain or
fine-tune a model specifically adapted to this dataset? Doing so could elicit clear
evidence that the general-purpose approach is confident-but-wrong, whereas the
SMC-specific approach could be confident-but-right."

This is the strongest single objection in the review set, because it proposes a
cheaper alternative explanation for our headline diagnosis. Böck et al. is the
canonical prior: style-specific models selected per input beat a single
general-purpose model. Either we run a leave-some-out SMC-adapted baseline, or we
argue explicitly why kernel mis-specification is a *different* failure from
train/test mismatch — the dithering artifact helps here, since it reproduces on a
**synthetic perfect metronome** with no musical style involved at all, which no
amount of retraining could explain.

## 3b. Krebs, Böck & Widmer (ISMIR 2015) — the quantisation is theirs; the estimation failure is NOT

**An Efficient State-Space Model for Joint Tempo and Meter Tracking.** ISMIR 2015, Málaga,
pp. 72–78. PDF: <https://www.cp.jku.at/research/papers/Krebs_etal_ISMIR_2015.pdf>
This is the paper madmom's DBN implements. Read 2026-08-21; quotes verbatim.

**They define the integer quantisation explicitly**, Eq. 7:
`M(T) = round(4 × 60 / (T · Δ))`, Δ = audio frame length. The `round()` is the artifact's origin.

**They discuss drawbacks of discretisation — but only resolution ones.** §2.2.1 *Time resolution*
(constant positions-per-bar means low tempi get lower time resolution than the frame rate
provides) and §2.2.2 *Tempo resolution* (constant tempo spacing vs the human JND of "around 2-5%
of the inter beat interval"). **Nothing about spurious tempo transitions, dithering, oscillation
between adjacent tempo states, or any bias in parameter estimation.**

**DECISIVE FOR OUR NOVELTY: λ was set by GRID SEARCH ON ACCURACY, never by maximum likelihood.**
§4: *"The maximum Cemgil values were obtained with **λ = 125**, and **λ = 95** respectively"*
(RNN-BeatTracker and GMM-BarTracker). §2.3.3: *"the rate parameter λ ∈ Z≥0 determines the
steepness of the distribution… In practice, for music with roughly constant tempo, we set
λ ∈ [1, 300]."*

They therefore **could not** have observed the corruption: grid-searching against an accuracy
metric is immune to it. The artifact bends the *likelihood* surface, not the *accuracy* surface.
ML was never attempted, so the failure was never visible.

**The consensus-vs-ML table — the paper's core argument in one figure:**

| estimator | λ |
|---|---|
| Krebs grid search, Cemgil accuracy (RNN-BeatTracker) | 125 |
| Krebs grid search, Cemgil accuracy (GMM-BarTracker) | 95 |
| madmom shipped default | 100 |
| CRF discriminative | 98.6 |
| dither-corrected mixture EM | 104.43 |
| **plain ML / exact EM, in-domain** | **40.3** |
| **plain ML / exact EM, on SMC** | **10.5, NOT CONVERGED (still falling at 8 iters)** |

Five independent estimators cluster in 95–125. Plain maximum likelihood lands at 40 and collapses
to 10.5 on SMC. **Claim: the tempo-transition prior has been hand-tuned since 2015 because
maximum-likelihood estimation of it silently fails, and it fails because the state space
quantises frames-per-beat to integers.**

⚠️ **Scope check still open:** the earlier prior-art pass flagged **Krebs's PhD thesis §4.3.2** as
the nearest neighbour to the dithering mechanism. The thesis has NOT been read. It may go further
than this paper. Read it before the novelty claim is finalised.

## 4. Also worth citing (not requested, but adjacent)

- **Davies, Böck et al., Evaluating the Evaluation Measures for Beat Tracking**,
  ISMIR 2014 — R2's specific comments on CMLt/CMLc, on the ±70 ms vs ±17.5 % window
  discrepancy, and on our F ≥ 0.8 "good" threshold all live in this paper's
  territory. Their line 359–361 comment (beats outside the F window but inside the
  CML window at slow tempi) is exactly this issue.
- **Krebs PhD thesis §4.3.2** — nearest prior art to the integer-interval dithering
  artifact per the VBPM prior-art check; must be cited when that artifact is
  claimed as novel.

## Action items falling out of this

1. **Rewrite the tempo-range and decoder-envelope claims as replications, not
   discoveries.** Sá Pinto Table 3.1 has the parameter move and Table 3.2 has the
   per-excerpt oracle envelope. Both must be cited and the delta stated explicitly:
   different and stronger activation models, systematic full-corpus evaluation rather
   than a 10-listener subjective study, and — the actual contribution — a *mechanism*
   plus a kernel whose shape is **learned by EM** rather than hand-set. This is the
   single most important revision falling out of the reviews.
2. Add a Related Work paragraph positioning us against the human-in-the-loop line
   (Sá Pinto) as the *automatic* alternative.
3. Move Schreiber (2020) into the motivation: "no well-defined global tempo" is a
   premise of the new argument, not a caveat on the old one.
4. Decide on the Böck (2014) response: SMC-adapted baseline, or the
   synthetic-metronome argument that the failure is not train/test mismatch.
5. Define CMLt/AMLt thresholds explicitly and justify the F ≥ 0.8 "good" cutoff
   (Davies et al. 2014) — cheap fixes, both explicitly requested by R2.
