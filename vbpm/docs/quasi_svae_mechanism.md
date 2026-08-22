# Why the chain posterior works and every amortized-location posterior nulled

**Question.** What exactly did the quasi-SVAE change (`schain`/`tchain`: encoder-emitted
per-frame potentials + exact forward–backward over a discretised (phase, tempo) state)
that took the model from F ~= null to within ~0.05 of Beat This on the same time axis
(0.498 vs 0.545, n=60 gtzan, `verify_align.py`) — and is the discretisation essential?

**Answer in one paragraph.** The change of variational family changed three things at
once, each provable from the model's own structure: (1) the chain family *contains the
exact posterior*, so the variational gap is exactly zero and training is (near-)exact
maximum likelihood with closed-form, sampling-free gradients; (2) the family carries the
posterior's *multimodality* (octave, alignment) through inference by summation instead
of asking a gradient to select a mode; (3) it changes *what function the encoder must
amortize* — from "the answer" (a global, discontinuous, error-amplified functional of
the whole input) to "local evidence" (a per-frame, smooth, error-forgiven one). The
discretisation itself is not one of the three: refining the trained model's grid from
96 to 384 bins moves the deployed path by 1e-4 rad and F by nothing — the bins are
quadrature nodes for a continuous-state model, not a modeling commitment. The
handcrafted prior is exonerated by a controlled A/B: both arms share it; one scores
0.65–0.98, the other 0.000.

Notation. Latent path z_{0:T} = (phi_t, omega_t): Markov prior p_theta(z) (von Mises
transition at kappa_physical = 383, log-tempo walk), per-frame emission p_theta(y_t |
z_t), audio features x, labels y. ELBO(q) = log p(y|x) − KL(q || p(z|y,x)).

---

## Part I — three exact properties of the chain family

The chain family is

    q_psi(z | x)  =  p_theta(z) · prod_t psi_t(z_t; x) / Z(psi),

with the encoder emitting one potential vector psi_t per frame and forward–backward
computing marginals and Z exactly.

**I.1 Containment, hence zero variational gap.** For any Markov prior with per-frame
emissions, the exact posterior is itself "prior × per-frame factors":

    p(z | y) = p(z) · prod_t p(y_t | z_t) / p(y).

So the family contains the exact posterior — set psi_t(·) = p(y_t | z_t = ·). At that
point ELBO = log p(y|x) exactly. Training `tchain` is therefore not variational in any
loose sense: it is maximum-likelihood training of a neural DBN, up to (a) the encoder's
ability to reproduce the label-likelihood tilt from audio and (b) quadrature error
(see Part III; both measured small). Nothing in Parts I–II uses discreteness per se —
only that exact marginalisation is available.

**I.2 Exact expectations, exact gradients.** Every ELBO term is a closed-form function
of the marginals: no sampling, no reparameterisation, no von Mises pathwise bias
(−48% at kappa=2), no one-sample responsibility noise, no gradient lottery. The
recurring seed-lottery / octave-hop phenomenology of the amortized arms is an estimator
property, and it leaves with the estimator.

**I.3 Locality: the family decides what the encoder has to compute.** The optimal
potentials are per-frame likelihood tilts — a *local, smooth* map from audio near frame
t. Measured on the trained `tchain` (probe_psi.py, 100 gtzan songs, 150k frames): the
learned potentials are statistically indistinguishable from uniform — per-frame entropy
4.564 nats vs log 96 = 4.564 — tilting at most ±10% relative to uniform, peaked exactly
at phase 0 on annotated-downbeat frames and at 180 deg elsewhere. Peak-picking this
"activation" directly gives **F = 0.000**; the same potentials through the sweep give
**F = 0.602** on the same songs (first-100 gtzan; corpus 0.75). AUC of the tilt against
the downbeat indicator: 0.823. That is the division of labour in one measurement: the
encoder contributes a whisper of local evidence per frame, individually worthless, and
*inference* — global, exact — converts 1500 whispers into placement. Errors in one
psi_t are averaged against the prior and every other frame (first-order forgiven).
Contrast the location families in Part II, where the encoder's output IS the answer and
every imperfection is charged at full price.

This also resolves the "why is it as good as Beat This" surprise: the trained object is
architecturally the classic activation + DBN decoder (Beat This activations feed the
encoder; the sweep is the decoder), but trained end-to-end by marginal likelihood. Its
ceiling is the frontend's evidence plus exact smoothing — the same ceiling Beat This's
own postprocessor lives under. There was never a mystery to within-0.05.

## Part II — three independent failure mechanisms of the amortized-location families

All retired arms (interval regression q, per-frame factorized q, mean-matched jitter q)
parameterise q by *locations* (means/offsets of phi_t, a regressed rate). Three
mechanisms, each sufficient, all measured:

**II.1 Representational floor (independence cost).** A factorized q must pay
KL(prod q_t || p(z|y)) >= the posterior's frame-to-frame coupling. Computed exactly in
the linear-Gaussian analogue of the bar-pointer chain (sigma_theorem.py, min over
diagonal q against the exact chain posterior): 0.66 nats/frame at sigma_phi = 1e-2,
3.4 nats/frame at 1e-4, linear in T — i.e. ~1000–5000 nats per 30 s window, against a
placement signal worth ~100 nats. The floor is theta-dependent, and its gradient on
theta rewards *decorrelating the posterior* — loosening the walk, flattening the
emission: it prices the model toward a metronome. A chain q zeroes this term by
construction (it inherits the prior's coupling).

**II.2 Amortization error is priced as physical jitter (smoothness pricing).** Even
with the right mode, a feedforward map emitting per-frame locations has per-frame
errors eps_t that the transition prior scores as innovations: cost ~ kappa_p · eps^2
per frame. Measured (rebuild verdict, 2026-07-27): the encoder's best placement is
~2000× jitterier than the prior's innovation scale; surrender (coast/metronome) is
483–589 nats/crop cheaper than jittery partial placement. The ELBO *correctly* sells
placement the family cannot deliver coherently. The objective is exonerated by the
same campaign: a free non-amortized q (SVI) under the identical ELBO holds placement.
The math is not rigged against the model — it is rigged against this parameterisation,
which converts encoder error into priced physical noise. (In the chain family, encoder
error is converted into slightly misweighted evidence instead.)

**II.3 Mode selection cannot be done by a gradient.** The exact posterior is multimodal
in the global variables. Measured on trained `tchain` (probe_posterior.py, n=100):
median 13% of posterior tempo mass sits an octave-class away (|d log rate| > 0.5); 57%
of songs hold >10% competitor mass; in 64% of songs the smoothed tempo decision differs
from the frame-0 marginal's argmax — the decision is made by accumulated evidence, not
readable off any prefix. Phase marginals hold 43% of mass >60 deg from their circular
mean throughout — the posterior keeps an *ensemble of coherent paths*, not a tube
around one. Under zero-forcing KL a unimodal q must pick one mode; the bound is equally
tight at every mode, the gradient toward the right one decays with mode separation, and
the measured lock basin is 0.3% of the rate axis. Enumeration (categorical rate,
summed phase) replaces mode *selection* by mode *integration*. Every positive result in
the project's history is this same move (coherence search, grid decode 0.899 vs
amortized 0.593, rate_grid 0.733, closed-form anchor).

**The controlled experiment (closes attribution).** `schain`, 2026-08-18, single
variable — `chain_posterior: chain` vs `frames`; same handcrafted prior, same triangle
emission, same encoder trunk, same potentials head, same exact KL accounting, same data
and epochs (abl_chain.log / abl_frames.log):

| arm | val F (ballroom/beatles/hainsworth/hjdb/rwc) |
|---|---|
| chain q | 0.973 / 0.943 / 0.794 / 0.977 / 0.654 |
| factorized q | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 |

The factorized arm does not even fail to a metronome: its MAP path has bin-step 0 at
100% of frames (mf_modes.log) — a frozen phase, prior-infeasible (−1.52 vs −0.67
nats/frame). The frozen configuration is a kinetic trap of frame-local gradients: to
start rotating, a contiguous span of frames must move together (two domain walls +
bulk), which a per-frame gradient cannot nucleate; forward–backward reaches the
coherent optimum in one sweep with no landscape to cross. Note the trap is *worse* than
the II.1 floor predicts — the mean-field arm does not even attain the mean-field
optimum.

**Corollary: the handcrafted prior is exonerated.** Both A/B arms use the fixed
physical prior; one works at 0.65–0.98. The F-null was never about prior trainability.
The Sohn conditional prior p_eta(z|x) remains the right *upgrade* (per-song tempo
level, per-song walk width — quantities the corpus says vary 20×), but it is an
accuracy axis, not the null's cause; a conditional prior grafted onto a location-family
q would null exactly as before.

## Part III — the discretisation is quadrature, not modeling

Refining the trained model's grid, reusing the SAME trained potentials
(Fourier-interpolated) and rebuilding the kernels at the finer resolution
(probe_posterior.py, P3):

| bins | F | mean |d path| vs 96 | scale-corrected logZ gap |
|---|---|---|---|
| 192 | 0.602 | 1e-4 rad | +0.69 nats (over ~1500 frames) |
| 384 | 0.602 | 1e-4 rad | +1.39 nats |

The 96-bin sweep already sits on the continuum limit: the von Mises transition's
Fourier coefficients decay like exp(−n²/2 kappa), so trapezoid quadrature on the circle
converges geometrically, and at kappa = 383 the tail beyond the 96-bin Nyquist is ~5%
and gone by 192. The model's state space is continuous; the *integrals* of exact
smoothing are evaluated numerically, exactly as every Bessel function in the vM density
is. The deployed output is already continuous-valued (circular-mean path, interpolated
crossings). "We discretised the state space" is the wrong description of `tchain`; the
right one is "we compute the continuous model's smoothing integrals on a grid whose
refinement changes nothing".

## Part IV — the continuous encoder the tutorial's Form 1 prescribes, and what it needs

The professor's factorisation answer (encoder factorises like the prior) is Form 1 of
the phase note: q(phi_0|c_0) · prod_t q(phi_t | phi_{t-1}, c_t), with a shared head
reading (cos phi_{t-1}, sin phi_{t-1}, c_t). Two structural facts say how far this can
go:

- **Chain-rule containment.** The exact posterior of a Markov chain factorises
  autoregressively with factors p(z_t | z_{t-1}, y_{t:T}) ∝ p(z_t|z_{t-1}) ×
  (likelihood-to-go). Form 1 is therefore *exactly sufficient in principle* — provided
  each head can read the backward evidence from the bidirectional context. Moreover,
  conditioned on the previous state, sharp transitions make each factor *unimodal*:
  Form 1's von Mises factors are adequate for every t >= 1.
- **The multimodality does not vanish; it concentrates at the base case.** All of
  Part II.3's mode mass lands in q(phi_0) and the tempo level. A unimodal boundary head
  inherits the 0.3%-basin problem intact. The base case must be *enumerated or mixed* —
  which is cheap, because it is one frame and one scalar, not the path.

None of the retired arms was Form 1: their per-step factors were mean-matched to the
prior ("the encoder chooses only how tightly each step is followed, never which way")
or offsets emitted from context alone — the feedback wire from the sampled phi_{t-1}
was never connected inside q. The user's intuition ("the encoder has h_{1:T} and
z_{t-1}, so the math cannot be fundamentally rigged") is correct — for a family that
actually conditions on z_{t-1}, with a multimodality-capable base case.

**The experiment and its outcome (2026-08-19, `vbpm/variants/archain.py`, branch
`archain-anchor`, jointly with a peer session).** Form 1 as written — shared step head
on (cos phi_{t-1}, sin phi_{t-1}, c_t), sequential sampling with the implicit von Mises
gradient, per-step closed-form KL, initialised exactly at the prior — iterated through
an overfit ladder (ballroom_0, 400 epochs/arm) that closed one failure channel at a
time. The ladder, in order:

1. lognormal regressed rate: harmonic-hops forever (registered prediction confirmed).
2. categorical rate alone: flatten-the-emission surrender (b falls, coast at the prior).
3. + pinned emission: clock-stop surrender (unbounded delta cancels the advance).
4. + absolute delta bound: clock-stop via the slowest bin financing the bounded delta.
5. + phi0 ENUMERATED (12-grid): first arm to hold the true mode 300 epochs (CMLt 0.69),
   in-tol capped ~10% by the rate grid's 10.5% spacing (in-tol needs ~0.5% precision).
6. phi0 by per-rate closed-form ANCHOR (peer session): same plateau, slower start —
   the anchor's amortization target is local but its gradient arrives through one
   pooled circular mean per candidate, a training-speed bottleneck, not a ceiling.
7. anchor + WITHIN-MODE RESIDUAL (rate = bin * exp(s(x)), |s| bounded at half a bin,
   residual-adjusted ramp fed to the anchor fold): MONOTONE convergence to truth —
   ratio 1.06 -> 1.02, med|err| 499 -> 129 ms, CMLt 0.92 — the first continuous
   amortized bar-pointer arm that converges. Family axis closed.

Then the decisive twist: from the converged-on-truth state the ELBO moves to the DOUBLE
octave and tracks it perfectly (AMLt 0.958, med|err| exactly half a bar; reproduced
exactly on a rerun). The defection is the OBJECTIVE's, not the family's, and it was
priced at the trained state with both hypotheses residual-tuned exact: the 2x
explanation beats 1x by +0.04 nats (recon +0.05, KL -0.01). The objective is FLAT
across metrical levels — not wrong, indifferent — and drift decides. The pinned-sharp
control confirms two-sidedness (it surrenders by UNDERCOUNT, ratio 0.40): no static
sharpness makes truth the argmax. This retro-explains the octave seed-lottery eras,
and the enabler matches era-3 lore: the octave bin sits 3% detuned, unreachable until
the residual made the harmonic precisely tunable — a harmonic the optimizer can tune
is a harmonic it takes.

The first-principles repair is a LIKELIHOOD change, not an auxiliary term: the binary
per-frame target discards exactly the level information the annotations carry. The
three-way (non-beat / beat / downbeat) emission prices a metrical level: under 2x,
half the claimed downbeats land on frames annotated as BEATS, ~log 2 per conflicted
event ~= 8 nats/window against a 0.04-nat status quo (~1 nat suffices).

**Capstone verdict (same day).** The class-emission arm converged and reached a PERFECT
lock — ep375: F 1.000, in-tol 100%, med|err| 38 ms, CMLt 1.000 — holding 1x through and
150 epochs past the binary arm's defection window. Then it defected anyway at ep399,
and the re-priced gap at that state explains how: 2x now beats 1x by +2.12 nats (all
recon). The 8-nat a-priori discrimination was CO-ADAPTED AWAY — the class shapes are
free Fourier series, and the optimizer reshaped them until the level swap paid. The
class vocabulary raised the defection barrier (ep250 -> ep399, via a perfect lock); it
did not make truth the global optimum, because nothing learnable can: any level
discrimination routed through free emission shapes is a discrimination the optimizer
can spend. Third instance of the day's second law (after sharpness two-sidedness and
the tunable-harmonic account): what the optimizer can tune, it will spend on the
defection.

**Where level identifiability must therefore live** — something the optimizer cannot
reshape, in order of derivability: (1) STRUCTURED emission: tie the beat class to
subdivision-shifted copies of the downbeat bump, so a level swap costs likelihood that
no reshaping can refund — ELBO-internal, and it is the road on which the meter latent
becomes consequential; (2) the CONDITIONAL PRIOR p_eta(rate | x) of the Sohn CVAE —
an audio-read tempo prior prices the level by nats the emission cannot touch; today's
result promotes it from accuracy-axis to identifiability mechanism, which is the
strongest first-principles argument yet for the professor's design; (3) p(N|phi) in
the interval vocabulary; (4) pre-registered validation checkpoint selection — outside
the model, legitimate as harness discipline (the capstone trajectory PASSES through
F 1.000; a select rule banks it), but it is selection, not identification.

**The fully continuous endgame, in order of principle:**
1. *Quadrature reading of tchain* (zero work): already continuous in the only sense
   that matters (Part III); write it up as exact smoothing computed in a quadrature /
   truncated-Fourier basis with geometric convergence.
2. *Form 1 + enumerated global mode* (archain): continuous states, amortized one-step
   factors, exact sum over the small discrete mode set. This is also the natural
   chassis for the Sohn CVAE — the conditional prior heads condition the same chain.
3. *Assumed-density continuous messages* (`vmchain`, built): von Mises belief per tempo
   bin — exact rotation/evidence closure, moment-matched mixing; its known gap is
   alignment multimodality within a tempo, i.e. it needs mixture messages.
4. *Variational SMC / FIVO*: particles over continuous z with the learned potentials as
   twist functions; the bound tightens to the likelihood as particles grow, recovering
   tchain's exactness without any grid — and it is the natural bridge to the SMC Blind
   Spot repair paper.

## Experiment index

- schain A/B (single variable, 2026-08-18): /tmp/scratch/abl_chain.log,
  abl_frames.log; mechanism probe /tmp/scratch/mf_modes.{py,log}
- analytic factorized floor: /tmp/scratch/sigma_theorem.py (0.66–3.4 nats/frame)
- posterior anatomy + quadrature: scratchpad probe_posterior.py (P1 octave mass /
  flips, P2 whisper-potentials, P3 refinement); probe_psi.py (entropy/AUC/profiles)
- Beat This same-axis head-to-head: /tmp/scratch/verify_align.py (0.498 vs 0.545)
- Form 1 arms: vbpm/variants/archain.py, vbpm/configs/archain.yaml; checkpoints to
  /disk4/jaehoon/vbpm_ckpt/archain_{cat,logn}
