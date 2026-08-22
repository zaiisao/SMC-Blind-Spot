# The case for enriching q: mixture and diffusion samplers of phase

**Claim.** The unimodal von Mises family currently used for the phase draw is
theoretically incapable of representing the posterior that the model itself defines.
Two independent arguments establish this — one about posterior shape, one about what a
learned corrector can be taught — and they motivate different instruments. The proposed
ablation ladder (fixed two-component mixture now, diffusion sampler as the limiting
instrument) is the experiment that assigns blame decisively among the three possible
culprits: the objective, the amortization map, and the variational family. Every branch
of its outcome tree changes what we do next, which is the standard an ablation must meet
to be *needed* rather than merely interesting.

Empirical results appear only in §8, clearly labeled. Everything before that is
derivable from the model's own structure.

---

## 1. Setting

The generative model is the bar-pointer process: bar phase `φ_t ∈ [0,2π)` (a wrap is a
downbeat), driven by a per-frame rate `ω_t` (a random walk in log-rate), with phase
innovations `Δφ_t` around the ramp, and an emission that scores the trajectory against
the observations. The amortized posterior factorizes as

    q(ω_{1:T}) · q(Δφ_{2:T}) · δ(θ)

with `q(ω_t)` log-normal, `q(Δφ_t) = vM(κ_q(x))`, and the anchor `θ` a deterministic
function of the input. **Every factor is unimodal.** The ELBO is exact in form; the
question this document addresses is whether the family can place `q` near `p` at all.

---

## 2. Proposition 1 — the posterior the model defines is not unimodal

Two mechanisms, both structural: they follow from the model specification, prior to any
data.

**(a) Comb symmetry of the emission.** Conditional on a rate hypothesis, translating the
whole trajectory by one beat period maps beat-consistent explanations onto
beat-consistent explanations. The emission is therefore near-periodic in the anchor, and
`p(θ | x, ω)` has approximately one mode per beat position in the bar. Jointly with the
octave ambiguity in rate (a trajectory at `2ω` with alternating strong/weak alignment is
a competing explanation by construction), `p(θ, ω | x)` is a ridge-and-comb surface.
This is the symmetry group of the problem, not an empirical accident.

*Honesty note (status of the premise):* on the current emission, the conditional anchor
basin at the true rate measured flat-top **unimodal**; the multimodality lives in the
joint with rate, where the lock basin occupies ~0.3% of the rate axis. The proposition
this document needs is only that the **joint** is not unimodal — which is both derivable
and measured.

**(b) Two-regime transition ⇒ mixture posterior at crossings.** The prior on the phase
innovation at bar crossings is two-regime: tight walk (`κ_physical`) with high
probability, wide redraw (`KAPPA_INTER`) otherwise. The exact single-site posterior is
proportional to (mixture prior) × (likelihood); a likelihood tilt acts on each component
separately and preserves mixture form. Therefore **the exact posterior innovation at a
crossing is itself a two-component mixture whenever the prior is** — a statement at
theorem level, not a modeling preference. And the two-regime prior is not a whim: the
measured increment law of real tempo trajectories is itself two-component, with a single
Gaussian ~80× too thin in the tail (§8).

---

## 3. Proposition 2 — a unimodal family facing a multimodal posterior fails in exactly two ways

`KL(q‖p)` is zero-forcing. For well-separated modes the stationary points of a unimodal
`q` are:

1. **Mode collapse** — all mass on one mode. Which mode is decided by initialization: a
   lottery the objective cannot see past, because the bound is equally tight at each
   basin. The posterior's genuine uncertainty across modes is unrepresentable.
2. **Variance inflation** — `κ → 0`, covering all modes and paying likelihood at every
   one of them.

Both are *family* pathologies: no optimizer, no schedule, and no added capacity in the
amortization map repairs them, because `inf_{q ∈ vM} KL(q‖p)` is bounded away from zero
by a constant that depends only on the mode separation.

**Corollary (attribution).** Any observed failure of the current model is confounded
across {objective, amortization map, family}. The family is the only member of the
triple that cannot be exonerated by more training or more compute. It can only be
exonerated **by construction** — by widening it until it is universal and observing that
nothing changes. That is what the ladder in §5 does.

---

## 4. Two independent theoretical roles for an enriched q

**Role A — posterior matching.** §2 + §3 directly: if the exact posterior is
mixture-shaped, the minimal family that can match it is a mixture, and the KL gap closed
by the enrichment is real bound improvement.

**Role B — coverage for the corrector (the denoising argument).** The corrector is a
learned reverse dynamics: it can only learn to repair states that occur under `q` with
nonzero mass *and* are scored by the emission while displaced. Under a concentrated
unimodal `q`, displaced states have measure ≈ 0 in training; the corrector's optimal
policy on states it never sees is a constant.

The mixture places the corruption **inside q**, and this is the theoretically clean way
to do it: every displaced sample is a legitimate draw of `q`, its log-density `log q` is
exact (the realized sample is scored under the full mixture density), so the estimator
remains an unbiased one-sample estimate of a valid lower bound. Contrast scheduled
sampling and forced kicks, which corrupt outside the measure and therefore produce
gradients of *no well-defined objective*. Role B is precisely the diffusion training
principle — a denoiser needs corrupted samples with a scoring rule — and the mixture is
its one-level, fixed-schedule degenerate case: one noise magnitude (`KAPPA_INTER`), a
Bernoulli(`mix_eps`) schedule, corruption only at crossings, and the reverse process
trained only implicitly through the ELBO gradient.

Roles A and B are logically independent: A can hold while B fails (posterior is
multimodal but the corrector is fine) and vice versa. An instrument that cannot separate
them produces uninterpretable nulls. The ladder separates them.

---

## 5. The instrument ladder

| rung | family for the phase draw | can show | cannot show |
|---|---|---|---|
| von Mises (mainline) | unimodal, 2 params | — (the control) | anything about the family |
| fixed 2-mixture (`mixture_q`) | 2 modes, fixed weight & width | whether *any* corruption coverage moves the corrector (Role B, minimal dose) | that expressiveness is sufficient — mode budget and shape are hardcoded |
| diffusion sampler | universal on the circle | whether the family axis, taken to its limit, changes anything at all | — (it is the limit) |

**Rigor at the diffusion rung.** An implicit sampler has no tractable entropy, and the
ELBO needs `log q`. The standard repair is the auxiliary-variable bound: treat the
diffusion's intermediate states `u` as latents, introduce the reverse-process factor
`r(u | z)`, and optimize

    E_q [ log p(x, z) + log r(u|z) − log q(z, u) ]  ≤  ELBO  ≤  log p(x).

The bound is looser, but **one-sidedly** so: a win under the looser bound is a fortiori
a win. A null requires the §6 controls to interpret, which is exactly why the ladder has
a middle rung.

**Why diffusion is the terminal rung and not a mainline proposal.** Phase and anchor are
one-dimensional and periodic. Anything an implicit sampler can represent on `S¹`, a grid
represents exactly and more cheaply at deployment. A diffusion win therefore does not
ship a diffusion — it licenses **structured inference** (enumeration, forward–backward)
on the axis it won, exactly the move that resolved the rate axis. The diffusion's role
is *measurement*: it is the least-assumptive instrument for the question "is the family
the binding constraint?", and its value is that its answer is decisive in both
directions.

---

## 6. Decision logic — why this is needed, not nice

The outcome tree, with the action each branch dictates:

1. **Mixture moves the corrector** (kick decay, placement) → Role B confirmed at minimal
   dose → escalate along the same axis: dense noise ladder with an explicit denoising
   target (the diffusion rung), or jump directly to structured inference over phase.
2. **Mixture null, diffusion wins** → the fixed mode budget/shape was binding; the
   posterior's multimodality is real (Role A) → structured inference over phase; no
   further amortized-family work.
3. **Both null** → the family is exonerated *by construction*. Blame moves to the
   objective's flatness in placement (independently measured, §8), and effort redirects
   to the emission. No family argument can be raised against the model again — this
   branch is the one that buys permanent clarity.

Attribution in every branch depends on controls already in place: the prior's crossing
gate is switchable independently of the q-mixture, so improvement is assignable to the
posterior side; an earlier probe that conflated the two produced results attributable to
neither and is inadmissible.

---

## 7. Anticipated objections

**"Phase is 1-D — just enumerate it on a grid now."** At deployment, yes; that is the
endgame this ladder converges to in branches 1 and 2. But the object of study is the
training-time gradient path through amortized `q`, and the *joint* over rate × phase ×
per-frame innovations does not grid — only 1-D marginals do. The ladder is how we learn
**which** axis deserves the grid before paying for it.

**"The diffusion rung breaks the ELBO."** It loosens it, one-sidedly, via the
auxiliary-variable construction (§5). Validity is preserved; only tightness is spent,
and it is spent to buy universality.

**"Learn `mix_eps` and the wide κ instead of fixing them."** For a mainline model,
eventually. For the ablation, fixed values are the point: a learned `mix_eps → 0`
collapse would be observationally indistinguishable from a null, destroying exactly the
attribution the experiment exists to provide.

**"This is scheduled sampling with extra steps."** No: the corruption's density is
charged to the bound, so the gradient is the gradient of a valid objective (§4). That
distinction is the difference between an ELBO method and a heuristic.

---

## 8. Empirical anchors admitted into this document

Clean-pipeline measurements only, each carrying exactly one premise above:

- **Free-form (non-amortized) q holds placement where amortized q fails.** The objective
  is exonerated on this axis; the gap is on the q side — family or map. (Motivates the
  whole program: the culprit is inside `q`.)
- **Corrector kick test:** the cell converges to a constant trim and forced kicks do not
  decay. (Motivates Role B; also warns that corrupted states alone may not suffice —
  hence the escalation path to an explicit denoising target.)
- **Increment law of real tempo:** two-component; a single Gaussian is ~80× too thin in
  the tail. (Grounds §2b's two-regime prior in data.)
- **Rate axis precedent:** the lock basin occupies ~0.3% of the rate axis; replacing
  regression with categorical search took corpus F from 0.16 to 0.733 with no other
  change. (Precedent that multimodal axes yield to enumeration, not to a better
  regressor — the exact template branches 1–2 would follow for phase.)
- **Placement flatness of the objective:** ~1.3 nats span the full range from ceiling to
  chance in-tolerance placement. (The risk that a Role-B null is ambiguous; this is why
  the ladder separates Role A from Role B rather than running one confounded probe.)

---
---

# Appendix — full context dump (machine-oriented; source for the LaTeX document)

Not written for readability. Every number below is from the clean pipeline (2026-07-30+)
or from the current working tree (branch `rate-init-and-clip`, uncommitted past
`3e06a42`). Dates given where attribution matters. Symbols follow the code.

## A. Current implementation, exactly (vbpm/model.py, vbpm/variants/interval.py)

**Latent state.** Bar phase `phi_t ∈ R` (unwrapped; a crossing of a multiple of 2π is a
downbeat), per-frame rate `dotphi_t > 0` (radians/frame), global rotation `theta` (=
"phi_0" in discussion; the anchor). fps=50 typically; window ~2250 frames (45 s).

**Constants (measured, not chosen):** `KAPPA_PHYSICAL = 383.0` (tight phase-innovation
concentration; fit), `KAPPA_INTER = 17.0` (wide/crossing concentration),
`TEMPO_PRIOR_MU = -2.5028`, `TEMPO_PRIOR_SIGMA = 0.5005` (log-rate initial prior),
`TEMPO_WALK_SIGMA = 0.00212` (Gaussian log-rate walk), mixture walk
`WALK_MIX_W = (0.687, 0.313)`, `WALK_MIX_SIGMA = (0.00029, 0.00377)` (corpus refit;
under the shipped single Gaussian ~20% of real knot moves were "impossible", tail ~80×
too thin), gated variant `WALK_INTRA_SIGMA = 0.00029`,
`WALK_INTER_W = (0.646, 0.354)`, `WALK_INTER_SIGMA = (0.0247, 0.198)` (crossing-gated).
`knot_stride = 25` frames, `kappa_place` annealed 3 → 300 over 0.7·epochs (default
`kappa_anneal = "3,300,0.7"`), `b_ratio = 0.1`, `dec_warmup = 15` epochs (corrector
frozen before).

**Amortized posterior (Encoder).** 2-layer TransformerEncoder, d_model=128, 4 heads,
`Dropout1d(0.1)` on input only (attention/FF dropout 0 — the dropout×walk tax, ~280k
nats, trained tempo flat; de-taxed 2026-08-15), optional sinusoidal PE (default off).
Four per-frame scalar channels: `rotation_weight_logit`, `phase_log_kappa` (plus
constant bias log 383), `tempo_log_mu` (bias init at prior mu), `tempo_sigma_logit`
(softplus; ceiling issue in C below). Factorization:

- `q(dotphi_t) `: lognormal, `dotphi = tempo_mu * exp(tempo_sigma * eps)`, eps ~ N(0,1),
  reparameterized.
- `q(Δphi_t)` : vM(0, kappa_q,t) innovations around the rate ramp — **mean-matched to
  the prior transition**. The encoder chooses only *how tightly* each step is followed,
  never *which way*. Sampler = Best–Fisher rejection with pathwise reparameterization
  (vbpm/vonmises.py; custom A'(κ) backward, exact below κ=50, series above — the exact
  float32 form flips gradient signs at κ~1e5; known pathwise bias −48% at κ=2,
  negligible above κ≈50, operating point 383).
- `theta` : **deterministic** (a Dirac). Closed-form circular mean
  `theta = −arg Σ_t a_t e^{i·ramp_t}` with `a_t = sigmoid(rotation_logit_t) · w_t` and
  the ramp **detached** (`Encoder._anchor(mean_ramp.detach(), rotation_weight)`;
  the undetached backdoor gradient measured −4096 nats vs the honest path's +2707).
  `a_t` is x-side only, so the read-out is deployable/label-free.

**Trajectory assembly (`VBPM._scan`).** `phi = theta + cumsum((dotphi·exp(corr) +
jitter)·pair_mask)`, where `corr` is a scalar log-rate trim emitted every
`knot_stride=25` frames by `ZDecoder` — an AR transformer over knot tokens
`(memory_frame, cos phase, sin phase)`; correction constant within each 25-frame
segment. `dot_eff = dotphi·exp(corr)` is what the walk prior scores (pre-bound; the
post-bound version laundered noise through the clamp and pointed the prior gradient
INTO both rails — fixed 2026-08-13).

**Gradient split (the straight-through trick).** `phi_place = phi.detach() +
(theta − theta.detach())`. The placement factor is evaluated on `phi_place`, so it
grads only into theta (rotation channel); the interval/Jacobian factors are evaluated
on `phi`, owning rate. Reason (measured): `phi_1 = mu0(t_1) + theta` gives
`d(κ cos phi_1)/d log k = −κ sin(phi_1)·mu0(t_1)` — with the first annotation ~42
frames in, ±476 nats/unit-log-rate of sign-flipping force against the interval ruler's
flat ±(N−1)/b = ±110; it reversed the net rate gradient exactly at k=1.25 and k=2.65,
the harmonics runs park on. Detach → single-song overfits went 2/10 → 10/10 holding
true tempo.

**Interval emission (`interval_loglik`), the current observation model.** Observation =
annotated downbeat times t_1..t_N in-window (not a per-frame indicator). The map
(t_1..t_N) → (phi_1, log r_1..log r_{N−1}), r_i = (phi(t_{i+1})−phi(t_i))/2π, is a
bijection while phi increases, so N coordinates carry exactly N density factors:

- `place` = vM(kappa_place) on the FIRST annotation's phase only. Scoring a vM at every
  annotation put 2N−1 factors on N coordinates → unnormalized with a latent-dependent
  normalizer (measured Z = 0.013, drifting 3.6 nats with the model's own rate).
- `interval` = N−1 Laplace(b_ratio) factors on log r_i (Huber optional; Laplace votes
  ±1/b at any error size).
- `jac` = Σ_i log dotphi(t_i) − Σ_i log(2π r_i), the exact change of variables.
  dotphi(t_i) read from the sampled path over a ±25-frame baseline (`path_dotphi`):
  at κ=383 per-frame jitter sd (0.072 rad) exceeds the bar rate (0.051 rad/frame), a
  one-frame difference measures noise, ~23% of slopes negative, donated 23–35 nats of
  octave margin to the 2× harmonic.
- optional `disp = kappa_place · disp_weight · resultant(all annotation phases)` —
  verdict NULL/NEGATIVE (10-nat demand vs 90-nat walk price; w≥3 re-corrugates;
  resultant is octave-blind: 0 mod 2π survives integer multiplication).

**KL terms.** Phase: closed-form KL(vM(μ,κ_q)‖vM(μ,κ_p)) per step — mean-matched, so it
prices concentration only and is **provably anchor-blind**. κ_p = 383, or KAPPA_INTER
at crossings when `kappa_gate` (independent switch). Tempo: per-frame lognormal entropy
+ walk log-prior on realized dot_eff (kinds: gauss / mix / gated). β=1, samples=1;
these are ELBO-faithful values, changing either breaks the bound.

**Deployment (`infer_phase`).** Mean path: ramp from tempo_mu, corrector on, no jitter;
theta from the closed-form anchor on the (undetached) mean ramp; downbeats = linear-
interpolated 2π-crossings (`downbeat_times`; frame quantization gave a +1-shifted
early bias, fixed in scoring 3e06a42). Encoder reads AUDIO ONLY (Beat This-style
frontend features, 512-dim penultimate); labels enter the loss only.

**ELBO derivation status (audited line-by-line 2026-08-14).** The objective IS
derivable as a Sohn CVAE over annotations, log p(y|x) ≥ L: tempo prior+entropy exact
(both in log-rate, Jacobians cancel), phase KL exact/closed-form, place normalized,
interval normalized (log 2b present), jac exact. TWO REAL HOLES: (1) q(phi_1) is a
Dirac at theta(x) — the −log q(phi_1) term is +∞ and silently dropped; not an ELBO in
that coordinate. (2) No p(N|phi) factor: emission is p(times|N, phi), nothing charges
for claiming the wrong bar count — this is the *derived* route to octave
identifiability (post-hoc count terms refuted; must be in training). Deliberate
departures: the detaches preserve the ELBO's value but make the gradient a surrogate;
kappa annealing means each epoch maximizes a valid ELBO of a *different* model.

**mixture_q variant (uncommitted, vbpm/variants/mixture_q.py).** As §4–5: at crossings
`q(Δphi) = (1−ε) vM(κ_q) + ε vM(17)`, ε=0.3 fixed; `lq` = exact mixture density at the
realized sample; phase KL becomes one-sample (`phase_kl_sampled`), so ε=0 reduces to
mainline in expectation, not per draw. Prior untouched; crossing gate on p stays
independently switchable (`kappa_gate`). Crossing indicator from the detached mean
ramp's 2π-floor changes. Diagnostic `wide_rate` = fraction |Δphi|>0.5.

## B. The phi_0 / anchor problem — complete verdict chain

The structural fact underneath everything: because q's phase chain is mean-matched to
the prior, **only two channels steer placement — the rate and phi_1**. Confirmed
causally: 600 epochs with random phi_1 never recovers placement on any song/seed.

1. **Landscape (2026-08-07, workflow-verified, 196 gtzan windows).** The anchor
   objective over constant shifts is UNIMODAL — 195/196 single peak, no beat-position
   modes (y is downbeat-only), peak at truth 92% within 70ms. But FLAT-TOPPED: 328 ms
   of shift within 20 nats; err ≤ 70ms costs median 2 nats, so SGD stops anywhere on
   the top. Anchor gradient at μ correctly signed in ~100% of wrong windows (median 40
   nats/rad) — the earlier "multimodal mode-averaging / gradient fixed point" story is
   REFUTED at this conditional level. (The corrugation/multimodality lives in the JOINT
   with rate: lock basin ≈ 0.3% of the rate axis, 0.3%-period oscillation outside it;
   Adam diffuses across octaves and catapults out of the well.)
2. **Representation, not enumeration, was the fix (2026-08-09).** Frozen-backbone lab,
   all heads trained on the same cached objective: pooled-trunk/frame-0 snapshot head
   206±9 ms / 25.7% in-tol; phase-binned → angle 38±2 ms / 61.2%; CLOSED FORM
   `−arg Σ a_t e^{iμ_t}` on the downbeat channel, ZERO params: 46 ms / 62.8%;
   enumeration bound 11 ms / 92.3%; deployed co-adapted v2 head 27 ms / 71%. The
   original encoder head failed because its INPUT was a BiGRU frame-0 snapshot, not
   phase-folded evidence. Half-bar flips 8.1% remain where a learned head should beat
   the first harmonic and doesn't.
3. **F ≡ anchor-within-tolerance (2026-08-09, decisive).** Read-out swap on banked
   checkpoints, no retraining: F 0.468 → 0.700 (closed-form moment) → 0.752 (learned k
   head). 82% of the +0.284 jump is deployment read-out alone. Identity holds to ~1
   point across checkpoints (6.7%→0.07, 45.6%→0.468, 70.8%→0.700): tempo/drift/emission
   were already right; F measured nothing but bar placement. Anchor error is a
   VARIANCE failure, not bias (non-flip mean +22 ms vs sd 85 ms; moment read-out cuts
   sd 85→50 ms, gross >250ms flips 26.9→15.7%).
4. **Making phi_0 a latent — both attempts dead.**
   - *uniform-phi0 (2026-08-14):* q(phi_0)=vM(anchor, κ_θ) sampled, uniform prior,
     exact KL, amortized κ_θ (5th channel). 3/3 seeds: κ_θ collapses 100 → 0.0–0.2 by
     ep 200; in-tol 54% → 4%; med err 110 → 325 ms; rate untouched (1.00–1.03).
     Mechanism: collapse forfeits ~300 nats of placement factor to save ~5 nats of KL —
     a gradient pathology, not an optimum: the KL push on log κ is first-order, the
     recon reward for sharpening second-order. **THE ELBO DOES NOT SEE THE DAMAGE**:
     ELBO ON−OFF = +15, +596, −1823 per seed — ON wins 2/3 while placing 3/3 worse.
     Never rank these arms on ELBO/recon. Untested: κ_θ tied to the anchor's own
     resultant; KL warmup.
   - *theta-as-latent with coherence κ (2026-08-14):* q(theta)=vM(−arg R, κ_θ(|R|)),
     the exact missing-KL repair. NULL, 12 runs: 54.5% vs 50.3% final. The coherence
     gradient has the RIGHT shape (∝|cos|, maximal on the bar line, vs the old μ-path's
     |sin| which vanishes exactly there) but is 6000× weaker in norm (5.1e-2 vs
     3.1e+2). No viable κ regime exists. Second recorded probe-pass/training-null
     dissociation.
5. **The PAD-vote anomaly (2026-08-14, mechanism NOT established).** `Encoder.forward`
   calls `heads(features)` without the mask, so rotation_weight is unmasked. On the
   probe window 33% of frames are padding; the ramp is frozen there (cumsum gated), so
   pad frames form one coherent constant-phase vote block in the circular mean. Passing
   the mask — the obviously correct one-liner — costs 45 pts in-tol (54% → 9%, 3/3
   seeds) while IMPROVING recon (−131.7 → −76.5). Ranking inversion again. Reverted;
   unexplained. Any LaTeX claim about the anchor must not silently assume the masked
   (clean) form matches reported numbers.
6. **The binding blocker — the objective is (near-)flat in placement (2026-08-14).**
   `place` scores exactly ONE annotation of ~23; `interval` and `jac` are rotation-
   invariant. Measured: 1.3 nats spans in-tol 100% vs 54% (ballroom_0: ep400 100% at
   recon −40.40; ep599 54% at recon −39.08 — best recon of the run = worst
   post-convergence placement). 12/12 runs × 2500 ep touch ≥83% in-tol then wander off
   to 9–92%: the model FINDS correct placement and leaves, because convergence is on
   something that isn't placement. `place` is not useless — uniform phi_1 gives 14.8%
   vs 49.8% mean in-tol — but placement evidence does not scale with N. Untested
   repair: move the anchor coordinate from u_1 = phi(t_1) to the MEAN phase across all
   annotations (same bijection, same normalization, zero new params; evidence scales
   with N, residual centered not end-anchored; independently supported by the
   closed-form result in B.2).

## C. The rate axis — solved, and the template it provides

- Interval term ALONE: ratio 1.00 on 6/6 single-song runs, mean |ratio−1| 0.002 (better
  than the full ELBO's 0.005). Displaced-rate gradient (noise-free, ±6%): points home
  at EVERY displacement, 3/3 songs; log-lik peak within 0.2% of truth. The ±200-nat
  ripple between adjacent 2% steps is a SAWTOOTH (level jumps as annotations reassign
  across cycles) — local slope always correct. Rate locks 1.00–1.01 even with phi_1
  drawn uniformly at random each step. Gradient-SNR methodology: never measure at a
  converged model (mean grad = 0 by definition); measure at a known displacement.
- History that motivated search: Bernoulli/BCE emission is distance-blind → rate axis
  corrugated + octave-degenerate (30 single-song runs, 0 retained, endpoints always
  harmonics; true rate beats 2× by only ~1 nat/song); upper rate clamp had exactly zero
  gradient (absorbing); five objective bugs fixed 2026-08-13 by gradient decomposition
  (anchor backdoor; post-bound walk scoring; per-bar entropy = bar-count subsidy;
  widened-y density triple-pay; emission gain pinned at p≈0.9 vs data's b~12, p~0.35)
  → truth becomes value-optimum by ~120 nats; remaining blocker was the 0.3% lock
  basin → **categorical q(rate)**: 1024 log-spaced candidates (0.29% spacing < basin),
  closed-form anchor per candidate, exact expectation over the grid, no rate gradient
  anywhere. Corpus: gtzan F 0.733 / CMLt 0.870 / AMLt 0.876 label-free.
- The amortization-gap bracket the ladder must respect: same objective family, gtzan —
  amortized regression q: F 0.163/0.134 (AMLt 0.49: right structure, wrong level);
  categorical search q: 0.733; untrained search read-out: 0.570. Precedent: multimodal
  axes yield to enumeration, not to better regressors, and training the evidence head
  under the search DOES amortize (0.570 → 0.733).
- Tempo-width pathology (independent 4-agent confirmation): per-frame tempo entropy is
  summed over 2250 frames while the window holds ~16 distinct bar-pooled values →
  +1149 nats for widening vs an emission spanning ±100; q's per-bar sd rails to the
  ceiling. Pinning sigma_ceil=0.01 rescues retention causally (41→95.5%, SNR 0.28→12.5)
  but is a band-aid; charge entropy per distinct latent. 100% of the rate gradient is
  the emission (KL contributes ±0.0).

## D. The corrector / drift channel — Role B's evidence base

- **Kick test (filter v1, 2026-08-14): NEGATIVE.** Phase kicked ±0.3/±1.0 rad at frame
  700 on the trained model: deviation stays EXACTLY at kick size at every horizon
  (+1..+738 frames), cell-ON ≡ cell-OFF. The cell is active (|corr| ≈ 0.097 pre = post
  kick) but learned a CONSTANT ~+10% trim: a phase-responsive correction costs
  thousands of walk-nats (Δc charged per frame), a constant costs ~0. Third independent
  measurement of the same ledger (dispersion verdict, anchor-coast anatomy): **the walk
  prior prices within-window correction out of the market**, and a free-form learner
  offered the wire buys the constant. Two fatal accounting flaws found and verified
  (dropped Cov(c_t, ε_{t−1}) cross-term — which SUBSIDIZES corrective cells, and the
  cell still declined; last-frame boundary bug). Strategic fork recorded: reprice the
  correction channel vs move correction to read-out (search/decode — the only
  corpus-validated placement mechanism).
- **Coverage premise for Role B:** under κ=383, q's own samples essentially never
  visit displaced states; forced kicks are off-measure (gradients of no objective);
  mixture_q is the on-measure minimal dose (ε=0.3/bar, one wide level).
- **Drift is real but currently dead capacity:** anchor+coast anatomy (drift channel
  untrained, corr≈0); rate_grid drift gap measured — steady tempo 100% in-tol, smooth
  1.61× in-window accelerando → compromise rate 0.88, ~50% in-tol, 13.9× rubato →
  commits to a 2.48× sub-pulse, 19%. Caveat: the evidence resultant stays 0.78–0.87
  even when half the downbeats are wrong (self-consistency, not truth; read in-tol).
- zdec wake destroys (transformer-decoder wake wrecks all arms 4–17%; hence
  dec_warmup and frozen-pre-wake arms); PLL read-out NULL on clean ground (PK 0.96 >
  PLL > OL only on leaked ground).

## E. The amortization through-line (why the family question is live at all)

Rebuild campaign (2026-07-27, 7 agents): under the IDENTICAL fixed-objective ELBO, a
FREE non-amortized q (per-crop SVI) holds honest placement (corr 0.634→0.635, F
0.758→0.762; honest ELBO 990.6 vs surrender 2523.5); EVERY amortized q collapses from
every init (10/10 cells, corr → 0.002 within 50 steps). Mechanism = smoothness
pricing: the prior's preference for truth is a preference for innovation ≈ 0; truth
needs 3.1e-5 rad steps, the encoder's best placement is 2000× jitterier (0.063), and
surrender is 483–589 nats/crop cheaper than jittery partial placement against ~108
nats of recon value. The ELBO *correctly* sells the placement. Analytic floor: honest
pinned-q KL is irreducible (2log(2+r)−log(4r) nats/frame; argmin ρ_q = 0.999725,
matches empirics to 4dp). Named-next-experiment then: make smoothness structural in q
(innovation-space / AR-residual amortized posterior, or semi-amortized SVI refinement)
— which is exactly the design axis the mixture/diffusion ladder now walks, since the
innovations ARE the coordinates being enriched.

Also standing: probes do not predict training (three dissociations now: supervised
probe 1.7° vs A/B null; closed-form probe strong vs co-adaptation worth 0.002;
theta-latent right-shaped gradient 6000× too weak). Any LaTeX claim must distinguish
trained-A/B evidence from probe evidence.

## F. Traps, retractions, and evaluation ground rules for the write-up

- y is ±1-frame widened for the Bernoulli path; `nonzero(y>0.5)` inflates derived
  rates 3.1×. The interval path consumes `downbeat_times` directly (unbiased since
  9fb05e5/3e06a42; frame-quantized wrap conversion had a +1-shift early bias).
- Single song+seed is noise: sd ≈ 27 F-points; rank on PERIODS/ratios, not
  in-tolerance; overfit_one 35ms/100% claims are single-snapshot — quote pooled
  windows.
- CMLt was actually AMLc before 1bf2fbe (fixed 2026-08-14); historical logged CMLt
  values are AMLc until re-scored. Track CMLt+AMLt beside F, never as a target.
- Dev protocol is phase-gated fold-honesty: single final0 fold + single seed, gtzan
  decides; folds+seeds must return before any baseline-comparable claim.
- Crop horizon: short crops make the ELBO PREFER a metronome; ~45 s+ separates truth
  from coast. 1/T normalization suppressed the tempo prior (untrainable) — fixed.
- ELBO/recon do NOT rank placement (B.4, B.5: three ranking inversions on record).
  In-tolerance and median |err| at annotated downbeats are the placement instruments;
  F(±70ms) at deployment; the F ≡ anchor-in-tolerance identity makes anchor error the
  sufficient statistic on this model family.
- The physical prior is vacuous as a truth-selector (charges 9.5e-07 nats; binding
  versions forbid real tempo steps); the walk constants are corpus-measured (A above);
  meter is hardcoded OUT of this stage (bars, not beats, are the latent's period —
  wrong-metrical-level selection shows up as AMLt ≫ F).
- gtzan-rigidity premise died 2026-08-15: rigid-grid ceiling is 0.902 gtzan / 0.716
  pooled FULL-SONG; the adaptive-structure thesis lives on long songs only; 45 s
  excerpts hide it. Eval is Beat This-style full-song excerpt datasets (no crop
  bridge) since 39abc4b.
