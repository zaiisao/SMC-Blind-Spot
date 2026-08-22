# VBPM — state of the project (2026-08-18)

Companion to `docs/q_family_case.md` (the theoretical case + full technical appendix).
This document is the narrative: where the project restarted, what each era of the
rebuild established, where it stands today, and what the papers are supposed to be.
Numbers are clean-pipeline (2026-07-30+) unless flagged as campaign-era.

## 0. Snapshot today

Package `vbpm/`, branch `rate-init-and-clip`. Mainline model: **`tchain`** — the
bar-pointer posterior as a STRUCTURED chain over phase (96 bins) with an exact forward
recursion, tempo in the state, triangle emission. The amortized regression q line
(`IntervalVAE`) is RETIRED: on this branch it scores at the null floor.

Scoreboard, `tchain`, rule-g label-free deploy, 45 s windows:

| dataset | F | note |
|---|---|---|
| gtzan (n=993, test-only) | **0.750 / 0.753** (seeds 0/1) | CMLt 0.70 / AMLt 0.85 |
| asap (n=62) | 0.745–0.782 | rubato: IBI CV 0.145, 1.9% half-to-half drift |
| rwc (n=15) | 0.797–0.839 | IBI CV 0.077, 2.0% drift |
| ballroom (n=83) | 0.968–0.973 | steady, 29 s |
| beatles (n=23) | 0.943–0.962 | long but steady (CV 0.023) |
| retired amortized `interval` q, same branch | 0.074 | == null (0.074 / 0.075) |
| rigid-grid oracle ceiling | 0.902 gtzan / 0.716 pooled full-song | |

Seed variance is negligible (0.750 vs 0.753). `tempo_revert=False` beats `True` on all
five datasets (gtzan +0.007, asap +0.037, rwc +0.042), largest exactly where tempo
moves; one seed each, needs a second. phase_err rises monotonically late in training in
both seeds (0.278 -> 0.312 over 20 epochs), so the last epoch is not the best epoch and
a pre-registered `--select` rule is probably free points.

What this settles: the STRUCTURED POSTERIOR is the thing that works. asap and rwc — the
two fluctuating-tempo corpora — sit at 0.75-0.84, so the adaptive-structure claim now
has direct per-dataset support rather than only the oracle argument.


## 1. Why we restarted with a phase-only VBPM

The pre-restart era (June–July: CHART/rungs/KVAE campaigns, ~55 memories archived) ended
with two audits that reset the ground truth:

- **Metric invalidation (2026-07-10):** all earlier deployment scores came from a broken
  pipeline; mechanism findings stood, numbers did not.
- **Rebuild verdict (2026-07-27), the founding result of the current line:** under the
  IDENTICAL ELBO, free non-amortized q (per-crop SVI) holds honest downbeat placement,
  while EVERY amortized q collapses from every init (10/10). Mechanism measured:
  smoothness pricing — truth needs ~3.1e-5 rad innovations, the encoder's best placement
  is 2000× jitterier, and surrendering placement is 483–589 nats/crop cheaper than
  holding it. The objective was exonerated; AMORTIZATION was named the breaker. The
  named next experiment — make smoothness structural in q — is the direct ancestor of
  today's family ladder.

The rebuild itself followed the professor's tutorial (VAEBPM_fin.pdf) with a staged
spec, tests written before the model (Stage-P acceptance suite: oracle, 29 properties,
25 mutants). Stage 0 chose METER as the first latent and measured it dead: corpora are
85–100% m=4, the latent carries no entropy (KL ≈ 0.02). Stage P inverted the choice:
**bar phase is the task** — full entropy on every corpus by construction (a crop begins
anywhere in the bar), and φ = 0 *is* the downbeat, so the model can finally be *wrong*
about placement. Meter stays hardcoded out until the phase machinery earns it
(meter-identifiability was separately measured ≈ chance from timing alone anyway).

## 2. The build-up, era by era

**Era 1 — bar-phase VAE and the collusion lessons (early August, `phase-min`).**
First working phase VAE (3ac1d0f): verified Best–Fisher von Mises sampler, un-aligned
crops, six controls. Two architectures live in the tutorial and mixing them was the
recurring trap: §7 (audio-only encoder, fixed physical prior, deploy encoder + rule g)
vs §9 (Sohn CVAE: q(z|x,b), conditional prior ψ, deploy the prior). Grafting §9's
transformer emission onto §7 scaffolding produced encoder–decoder collusion (downbeats
carried in phase micro-modulations only the co-trained decoder reads; rule-g at chance
while recon is excellent) — a seed lottery even under constrained q. Verdicts that
stuck: reader-side dumbness is load-bearing (triangle emission), bar-rate q family =
parity at fewer DoF, and the emission-D ≈ rule-g agreement is the honesty dashboard.
The §9/ψ mainline was then built properly and killed on evidence: the discrete
rotation-mixture head never trains (uniform or confidently arbitrary in every arm,
clip-exempt 10× lr included), distillation transfers only through the continuous path,
and even working ψ (0.45) < no-ψ encoder (0.505). User removed ψ; K=1 reached parity
and was removed on parsimony. Mainline = no-ψ bar-rate + triangle, gtzan rule-g ~0.505.

**Era 2 — the anchor discovered (2026-08-07..09).** The anchor (phi_0) became the named
problem: F 0.468 → 0.752 via the anchor_k line, then the decisive decomposition — 82%
of that jump is a zero-parameter deployment read-out swap (closed-form circular mean of
downbeat evidence folded under the ramp), no retraining; F ≡ anchor-within-tolerance to
~1 point. The failure was representation (BiGRU frame-0 snapshot head), not landscape
multimodality (conditional basin measured flat-top unimodal) and not enumeration.
Meanwhile the eval stack went Beat This-style: plug-and-play frozen frontends, excerpt
datasets consumed directly (39abc4b), frontend checkpoint pinned as part of the model
(the final0-routing artifact cost gtzan 0.43→0.08 once), and the last oracle left the
inference path (bar rate from audio; x-only ACF reaches oracle parity).

**Era 3 — the objective autopsy and the search turn (2026-08-11..13).** Overfit-one
forensics on the Bernoulli emission found the octave-degenerate likelihood (true rate
beats 2× by ~1 nat/song; 30 runs, 0 retained, endpoints always harmonics) and then five
measured objective bugs, fixed by gradient decomposition; after repair, truth is the
value-optimum by ~120 nats but sits in a lock basin 0.3% of the rate axis wide, outside
which the surface corrugates and Adam octave-hops. Gradient arm closed (0/30 retained).
The era's lasting finding is the corrugation itself: a rate the optimizer must reach by
following a slope is a rate it loses to harmonics.

**Era 4 — the interval emission and the placement verdicts (2026-08-13..14).** The
observation model moved from per-frame Bernoulli to the downbeat TIMES (bijective
change of variables; distance-aware placement + interval rulers + exact Jacobian). One
detach (separating rotation from rate gradients) took single-song tempo from 2/10 to
10/10 — rate closed. The corpus run then exposed the same amortization gap as ever
(corpus F 0.16 against read-outs several times that on the same weights). A one-night blitz closed every placement side-door with
measured verdicts: uniform-phi0 latent (κ collapse, ELBO blind to the damage),
theta-as-latent (right-shaped gradient, 6000× too weak), AR correction cell (kick test:
learns a constant trim; walk prior prices feedback out — third independent
measurement), dispersion term (octave-blind), PLL read-out (null on clean ground). What
remained standing is the current blocker: the ELBO is ~flat in placement because
placement evidence does not scale with N (one vM factor on the first annotation). Same
era: ELBO derivation audited line-by-line (two real holes: Dirac q(phi_1), missing
p(N|phi)); CMLt-was-AMLc metric bug fixed; package renamed vbpm.

**Era 5 — now (2026-08-15).** Physics measured and audited (walk constants refit from
the corpus; dropout×walk tax removed; downbeat times de-biased; scorer re-baselined).
The rigid-grid ceiling was re-measured and CORRECTED: 0.902 gtzan full-song — the old
"rigidity caps gtzan at 0.639" leg of the thesis is dead, and the adaptive-structure
claim relocates to LONG songs (asap 0.164 / rwc 0.258 / beatles 0.349 full-song vs
0.713 at 45 s), which mandates full-song evaluation. The live experiment is the
q-family ladder (mixture_q in the tree; diffusion as the terminal instrument), which is
Era-4's blocker attacked on the axis the rebuild verdict named a month earlier.

**Era 6 — the structured posterior (2026-08-17..18).** The q-family question was
settled by building the family rather than laddering toward it: `tchain` replaces the
per-frame factorized q with a chain over 96 phase bins and an exact forward recursion,
tempo inside the state. Four runs, two seeds: gtzan 0.747/0.750/0.753/0.754, asap
0.745-0.782, rwc 0.797-0.839, ballroom ~0.97, beatles ~0.95; phase_err 0.28-0.31 rad
against the retired amortized line's 1.52 (chance 1.571). Same branch, same frontend,
same rule-g read-out. The amortized `interval` arm was re-run on this branch as the
control and lands at the null floor (0.074 vs nulls 0.074/0.075), which retires it.

Also measured this era, inside the retired arm and therefore only mechanism, not
scoreboard: making placement evidence scale with N (the tutorial's SS7.5 Gaussian over
beat times, `interval_kind: gauss_time`) lifts continuity a long way under a factorized
q -- gtzan CMLt 0.115 -> 0.467, AMLt 0.184 -> 0.556, est/ref 1.313 -> 1.059 -- while F
stays at ~null. The audited `p(N|phi)` hole alone changes nothing (identical to control
on every metric). And `dec_warmup` is confirmed destructive with its own control: the
zdec wake at epoch 15 takes phase_err 1.21 -> 1.50 and it never recovers, while the
frozen-decoder arm holds 1.17-1.24 for twenty further epochs.

Read together: an under-specified placement likelihood costs continuity, but it is the
POSTERIOR FAMILY that decides whether the model works at all. A repaired likelihood
under a factorized q reaches F ~0.13; a chain q reaches 0.75.

## 3. The through-line (what the rebuild proved, four times each)

1. **Structure/enumeration beats regression on multimodal axes.** The chain posterior
   (0.75 vs a null-floor amortized q), the anchor (closed-form/argmax vs slope-trained
   heads), per-song lambda (user's oracle measurement on SMC), and the correction cell
   (offered a wire, bought a constant). This is the unifying mechanism claim of the
   paper line.
2. **The objective keeps being exonerated; the amortized q keeps failing.** SVI holds
   placement; search holds rate; every amortized regression collapses or coasts.
3. **The ELBO does not rank placement.** Three recorded inversions (uniform-phi0,
   PAD-vote masking, best-recon-worst-placement). Placement instruments are in-tol /
   median error / F, never recon — and no September number is defensible without a
   pre-registered checkpoint rule.
4. **Probes do not predict training.** Three dissociations on record; only trained
   A/Bs settle a mechanism claim.

## 4. Paper goals

**Venue plan.** ICME 2027 (Xiamen) is primary — 6 pages holds the full arc
(diagnosis + mechanism + repair) that ICASSP's 4 could not; A-tier, music-receptive,
CFP deadline ~Dec 2026 (UNVERIFIED — check, do not plan on the guess). Fallbacks:
AISTATS (Oct) framed as inference pathology; ICML (late Jan) as upgrade; TASLP rolling
for long-form. **The REAL deadline is early September 2026** — departure for a
six-month NTU (Taiwan) visit, after which iteration is slow. Before departure:
certified numbers, the working version, remote-drivable harness (folds+seeds
restoration and re-scored metrics are the remote infrastructure). After: writing,
framing, related work, slow re-runs. Anything with a 16-hour loop not done by
departure is not in the paper.

**The three-stage goal (user, 2026-08-14).**
1. Sprint to SOME working version (not the best one). THIS NOW EXISTS: `tchain` —
   BT frontend → chain posterior with exact recursion → rule-g decode, 0.75 gtzan
   label-free, 0.78/0.84 on the rubato corpora, two seeds, no gradient lottery.
2. Revisit the SMC Blind Spot paper (arXiv 2605.12287 — the user's own diagnosis
   paper: SOTA SMC failures cluster into octave / continuity / complete; DBN 55-BPM
   floor forces confident-but-wrong octave errors) with the working version as the
   REPAIR leg. Standing mission: SMC beat F ≥ 0.7. Honest bar after the provenance
   audit: structured inference has NEVER deployably beaten peak-pick beat-F on SMC
   (0.589 < 0.620); the real targets are beating 0.620 deployably or realizing the
   oracle headroom (0.654 σ-oracle / 0.703 TTA ceiling) without oracle selection.
   Known dead end: naive tempo augmentation (user already fixed BT's pitch co-adjust —
   didn't help; mel-interp stretch traded octave errors for continuity errors).
3. Distill to the minimum that still trains END-TO-END while applying music-theoretic
   priors to the FINAL beats: keep the amortized network only as a per-song conditioner
   (initial prior / transition kernel / meter), delete the phase-generating machinery,
   let exact inference produce the beats (the R4-spec shape, ~0.5M params).

**The thesis the papers carry.** Adaptive structure: rigid structure < no structure <
adaptive structure (user's per-song-lambda measurement on SMC, ~0.67 F-optimal, above
peak-pick). Premise corrected 2026-08-15: rigidity is demonstrably costly on LONG
songs and rubato, not on 45-s excerpts of steady-tempo corpora — so the evaluation
protocol itself (full-song) is part of the claim, and gtzan/ballroom/hjdb rigidity must
never be cited as evidence. The label-free bridge from oracle to claim is per-song
lambda selection by marginal likelihood (evidence-optimal ≈ F-optimal in the R2
measurement; single measurement, needs folds+seeds).

**Claim-discipline gates (all standing):** dev = single fold single seed, gtzan
decides, folds+seeds return before any baseline-comparable number; report CMLt+AMLt
beside F, never train on them; historical CMLt logs are AMLc until re-scored;
single-song/seed differences under ~27 F-points are noise; label-free means x-only —
any read-out that touches y is an oracle bound and must be labeled as one.

**Where the current experiment fits.** The q-family question is ANSWERED: the family
was the breaker, and `tchain` is the answer, not a rung on the ladder toward it. The
open attribution is now narrower and worth one clean A/B — `tchain` differs from the
retired line in more than its posterior (emission, rate handling), so "the family did
it" is inference from a large effect, not yet a single-variable measurement. The
experiment that would close it: same emission, factorized q vs chain q. The other live
question is whether N-scaled placement evidence adds anything ON TOP of a chain q,
which must be tested inside `tchain`, never in the retired arm.
