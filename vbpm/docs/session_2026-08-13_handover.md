# Handover — 2026-08-13: the anchor purge autopsy, five objective bugs, and the corrugation verdict

Overfit-one-song (ballroom_Albums-AnaBelen_Veneo-01, true tempo 0.0512 rad/frame,
period 2.454 s) was the instrument for everything below. Committed at session start:
`bd40b35` (anchor/evidence purge; configs to attic/; rate_init -> tempo_init).
**Everything after that commit is UNCOMMITTED** in `phasevae/model.py` and
`phasevae/checks/overfit_one.py`.

## Arc

Session start state: the purged (a-free) model executes coherently but pins tempo at a
rail. Both init fixes (bias at prior mean; _ramp centered at raw 0 — the latter is what
is in the file) put birth in-band; the model railed anyway, even from ORACLE tempo init.
Diagnosis proceeded by per-term gradient decomposition (chain rule, measured not argued),
finding and fixing five independent objective pathologies, then hitting the real wall.

## Verified findings, in causal order

1. **No offset = recon prefers silence.** With rotation pinned at frame 0 no rate places
   downbeats correctly, so recon-only training walks to the SLOW rail (fewer claims =
   less BCE liability). recon grad into tempo bias at birth: +114 (slower). KL-only
   training sat sanely at the prior mean.
2. **Anchor restored, 3 lines** (in heads): `a = sigmoid(downbeat_logit)` (the dead
   channel 0), `_anchor` circular mean, `mu = mu0 + offset`. Tempo-pinned control:
   evidence head learns to 5–15 ms median error, in-tol 100%, recon −351 → −240.
   **Recon rewards aligned truth by ~105 nats.** The likelihood was alignment-starved,
   not rate-blind.
3. **Anchor backdoor (fixed: detach).** Path decomposition at oracle: ramp path
   d/d_tempo = +2707 (toward truth!), offset path = −4096 (garbage estimator used as a
   remote control for rotation). Fix: `_anchor(mu0.detach(), a)` — evidence still
   trains via `a`, tempo keeps only the honest path.
4. **Rail-attracting prior (fixed: pre-bound scoring).** Walk/prior were scored on
   post-`_ramp` (squashed) tempo while q noises the pre-bound latent: on the shoulders
   the squash launders the noise, so d(prior)/d_tempo pointed INTO both rails (+3.7 at
   fast, −0.8 at slow). Fix: `_tempo_log_prior(log_dotphi_presquash, ...)`. Rails now
   repel — later runs visit 0.2000 and LEAVE.
5. **Entropy bar-count subsidy (fixed: per-frame measure).** Per-bar entropy/walk terms
   made latent dimension a function of tempo (fast = ~70 bars = ~1.7 nats harvested per
   bar; kl −117 signature at the fast rail; rail beat perfectly-aligned truth by ~17
   nats of ELBO). Fix: entropy weighted per frame, walk charged on every adjacent pair.
   NOTE: per-frame accounting ALONE made things worse (kl-only railed fast) until #4
   closed the laundering — the two interact.
6. **Widened-target triple payment (fixed: per-event recon).** y is ±1-frame widened;
   factorized Bernoulli pays each of 3 positive frames independently -> density is paid
   3x per event. At aligned truth recon STILL wanted faster (+629 pos, +182 neg);
   recon's rate optimum was ~3–4x (−330 truth vs −315 at 4x). Fix: per-event credit,
   `log(1 − prod(1−p))` once per contiguous positive run; negatives stay per-frame.
   Result: rate profile flat to half a nat; gradients collapse ±600 -> ±100.
7. **Amplitude compression escape (fixed in-check: pinned gain).** Per-event reward
   saturates, per-frame FP cost is unbounded -> optimum is timid teeth (hit prob ~0.28,
   9/12 events under 50% even aligned), flattening the landscape. Grid search
   (sharpness x pos_weight): sharpness moves endpoints off the 4x shelf; sharp=8/pw=3
   visited ratio 1.01 but endpoints are coin flips (CUDA nondeterminism). Fix probed:
   pin emission peak at p≈0.9 (`--pin-gain`). Recon then ranks truth FIRST for the
   first time (−394 near truth vs −432..−440 at 4x).
8. **Oracle-prior probe: FAILED (prior exonerated as binding constraint).** Prior mean
   0.0687 = 1.34x truth is where unaligned runs park; but setting TEMPO_PRIOR_MU to the
   true tempo did not produce capture. sigma-cap probe (ceil 0.05) also failed.
9. **The real wall: lock-basin geometry.** Coherent evidence accumulation needs rate
   within tooth_halfwidth / total_phase = 0.37/115 ≈ **0.3% of truth**. Outside it the
   rate surface is CORRUGATED (loss oscillates with 0.3% period, sign-random gradients,
   T·b lever arm, sigma-ceiling ×1.3/bar sampling jitter) -> Adam diffuses across
   octaves. Coarse-to-fine (b ramp 2->10 over 700 of 1000 epochs, gain tied to keep
   peak p≈0.9): wide phase parks at 4x (wide teeth = cheap FPs), sharpening unhooks it,
   **ep 700 capture: ratio 1.07, 110 ms, recon −311 vs ~−430 elsewhere (~120-nat
   well)** — then a single-step Adam catapult ejects it; keeps falling back (ep 999:
   1.14, recon −310) but never holds.

## Where this lands

Objective: REPAIRED — truth is now the global value optimum (~120 nats). Remaining
failure is optimization geometry: a 0.3%-wide basin in a corrugated landscape is not
gradient-findable/holdable at exploration step sizes. This independently re-derives the
banked verdicts `project_rate_is_the_bottleneck` ("rate needs SEARCH not regression")
and the search-readout result, now WITH a fixed objective for search to inherit.

## Next steps (in intended order)

1. Cheap: lr anneal (x0.1 at sharpness-ramp end) on the coarse-to-fine schedule — tests
   whether the catapult is the last gradient obstacle. Not yet run.
2. The real build: **categorical q over a log-spaced rate grid** (per-window candidates,
   closed-form anchor per candidate, softmax over scores; corrugation becomes evaluated
   scores instead of descended gradients). This is the anchor-mechanism-verdict
   categorical design arriving from a second independent derivation.
3. Promote the check-only pins into config if kept: emission gain pinning (peak p≈0.9,
   `--pin-gain` semantics: b_raw frozen, a tied to scheduled b), pos_weight=3,
   sharpness=8 with slow warmup.
4. Decide fate of per-event recon vs tests (`test_phasevae.py` etc. not run since the
   recon change; expect breakage in anything asserting per-frame BCE).

## Uncommitted diff inventory

`phasevae/model.py`: _ramp centered at raw 0; anchor restored + mu0.detach();
per-frame entropy (`* w` not per_bar_weight — per_bar_weight now dead); walk on all
adjacent pairs, scored pre-bound; per-event recon block in forward.
`phasevae/checks/overfit_one.py`: flags `--oracle-tempo`, `--kl-only`, `--pin-tempo`
(hard channel override), `--pin-gain` (frozen b_raw=2.0, a tied to 2.2−b each epoch).
Plots: /tmp/overfit_emission.png, /tmp/overfit_coarse2fine.png, /tmp/overfit_oracle_prior.png,
/tmp/overfit_sigma_cap.png. Rail-run kl signature −117 (per-bar era) / parked-fast kl
≈ −3500 (per-frame era) are diagnostic constants to recognize in logs.

## Traps hit today (do not repeat)

- Freezing out.weight[2]/bias[2] does NOT pin tempo (trunk leaks); pin the channel.
- Oracle bias must be `log(rate) − mid` under the centered _ramp, not `log(rate)`.
- Endpoint-only grid readouts are coin flips; read trajectories.
- Per-frame measure without pre-bound scoring made the fast rail WORSE (laundering).

## Addendum (late session): the gradient arm is CLOSED

Three 10-song ladders (1000ep; 3000ep; 3000ep + lr x0.1 at ramp-end), difficulty spread
ballroom->asap: **0/30 retained captures**. Transient TRUE locks occurred (beatles100
30ms/78% ep1500; hainsworth30 38ms/88% ep2500; ballroom0 90ms/46% POST-lr-drop) and
every one was subsequently destroyed -- including ballroom0's at 0.1x lr. Endpoints are
always comb harmonics (0.15x/0.75x/1.5x/2.6x/3-4x), identical on metronomic hjdb and
rubato Bach: failure is geometric, not musical. Root statement: BCE is distance-blind;
a global (rate,offset) latent needs a long-range ruler the loss cannot provide (Beat
This's shift-tolerant BCE = local ruler only; their supervised frame classifier needs no
global alignment, which is why it suffices there). Offset has a closed-form ruler (the
anchor); rate does not -> enumeration (rate_grid).

Supervised-Bernoulli control (oracle phase, MLE over a,b): every song fits to -6..-28
nats per-event (family sufficient, incl. Bach); data chooses b~11-12 SHARP but peak
p~0.25-0.46 TIMID (triangle's linear shoulders punish tall peaks; Bach 0.78 because fast
bars). The p~0.9 pin overpays shoulders by construction; use a~-12.5,b~12, or move to a
concave bump emission. Trained best-ever states sit ~300 nats above these ceilings.

rate_grid smoke: NaN root cause = sqrt(real^2+imag^2) gradient at origin on
zero-evidence windows (anomaly-run confirmed; other session patched with norm2 guard +
eps). Relaunched on fixed code, past the old crash point. Watch: advance stuck at 0.079
vs true med 0.064, phase_err/res frozen while recon climbs, b drifting down -- config
has sharpness 0 / pos_weight 1 / NO gain pin; the timid-amplitude escape is open.

---

# Addendum (late evening, same session): the smoke result and the cross-session state

## rate_grid smoke: gtzan F 0.733 (best-ever class, single seed)

The other session built `phasevae/variants/rate_grid.py` + `configs/rate_grid.yaml`
(categorical rate search: log-spaced candidate grid, closed-form circular-mean
offset+resultant per candidate, softmax(scale*resultant + log prior) picks the rate;
evidence head trained under the repaired ELBO). Its first smoke launch NaN'd at epoch 8.

**NaN root cause (found via detect_anomaly, FIXED in rate_grid.py):**
`sqrt(real**2+imag**2)` has an unbounded gradient at 0; by epoch 8 the trained evidence
head produced an exactly-cancelling fold for some (window, candidate) and the backward
poisoned the trunk. atan2(0,0) has the same origin pathology. Guard now in place:
`norm2 = real**2+imag**2`, `sqrt(norm2 + 1e-12)`, and
`atan2(imag, where(norm2 > 1e-12, real, 1))`. Forward scores unchanged; only gradients
were pathological. (Legit near-zero resultants = candidates with no support; they should
score ~0 and now safely do.)

**Relaunched same config/seed; completed 60 epochs, zero NaNs. Results:**
- gtzan (999 held out, label-free rule-g): **F 0.733**, CMLt-col 0.870, AMLt-col 0.876
  (CMLt column subject to the unfixed AMLc-mislabel bug — verify before quoting)
- val: hjdb 0.999, ballroom 0.900, beatles 0.766, hainsworth 0.657, rwc 0.573,
  **asap 0.326** (rubato fails exactly as the stiffness thesis predicts; its AMLt 0.694
  says structure right, rigidity fatal)
- vs search read-out with UNTRAINED evidence: 0.570 -> 0.733 (**training the evidence
  head under the repaired objective is worth +0.16**)
- vs anchor_k v2 (project best): 0.752 — matched by a first run of a day-old variant
- b (emission amplitude) dipped 1.30->1.09 by epoch 20 (timid-amplitude signature),
  then RECOVERED to 1.65 by epoch 59 on its own — the escape was transient here
- caveats: single seed, dev protocol (gtzan fold-honest by construction: final0 +
  gtzan excluded), SMC not evaluated, checkpoint at checkpoints/rate_grid_smoke

## Cross-session reconciliation needed

A parallel session declared the Bernoulli emission dead (30 single-song runs, zero
retained solutions; indicator can't measure how-far-off; 2x comb hits every annotation)
and pivoted to an interval emission (vM placement + Laplace on log interval ratios,
~73-nat octave margin after fixing two audit-caught math errors: a Jacobian
differentiating jitter noise, and 2N−1 density factors on N free coordinates).

These results do NOT contradict, but the reconciliation must be explicit: the
cold-shower table measured single-song GRADIENT RETENTION of the rate (which this
handover's corrugation analysis says can't work), while 0.733 is DEPLOYMENT through the
categorical grid, where rate is selected not retained. The per-event Bernoulli was
sufficient to train an evidence head whose folds select the right rate 73% of the time
on gtzan. The interval emission's corpus runs should therefore be judged against
**0.733**, not against the search read-out's 0.570 or the single-song wreckage.

## Also decided/learned this evening (not in the main doc)

- ICASSP plan (deadline unverified, assume ~Sept 17): Gate 1 Aug 20 = rate-grid
  amortization verdict — **effectively PASSED early with 0.733**; Gate 2 Sept 1 =
  certified numbers for chosen story; Gate 3 Sept 10 = writing freeze. Fallback ladder:
  results paper (A) -> mechanism paper (B: amortization fails/search wins) -> stiffness
  paper (C: continuity priors vs rubato; madmom-DBN result already in hand from the
  parallel session: BT peak-pick 0.810 vs BT+DBN 0.437 on Beethoven rubato).
  Do-regardless: CMLt fix, fold/seed restoration, paper skeleton. AISTATS (Oct) is the
  best-fit alternative if framed as inference-pathology paper; ICML (late Jan) the
  upgrade path; ICME (Dec/Jan) insurance only; TASLP rolling for the long-form story.
- The tolerance/tooth-width insight: emission tooth width IS the rate-capture range
  (basin ≈ toothwidth/total-phase). Coarse-to-fine b ramp produced the best gradient-
  training captures (ep-700 lock at 110 ms) but retention still fails (Adam catapult;
  next-seed run never captured at all). Gradient training of rate: formally a lottery.
- event_recon refactor (user's, uncommitted): behavior-identical extraction out of
  forward; gives the rate-grid scorer and unit tests a standalone likelihood.

---

# Addendum 3 (2026-08-14, ~01:40): the interval variant lands in the repo, and the corpus verdict

## THE FIX THAT MATTERED: separate the rotation gradient from the rate gradient

Chain-rule autopsy at the states runs actually die in (trained evidence head, tempo swept
across the band, every term decomposed as d(term)/d(log k)) found the saboteur:

      k     interval  placement  jacobian   prior    NET
    0.50     +109.97    -45.90     0.82     0.00   +64.88
    1.00      +30.83     -1.90     0.89     0.00   +29.82
    1.25     -109.98   +267.18     0.91     0.00  +158.12   <- pushed AWAY from truth
    2.65     -109.99   +476.22     0.96     0.00  +367.19   <- pushed AWAY (hjdb parks here)

The interval ruler is flawless: it saturates at exactly +-(N-1)/b = +-110 and points at
truth from every k. The PLACEMENT factor was leaking into the rate channel, because
phi_1 = mu0(t_1) + theta, so d(kappa cos phi_1)/d log k = -kappa sin(phi_1) * mu0(t_1);
with the first annotation ~42 frames in, mu0(t_1) is 2.2 rad at k=1 and 5.8 rad at
k=2.65, giving +-580 nats of SIGN-FLIPPING force against the ruler's +-110.

FIX (phasevae/variants/interval.py): the placement factor scores a phase whose ramp is
detached -- phi_place = (mu - offset).detach() + offset + jitter. It still trains theta
(hence the evidence head); it contributes exactly 0.00 to the rate at every k. Verified
by re-running the same decomposition: placement column all zeros, NET +110 below truth,
+32 at truth, -109 above. Monotone and correct from everywhere.

RESULT, single-song overfit ladder, 10 songs, final epoch (was 2/10 before):
  ALL TEN hold the correct tempo: ratios 0.95 0.96 0.97 0.99 0.99 0.99 1.00 1.00 1.00 1.01
  Rotation is now the weak half: median err 44-555 ms, in-tol 4-52%.

## CORPUS VERDICT (100 epochs, fold 7 val, gtzan held out of every checkpoint)

  seed 0: gtzan rule-g F 0.163  (AMLt 0.493, CMLt 0.463)   null-zero 0.070
  seed 1: gtzan rule-g F 0.134  (AMLt 0.437, CMLt 0.392)   null-zero 0.071
  val s0: beatles .? / hainsworth 0.172 / hjdb 0.096 / rwc 0.132   (AMLt 0.41-0.52)

So: single-song rate inference is SOLVED and corpus deployment is NOT. F 0.13-0.16 against
rate_grid's 0.733 on the same held-out set, and barely above the 0.070 null. The AMLt-F gap
(0.49 vs 0.16) says the structure is roughly right at a wrong metrical level -- the same
disease the peer session diagnosed in rate_grid, but far worse. Two seeds agree, so this is
not seed luck. AMORTISATION, not the objective, is what fails here: the encoder can fit one
song's rate but does not generalise the mapping.

## INDEPENDENT CONFIRMATION + a new accounting bug (converge-or-not workflow, 4 agents)

- 0/20 oracle-initialised runs retained truth (pre-fix), median exit from +-5% by epoch 10.
- CAUSAL: pinning q's per-bar tempo sd rescues retention on its own -- held 41% -> 95.5%,
  F@end 0.110 -> 0.625 (medians over 5 songs x 2 emissions). Independent of my SNR probe.
- WHY sigma rails, quantified: the window holds ~16 distinct bar-pooled tempo values but
  tempo_entropy is summed over all 2250 FRAMES. Going from sigma 0.15 to the 0.25 ceiling
  buys +1149 nats while the whole emission spans +-100. A per-bar charge would be 141x
  smaller. sigma ended >= 0.245 in 38/40 runs. **This is the per-frame entropy "repair"
  from earlier today over-correcting: it removed the bar-count subsidy and installed a
  141x entropy overcharge in its place.** Capping sigma_ceil is a band-aid; the real fix
  is charging entropy per distinct latent while keeping the measure tempo-independent.
- 100% of the rate gradient is the EMISSION; kl_jitter, tempo_prior and tempo_entropy each
  contribute +-0.0 to d/d(log rate). The tempo prior is exonerated as a cause.
- STRIKING: with the offset RE-OPTIMISED per candidate rate, the rate profile is smooth,
  monotone, and argmax = k=1.000 exactly for BOTH emissions. The corrugation appears ONLY
  when the offset comes from the circular-mean anchor. Same finger the gradient-separation
  fix points at, from a second direction.

## DENSE-ANCHOR (BeatFCOS-as-likelihood) VERDICT

Built and characterised (scratchpad/dense_da_emission.py). It collapses the harmonic ridge:
k=2's prominence falls from 93%/92% of truth's advantage (sparse) to 36%/31% (dense
log-ratio) and 15%/13% (dense GIoU); k>=3 becomes WORSE than a random wrong rate. It does
NOT widen the lock basin (13.4% vs 12.5%; the basin is ~0.73 / number-of-bars regardless of
emission) and it is 5-7x more corrugated in local-maximum count though with half the ripple
depth. Sparse rejects halving well and doubling barely; dense does the reverse --
complementary, and naive addition does not work (needs ~18x reweighting). 1D GIoU degenerates
to plain IoU here (the anchor lies inside both intervals by construction, non-overlap
fraction measured 0.0000), so it is bounded but not a density. Guard must be ADDITIVE:
clamp(min=eps) gives exactly zero gradient for L^ < eps, the same species as the tempo clamp.

## REPO STATE (all uncommitted, branch rate-init-and-clip)

NEW: phasevae/variants/interval.py, phasevae/configs/interval.yaml
CHANGED: model.py (measured priors + 2-component TEMPO_WALK_MIX; Encoder gains sigma_ceil;
  heads() exports "offset" in aux), run.py + checks/overfit_one.py (2-line wants_raw hook so
  a variant can receive the batch's downbeat_times; no other variant affected)
TESTS: 83 pass; 2 fail and BOTH are pre-existing at HEAD (verified by stashing) --
  test_rate_bound_is_identity_in_the_interior (predates the _ramp centering in e0e31a8) and
  test_learned_sigma_one_draw_per_bar_and_entropy_formula (predates the per-frame entropy).

## WHAT I WOULD DO NEXT

1. The amortisation gap is the whole story now: overfit 10/10 vs corpus 0.16. Probe whether
   the encoder can even express per-song rate across the corpus (freeze everything but the
   tempo channel and fit; or check whether a per-song embedding closes it).
2. Fix the entropy accounting properly rather than capping sigma.
3. Give the count/compensator term to rate_grid's TRAINING score (peer session's conclusion;
   scratchpad/rate_grid_count.py is written and smoke-tested, needs a retrain).
4. Combine sparse + dense interval terms with the measured ~18x reweight (kills both the
   doubling and the halving ridge).
