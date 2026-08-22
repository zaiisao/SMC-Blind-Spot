# The two KVAE generations — what they were, what they encoded, what survives

Compiled 2026-08-20. Read-only archaeology over `/home/sogang/jaehoon/VBPM` (git intact),
`/home/sogang/jaehoon/Analyze-SMC/vbpm/docs/`, and
`/home/sogang/.claude/projects/-home-sogang-jaehoon-VBPM/memory/`.

**Headline correction to the framing.** The user recalls "an original KVAE" and "a recent KVAE."
Only the first is a Kalman-VAE. The literal Kalman-VAE code was written 2026-06-29 → 2026-07-02
and **deleted from the working tree on 2026-06-30 / never reinstated**; `grep -ril kvae|kalman`
over the current tree returns only `anchor_swap.py` and `external/README.md`, neither of which is a
model. The "recent" effort of 2026-08-14 → 2026-08-19 is the **von Mises / discrete chain family**
(`tchain`, `vmchain`, `schain`, `archain`, now `vbpm`). It is not a Kalman filter and contains no
linear-Gaussian SSM. A plan to resurrect the real KVAE exists — `memory/project_kvae_ladder_plan.md`
— but it is explicitly **"Deferred by the user 2026-08-18 (low priority, credits). Do NOT start
without being asked."** It was never started. The two generations are therefore cleanly separable,
but the second one is a KVAE only in the user's memory, not in the code.

---

## Generation 1 — the actual Kalman-VAE (2026-06-29 → 2026-07-02)

### Dates, commits, code paths

Two sub-campaigns, distinct code and distinct numbers:

| | Campaign A "M1 / kvae_barpointer" | Campaign B "kvae_prototype" |
|---|---|---|
| dates | 2026-06-29 / 06-30 | 2026-07-01 / 07-02 |
| added at | `7301b92` (2026-06-30) | `ce4a4ea` (2026-07-02) |
| deleted at | `2473325` (2026-06-30) | never re-added after 07-02; line abandoned |
| paths | `experiments/kvae_barpointer/` (30 scripts, 2,743 LOC) | `model/kalman_vae_bar_pointer.py` + `experiments/kvae_prototype/` (24 scripts, 3,712 LOC) + `third_party/kalman-vae/` |

Campaign B is a *from-scratch reproduction* of M1 against the post-refactor codebase, not a rerun.
Later commits: `29b83f4` (2026-07-02, the phase variant), `7c32937` (comment-only cleanup). Nothing
after that. Recover any file with `git show <sha>:<path>`.

Provenance of the backbone: official `simonkamronn/kvae` (TF1) used as a cross-check reference,
PyTorch port `nkiyohara/kalman-vae` (**MIT**, `third_party/kalman-vae`) reused **verbatim** for the
filter/smoother. Vendored as plain committed files, *not* a git submodule (the only submodule at that
sha is `external/beat_this`), so it is directly liftable.

### What the latents actually mean — **no musical prior whatsoever**

This is the decisive finding, and it is unambiguous in the code.

- **z is an unstructured 8-dimensional continuous vector.** `z_dim: int = 8`, `a_dim: int = 8`. Flat
  Euclidean. No reserved coordinates, no phase/tempo/meter partition, no circle.
- **A and C are free learned matrices.** In `third_party/kalman-vae/kvae/state_space_model.py:41-47`:
  ```
  self._mat_A_K = nn.Parameter((1.0 - 0.9) * torch.randn(K, z_dim, z_dim) + 0.9 * torch.eye(z_dim))
  self._mat_C_K = nn.Parameter((1.0 - 0.9) * torch.randn(K, a_dim, z_dim) + 0.9 * torch.eye(a_dim, z_dim))
  ```
  That is `0.9·I + 0.1·randn` — an identity-biased random init, i.e. a numerical "start near a random
  walk" prior. No rotation block, no companion form, no tempo parameterisation, no constraint applied
  after init. `mat_Q = LLᵀ + Q_reg·I`, `mat_R = LLᵀ + R_reg·I` are free learned covariances with an
  additive floor.
- **K=5 corresponds to nothing musical.** Per frame `weight = LSTM(a_{1:t})` over K, then
  `mat_A = einsum("tbk,kij->bij", weight, mat_A_K)` (lines 225-228) — a learned soft interpolation
  between 5 randomly-initialised linear regimes, gated by a 1-layer hidden-64 LSTM. There is no tempo
  binning, no BPM grid, no assignment of component k to any tempo or meter. Empirically confirmed
  generic: mixture entropy 1.104 in-domain vs 1.112 on SMC (max ln 5 = 1.609), all 217 SMC songs use
  all 5 components at >10% weight, and no entropy difference between low-F and high-F songs.
- **Read-out is a learned MLP head, not a geometric phase wrap.**
  `nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, 2))` → (beat, downbeat) logits →
  sigmoid → peak-pick.
- **There is no tempo state, no beat-phase state, no bar-position pointer, and no transition kernel
  over tempo.** All the musical content lives in (a) the frozen Beat-This features it observes and
  (b) the supervised BCE head on top.

The project said this itself at the time, in answer to a direct user question
(`memory/project_smc_mirex_ood_generalization.md`, "NAMING NOTE, user asked directly 'is this still a
true VBPM'"):

> it is **NOT** literally the ELBO_for_DBN paper's VBPM spec — **no explicit meter/tempo/phase
> latents, no von Mises circular dynamics**, and the geometric phase-wrap read-out was tried and
> documented as structurally failing on this posterior … the working read-out is a learned MLP head
> on filtered z instead.

### Marginalisation — **yes, genuinely, and this part is the good news**

This is the one property of generation 1 that *does* match the paper's thesis. The ELBO in
`model/kalman_vae_bar_pointer.py` runs `ssm.kalman_filter` followed by `ssm.kalman_smooth` (a full
RTS smoother) and scores the four standard KVAE terms — reconstruction, regularization,
`kalman_observation`, `kalman_transition`, minus `kalman_posterior_entropy`. The latent posterior
`q(z|a)` is computed **exactly** by the Kalman recursion rather than amortized, which was the entire
point of the pivot: "gradient descent cannot shortcut an exact conditional the way it can an
amortized approximation."

So generation 1 is: **real latent temporal state + real marginalisation, over a latent that means
nothing musically.** It is precisely the "generic linear-Gaussian state-space VAE over audio
features" case named in the brief.

### The variant that *did* try to encode a musical prior — and never ran

`model/kalman_vae_bar_pointer_phase.py` (418 LOC, `29b83f4`, 2026-07-02) dedicates `z[..., 0:2]` to
`(cos φ, sin φ)`, overrides the phase sub-block of `mat_A` with an exact 2×2 rotation by a `φ̇`
computed from a differentiable autocorrelation tempo head, and recovers `φ = atan2(z₁, z₀)` for a
geometric wrap read-out. It takes `beats_per_bar: int = 4`. **That would be a musical prior.**

**It was never trained.** Its gate `verify_phase_kvae_correctness.py` (218 LOC) checks only rotation
invariants, a 4-frame recursion match, and gradient flow — no F-measure, no beat evaluation. Its
docstring says "MUST PASS before any training run proceeds." The memory note dispatching the
overnight run ends: *"STATUS AS OF THIS NOTE: dispatched, not yet landed … Do not assume success or
failure before real numbers exist."* That note was never updated; it is the last entry in the file.
No checkpoint, no log, no result. A grep of all ~180 memory files for
`kalman_vae_bar_pointer_phase` / `KalmanVAEBarPointerPhase` returns nothing.

The *earlier* rotational-z attempt, `experiments/kvae_barpointer/kvae_geom.py` (126 LOC), **did**
produce numbers, and they are failures — see the M2 rows below.

### Results, with provenance

All in-domain numbers: **bt_val_rich, n = 40 songs**, frozen Beat-This `final0` 512-d activations
@ 86.13 fps, mir_eval, 0.07 s tolerance.

**Campaign A (`git show 7301b92:experiments/kvae_barpointer/CAMPAIGN.md`)** — beat-F / downbeat-F only:

| result | beat-F | db-F | status (record's own words) |
|---|---:|---:|---|
| **M1 KVAE, filter + learned head** | **0.878** | 0.833 | **"M1 (certified): WALL BROKEN"**; leak shuffled 0.293/0.126, zero 0.000/0.000 → **verified** |
| frozen frontend peak-pick (ceiling) | 0.929 | 0.879 | verified reference |
| frozen frontend + madmom DBN | 0.922 | 0.904 | verified reference |
| M1-scaled (2000 songs, 3000 steps, a/z=16, K=8) | 0.868 → peaked 0.888, final 0.848 | 0.825 | **preliminary** ("climbing", "mild late fluctuation") |
| seed variance, 3 seeds | 0.836 ± 0.009 | — | verified ("numbers are noise? REFUTED") |
| clamp test, GT φ fed in | 0.975 | — | verified diagnostic (read-out is sound) |
| free-run `predict_future`, open-loop | 0.289 (vs filter 0.830) | — | verified ("free-run collapses? CONFIRMED", ratio 0.35) |
| **M2a geometric (`kvae_geom.py`)** | **0.345** | **0.000** | **dead**: "FAIL: φ does not rotate (phi-revs 0)"; the 0.345 is "φ *jitter* … NOT a rotating pointer" |
| M2b geometric (recon_w = 0) | 0.34 | 0.000 | FAIL, "killed; redundant w/ M2a" |
| M2-CV (integrate tempo from filter) | 0.31 | 0.09 | **confounded** — "stuck ~230 BPM, input-independent grid"; **real < zero** |
| `phi_capacity` pure-φ-supervision | 0.375 | — | verified negative (loss pinned ~1.0 over 1200 steps) |
| e2e from-scratch TCN + filter | 0.473 | 0.302 | verified (leak collapses); "frontend-limited" |

**Campaign A OOD (SMC-MIREX, n = 217)** — the only Campaign-A table with AMLt:

| arm | beat-F | AMLt | status |
|---|---:|---:|---|
| e2e from-scratch KVAE | 0.135 | 0.061 | verified failure ("FAILS OOD") |
| Beat-This peak-pick (harness check) | 0.620 | 0.605 | verified ("matches published 0.626 → harness valid") |
| Beat-This + **our DBN inference** (not KVAE) | 0.589 | **0.656** | verified; "beats madmom-DBN (0.575)" |

**Campaign B, checkpoint `kvae_m1_repro_400ep1000.pt`** (400 train songs / 1000 steps / batch 16 /
crop 256 / a=z=8 / K=5 / lr 1e-3). Harness `experiments/kvae_prototype/continuity_metrics.py`
(143 LOC), `mir_eval.beat.continuity` + `readout.f_measure`, 0.07 s tol, via
`fast_deploy.batched_kalman_filter_deploy`:

| set | n | F | CMLc | CMLt | AMLc | AMLt |
|---|---:|---:|---:|---:|---:|---:|
| in-domain bt_val_rich | 40 | **0.895** | 0.846 | **0.875** | 0.875 | **0.906** |
| SMC, zero-shot | 217 | **0.654** | 0.420 | **0.587** | 0.440 | **0.616** |

I **independently recomputed** the SMC F from the one committed data artifact in the whole effort,
`git show ce4a4ea:experiments/kvae_prototype/logs/smc_per_song_diagnostics.csv` (218 lines = header +
217 songs): **n = 217, mean F = 0.654038, median F = 0.709700**. Matches the claim exactly. 67.28% of
songs have a tempo ratio within 4% of 1.0; only 5.99% exceed 1.8×.

Other Campaign-B numbers (all "verified independently by rerunning the eval scripts" per the record):

| result | value | dataset / n | status |
|---|---|---|---|
| GTZAN zero-shot | beat 0.879 / db 0.789; shuffle 0.279/0.074; zero 0.000/0.000 | GTZAN, n=993 | verified (F only, no CMLt/AMLt) |
| classifier-only ceiling (no filter) | SMC 0.407, in-domain 0.705 | 217 / 40 | verified → **filter adds +0.247 F on SMC vs +0.190 in-domain** |
| Q_reg sweep 1e-3→1e-1 | in-domain 0.895→0.877; SMC 0.654→0.620 | 40 / 217 | verified **NEGATIVE**, "RULED OUT" |
| rubato augmentation | in-domain →0.879/0.841; SMC →0.640; CMLt−CMLc 0.166→0.162 | 40 / 217 | verified NEGATIVE |
| adaptive Kalman noise | in-domain →0.861/0.844; SMC →0.630; CMLt−CMLc →0.159 | 40 / 217 | verified NEGATIVE |
| octave errors | 0% of the 23/217 failing SMC songs show a clean 2× ratio; AMLt−CMLt ≈ 0.03 both sets | 217 | verified — **ruled out** |
| tempo distribution | SMC median 70.9 BPM, Q1 57.3, 45/217 songs < 55 BPM; bt_train_rich median 112.3, Q1 95.7, ~0 songs < 55 | — | verified |
| **the causal story** | "tempo shift → continuity errors" | — | **explicitly downgraded**: "BEST-SUPPORTED HYPOTHESIS, NOT PROVEN CAUSALLY … previously written as RESOLVED/CONFIRMED, which overstated the epistemic status … Do not treat this as closed" |

**The continuity signature, which is the single most paper-relevant number in the whole record:**
`CMLt − CMLc = 0.166 on SMC vs 0.030 in-domain` — "the filter gets the tempo right but loses lock
intermittently." And note it is a *different* mechanism from the SMC-blind-spot paper's DBN min-tempo
floor: "our KVAE has no explicit BPM floor."

### Caveats on generation 1's numbers

- **Not covered by `project_metric_invalidation_audit.md`.** That audit's scope is "deployment F
  scores before 2026-07-10 … through a broken pipeline (side-channel emission + naive filter)." The
  KVAE deploy path is Kalman filter → learned head → peak-pick, with no particle filter, no emission
  side-channel. The audit in fact cites KVAE on the winning side: *"KVAE/SMC won by side-channel
  immunity."* So the blanket pre-07-10 invalidation does **not** apply.
- **But `memory/project_kvae_ladder_plan.md` still says, conservatively:** *"Do not treat 0.878/0.833
  as a target … under [[feedback_distrust_project_history]] it is a hypothesis to re-derive, not a
  number."* Honour that.
- **Open, unreconciled contamination risk.** The 2026-07-10 "fold-honesty reckoning"
  (`project_wall_broken_kappa2.md`) established that the `final0` Beat-This checkpoint was **trained
  on SMC**. The KVAE's "zero-shot" SMC 0.654 rides on `final0` features. No record applies that
  correction back to these numbers. **Treat 0.654 and 0.879/0.789 as OOD-with-an-asterisk** until
  re-run on a fold-honest frontend.
- **No log or checkpoint was ever committed.** The only committed data artifact in the entire effort
  is the SMC per-song CSV. Everything else is record-verified but artifact-unverifiable today.
- The `0.878/0.833` vs `0.895/0.845` apparent conflict **is not one**: they are two different runs
  (Campaign A, 800 steps / 200 songs; Campaign B, 1000 steps / 400 songs). Every file that quotes one
  or the other is internally consistent about which it means.

### Does it run today? **Yes — verified, unmodified, on the current env.**

Extracted `ce4a4ea` into `/tmp/kvae_smoke` and ran on `/disk4/anaconda3/envs/vbpm/bin/python`
(torch 2.6.0+cu124), CPU, no GPU job:

```
constructed; params= 421663
elbo= -222.7642059326172
z shape (200, 2, 8)
head logits (200, 2, 2)
backward OK
```

Full forward + Kalman filter + RTS smoother + ELBO + backward, on random `[T=200, B=2, 512]` input.
One gotcha, and it is a guardrail rather than a bug — with `state_transition='mean'` the vendored SSM
raises verbatim:

```
ValueError: sample_control.state_transition must be 'sample' for training
```

Setting `SampleControl(encoder='sample', decoder='mean', state_transition='sample', observation='sample')`
works. The **phase variant also imports and runs**: `KalmanVAEBarPointerPhase()` constructs with
454,561 params and `phase_kvae_elbo(...)` returns `-320.13983154296875` and backwards cleanly, once
`model/__init__.py`'s unrelated `bar_pointer_vae` import is stubbed (that module was deleted in a later
refactor; nothing in the phase KVAE needs it). **Both generation-1 models are alive.**

### Rebuild cost in `/home/sogang/jaehoon/Analyze-SMC`

LOC, counted:

| component | LOC | needed? |
|---|---:|---|
| `kvae/state_space_model.py` (filter + RTS smoother + K-mixture) | 575 | yes |
| `kvae/dynamics_parameter_network.py` | 90 | yes (transitive) |
| `kvae/misc.py` | 41 | yes (one function) |
| `kvae/sample_control.py` | 8 | yes |
| our `kalman_vae_bar_pointer.py` (model + ELBO) | 157 | yes |
| `train_kvae.py` | 264 | yes |
| `fast_deploy.py` (batched deploy; 138× speedup) | 310 | for tractable eval |
| `continuity_metrics.py` (CMLt/AMLt) | 143 | for the paper's metrics |

- **Minimum viable (model + train + vendored SSM, using the in-loop leak-test eval): ≈ 1,135 LOC.**
- **With batched deploy + one OOD eval: ≈ 1,604 LOC.**
- **With continuity metrics + OOD feature extraction: ≈ 1,848 LOC.**
- Of that, **714 LOC is vendored MIT code you copy, not write.** Genuinely new/ported: ~575–1,134.

**Dependencies:** `torch` (incl. `torch.distributions`), `numpy`, `mir_eval`. That is all for the
model itself — I confirmed the vendored SSM runs on torch alone; the heavy deps in its `pyproject.toml`
(opencv, pygame, pymunk) are for the pendulum demo only. Feature extraction additionally needs the
Beat-This frontend (`external/beat_this`) + `torchaudio`/`einops`/`rotary_embedding_torch`. **No
TensorFlow.** The `analyze-smc` env already satisfies the model side: torch 2.7.1+cu118 with CUDA
available, numpy 1.24.3, mir_eval present — **verified**.

**GPU-hours: could not determine from the record.** No training log survives. The recorded budgets
are: M1 = 800 steps, batch 16, crop 256 frames, a_dim 8, 200 train songs; Campaign B repro = 1000
steps, batch 16, crop 256, 400 train songs; the phase experiment was given a stated **6-hour
unsupervised budget**. Campaign A ran four jobs concurrently on GPUs 0–3 overnight. On 8-dim latents
at that budget this is a **low-single-digit GPU-hours** job per arm, but that is my inference, not a
recorded measurement. The only surviving timings are eval-side: full `bt_val_rich` leak test
295 s → 2.1 s (138×); GTZAN 185+ min → < 1 min.

---

## Generation 2 — the August chain family (2026-08-14 → 2026-08-19). **Not a KVAE.**

### What it actually is

55 commits between 2026-08-13 and 2026-08-20. The modules live at
`/home/sogang/jaehoon/VBPM/vbpm/variants/`:

| file | LOC | what it is |
|---|---:|---|
| `tchain.py` | 349 | discrete phase × tempo grid, **exact forward-backward** |
| `vmchain.py` | 369 | continuous von Mises phase, discrete tempo, **assumed-density filter** |
| `schain.py` | 236 | single enumerated tempo for the window |
| `vbpm.py` | 96 | "a downbeat chain whose tempo is fixed between downbeats" — the current mainline |
| `archain` | — | AR von Mises chain; the subject of the 2026-08-19 audit |

Key commits: `066daa5` "tempo joins the state, so the posterior can bend"; `ae39b88` "a chain
posterior over phase, with the rate still enumerated outside it"; `7d5cb0b` (2026-08-19) "the
factorized model is deleted, and the chain takes the name VBPM"; `50a8acb` (2026-08-20) "tempo is
fixed between downbeats and free at them, and the marginal chain becomes the baseline."

**None of these is a Kalman filter or a linear-Gaussian SSM.** There is no `mat_A`, no RTS smoother,
no Gaussian state. The inference is either exact discrete DP or von Mises moment-matched ADF.

### What the latents mean — **yes, explicit musical priors, and this is the contrast that matters**

`tchain.py` docstring, verbatim:

> Bar phase AND tempo in the chain state, marginalised exactly. … Here the state is a pair,
> `z_t = (phase bin k, tempo bin m)` and forward-backward sums over tempo PATHS rather than picking a
> tempo. … The transition factorises:
> `p(k', m' | k, m) = vM(theta_k' - theta_k - rate_m ; kappa) * N(log rate_m' - log rate_m ; sigma^2)`

That is an explicit **beat/bar-phase latent**, an explicit **tempo latent**, and an explicit
**transition kernel over tempo** (a log-normal walk band around the diagonal). `vmchain.py` keeps
phase continuous on the circle as a von Mises belief closed under rotation and evidence, with tempo
discrete "because the octave ambiguity is genuinely multimodal and a unimodal family cannot hold mass
at v and 2v at once."

So generation 2 **does** encode musical priors — exactly the ones the paper wants. Its problem is not
the model class; it is that almost nothing it measured survived audit.

### Marginalisation — yes, and it is the family's stated selling point

`tchain` does exact forward-backward over phase × tempo (cost = M phase-shifts + one M×M mix per
frame, not a dense (KM)×(KM) matrix). `vmchain` is an assumed-density filter — "the family is
continuous, the recursion is approximate. tchain is the reverse — discrete family, exact recursion."
`archain` does exact categorical marginalisation over the rate/alignment components with a closed-form
von Mises step KL.

### What the 2026-08-19 audit refuted vs what survived

Source: `/home/sogang/jaehoon/Analyze-SMC/vbpm/docs/audit_2026-08-19.md`. It opens: *"nothing here
rests on an F score."* The claims ledger runs A–Q. Summary:

**SURVIVED (independently re-derived, no action needed):**
- Form-1 AR von Mises chain and its **exact closed-form per-step KL** (the vM KL depends on the means
  only through `cos(mu1−mu2)`); phase KL genuinely starts at ~0.
- **Exact categorical marginalisation** — `ELBO = Σ_c q(c)[recon_c − KL_chain_c] − KL(q(c)‖p(c))`;
  per-component recon verified genuinely per-component.
- The **closed-form anchor**: `mu0 = −atan2(Σ a_t sin(rt), Σ a_t cos(rt))` is the exact argmax —
  closed form **2.899553** vs 20001-point grid argmax **2.899690**.
- Wrapped-Cauchy density correct and normalised; unbiased KL estimator with unbiased pathwise
  gradients. Inverse-CDF von Mises sampler with implicit `dS/dκ` — the correct branch, avoiding the
  recorded **−48% Best-Fisher κ bias at κ=2**.
- Deployment is label-free and mode-consistent; no `detach()` anywhere in the file.
- Three departures from the tutorial are **justified**: audio-only encoder (by §7), exact enumeration
  over Gumbel-softmax (strictly better), fixed φ-only emission over the emission Transformer.

**REFUTED — including the claim named in the brief:**
- **(O) "The chain family contains the exact posterior, so the variational gap is exactly zero"** —
  **REFUTED for the object actually trained.** Containment holds for the *non-amortized* family, but
  the trained encoder reads `x` only while the optimal potentials `ψ_t(·) = p(y_t|z_t=·)` are
  functions of `y`. Measured consequence: learned potentials have per-frame entropy **4.564** against
  **log 96 = 4.564** — "statistically indistinguishable from uniform," tilting at most ±10%. Two
  downstream casualties: *"training tchain is maximum-likelihood training of a neural DBN" is
  unsupported*, and the attribution of the 2× defection to a flat objective requires the gap to be
  zero. **Do not cite the zero-gap or exact-ML claim.**
- **(D)** The "4 ms Wing–Kristofferson annotation jitter" and the "κ_physical is 100–1000× too loose"
  factor — **RETRACT ENTIRELY**. The WK estimator is at its floor on 30.6% / 0.5% / 84.0% / 50.5% /
  78.4% / 95.3% of ballroom / hjdb / beatles / hainsworth / rwc / asap; pooled median jitter 0.0 ms.
- **(E)** "Tightening κ fixes placement, 60% → 100% in-tolerance" — REFUTED; the two logs are the same
  configuration, bit-identical through ep200, diverging on nondeterminism alone.
- **(H)** "MONOTONE convergence to truth, 499 → 129 ms" — REFUTED as stated; two logs spliced. The
  quoted run goes to ratio 2.01 / 612 ms and never recovers (ep399: 615 ms, **CMLt 0.000**).
- **(I)** "Capstone: perfect lock, F 1.000" — REFUTED; that is the max of 17 draws. Tail F: 0.583,
  0.583, 0.417, 0.833, 0.750, 0.167, 0.154, 0.083, 1.000, 0.667 — **mean 0.491, sd 0.299**.
- **(Q)** The 0.973-vs-0.000 controlled A/B as a representational floor — REFUTED; it is an
  optimization trap (bin-step 0 at 100% of frames), not a representational one.
- **(K) The noise floor, CONFIRMED and binding on everything:** `archain_lab.py:53` hardcodes
  `torch.manual_seed(0)` with **no override — every archain result on record is single-seed**.
  Within-run tail sd of a final-epoch F snapshot ≈ **0.16** overall, **0.30** on locked songs.
  Adjacent-checkpoint |ΔF| across all 73 logs: **median 0.076, mean 0.142, p90 0.344, max 1.000**.
  One run traverses F = 1.0 / 0.733 / 0.4 / 0.867 / 0.0 / 0.733 / 0.133 internally. **NULL, all of
  them:** κ 3000 vs 30000; the σ sweep; stride 8 (0.466 ± 0.401) vs stride 25 (0.314 ± 0.077).
- **(L)** The one between-arm effect exceeding the floor is **base vs tempered-walk, and it is
  NEGATIVE** (0.787 → 0.148 / 0.272 on asap251, etc.).
- Also: **9 bugs ranked S1–S9, "all must be fixed before any further run,"** three of them blocking —
  an absorbing `log_rate` clamp with exactly zero gradient at the rail, a Cauchy/von Mises stride
  composition off by a factor **2.92**, and an anchor/rollout tempo disagreement on edge bins.

The `q_family_case` / "q-family ladder" material is, per the record, **not testable and a liability
to cite** — the audit's items P and Q are the specific reasons. Steer clear.

### Does it run today? **It imports and its tests pass — but it does not track beats.**

Verified on `/disk4/anaconda3/envs/vbpm/bin/python`:

```
import vbpm  → OK (/home/sogang/jaehoon/VBPM/vbpm/__init__.py)
vbpm.run, vbpm.nets, vbpm.observation, vbpm.readout, vbpm.specs → OK
python -m pytest vbpm/tests -q  →  54 passed, 1 warning in 5.22s
```

So the "can't even track beats right now" claim is **not an import or crash failure** — the package is
healthy. It is a *results* failure, and the audit is the evidence: 9 pre-run blocking bugs, every
result single-seed inside a ±0.16–0.30 F noise floor, and the capstone/convergence claims retracted.
Corroborating the state of play: the newest artifacts in the tree are single-song **overfit-one**
diagnostics on 4 songs (`logs/overfit/`, `ballroom103`, `rwc8`, `asap466`, `asap468`, timestamped
2026-08-20 18:52). The most recent corpus-level CSVs are from **2026-08-15**
(`logs/fixed_2026-08-15/`). Generation 2 has been at the single-song-overfit stage for five days.

### Traceable results

**I could not find a corpus-level F/CMLt/AMLt for any August model that survives its own audit.**
The numbers that exist are single-song, single-seed, and the audit ranks them at or below the noise
floor. The only figures I would repeat are the *negative* ones, quoted above with the audit's own
language: tail-mean F 0.491 ± 0.299 on the capstone; CMLt 0.000 at ep399 on the "convergence" run;
tempered-walk consistently negative on 3/3 songs. The memory claim that `vmchain` scores **F 0.000**
("zero downbeats emitted, because the phase never advances") appears in
`memory/project_kvae_ladder_plan.md` with a mechanism (a constant-direction von Mises potential is a
stable fixed point; verified twice at 0.248/8.05 = 0.0308 and 0.237/7.97 = 0.0297 against a true rate
of 0.0300). **I did not locate an independent scoring artifact for the F 0.000 figure itself** — the
fixed-point measurements are what is verified; treat the F as record-only.

---

## Comparison

| | **Gen 1 — Kalman-VAE** | **Gen 2 — August chain family** |
|---|---|---|
| dates | 2026-06-29 → 2026-07-02 | 2026-08-14 → 2026-08-19 (lineage from 08-03) |
| is it a KVAE? | **Yes**, literally (Fraccaro 2017, vendored `nkiyohara/kalman-vae`, MIT) | **No.** No Kalman filter, no linear-Gaussian SSM |
| code today | **deleted** from tree; recover via `git show ce4a4ea:…` | live at `vbpm/variants/{tchain,vmchain,schain,vbpm}.py` |
| latent | unstructured **8-dim** continuous vector | **(bar phase, tempo)** — explicit, named |
| tempo state? | **No** | **Yes** (discrete grid or log-normal walk) |
| beat-phase state? | **No** | **Yes** (96-bin grid, or continuous von Mises on the circle) |
| tempo transition kernel? | **No** — free `A`, `0.9·I + 0.1·randn` | **Yes** — `N(log r' − log r; σ²)` |
| K=5 mixture means…? | nothing musical (LSTM-gated random regimes) | n/a |
| read-out | learned MLP head on `z` | geometric phase wrap |
| **marginalisation?** | **Yes** — exact Kalman filter + RTS smoother | **Yes** — exact forward-backward (tchain) / vM ADF (vmchain) |
| **musical prior?** | **NO** | **YES** |
| best traceable result | SMC n=217 **F 0.654038** (recomputed), CMLt 0.587, AMLt 0.616; in-domain n=40 F 0.895, CMLt 0.875, AMLt 0.906 | none that survives its own audit |
| epistemic status | verified, but OOD-with-an-asterisk (`final0` saw SMC) | largely **retracted / single-seed inside a ±0.16–0.30 noise floor** |
| runs today? | **Yes** — verified forward+backward, 421,663 params | imports + 54/54 tests pass, but does not track beats |
| simple? | **Yes** — 157 LOC + 714 vendored | No — 1,050 LOC of variants + 9 open blocking bugs |

---

## Verdict on the user's question

The user's instinct was right on both counts, and sharper than they gave themselves credit for.

**(a) Is the original KVAE simple enough to mention in passing as prior work we did? — YES, and it is
genuinely ours.** 157 LOC of our own model + ELBO on top of 714 LOC of vendored MIT code, one
frozen-feature encoder, one MLP head. It runs today, unmodified, on the current environment — I
verified the full filter/smoother/ELBO/backward path. It has one clean, leak-controlled, independently
recomputable headline: **SMC n=217, F = 0.654038**, with CMLt 0.587 / AMLt 0.616, against in-domain
n=40 F 0.895 / CMLt 0.875 / AMLt 0.906. Two or three sentences will hold it. Two caveats to carry in
the sentence: the frontend (`final0`) had seen SMC, so "zero-shot" needs an asterisk; and no log or
checkpoint was ever committed, so everything but the SMC per-song CSV is record-verified rather than
artifact-verified.

**(b) Does it encode musical priors, and does it therefore match the current goal? — NO. It does not
match, and we must say so plainly.** Its `z` is exactly the case the brief names as disqualifying: a
generic 8-dimensional linear-Gaussian state with free learned `A`/`C`/`Q`/`R`, a 5-way LSTM-gated
mixture that corresponds to nothing musical, and a learned MLP head. There is no tempo latent, no
beat-phase latent, no bar-position pointer, and no transition kernel over tempo. The project said this
itself, in writing, in answer to the same question: *"no explicit meter/tempo/phase latents, no von
Mises circular dynamics."* The one variant that *would* have encoded a musical prior — the unit-circle
`(cos φ, sin φ)` sub-block with rotation by a computed `φ̇` — **was gated for correctness and never
trained**, and the two rotational-z attempts that *did* run both failed structurally (φ-revs = 0;
"real < zero" on the M2-CV arm).

**What it does give us, and this is the part worth keeping.** Generation 1 has the *other* half of the
paper's property: genuine latent temporal state with **exact marginalisation** — a Kalman
filter/smoother, not a smoothness penalty. So the honest framing for ICASSP is a **two-axis** one, and
the KVAE is the clean off-diagonal cell:

> marginalisation without a musical prior gets you F 0.654 / CMLt 0.587 / AMLt 0.616 on SMC, and its
> failure signature is **CMLt − CMLc = 0.166 on SMC vs 0.030 in-domain** — the tempo is right and the
> *lock* is what breaks.

That is a continuity failure, measured, on our own model, and it is a **different mechanism** from the
SMC blind-spot paper's DBN min-tempo floor ("our KVAE has no explicit BPM floor"; octave errors ruled
out at 0% of the 23 failing songs). It motivates the latent-tempo-state fix rather than substituting
for it. It also carries a hard negative that is directly on-thesis: reading a phase off a *freely
filtered generic* latent structurally does not rotate — the observation update corrects `z` back
toward a static estimate. The structure has to be in the state space, not read out of it afterwards.

**One correction to the premise the user should hear.** The "recent KVAE from a few days ago" is not a
KVAE. Nothing named `kvae`/`kalman` exists in the current tree; the KVAE line was abandoned 2026-07-02
and the plan to resurrect it (`project_kvae_ladder_plan.md`) was **deferred by the user on 2026-08-18
and never started**. What happened in mid-August is the von Mises / discrete chain family, which
*does* encode the musical priors (explicit bar-phase and tempo latents, an explicit tempo transition
kernel, exact forward-backward) — but whose results were largely dismantled by its own
`audit_2026-08-19.md`, including the "zero variational gap / exact ML training of a neural DBN" claim,
which is **refuted for the object actually trained** and must not be cited. Every archain result on
record is single-seed against a ±0.16–0.30 F noise floor, and nine bugs are flagged "all must be fixed
before any further run."

**Practical recommendation.** If you want the KVAE cell for the paper, rebuilding it cleanly in
`Analyze-SMC` is cheap: **≈ 1,135 LOC minimum** (of which 714 is vendored MIT code you copy, not
write), **≈ 1,604** with a batched deploy and one OOD eval, **≈ 1,848** with CMLt/AMLt. Dependencies
are just `torch` + `numpy` + `mir_eval`, all already present and CUDA-enabled in the `analyze-smc`
env — verified. Training budget on record is 1000 steps at batch 16 / crop 256 / 8-dim latents, i.e.
low-single-digit GPU-hours per arm, though **no wall-clock was ever logged, so that is inference, not
a measurement**. The one thing a rebuild would *fix* is the asterisk: run it on a fold-honest frontend
so the SMC number is genuinely zero-shot.
