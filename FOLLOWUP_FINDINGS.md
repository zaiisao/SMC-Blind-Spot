# Follow-up analysis: blind spots and inference-time gains on SMC

Companion notes to *The SMC Blind Spot*. Everything here was recomputed with a
single shared harness (`scripts/_harness.py`) that reproduces the paper's
pipeline exactly — Beat This raw peak-pick = **0.6270**, identical to the paper.
All rows use the same eval (`mir_eval` 5 s trim, ±70 ms). Env:
`/home/sogang/mnt/db_2/anaconda3/envs/analyze-smc`.

## 0. The paper's numbers reproduce

Every headline figure checks out against the committed CSVs / activations:
median 70.9 BPM, IQR 57–93, 45 tracks (21 %) < 55 BPM, 61 (28 %) < 60;
error taxonomy 62 / 17 / 38 / 11 / 89 (sequential); global-best fixed λ = 5 → 0.603,
λ = 100 → 0.592, per-track optimal λ → 0.642; Spearman(act@GT, F) = 0.784
(p = 2e-46); act@GT ratio good/total-fail = 3.1×. The analysis is sound. The
items below are *unexplored levers* and *one overstated claim*, not arithmetic bugs.

---

## Blind spot 1 — "F = 0.673 is an upper bound on any decoder" is not correct

The paper derives 0.673 from a per-track **threshold** sweep and states it bounds
"any decoder operating on the real activations" (§5.3, §5.4.5, Conclusion). But
the threshold sweep only bounds the *thresholded-peak-picker family*. Recomputed
in one harness on Beat This's real activations, **no GT tempo**, oracle
hyper-parameter selection (the same kind the threshold sweep already uses):

| decoder (per-track oracle hyperparam) | F |
|---|---|
| peak-pick, oracle threshold | 0.670 |
| wide DBN [30,215], oracle λ | 0.653 |
| **per-track best of the two** | **0.691** |

On **78 / 217** tracks the DBN beats the best peak-picking threshold — different
decoders fix different tracks. No single decoder exceeds 0.673, but their
*envelope* reaches 0.691 (0.685–0.702 if you instead reuse the paper's own
`transition_lambda` / GT-tempo CSVs). Suggested fixes:

- Scope the claim: "0.673 upper-bounds thresholded peak-picking," not "any decoder."
- The §5.4.5 split "85 % activation / 15 % decoder" should be ~**78 % / 22 %**
  (decoder gap = 0.691 − 0.627 = 0.064 of the 0.297 total). **The qualitative
  conclusion — the activation function is the dominant bottleneck — still holds.**
- This actually *strengthens* the §6.1 "adaptive decoder" argument: the
  no-single-setting tension you show for λ also holds at the decoder-family level.

## Blind spot 2 — model fusion was never tried; the "activation ceiling" is model-specific

Fusing Beat This + Beat Transformer activations (resampled to a common 50 fps grid,
mean of probabilities) — free, no retraining, no GT:

| | realistic peak-pick F | oracle-threshold ceiling |
|---|---|---|
| Beat This alone | 0.627 | 0.670 |
| Beat This + Beat Transformer (mean) | 0.635 | 0.686 |

So the "~0.67 activation ceiling" is a property of **one model**, not of activation
functions in general. Caveat that *supports your thesis*: the two models' per-track
F correlate ρ = 0.75 and the "pick the better model per track" oracle is only 0.664
— fusion beats selection because averaging suppresses spurious peaks, but the gain
is capped by the shared percussion/steady-tempo training bias. Fusing two
similarly-biased models can't escape the shared blind spot.

## Blind spot 3 — test-time augmentation improves all three metrics for free

Pitch-shift TTA on Beat This (mean activation over shifts {−2,−1,0,+1,+2} semitones,
each track's correct held-out fold checkpoint, timing preserved so no un-warping):

| metric | Beat This | TTA (p = 0.5, no tuning) |
|---|---|---|
| F | 0.627 | **0.633** |
| CMLt | 0.514 | **0.531** |
| AMLt | 0.610 | **0.625** |

Unlike the DBN (trades F for AMLt) or fusion (trades CMLt), TTA improves **all
three at once**. This is direct positive evidence for the "confident-but-wrong"
diagnosis: a subset of wrong peaks are unstable to small input changes and average
out, while genuine beats are shift-stable.

## Blind spot 4 — time-stretch TTA is the strongest free lever, and it gives causal evidence for the tempo thesis

Same idea as pitch-shift TTA but perturbing **tempo**: stretch the audio by rate
r (librosa, r>1 = faster), run Beat This, un-warp the activation back to the
original 50 fps grid (a stretched-time τ sits at original-time τ·r — independently
verified; rate=1.0 reproduces 0.627 to |diff|=0.0), then average across rates.
Scripts: `scripts/exp_ts_worker.py` (4-GPU sharded inference), `scripts/eval_timestretch.py`.

**Per-rate competence curve (single rate, peak-pick @0.5) — causal evidence:**

| GT tempo band | n | rate 0.7 | 0.85 | **1.0** | 1.2 | 1.5 |
|---|---|---|---|---|---|---|
| < 55 BPM | 45 | 0.447 | 0.495 | 0.537 | **0.562** | 0.534 |
| 55–90 BPM | 110 | 0.534 | 0.598 | 0.623 | **0.625** | 0.580 |
| > 90 BPM | 62 | 0.705 | **0.735** | 0.700 | 0.678 | 0.628 |

Each tempo band peaks at a *different* rate — slow tracks improve when sped up,
fast tracks improve when slowed down — i.e. activation quality rises when the
displayed tempo moves toward the ~80–140 BPM training zone. This is a *causal*
demonstration of the training-distribution-mismatch thesis (intervene on tempo →
the activation function itself gets better), stronger than the correlational
evidence in the paper.

**Realistic time-stretch TTA (average un-warped activations, no GT, no tuning, p=0.5):**

| config | F | CMLt | AMLt |
|---|---|---|---|
| Beat This | 0.627 | 0.514 | 0.610 |
| pitch-shift TTA | 0.633 | 0.530 | 0.626 |
| **time-stretch TTA {0.85,1,1.2}** | **0.645** | **0.550** | **0.649** |

+0.018 F and large continuity gains (+0.036 CMLt, +0.039 AMLt), all free. This is
the best single realistic lever found — bigger than pitch-shift or fusion, because
it perturbs the exact axis (tempo) where the model is weakest.

**Where the +0.018 comes from (peak-pick @0.5, by GT tempo band):**

| band BPM | n | base | TTA | ΔF | % of gain | +/=/− |
|---|---|---|---|---|---|---|
| 0–55 | 45 | 0.537 | 0.569 | +0.033 | 38 % | 28/3/14 |
| 55–70 | 58 | 0.587 | 0.622 | +0.035 | 53 % | 38/4/16 |
| 70–90 | 52 | 0.663 | 0.662 | −0.001 | ~0 % | 23/6/23 |
| 90–120 | 44 | 0.737 | 0.743 | +0.006 | 6 % | 19/14/11 |
| 120+ | 18 | 0.610 | 0.620 | +0.010 | 5 % | 10/3/5 |

**74 % of the gain comes from tracks below 64 BPM** — beyond Beat This's own
±20 %-augmented training range (~64–168 BPM); the two slow bands (<70) give 91 %.
The 70–90 band is a dead wash (ΔF −0.001, 23 up / 23 down) and in-cluster tempos
(90–120) gain almost nothing. Spearman(baseline F, ΔF) = −0.075 (p=0.27, n.s.), so
this is a *tempo*-coverage effect, not a generic "rescue the hard tracks" effect.
Implication: the deployable TTA gain is almost entirely **slow-tempo coverage** that
training-time ±20 % tempo aug cannot reach — a wider training tempo range or curated
slow/expressive data would likely capture most of it (and a generic
consistency/invariance objective over in-range tempos would not). Not free, though:
~32 % of tracks (69/217) get slightly worse; for in-range tempos it is a coin flip.

**Two caveats, stated honestly:**
- *Oracle* per-track rate selection reaches **0.712** (uses GT-F to pick the rate
  — same class of oracle as the paper's per-track threshold/λ). So stretch-rate is
  yet another complementary axis with real headroom (it exceeds the 0.691 decoder
  envelope of Blind Spot 1).
- BUT that 0.712 is **not** unlockable by tempo estimation: routing each track to a
  fixed "sweet-spot BPM" via *GT tempo* does not lift F (≈0.61–0.63), because the
  best rate per track is not a clean function of tempo (phase-vocoder artifacts,
  peak blur). And routing by *estimated* tempo (Beat Transformer, 59% acc) gives
  only 0.631 — **averaging across rates (0.645) beats routing**, because it needs
  no tempo estimate and is robust to tempo-estimation error (the paper's known
  weak spot). So the realistic recipe is "average a few mild stretches," not
  "estimate tempo and normalize."

## Stacking the levers (peak-pick, p=0.5 unless noted)

| config | F | CMLt | AMLt | oracle ceiling |
|---|---|---|---|---|
| Beat This (paper baseline) | 0.627 | 0.514 | 0.610 | 0.670 |
| pitch-shift TTA | 0.633 | 0.530 | 0.626 | 0.676 |
| Beat This + Beat Transformer fusion | 0.635* | — | — | 0.686 |
| **time-stretch TTA** | **0.645** | **0.550** | **0.649** | **0.703** |
| pitch + time-stretch TTA | 0.640 | 0.545 | 0.644 | 0.692 |
| pitch+TS TTA + BTrans fusion | 0.633 | 0.517 | 0.609 | 0.697 |

(*one global threshold.) **Time-stretch TTA alone is the winner** — stacking
pitch-shift, or fusing Beat Transformer on top, does *not* improve it (the extra
sources are correlated/weaker and dilute the average). So the recommended recipe
is the single cheapest lever: **average a few mild time-stretches of one model.**
It beats the paper's best realistic number by **+0.018 F / +0.036 CMLt / +0.039
AMLt** with zero retraining and no ground truth, and raises the achievable
activation-decoder ceiling from ~0.67 to ~0.70.

---

## Blind spot 5 — TTA (realistic activation lever) compounds with oracle tempo

The paper frames F (placement) and CMLt (coherence) as *two independent problems*
and shows the best config is raw activations + GT-tempo DBN + per-track optimal λ
= F 0.667 / CMLt 0.735 (Table 3). TTA attacks the activation (placement); oracle
tempo attacks coherence. Do they stack? Feeding the TTA activations (beat **and**
downbeat) through the same GT-tempo ±20 % DBN with per-track optimal λ
(`scripts/exp_tta_full_worker.py`, `scripts/eval_tta_oracle_tempo.py`):

| activation | decoder | F | CMLt | AMLt |
|---|---|---|---|---|
| raw Beat This | peak-pick @0.5 | 0.627 | 0.514 | 0.610 |
| raw Beat This | GT-tempo, λ=100 | 0.618 | 0.700 | 0.729 |
| raw Beat This | GT-tempo, opt-λ | 0.667 | 0.735 | 0.755 |
| pitch-shift TTA | GT-tempo, opt-λ | 0.672 | 0.739 | 0.764 |
| **time-stretch TTA** | **GT-tempo, opt-λ** | **0.681** | **0.753** | **0.773** |
| pitch+TS TTA | GT-tempo, opt-λ | 0.680 | 0.746 | 0.768 |

(Sanity: raw + GT-tempo + opt-λ reproduces the paper's 0.667/0.735/0.755 exactly;
raw + GT-tempo λ=100 reproduces §5.4.5's 0.618/0.700/0.729 exactly.)

**They compound, almost perfectly additively.** Decompose the gains from raw
peak-pick (0.627): TTA adds +0.018 F; GT-tempo+opt-λ adds +0.040 F; doing both
gives +0.054 (vs +0.058 if perfectly additive — interaction only −0.004). This is
quantitative confirmation of the paper's "two independent problems" claim, and
shows **both ceilings can be pushed at once**:

- F = **0.681** exceeds the paper's stated activation ceiling (~0.67, §6.1) —
  because TTA yields better activations than the single static forward pass
  (consistent with Blind Spots 1 & 4: that ceiling is inference-specific).
- CMLt = **0.753** / AMLt = **0.773** exceed the paper's best (0.735 / 0.755).

Caveat: GT-tempo and opt-λ are **oracles** (ground-truth tempo; per-track λ chosen
on GT-F). TTA is the only *realistic* lever in that row. So 0.681 is "realistic
activation improvement + perfect tempo/decoder" — the new upper end of Table 3,
not a deployable number. The deployable number is still **TTA peak-pick = 0.645**.
The takeaway for the paper: the activation lever you can't see (TTA) and the tempo
lever you already study are orthogonal — a learned tempo-aware decoder (your §6.3
proposal) applied to TTA activations would target both at once.

## Deployable check: TTA with NO ground truth (no GT tempo, no per-track oracle)

Stripping every oracle — only native peak-picking, the standard wide DBN (default
λ=100 or one global-best λ), and Beat Transformer's *predicted* tempo (59% acc, no
GT) — on the cached TTA activations (`scripts/eval_tta_realistic.py`):

| activation | peak-pick | wide DBN (best global λ) | BT-pred-tempo DBN (best) |
|---|---|---|---|
| raw Beat This | 0.627 | 0.600 | 0.565 |
| pitch TTA | 0.633 | 0.606 | 0.574 |
| **time-stretch TTA** | **0.645** | 0.621 | 0.583 |
| pitch+TS TTA | 0.644 | 0.620 | 0.580 |

**No realistic decoder beats native peak-picking on any activation set** — the
standard DBN and even Beat-Transformer-predicted-tempo both *reduce* F (the
DBN's rigidity hurts placement; the predicted tempo is too inaccurate to help),
exactly as the paper found for raw activations. So the honest deployable result is:

> **time-stretch TTA + peak-picking = F 0.645 / CMLt 0.550 / AMLt 0.649**
> (+0.018 F, +0.036 CMLt, +0.039 AMLt over raw Beat This; no retraining, no GT)

The 0.681 in Blind Spot 5 required GT tempo and is *not* achievable today — it only
marks how much further the TTA activations could go if tempo estimation improved.

## Bottom line

The paper's core claims survive and are *reinforced*: the activation function is
the dominant bottleneck (~78 % of the gap, vs the stated 85 %), and training-data
tempo mismatch is the root cause — now shown *causally* (Blind Spot 4: moving a
track's tempo toward the training zone improves its activation), and supported by
the ρ = 0.75 correlated errors and limited fusion gain. Corrections / additions
for a revision:

1. Re-scope the 0.673 "any decoder" upper bound → thresholded-peak-picking only;
   adjust the §5.4.5 split from 85/15 to ~78/22 (qualitative conclusion unchanged).
2. Add **time-stretch TTA** as a cheap inference-time baseline (+0.018 F, large
   CMLt/AMLt gains, no retraining/GT) and report the per-band competence curve as
   causal support for the training-distribution argument.
3. Note that the tempo-normalization headroom (oracle rate 0.712) is gated by
   tempo-estimation accuracy and is *not* realized by routing on an estimated
   tempo — averaging is the robust route. This sharpens the paper's "tempo at
   inference time" discussion (§6.3).
4. Extend Table 3: TTA activations + GT-tempo + opt-λ = **0.681 F / 0.753 CMLt /
   0.773 AMLt**, the new best in any configuration, beating both of the §6.1
   ceilings. The activation lever (TTA) and the tempo lever are nearly perfectly
   additive — direct quantitative support for the "two independent problems" claim.

## Reproduce

```bash
PY=/home/sogang/mnt/db_2/anaconda3/envs/analyze-smc/bin/python
$PY scripts/exp_decoder_ceiling.py     # blind spot 1  (CPU, ~few min)
$PY scripts/exp_ensemble.py            # blind spot 2  (peak-pick sweeps)
$PY scripts/exp_ensemble_dbn.py        # blind spot 2  (realistic decoders)
$PY scripts/exp_tta.py                 # blind spot 3  (GPU, ~8 min; writes data/tta_activations/)
$PY scripts/exp_tta_ensemble.py        # stacking pitch-TTA + fusion (needs exp_tta.py)
# blind spot 4 — time-stretch TTA (4-GPU sharded inference, then evaluate):
for f in "0,1 cuda:0" "2,3 cuda:1" "4,5 cuda:2" "6,7 cuda:3"; do set -- $f; \
  $PY scripts/exp_ts_worker.py --folds $1 --device $2 & done; wait
$PY scripts/eval_timestretch.py        # writes nothing; prints Phases A/B/C
```
