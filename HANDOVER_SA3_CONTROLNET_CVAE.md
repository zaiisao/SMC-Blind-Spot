# Handover: Stable Audio 3 + ControlNet + CVAE for beat-tracking data synthesis

Status as of 2026-08-22. Written so someone (including future us) can pick this up without
re-deriving the reasoning or repeating the measurements.

---

## 1. Why this exists — the argument in one page

SMC beat tracking is stuck at F ≈ 0.627 (peak-picked Beat This). Two days of work closed every
decoder-side and representation-side route:

| route | best case | realistic |
|---|---|---|
| beat-only decoding | — | **+0.0255 F / +0.0570 CMLt** (the one clear win, in our stack) |
| weight rebalance + TTA | — | **+0.022 F** (in stack) |
| tempo, injected softly (4 heads, 49–58% acc.) | oracle +0.057 F | +0.010–0.017 CMLt, ~0 F |
| tempo, hard state-space pruning | oracle +0.044 F / +0.170 CMLt | **exactly 0.0000** — gains and octave-error damage cancel |
| per-track decoder hyperparameters | apparent +0.077 F | **−0.017 F** vs best global, after split-half correction |
| second frontend (Beat Transformer) fusion | oracle +0.039 F | +0.006, p=0.89 |
| meter sets, λ, tempo grid density, tempo floor | — | ≤ 0 (grid densification significantly *worse*, p=0.042) |
| MERT / foundation-model features | — | below baseline; BeatFM's published SMC best is 61.3 vs our 63.89 |
| MMI through the DBN (4 settings) | — | never beat its own control |

**The conclusion those force:** a decoder cannot out-decode its input. Every bound above is a
statement about information present in the activation stream. The remaining deficit is *frame-level
evidence on soft-onset expressive material*, and the fold checkpoints prove capacity is not the
limit — they score **~0.71 F on SMC tracks they trained on vs 0.63 held out**. That gap is the
target, and it says the binding constraint is **training-distribution coverage**.

**Why synthesis rather than collection:** the missing regime is *slow AND expressive*, which is
12.4% of SMC and 1.6% of the corpus (measured, `scripts/diagnostics/tempo_curve_stats.py`). You cannot collect
your way out of that — annotating expressive rubato is exactly what SMC's authors found hardest.
But you can synthesize it, because beat labels transform exactly through a known warp.

**Novelty check (done):** beat/rhythm-conditioned generation exists (Music ControlNet, MusiConGen,
MusTango, JASCO, Coco-Mulla). Augmentation for beat tracking exists (Chiu et al. 2021 —
*source separation*, not generation). **Generating audio with a beat-conditioned diffusion model in
order to train a beat tracker: no prior work found.** Position against Chiu et al.: same problem
(label scarcity), different instrument, and separation cannot synthesize a regime absent from the
source recordings.

---

## 2. What has been established empirically (do not re-run these)

### 2.1 SA3 loads and the autoencoder works
- Weights: `/disk1/jaehoon/sa3_small_music` (`stabilityai/stable-audio-3-small-music`, gated=auto,
  token on this machine has access).
- Env: `/disk1/jaehoon/sa3env` — a `--system-site-packages` venv inheriting torch 2.7.1+cu118.
- **SAME autoencoder: 108.2M params, loads with 0 missing / 0 unexpected keys.**
- `downsampling_ratio=4096` at 44.1 kHz → **latent 10.77 Hz = 92.9 ms per frame**.

### 2.2 Timing survives the codec far better than the compression ratio suggests
`scripts/sa3/sa3_roundtrip.py`, encode→decode on all 217 SMC tracks:
- matched-onset displacement **median 11.6 ms**, 88% of tracks under 20 ms
- but per-track **p90 ≈ 81 ms**, and onset *counts* change >10% on 154/217 tracks
- SNR ≈ 12.6 dB, mel-L1 ≈ 2.66 dB → audible degradation, acceptable for training data

### 2.3 The timing loss is ENCODER-side, not decoder capacity — this is the key result
`scripts/sa3/sa3_latent_opt.py`. Freeze the decoder, optimise `z` directly (GAN-inversion as a capacity
probe), 12 hardest tracks:

```
                       onset med    onset p90
encoder                  22.3 ms      202.1 ms
optimised from encoder    7.3 ms       93.6 ms
optimised from noise      13.5 ms      58.4 ms   (n=3, matched subset: beats encoder-init)
```

**The decoder can render timing the encoder throws away.** Since *generation never uses the
encoder*, the round-trip tail does not bound a ControlNet plan. Caveat: improved p90 on only 7/12
tracks, so the mean is heavy-tailed — "clearly encoder-side on the majority, unresolved on the rest."

### 2.4 Two metric traps already paid for — do not repeat
- **Nearest-neighbour reassignment.** A "beat-anchored displacement" metric (distance from each true
  beat to the nearest onset, before vs after) reported 128 ms while matched-onset displacement was
  11 ms. Cause: onset counts change, so "nearest" flips to a different event. Only matched-onset
  metrics are valid.
- **Phase-blind objectives smear transients.** Optimising multi-resolution *magnitude* spectrogram
  L1 alone reduced the loss (1.88→1.31) while making onset p90 *worse* (48.8→106.8 ms). The
  objective must contain a positive-spectral-flux (onset-envelope) term — see `mel_loss()` in
  `sa3_latent_opt.py`. **This is very likely why SAME's encoder discards timing in the first place:
  its training loss had no reason to preserve it.**

---

## 3. The design

### 3.1 Where conditioning goes, and why

| component | conditioned? | reason |
|---|---|---|
| **Encoder** | yes (CVAE) or onset-aware fine-tune | it discards timing — measured (§2.3) |
| **DiT** | yes (ControlNet) | it decides what generated/edited latents contain |
| **Decoder** | **no — freeze** | already proven capable at 7.3 ms |

### 3.2 The CVAE objective, and the two deliberate non-conditionings

```
L  =  E_q(z|x,b)[ log p(x|z) ]  −  KL( q(z|x,b) ‖ p(z) )
        ↑ encoder conditioned      ↑ decoder UNconditioned   ↑ prior UNconditioned
```

- **Decoder stays `p(x|z)`.** If it could see `b` directly, the encoder would be free to *not* put
  timing into `z` — the beats would arrive by the side channel. Leaving it unconditioned **forces
  `z` to carry the beats**, which is required because the DiT only ever produces `z`.
- **Prior stays `p(z)`.** The KL against the *unconditional* prior is what keeps new latents near
  the manifold the DiT was trained on. Conditioning the prior licenses drift.
- **The reconstruction term must include an onset/flux component** (§2.4).

Caveat to verify: SAME may not be a true VAE — the checkpoint has `bottleneck.bias` and
`bottleneck.noise_scaling_factor`, suggesting a noise-regularised bottleneck rather than a Gaussian
posterior with KL. If there is no KL, the ELBO framing is decorative; keep the existing regulariser
active unchanged (to stop latent-scale drift) and modify only the reconstruction term.

### 3.3 The diffusion objective does not change

```
L = E ‖ v_θ(z_t, t, c) − v ‖²        c = (text, duration)
                ↓
L = E ‖ v_θ(z_t, t, c, C_φ(b)) − v ‖²    θ frozen, only φ trained
```

Same loss, same schedule, same targets — the conditioning set grows by one signal. `C_φ`
zero-initialised so training begins as the identity.

**Alternative that changes nothing in training:** classifier guidance,
`∇log p(z_t) + s·∇log p(b|z_t)`. Cheaper (train only a beat classifier on noisy latents) but the
guidance signal is a beat predictor, and beat predictors are weakest on exactly the slow expressive
material we are targeting. ControlNet trains on ground-truth annotations and avoids that
circularity.

### 3.4 Do pretrained weights survive the CVAE surgery? Yes — with zero-init

- **Extra input channel:** first conv `Conv1d(2, C, k)` → `Conv1d(3, C, k)`; copy the pretrained
  `[C,2,k]` slice, **zero-init the new channel's slice**. At step 0 the model is bit-identical to
  the original; gradient still flows into the new slice.
- **FiLM alternative:** per-block scale/shift from a beat embedding, projection zero-initialised →
  scale 1, shift 0 → identity at init.
- Encode beat phase as **sin/cos of continuous phase, not a binary pulse**, so sub-frame position
  is expressible despite the 10.77 Hz grid.

### 3.5 The coupling problem — the main architectural risk

Fine-tuning the encoder changes the latent distribution the **DiT** was trained on.

- **Decoder** — self-solving. Fine-tune the encoder with the **decoder frozen**: the training signal
  is "produce latents this decoder reconstructs well," which pins the encoder to the decoder's
  manifold by construction.
- **DiT** — *not* self-solving. It learned the marginal over *old* latents.

Three ways out:
1. **Freeze the encoder**, fix timing only in the ControlNet. Cheapest. Viable only if the DiT's
   prior actually represents sharp beats — **untested, see §5.1**.
2. **Fine-tune encoder + DiT together.** Correct, expensive.
3. **Skip the DiT for editing** — warp in the waveform domain, timbre-transfer another way.

**Practical ordering that avoids debugging two distribution shifts at once:** condition the encoder,
fine-tune with the decoder frozen, verify the pipeline works *without* the DiT in the loop, and only
then bring the DiT in.

---

## 4. Why editing (not pure generation) is our use case

- **Generation from scratch:** labels come from conditioning → requires trusting adherence →
  unverifiable on slow expressive music, because our verifiers are the trackers that fail there.
  Circular.
- **Editing real audio:** labels come from the source plus a *known* warp → **exact by
  construction**. No trust required.

Intended pipeline: well-annotated corpus track → **time-warp** to SMC-like rubato (labels transform
exactly, no model involved) → **re-timbre** toward SMC's acoustic character (solo piano, quiet, soft
onsets). The second step goes through the encoder, which is why §2.3/§3.2 matter.

This is also the direction VBPM is going — a conditional latent-variable model with musical
structure in the posterior — so the CVAE work is shared between projects.

---

## 5. Open questions, in priority order

### 5.1 Did the DiT inherit the encoder's blind spot? (cheapest, do first)
The DiT was **trained on encoder outputs**. If those have smeared timing, the learned prior may not
represent sharp beats, and a ControlNet would be steering toward regions the prior never saw.
**Test:** generate audio from SA3 unconditionally, measure onset sharpness and inter-onset
regularity vs real audio. Needs the full model (DiT + T5Gemma text encoder — not yet downloaded),
no training. **This decides between path 1 (small job) and path 2 (large job) in §3.5.**

### 5.2 Does SA3's prior even contain expressive solo piano?
Trained on AudioSparx + Freesound (stock music and SFX). If the base model cannot render quiet
expressive classical piano, the text prompt cannot supply the style no matter how good the
conditioning. **Test:** prompt it and listen. Minutes.

### 5.3 Non-circular adherence measurement
Before trusting any generated label, calibrate the verifier on *real* audio with known annotations,
then apply it to generated audio. Without this we cannot distinguish "the model followed the beat"
from "our tracker agrees with our tracker."

### 5.4 Is the residual p90 a decoder floor or unconverged optimisation?
§2.3 left optimised p90 at 93.6 ms, but noise-init beat encoder-init on the matched subset (58.4 vs
77.4), implying encoder-init is stuck in a local optimum and the true floor is lower. More steps /
better init would settle it.

---

## 6. Training data for the ControlNet

~3,100 annotated tracks (ballroom 672, GTZAN 999, Hainsworth 222, Beatles 179, ASAP 469, RWC 126,
HJDB 232, SMC 217). **Spectrograms for all 16 Beat This datasets are already on disk at
`/disk1/jaehoon/beat_this_spect/done/*.npz`** (170 GB, 22 augmentation variants per track).

**Fold discipline.** Two valid options:
- **Exclude SMC entirely, train the ControlNet once.** Costs 7% of data. Keep fold discipline
  downstream (tracker trains on synthetic + real folds ≠ f, evaluates on f).
- **8-fold CV on the ControlNet** — valid but 8× training cost.
- **Middle path (recommended):** keep SMC *audio* out, but sample the conditioning beat sequences
  from SMC-like tempo-curve statistics computed **per fold** (folds ≠ f). A few distributional
  parameters rather than audio; fold-honest by construction; and it is what actually steers
  generation toward the missing regime.

**The division of labour that makes this work at all:** the ControlNet learns only *rhythmic
control*, which is easiest to learn from percussive well-annotated material. The *acoustic content*
comes from the frozen base model's prior plus the text prompt. We compose two things that neither
corpus has together. (Risk: §5.2.)

**Note:** SMC's rubato is not the missing axis in isolation — ASAP is *more* expressively timed
(σ_d 0.213 vs SMC's 0.152) with 473 annotated tracks. The missing thing is the **conjunction** of
expressive *and* slow (ASAP sits at 106 BPM median, SMC at 71).

---

## 7. Target and kill criteria

**Target: F > 0.70 on SMC.** Grounded, not arbitrary: the fold checkpoints already reach ~0.71 F on
tracks they trained on. You are trying to close a *measured* gap, not hoping for an unprecedented
number. Discount: memorisation ≠ generalisation, and every field that tried synthetic training data
hit a domain gap.

**Kill criteria, written down in advance:**
- If the warp-only augmentation (no diffusion) and the first synthetic run both come back flat, the
  data hypothesis is **falsified**, not under-resourced.
- If §5.1 shows the DiT inherited smeared timing, the job is path 2 in §3.5 (encoder + DiT
  retraining) — re-scope before committing.
- If §5.2 shows the base model cannot render expressive piano, the style cannot come from the prior
  and the whole composition argument collapses.

**Known negative already:** fine-tuning on slowed-ASAP *alone* degrades SMC (−0.067 F). Confounded
by distribution narrowing (ASAP is already in Beat This's training set), so it falsifies "fine-tune
on slowed ASAP alone," **not** "slow expressive data helps." The clean version trains on the full
corpus with slowed ASAP mixed in.

---

## 8. Environment / reproduction notes

- **SA3 env:** `/disk1/jaehoon/sa3env/bin/python` (venv with `--system-site-packages`).
  `pypi.ngc.nvidia.com` is **DNS-blocked** here, so torch cannot be freshly installed — the venv must
  inherit it. `PyWavelets` must be pinned to **1.4.1** (current wheels are built against numpy 2.x;
  this environment is numpy 1.24.3). Install `stable-audio-tools` with `--no-deps` and add missing
  pure-Python packages individually.
- **Loading just the autoencoder** (avoids `transformers`/`lightning`):
  `create_autoencoder_from_config({'model': cfg['model']['pretransform']['config'], 'sample_rate': cfg['sample_rate']})`,
  then load keys prefixed `pretransform.model.` from `model.safetensors`.
- **Gradient through decode:** `AudioAutoencoder.decode()` routes its *nested* pretransform (a
  wavelet stage) through a `no_grad` branch unless `ae.pretransform.enable_grad = True`. Without
  that, `loss.backward()` raises "does not require grad" and it looks like the model is broken.
- **AE pads to a multiple of 4096 samples**, so decoded output is longer than input (176400 →
  180224). Trim before any waveform comparison or every displacement carries a constant offset.

## 9. Scripts

| script | what it does |
|---|---|
| `scripts/sa3/sa3_roundtrip.py` | encode→decode fidelity, matched-onset displacement, n=217 |
| `scripts/sa3/sa3_latent_opt.py` | latent optimisation capacity probe (encoder-side vs decoder-side) |
| `scripts/diagnostics/tempo_curve_stats.py` | SMC vs corpus rubato/tempo statistics; defines the missing regime |
| `scripts/experiments/exp_slow_asap.py` | slowed-ASAP fine-tune (the known negative, §7) |

All committed on branch `archive/2026-08-experiments`.
