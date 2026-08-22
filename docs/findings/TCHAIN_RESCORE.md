# tchain re-score: verifying "F 0.747 / CMLt 0.704 on GTZAN"

Inference only. No training was launched. All work done in /tmp; nothing under
`Analyze-SMC/vbpm/` or `VBPM/` was modified.

> **EVERY NUMBER IN THIS DOCUMENT IS A DOWNBEAT NUMBER.** `tchain` is downbeat-only.
> There is no beat read-out anywhere in the chain family. See §1.

---

## 1. What the metric actually is (verified in code, not assumed)

| property | value | evidence |
|---|---|---|
| event type | **DOWNBEAT only** | `vbpm/scoring/evaluation.py` header prints `==== downbeat F (+-70 ms) ====`; the reference in `evaluate()` is `truth = np.asarray(crop["downbeat_times"])`; the training target `y` in `vbpm/data/excerpts.py::_targets` is built **only** from `downbeat_times` |
| read-out | **rule-g**, label-free | `vbpm/readout.py::downbeat_times` — fractional-frame 2π crossings of the bar-phase path, linearly interpolated; called via `evaluation.rule_g_times` |
| latent | **bar** phase | `tchain.py` rate grid `tempo_lo=0.030 … tempo_hi=0.180` rad/frame at 50 fps ⇒ **bar** period 0.70–4.19 s; prior mean `tempo_prior_mu=-2.6827` ⇒ 0.0684 rad/frame ⇒ 1.84 s bar |
| F-measure | greedy one-to-one match, tol **±0.070 s** (`TOLERANCE_S`) — the repo's own implementation, *not* `mir_eval.beat.f_measure` | `evaluation.py::f_measure` |
| CMLt / AMLt | `mir_eval.beat.continuity(annotated, predicted)`, fields 2 and 4; returns (0.0, 0.0) if either sequence has ≤1 event | `evaluation.py::continuity_scores` |
| fold | **`None`** = the test-only group, held out of every checkpoint | `dataset.py::split_songs`; GTZAN is the only dataset in the store with **no** `8-folds.split` file, so it lands in fold `None` |
| window | **deterministic middle 30.0 s** (`excerpt_seconds: 30.0`), one window per song | `excerpts.py`: `start = longer // 2` when `deterministic=True`; `evaluate()` asserts `dataset.deterministic` |
| n | **993** (999 GTZAN songs, 6 rejected for `< MIN_DOWNBEATS = 4` downbeats) | measured this run |
| overlap | **0 songs** shared between (train ∪ val) and the fold-`None` test set | measured this run; train 1850 / val 267 (fold 7) / test 999 |

**SMC was not evaluated and must not be.** SMC is beat-only; a downbeat-only model
is structurally incompatible with it. Nothing in this report touches SMC.

## 2. Code revision used

The checkpoints are dated 2026-08-18 04:27–07:23. The tree state of that day is
commit **`87a9182`** (2026-08-18 05:10, the last commit before the 17:48 batch);
`066daa5` is its immediate parent-side sibling in the same push and introduced
`tchain.py`. I extracted `87a9182` with `git archive` into `/tmp/vbpm87` (the live
repos were never touched) and scored against **that** revision.

`vbpm/scoring/evaluation.py` at `87a9182` differs from HEAD by exactly one line
(an import path move `..model` → `..readout`); the metric code is byte-identical.
`vbpm/configs/tchain.yaml` at `87a9182` differs from HEAD by one added key
(`use_graphs`). So the metric implementation is not in question.

**The checkpoints DO load, cleanly.** Contrary to the open "full 60-epoch gtzan runs
do not reproduce from the saved checkpoints" item, all four surviving tchain
checkpoints load `strict=True` with `<All keys matched successfully>` for **both**
the model and the frozen frontend. Each blob also carries its own `config` dict, so
I rebuilt the config from the blob rather than re-reading today's YAML — that
removes config drift as a confound entirely.

## 3. Checkpoint inventory

| dir | file | status |
|---|---|---|
| `tchain/` | — | **EMPTY DIRECTORY, no checkpoint** |
| `tchain_full/` | — | **EMPTY DIRECTORY, no checkpoint** |
| `tchain_s0/` | `seed0.pt` | loads, scored |
| `tchain_s1/` | `seed1.pt` | loads, scored |
| `tchain_true/` | `seed0.pt` | loads, scored |
| `tchain_false/` | `seed0.pt` | loads, scored |

Two directories, `tchain/` and `tchain_full/`, are empty — those runs' checkpoints are
gone — and no corpus log survives for any tchain run (`grep` over `VBPM/logs/` and
`*.log` finds no tchain gtzan table, confirming the "no log file on disk" item). That
turned out not to matter: **the 0.747 is carried by `tchain_true/seed0.pt`, which does
exist and does re-score to it** (§4).

Overrides recorded in the blobs:
- `tchain_s0`, `tchain_s1`: `batch_size=8 epochs=20 beta_warmup=0 chain_sigma=0.03 tempo_revert=true`
- `tchain_true`: `batch_size=16 epochs=15 tempo_revert=true chain_sigma=0.03`
- `tchain_false`: `batch_size=16 epochs=15 tempo_revert=false chain_sigma=0.03`

Note the recipes here are **15–20 epochs / batch 8–16**, *not* the "stride 8 / class
emission / 24 epochs" archain protocol. The protocols are **not** identical; the open
"UNVERIFIED whether the protocols are identical" item resolves to **NOT IDENTICAL**,
but that only ever concerned archain-vs-tchain, and the Beat This comparison below
is run inside the same forward pass on the same crops, so it is unaffected.

## 4. Results — all DOWNBEAT, GTZAN fold `None`, n = 993

Read-out rule-g, deterministic middle 30 s window, tolerance ±0.070 s. Beat This is
scored **in the same forward pass, on the same crop, against the same reference**,
using its own official minimal postprocessor on its own downbeat head.

| checkpoint | recipe | **downbeat F** | **downbeat CMLt** | **downbeat AMLt** | n |
|---|---|---|---|---|---|
| `tchain_true/seed0.pt` | 15 ep, bs 16, `tempo_revert=true` | **0.7470** | **0.7038** | **0.8489** | 993 |
| `tchain_false/seed0.pt` | 15 ep, bs 16, `tempo_revert=false` | 0.7537 | 0.7079 | 0.8402 | 993 |
| `tchain_s0/seed0.pt` | 20 ep, bs 8, seed 0 | 0.7498 | 0.7011 | 0.8483 | 993 |
| `tchain_s1/seed1.pt` | 20 ep, bs 8, seed 1 | 0.7533 | 0.7017 | 0.8484 | 993 |
| `tchain/` | — | **no checkpoint on disk** | — | — | — |
| `tchain_full/` | — | **no checkpoint on disk** | — | — | — |
| *Beat This (final0, minimal), identical crops* | — | **0.7844** | **0.6717** | **0.7910** | 993 |

**`tchain_true` reproduces the claim to 4 decimal places: F 0.7470 vs claimed 0.747,
CMLt 0.7038 vs claimed 0.704.** The headline run is the `tempo_revert=true` arm, which
is exactly what the `066daa5` commit message is arguing for. The claim's provenance is
therefore now closed: it is `tchain_true/seed0.pt`, and it re-scores.

The other recorded line — `n=993  ours 0.7498  beat-this 0.7844  delta -0.0347` — is
reproduced **exactly** by `tchain_s0` (0.74978 / 0.78445 / −0.03467). Two independent
recorded numbers land on two different checkpoints, both to 4 dp. Nothing here is a
harness accident.

Spread across all four scored tchain checkpoints (different recipes, different seeds):
F range 0.0066 (sd 0.0032), CMLt range 0.0068 (sd 0.0031), AMLt range 0.0088 (sd 0.0042).

`tempo_revert` true vs false, paired over the 993 songs: F −0.0066 ± 0.0033,
CMLt −0.0040 ± 0.0083, AMLt +0.0088 ± 0.0051 (mean paired delta, 95 % CI). The
mean-reverting kernel is **not** the better arm on F or CMLt at this budget; it wins
only on AMLt. That is a caveat on the commit message's framing, not on the number.

## 5. Seed spread

`tchain_s0` (seed 0) and `tchain_s1` (seed 1) are the same recipe, two seeds. Paired
over the identical 993 songs:

| metric (downbeat) | seed 0 | seed 1 | spread | paired 95 % CI on the difference |
|---|---|---|---|---|
| F | 0.7498 | 0.7533 | **0.0035** | ± 0.0022 |
| CMLt | 0.7011 | 0.7017 | **0.0006** | ± 0.0028 |
| AMLt | 0.8483 | 0.8484 | **0.0001** | ± 0.0030 |

**The seed-to-seed noise floor on the corpus mean is 0.0035 F and 0.0006 CMLt, n=993 —
not 0.16.** Audit item K's "~0.16 F within-run tail standard deviation" is a
*within-run, epoch-to-epoch* quantity (and its 0.30-on-locked-songs / |dF| median 0.076
figures are per-song or per-epoch), not the seed-to-seed variance of the corpus mean at
n=993. Averaging 993 songs is what buys the two orders of magnitude. Item K does not
bound the number under test, and should be re-worded so it stops appearing to.

**Is the +0.031 CMLt advantage resolvable? Yes, decisively.** The advantage is
+0.0294 (seed 0) / +0.0300 (seed 1) with a paired 95 % CI of ± 0.0121, against a
seed-to-seed spread of 0.0006 CMLt. That is roughly 50× the seed noise and about 2.5×
its own sampling CI. Caveat: this is two seeds, so it constrains the seed variance only
loosely; but the point estimate is stable to the fourth decimal across them, and the
comparison against Beat This is *paired within each run*, which removes the seed term
from the contrast almost entirely.

## 6. Matched Beat This comparison

### How it was made apples-to-apples

Beat This is scored **inside the same loop as tchain**, on the same batch, the same
deterministic 30 s crop, the same 993-song list, the same reference downbeat array, and
the same `f_measure` / `continuity_scores` implementations. Its downbeat times come from
`beat_this.model.postprocessor.Postprocessor(type="minimal", fps=50)` applied to the
`downbeat` head of the very frontend tchain reads features from — i.e. **Beat This's own
published system**, not a peak picker written here. There is only one harness, so the
"two harnesses landing close" failure mode cannot occur.

### Fold-honesty — established, not assumed

- The frontend checkpoint is **`final0`** (recorded in all four blobs). `final0` is the
  Beat This *final* model, not one of the 8 CV fold models, so no fold selection error
  is possible.
- GTZAN is a **test-only dataset in the Beat This protocol**: it is the only dataset in
  `dataset_store/beat_this_annotations/` with **no `8-folds.split` file**, so
  `load_catalog` assigns it fold `None`, and `split_songs` puts fold `None` in the test
  split and in neither train nor val.
- Measured this run: train 1850 / val 267 (fold 7) / test 999 songs, and the
  **(dataset, song_id) overlap between train ∪ val and the test set is 0**.
- Independent corroboration: my measured Beat This downbeat CMLt of **0.6717** sits on
  top of the published GTZAN downbeat CMLt of **0.673**, and my downbeat F of **0.7844**
  on top of the published **0.783** — on a 30 s crop rather than full songs. A
  contaminated (wrong-fold) checkpoint would score materially *higher* than published.
  Fold-honesty is therefore established; the comparison is **not** provisional.

### The comparison, downbeat-to-downbeat, n = 993

| | downbeat F | downbeat CMLt | downbeat AMLt |
|---|---|---|---|
| tchain (`tchain_true`) | 0.7470 | 0.7038 | 0.8489 |
| tchain (`tchain_s0`) | 0.7498 | 0.7011 | 0.8483 |
| tchain (`tchain_s1`) | 0.7533 | 0.7017 | 0.8484 |
| Beat This final0, minimal | 0.7844 | 0.6717 | 0.7910 |
| Beat This, **published**, full songs | 0.783 | 0.673 | 0.791 |

Paired per-song deltas (tchain − Beat This), mean ± 95 % CI, n = 993:

| run | ΔF | ΔCMLt | ΔAMLt |
|---|---|---|---|
| `tchain_s0` | **−0.0347 ± 0.0055** | **+0.0294 ± 0.0121** | **+0.0573 ± 0.0101** |
| `tchain_s1` | −0.0311 ± 0.0055 | +0.0300 ± 0.0123 | +0.0574 ± 0.0101 |
| `tchain_true` | −0.0374 | +0.0321 | +0.0579 |

All three deltas exclude zero comfortably. **The F-down / CMLt-up trade survives, and
AMLt-up is the largest and cleanest effect of the three** (+0.057, CI ±0.010) — the
paper should quote AMLt alongside CMLt, because continuity-at-allowed-metrical-levels is
where the latent tempo state pays off hardest.

## 7. Verdict

### **CONFIRMED** — with one mandatory relabelling.

| claimed | measured | source |
|---|---|---|
| F 0.747 | **downbeat F 0.7470** | `tchain_true/seed0.pt`, n = 993 |
| CMLt 0.704 | **downbeat CMLt 0.7038** | same |
| (not previously stated) | **downbeat AMLt 0.8489** | same |
| Beat This F 0.7844 | **downbeat F 0.7844** | final0 minimal, same 993 crops |
| ours 0.7498 (the recorded compare line) | **downbeat F 0.74978** | `tchain_s0/seed0.pt` |

The number reproduces exactly from a saved checkpoint against the code of the day it was
produced. n = 993, GTZAN fold `None`, zero overlap with the 2117 train+val songs.

**Mandatory relabelling:** the claim must be written as **downbeat F 0.747 / downbeat
CMLt 0.704**. Written unqualified as "F 0.747" it invites comparison against Beat This's
**beat** F of 0.891, which would be a 0.14-point error in the paper's favour-losing
direction and a category mistake either way. The correct opponent is Beat This's
**downbeat** row: F 0.783 / CMLt 0.673 / AMLt 0.791.

**Resolved side-issues:**
- "Checkpoints do not reproduce" — **not true of tchain.** All four surviving tchain
  checkpoints loaded `strict=True`, model and frontend, and re-scored to 4 dp against
  the record. Whatever the 60-epoch non-reproduction was, it is not a tchain problem and
  should be re-scoped in the audit.
- "Protocols may not be identical" — **they are not** (tchain 15–20 epochs / bs 8–16 vs
  archain stride-8 / class-emission / 24 epochs), so archain-vs-tchain rows must not be
  put in one table without a re-run. **This does not touch the Beat This comparison**,
  which is computed in the same forward pass as tchain and needs no protocol matching.
- Single-seed concern — retired by the s0/s1 pair (§5).

**Remaining caveats, stated plainly:**
1. Two seeds only. The seed variance is estimated from a sample of 2.
2. `tempo_revert=true` (the arm the commit argues for) is the *worse* arm on F and CMLt
   here (§4) and wins only on AMLt. The commit message's causal story about
   mean-reversion is not supported by these four checkpoints.
3. All numbers are on a **30 s middle crop**, not full songs; published Beat This
   figures are full-song. The Beat This numbers agreeing to ~0.001 suggests the crop
   costs little, but it is not the published protocol.
4. `tchain/` and `tchain_full/` are empty directories — two runs whose checkpoints are
   gone. Nothing downstream should cite a `tchain_full` number.

## 8. What a beat read-out would take

Assessment only — nothing was implemented, and this would require retraining, which was
out of scope.

**Yes, the natural move is exactly the madmom bar-pointer factorisation: make the phase
wrap per BEAT, and add a discrete beat-in-bar counter c ∈ {0 … B−1} that increments on
each wrap, with downbeats read off as the frames where c wraps to 0.** The current code
is unusually close to this because the machinery it needs is already there in a
different role.

What it touches, concretely:

1. **`vbpm/nets.py::PriorModel`** already carries a second discrete state alongside
   phase and already implements "this factor may only change at a *wrap*" — see
   `k_wrap` / `k_stay` and `fwd()`, where the rate index may switch only through a
   wrapping phase transition. Today that second state is the **rate**, not a counter.
   A meter counter is the same code shape: split the kernel at the wrap, and on the wrap
   apply a deterministic (or near-deterministic) `c → (c+1) mod B` permutation instead of
   the wrapped-Cauchy `switch` matrix. Cost stays O(C·N²)-ish; the state becomes
   (phase, rate, counter), so memory grows by ×B.
2. **Rate grid**, `vbpm/configs/tchain.yaml` `tempo_lo/tempo_hi` = 0.030/0.180 rad/frame,
   and `tempo_prior_mu = -2.6827`, `tempo_prior_sigma = 0.3903`. These are **bar**-rate
   constants fitted on bar periods (0.70–4.19 s). A beat-phase chain needs the beat-rate
   grid (roughly 0.084–0.50 rad/frame for 0.25–1.5 s beats) and a re-fitted log-normal
   prior. These are fitted numbers, not knobs — re-derive them from the annotations, do
   not divide the bar constants by 4.
3. **`vbpm/data/excerpts.py::_targets`** already computes `cls` (0 non-beat / 1 beat /
   2 downbeat) from `beat_times` *and* `downbeat_times`; only `y` (downbeat-only) is
   consumed today. The beat target already exists — the emission would read `cls`, and
   the natural emission is per-counter-value: p(y_t | phi_t, c_t), i.e. the beat window
   of `EmissionModel` gated by whether c = 0.
4. **`vbpm/readout.py`** gains a second function: beats = 2π crossings of the (now beat)
   phase; downbeats = the subset of those crossings where the decoded counter wraps.
   `downbeat_times` itself is reusable verbatim for the beat read-out.
5. **`vbpm/scoring/evaluation.py`** would need a beat row beside the downbeat row
   (`crop["beat_times"]` is already in the collated batch as `raw` beat times via
   `_targets`), and the printed header must stop saying "downbeat F" unconditionally.
6. **The thing that will actually bite:** commit `29e7c97` deleted the tied and class
   emissions specifically because they "read a beat-in-bar structure at a fixed
   subdivision" — i.e. a hardcoded meter. Re-introducing a counter re-introduces
   exactly that objection, and the honest answer is that B must be either inferred
   (a per-song latent, marginalised, which is what madmom does with parallel 3/4 and 4/4
   state spaces) or explicitly declared a modelling assumption. Reviving the deleted code
   is not enough; the reason it was deleted has to be answered.

Effort estimate: the prior/counter change is small (tens of lines in `PriorModel`), the
constants re-fit is a data job, and everything after it is a full retrain plus a fresh
corpus run. Nothing about it is inference-only.

---

## Appendix: reproduction recipe

```
git -C /home/sogang/jaehoon/VBPM archive 87a9182 | tar -x -C /tmp/vbpm87
ln -s /home/sogang/jaehoon/VBPM/dataset_store /tmp/vbpm87/dataset_store
cd /tmp/vbpm87 && PYTHONPATH=/tmp/vbpm87 /disk4/anaconda3/envs/vbpm/bin/python \
  /tmp/rescore_tchain.py --ckpt /disk4/jaehoon/vbpm_ckpt/tchain_true/seed0.pt \
  --tag tchain_true --gpu 1
```

Script: `/tmp/rescore_tchain.py`. Per-song outputs (993 rows each, both systems):
`/tmp/rescore/tchain_{true,false,s0,s1}.csv`; summaries in the matching `.json`;
stdout in the matching `.log`. Env `/disk4/anaconda3/envs/vbpm` (torch 2.6.0+cu124),
GPU 1. `beat_this` resolved from that env's editable install of
`/home/sogang/jaehoon/VBPM/external/beat_this` (the submodule is empty in `git archive`,
so it is the installed package, not the archived path).

One config key, `phase_nu`, is absent from the `tchain_true` / `tchain_false` blobs
(they predate it) and was filled from the variant's `DEFAULTS` value 4.0. It is inert
here: `phase_kernel = "vonmises"` in every blob, and `phase_nu` is read only by the
Student-t branch.
