# Late-breaking facts the claim-audit workflow does not know

Established 2026-08-20 **after** the audit workflow (`smc-claim-audit-v2`) launched, so its
agents could not use them. **Merge these into `CLAIM_AUDIT.md` when it lands.** Two of them
change conclusions the workflow reached.

---

## 1. The Analyze-SMC SMC pipeline is ALREADY FOLD-HONEST

`SMC_Beat_Tracking_Analysis.ipynb` builds `beat_this_activations_cache/` (217 tracks) by
*"Run beat_this on all 217 SMC tracks using 8-fold cross-validation checkpoints"*, reading
`beat_this_annotations/smc/8-folds.split` and instantiating `Audio2Frames` per fold.
`scripts/run_folds.py` states the contract directly: *"for each fold N, run the held-out songs
using the fold{N} checkpoint so every song is predicted by a model that never saw it during
training."* `scripts/exp_tta_full_worker.py` reuses the same pattern
(`Audio2Frames(checkpoint_path=f"fold{fold}")`).

**Consequences.**
- **F = 0.6270 is a fold-honest reproduction of Beat This's published SMC 62.7**, not a
  `final0` number. Everything reading `scripts/_harness.py` inherits this: the fixed-λ=100 row
  (0.5827), the oracle-λ row (0.6495), the TTA tables, and `results_smc3.json`.
- **The `final0` contamination warning does NOT apply to any Analyze-SMC SMC number.** It
  applies only to (a) the KVAE's SMC F 0.654038, and (b) `tchain`'s GTZAN comparison against
  Beat This `final0`. Any workflow output that flags Analyze-SMC SMC rows as contaminated is
  wrong and must be corrected.
- Therefore the **+0.0668 F oracle-λ-over-hand-set-λ=100 measurement is uncontaminated**, same
  harness, same activations, fold-honest on both rows.

## 2. The FULL Beat This training corpus is on disk, preprocessed

`/disk1/jaehoon/beat_this_spect/done/` — **163 GB, 16 precomputed spectrogram archives**:
`asap, ballroom, beatles, candombe, filosax, groove_midi, gtzan, guitarset, hainsworth,
harmonix, hjdb, jaah, rwc, simac, smc, tapcorrect`.

Annotations at `/disk1/jaehoon/dataset_store/beat_this_annotations/`, and **every dataset ships
its own `8-folds.split`**, SMC included — so per-fold checkpoint honesty extends to all of them
with no new machinery, only more inference.

Counts verified: gtzan 999, harmonix 911, ballroom 685, **asap 473**, hjdb 235, rwc 226,
hainsworth 222, smc 217, beatles 180 `.beats` files.

**This removes the data-preparation blocker on every retry design in the audit.** Any
"REQUIRES-RERUN" item costed on the assumption that data must be prepared should be re-costed.

## 3. `asap` is the corpus that can actually discriminate — and the workflow does not mention it

The e2e CRF negative was worthless because it ran on 60 ballroom tracks where an **oracle
constant-period metronome scores F 0.9843 / CMLt 0.9597**, above the winning arm. There was no
tempo variation for a latent-tempo model to exploit.

Recorded full-song grid ceilings on the other corpora: **asap 0.164**, rwc 0.258. A
constant-tempo baseline is nearly useless on asap — exactly the discriminating regime the e2e
forensic said was missing.

| corpus | n | metronome ceiling | role |
|---|---|---|---|
| ballroom / hjdb | 685 / 235 | 0.984 / 1.000 | no-harm check ONLY — cannot discriminate |
| gtzan | 999 | — | standard comparison; Beat This's own headline corpus |
| hainsworth / beatles | 222 / 180 | 0.894 / 0.715 | mildly discriminating |
| **asap** | **473** | **0.164** | **where the thesis can be proven, with power** |
| **smc** | **217** | — | the target: beat-only, hardest, no downbeats |

asap is >2x SMC's size, expressive piano, **and has downbeat annotations** (SMC does not). So
the mechanism can be demonstrated there at full statistical power and then shown to transfer to
SMC — which alone cannot resolve a +0.02 effect confidently.

## 4. Baseline choice is a live decision, not a detail

The "+0.05" figure is real but is measured **against the DBN at λ=100**, not against the state
of the art:

| comparator | oracle-λ delta |
|---|---|
| fixed λ=100, paper grid (0.592 → 0.642) | **+0.050** |
| fixed λ=100, wide grid (0.5827 → 0.6495) | **+0.0668** |
| Beat This peak-pick (0.6270 → 0.6495) | **+0.0225** |

Both are defensible; they answer different questions. The paper must show **both rows in the
same table** or a reviewer will read it as improving a self-chosen strawman. Recommended
framing: the DBN at shipped settings loses to peak-picking on SMC (0.5827 vs 0.6270) — that is
what drove the field away from it — and a per-song λ recovers +0.0668 F, +0.0663 CMLt and
+0.0484 AMLt, overtaking peak-picking on all three. The decoder was never the problem; the
**homogeneous scale** was.

Standing caveat: all oracle. Best label-free selector = 0.6021, *below* peak-picking. The one
deployable data point is the EM-learned λ transferred to SMC: **+0.0211 F / +0.0115 CMLt** over
hand-set — roughly a third of the oracle budget, with no labels.
