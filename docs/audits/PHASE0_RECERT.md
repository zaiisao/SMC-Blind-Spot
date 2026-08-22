# Phase 0 — re-certification of the VBPM machinery inside Analyze-SMC

Run 2026-08-20 against `/home/sogang/jaehoon/Analyze-SMC/vbpm/rungs_lab` after the move
from `/home/sogang/jaehoon/VBPM/rungs_lab`. Envs: rungs_lab code =
`/disk4/anaconda3/envs/vbpm/bin/python` (the retired `chart` env is still referenced by one
dead script, see §1.3); Analyze-SMC harness =
`/home/sogang/mnt/db_2/anaconda3/envs/analyze-smc/bin/python`. Both envs carry
madmom 0.17.dev0, so the rungs_lab scripts that import `scripts/_harness.py` run fine
under the vbpm env.

**Headline: the move is sound, but the handover's numbers table is not. §3 shows that
`R0_shipped = 0.6266` and `Beat This peak-pick = 0.6270` are measurements of two
different neural frontends at two different frame rates with two different decoders.
They agree by coincidence.**

---

## 1. Move integrity

### 1.1 Symlinks — all three resolve

| link | target | status |
|---|---|---|
| `rungs_lab/external` | `/home/sogang/jaehoon/VBPM/external` | RESOLVES |
| `rungs_lab/dataset_store` | `/home/sogang/jaehoon/VBPM/dataset_store` | RESOLVES (→ `/disk1/jaehoon/dataset_store`) |
| `rungs_lab/cache/bt_demix` | `/disk4/jaehoon/VBPM_cache/bt_demix` | RESOLVES, 217 SMC shards present |

`/home/sogang/jaehoon/VBPM/` still exists and still holds `external/`, `dataset_store/`,
`checkpoints/`, `vbpm/`, `environment.yml`, `pyproject.toml`. **`VBPM/rungs_lab/` does
not exist any more** — the move was a move, not a copy. That is what makes the hardcoded
paths below hard failures rather than silent stale-code reads (the more dangerous
outcome, which we do not have).

Other assets verified present under the new root: `checkpoints/bt_fold{0..7}_repacked.pt`
(8 files), `runs/r2mix_seed0.pt`, `runs/r3_{wt,lamt}_seed0.pt`, `mert/runs/*.pt`,
`r4/runs/*.pt`, all `results_*.json`, `tests/` (6 files + `v2/`),
`experiments/bt_e2e/RESULTS.md` (304 lines).
Annotations: `rungs_lab/dataset_store/beat_this_annotations/smc/annotations/beats`
is **byte-identical** to `Analyze-SMC/beat_this_annotations/smc/annotations/beats`
(217 files, `diff -rq` clean), so the two projects grade against the same ground truth.

### 1.2 Hardcoded paths found and FIXED (unambiguous)

`grep -rn "/home/sogang/jaehoon/VBPM"` over `*.py`/`*.sh` found **50 occurrences in 27
files**. All of the following pointed into `VBPM/rungs_lab`, which no longer exists, so
every one was a hard `FileNotFoundError` / `ImportError`; all were rewritten to
`/home/sogang/jaehoon/Analyze-SMC/vbpm/rungs_lab`:

- `smc3_certify.py:8` — `sys.path.insert` for the `rungs` package
- `mert/audit/` — 21 files: `_base.py` (`HERE`, `LAB`), `oracle.py`, `w.py`, `balarm.py`,
  `octave_all.py`, `octave_all_short.py`, `octave_graded.py`, `octave_paired.py`,
  `hazard_model.py`, `gr.py`, `cache_check.py` (incl. an output path), `binsweep.py`,
  `paired.py`, `v_pairedF.py`, `train_shortcrop.py`, `nll.py`, `abl.py`, `ablate.py`,
  `build_race.py` (incl. an `np.savez` output path), `se.py`, `oct_audit.py`,
  `race_confound.py`, `ballroom.py`
- `experiments/bt_e2e/run_mixture_chain.sh`, `run_r2_em_chain.sh` — `cd` target
  (`VBPM/experiments/bt_e2e`, a directory that never existed at that path) and
  `PYTHONPATH`
- `tests/test_inference.py:21`, `tests/test_structured_dp.py:16` — these read
  `sys.path.insert(0, "/home/sogang/jaehoon/VBPM")` and then
  `from rungs.bar_pointer.inference import ...`. `VBPM/` root still exists but has no
  `rungs` package (it lives in rungs_lab), so these were broken; repointed at the new
  rungs_lab root.
- `experiments/bt_e2e/smc/eval_smc.py:10` — `ROOT = Path("/home/sogang/jaehoon/VBPM")`,
  used for `rungs`, `experiments/bt_e2e`, and `cache/bt_demix`; all three now live under
  rungs_lab, so unambiguous.

### 1.3 Left alone (ambiguous or dead) — 1 occurrence

- `experiments/bt_e2e/launch.sh:3` — `cd /home/sogang/jaehoon/VBPM`. That directory
  still exists, so intent is ambiguous, but the script is **dead** regardless: it invokes
  `/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python`, and the `chart` env was
  retired on 2026-07-31 (VBPM memory `reference_vbpm_env.md`). It launches the 2026-07-18
  `train_bt.py` campaign, which is already concluded in `RESULTS.md`. Not fixed; flagged.

### 1.4 Relative-root assumptions (not broken by the move, but cwd-sensitive)

None of the `Path(__file__).resolve().parent[...]` roots break — `smc_data.py`
(`ROOT = parent`) and `training/frontend.py` (`ROOT = parent.parent`, which then appends
`external/beat_transformer/code`) both still land on rungs_lab, and the symlinks are
inside it. But the entry-point scripts use bare relative paths
(`torch.load("runs/r2mix_seed0.pt")`, `checkpoints/bt_fold{fold}_repacked.pt`,
`json.dump(..., open("results_smc.json","w"))`). **They must be run with cwd =
`vbpm/rungs_lab`.** This was true before the move too; it just matters more now that
the tree sits two levels deeper.

### 1.5 One pre-existing (move-unrelated) defect found while smoke-testing

`hygiene_check.py` is **flaky: it passed 1 of 4 runs.** The failure is real, not a move
artifact — the script builds its probe input with an *unseeded*
`torch.rand(400, 2) * 0.9 + 0.05`. Characterised over 20 fixed seeds:

| arm | seeds with non-finite kernel | total `nan` | total `inf` |
|---|---|---|---|
| `R3_wt` | **6 / 20** | 0 | 7140 |
| `R3_lamt` | 0 / 20 | 0 | 0 |

Only the `w_t` arm, and only `-inf` (never NaN). Mechanism is the one `RESULTS.md`
already documents for that arm ("net on `w_t` → saturated, med 0.93"): when the per-frame
head drives `w_t → 1`, the exponential component's weight `(1 - w_t) → 0`, `log(0) = -inf`,
and it no longer covers the dither component's structurally-forbidden `-inf` entries.
R3 is a retired rung so nothing standing depends on this — **but Phase-2 candidate 3
(heterogeneous / per-frame scale) would reuse exactly this mixture-weight head, so it
needs a clamp or a `log1p`-style parameterisation before it is trained.**

---

## 2. The core certification: R0 vs R1 on SMC

**Verdict: PASSES on today's code, on BOTH frontends, and the certificate is now stronger
than the handover claims — it covers downbeats too.**

| frontend / config | beats identical | downbeats identical | R0 F / CMLt / AMLt | R1 F / CMLt / AMLt |
|---|---|---|---|---|
| **Beat Transformer**, 43.066 fps, `bounding="none"`, min55 num_tempi=60 shipped (= the handover's certificate) | **216 / 217** | **217 / 217** | 0.6266 / 0.5369 / 0.7110 | 0.6266 / 0.5369 / 0.7110 |
| **Beat This**, 50 fps, `bounding="squeeze"`, min55 num_tempi=60 shipped | **217 / 217** | **217 / 217** | 0.5757 / 0.4738 / 0.6540 | 0.5757 / 0.4738 / 0.6540 |
| **Beat This**, 50 fps, `bounding="squeeze"`, **wide** min30 num_tempi=60 shipped | **216 / 217** | **216 / 217** | 0.5851 / 0.5102 / 0.6735 | 0.5851 / 0.5102 / 0.6735 |

(Beat This rows also: min55 CMLc 0.3728 / AMLc 0.4957; min30 CMLc 0.4162 / AMLc 0.5371 —
identical between R0 and R1 to all 16 printed digits in both rows.)

Cross-check, `smc3_parity.py` as shipped (first 60 SMC ids, Beat This, squeeze):
`min55_num60_shipped 60/60`, `min30_num60_shipped 59/60`, `min30_numNone_shipped 60/60`,
`min30_numNone_bare 60/60`; madmom F and ours F equal to 4 dp on every row
(0.6197 / 0.6380 / 0.6343 / 0.6203).

### 2.1 What "reproduces" is actually measured as

**Event identity of the emitted beat times, to `atol=1e-6` s, with a length check first.**
Not F, not the Viterbi state path. `smc_diagnose.py:62-66` is the certificate the handover
quotes, verbatim:

```python
if n == "R1_matched_shipped":
    total += 1
    if (len(ev["beats"]) == len(ev0["beats"])
            and np.allclose(ev["beats"], ev0["beats"], atol=1e-6)):
        identical += 1
```

1e-6 s is 1/23000 of a frame at 43 fps, so this is effectively "the same frame index, and
the same peak-snap inside it" — far tighter than the ±70 ms F tolerance. It is an
event-level, not a path-level, claim: two different state paths that read out the same
beat frames would pass. The nearest path-level evidence is not on SMC — it is the
`r1_2016_dbn.py` docstring's re-certification note (2026-07-17, 30 songs across
ballroom/beatles/gtzan/hainsworth/hjdb): *"event-identical to R0 30/30 shipped AND 30/30
bare; per-meter Viterbi path agreement 1.0 against madmom's own hmm.viterbi; scores match
to ~1e-4"*. `certify.py` runs that 30-song event-identity check (it does test downbeats:
`certify.py:39-41`), but on 5 non-SMC datasets. So: **path parity is certified off-SMC and
30 songs; event parity is certified on SMC and 217 songs.**

### 2.2 The one Beat Transformer mismatch — a one-frame tie-break, nothing worse

`smc_085` (fold 2). Both engines emit **37 beats and 9 downbeats**; the downbeats are
bit-identical. Exactly **4 of the 37 beats differ, each by exactly one frame**:
max |Δt| = 0.0232200 s, which is 1/43.06640625 s to the digit; mean |Δt| over all 37 beats
= 0.0025103 s (= 4 × one frame / 37). Scores are unchanged in every metric:
**F 0.40625 / CMLt 0.71875 / AMLt 0.71875 for both R0 and R1.** This is the score-tie
plateau the R1 docstring predicts (madmom accumulates observation densities in float32,
R1 in float64; equal-score paths shifted by one frame). It is a tie-break artifact, not a
model difference.

The wide-grid Beat This mismatch is the same species: `smc_044`, 49 beats and 13 downbeats
on both sides, 2 beats and 1 downbeat displaced by 0.060 s (3 frames at 50 fps), and
F / CMLt / AMLt identical at 0.8837 / 0.9767 / 0.9767. Aggregate scores unaffected.

### 2.3 Which frontend — and a correction to the previous state of the record

Both, now. **Before today the certificate existed on the Beat Transformer branch only**:
the only "N/217 event-identical" line recorded anywhere in the tree is
`smc_diag.log:4` ("216/217"), written by `smc_diagnose.py`, which per §3 is Beat
Transformer at 43.066 fps. `smc3_certify.py` (the Beat This branch) prints a parity count
but its output was never persisted, and `results_smc3.json` stores per-track scores only.
So the handover's load-bearing fact was, as of this morning, a Beat-Transformer-only fact.
The two Beat This rows above close that gap: **R1 is now certified faithful on the Beat
This activations the ICASSP chassis will actually be built on**, at the shipped grid
(217/217) and at the wide [30, 215] grid the SMC work uses (216/217).

**Caveat on `smc3_certify.py` — it under-reports parity, and its number should not be
quoted.** It constructs R1 with `bounding="none"` (`smc3_certify.py:24`) while feeding
madmom activations squeezed and decorrelated by `_harness.dbn_beats`, whose floor is
`eps/2 = 5e-6`. `rungs/base.py:145-147` gives `bounding="none"` a decorrelation floor of
**1e-12** instead — a different no-beat log-density wherever the beat channel falls below
the downbeat channel. Run that way on all 217 tracks (wide min30, shipped heuristics) the
count collapses to **beats 135/217, downbeats 146/217**, and R1's aggregate F falls to
0.5660 / CMLt 0.4805 / AMLt 0.6417 against R0's 0.5851 / 0.5102 / 0.6735 — a spurious
−0.0191 F. Identical counts in float32 and float64 (135/217 and 146/217 both times), so
this is a config mismatch, not precision. At min55 the same mismatch gives 122/217 beats.
Matching the bounding (the three rows in the table above) is what restores 217/217.
`smc3_certify.py` was left unmodified, per the audit constraint; this note is the fix.

### 2.4 Downbeats

**Yes — R1 emits downbeats, and they are now certified.** Both rungs return
`{"beats": ..., "downbeats": ...}`; R0 derives downbeats from madmom's bar-position column
(`r0_madmom_dbn.py:124-127`, `is_downbeat = beats_with_bar_positions[:, 1] == 1`) and R1
from the same bar-pointer position class. The shipped SMC certificate **did not test
them** — `smc_diagnose.py` compares `ev["beats"]` only (quoted in §2.1) — so the
handover's "216 of 217" was a beat-only claim. Adding the downbeat comparison gives
**217/217 on Beat Transformer**, **217/217 on Beat This min55**, **216/217 on Beat This
min30**. `certify.py` already tested downbeats but only on the 5 non-SMC datasets.

### 2.5 Reproduce

```bash
PY=/disk4/anaconda3/envs/vbpm/bin/python
cd /home/sogang/jaehoon/Analyze-SMC/vbpm/rungs_lab
CUDA_VISIBLE_DEVICES=1 $PY /tmp/recert_bt_transformer.py            # Beat Transformer row
CUDA_VISIBLE_DEVICES=2 $PY /tmp/recert_bt_this_squeeze.py 55 60 min55   # Beat This shipped
CUDA_VISIBLE_DEVICES=3 $PY /tmp/recert_bt_this_squeeze.py 30 60 min30   # Beat This wide
CUDA_VISIBLE_DEVICES=1 $PY smc3_parity.py                           # 60-song config bisect
```

No further hardcoded `/home/sogang/jaehoon/VBPM/rungs_lab` paths were hit; the 50 fixed in
§1.2 were sufficient for every entry point above.

---

## 3. Harness reconciliation — the verdict

### 3.1 The eval layer is identical. It is not the source of any difference.

`Analyze-SMC/scripts/_harness.py::ev` calls `mir_eval.beat.evaluate(ref, est)` with no
kwargs. `rungs_lab`'s `score()` (identical copy in `eval_smc.py`, `eval_smc2.py`,
`smc_diagnose.py`, `lambda_select.py`, `lambda_select2.py`) calls
`meb.trim_beats(...)` on both sequences and then `meb.f_measure` / `meb.continuity`.
Checked against the installed `mir_eval` signatures:

| knob | `_harness.ev` (via `evaluate`) | `rungs_lab.score` | same? |
|---|---|---|---|
| trim | `trim_beats(min_beat_time=5.0)` on ref **and** est | `trim_beats` default = 5.0 on ref and est | YES |
| F tolerance | `f_measure_threshold=0.07` (±70 ms) | default 0.07 | YES |
| CMLt / AMLt | `continuity(phase=0.175, period=0.175)` | defaults 0.175 / 0.175 | YES |
| CMLt index | `r["Correct Metric Level Total"]` | `continuity(...)[1]` | YES |
| AMLt index | `r["Any Metric Level Total"]` | `continuity(...)[3]` | YES |
| empty-input | returns 0.0 for all metrics | returns 0.0 for all metrics | YES |
| n | 217 (`track_ids()` = 217 `.npz` in the BT cache) | 217 (`load_smc()`; 217/217 stems have both a fold and a mel shard) | YES |

Both cover the same 217 tracks, verified by set intersection: `|cached ∩ live| = 217`.
So **trim, tolerance, metric definitions and track set are not where the two projects
differ.** The difference is entirely upstream of the metric.

### 3.2 Where they actually differ: the frontend is a different neural network

This is the finding that matters, and it contradicts the handover.

`rungs_lab/training/frontend.py` — the module `eval_smc.py`, `smc_diagnose.py`,
`lambda_select.py` and `lambda_select2.py` all get their activations from — has the
docstring *"Build/resume/freeze the **Beat Transformer** …"* and builds
`Demixed_DilatedTransformerModel` from
`external/beat_transformer/code/DilatedTransformer.py`. `training/data.py` sets
`FPS = 44100 / 1024 = 43.06640625`. The `bt_fold*_repacked.pt` checkpoints are
**Beat Transformer** folds, and `cache/bt_demix` is the Beat-Transformer demixed-mel cache.
(The "bt" prefix throughout that subtree means Beat Transformer, not Beat This — the
docstrings in `eval_smc.py` use "BT" for both, which is how the conflation happened.)

`Analyze-SMC/scripts/_harness.py` loads **Beat This** logits from
`beat_this_activations_cache/*.npz` at `FPS = 50`.

Measured directly (`/tmp/act_compare.py`, both activation sets for all 217 stems side by
side):

- length difference is a constant **+274 frames** (cached minus live) on every track —
  exactly the 50 vs 43.066 fps ratio on ~39.5 s clips
- mean `|cached − live|` beat probability: **mean 0.16310, median 0.15297, max 0.79241**
- tracks where the two activation arrays agree to 1e-6: **0 / 217**
- the *same* peak-picker (`_harness.peakpick`, threshold 0.5, width 7) on each:

| activations | F | CMLt | AMLt |
|---|---|---|---|
| Beat This cached (Analyze-SMC) | **0.6270** | 0.5138 | 0.6100 |
| Beat Transformer live (rungs_lab) | **0.5055** | 0.2974 | 0.3361 |

A **0.1215 F** gap between the two activation sets under one identical decoder and one
identical metric. These are not interchangeable inputs.

### 3.3 Are 0.6270 and 0.6266 the same measurement? — **No. Definitively no.**

| | Analyze-SMC 0.6270 | VBPM `results_smc.json` 0.6266 |
|---|---|---|
| frontend | Beat This, cached logits | Beat Transformer, live fold-honest inference |
| frame rate | 50 fps | 43.066 fps |
| decoder | max-pool-7 peak-pick @ 0.5 | madmom `DBNDownBeatTrackingProcessor`, shipped defaults |
| bounding | n/a | `"none"` |
| grid | n/a | madmom default `min_bpm=55`, `max_bpm=215` |
| eval | mir_eval, 5 s trim, ±70 ms | mir_eval, 5 s trim, ±70 ms (identical) |

They differ in the network, the sample rate and the decoder, and agree in the metric.
The 0.0004 agreement is a **coincidence**.

Three independent pieces of evidence, not inference:

1. **Direct measurement (§3.2).** Beat Transformer activations peak-picked give 0.5055,
   not 0.6270. Beat This activations are what give 0.6270.
2. **The cross-harness bridge already exists and gives different DBN numbers.**
   `smc3_ladder.py` and `smc3_certify.py` do `sys.path.insert(0,
   "/home/sogang/jaehoon/Analyze-SMC/scripts")` and import `_harness` directly — i.e.
   they run the rungs_lab engine on the Analyze-SMC Beat This activations under the
   Analyze-SMC eval. `smc3_ladder.log` reproduces **peak = 0.6270 exactly**, and on the
   *same* activations the DBN row is `fixed100 F 0.5827 CMLt 0.5091 AMLt 0.6718` — not
   0.6266. Under one harness, Beat This + DBN ≠ 0.6266.
3. **`smc_diagnose.py` never computes a peak-pick row at all.** The 0.6266 line in
   `smc_diag.log` is `R0_shipped`, a DBN row. There is no peak-pick measurement anywhere
   in the Beat Transformer branch of the codebase.

### 3.4 Corrections the handover needs

- The block headed "From **VBPM** `rungs_lab/results_smc*.json` (n=217)" lists
  `R0_shipped` 0.6266, `R1_obs6_shipped` 0.6389 and `R1_wide30_notempi` 0.6328 /
  0.5667 CMLt. **All three are Beat Transformer**, and they sit directly beneath a Beat
  This block, inviting exactly the merge the handover warns against. Label them.
- The `smc_001` line ("peak-pick 0.594 vs DBN fixed100 **0.411**") is from
  `results_smc3.json`, i.e. it *is* Beat This under the Analyze-SMC harness (exact values
  0.5952 / 0.4110, bpm 48.2) — that row is safe to merge, unlike the three above it.
  Note the same song's Beat Transformer `R0_shipped` is F 0.506 / CMLt 0.000 / AMLt 1.000.
- "max tempo change 1.57×" is a unit error: `max_tempo_change` is
  `max(local BPM) − min(local BPM)` over 8-beat windows, so it is **1.57 BPM**, not a
  ratio.
- **Rule for Phase 1 onward: only `results_smc3.json` (and anything else that imports
  `_harness`) is directly comparable to `docs/findings/FOLLOWUP_FINDINGS.md`.** `results_smc.json`,
  `results_smc_diagnosis.json` and `results_lamselect_smc.json` are a separate
  Beat-Transformer world and must be relabelled or re-run on Beat This before entering
  any shared table.

---

## 4. Phase-1 gate: is scale tuning exhausted? — **CONFIRMED, and more strongly than
Caution 2 states**

### 4.1 The two lamselect files answer different questions

`results_lamselect_smc.json` (from `lambda_select.py`) is **not** the file that supports
Caution 2's "0.642". It is Beat Transformer, mixture chassis (`w = 0.390`,
`observation_lambda = 6`), λ on a 16-point log grid **whose floor is 20**:

| row | F | CMLt | AMLt |
|---|---|---|---|
| fixed100 | 0.4790 | 0.5163 | 0.6659 |
| peak (find_peaks h=0.2) | 0.6852 | 0.5657 | 0.6054 |
| ML (label-free per-song λ) | 0.5014 | 0.5231 | 0.6615 |
| **ORACLE per-song λ** | **0.5274** | 0.5282 | 0.6827 |

**95 of 217 tracks (43.8 %) pin λ_oracle at the grid floor of 20.** The grid is censored
from below on nearly half the corpus, so 0.5274 is not a real ceiling — it is a
truncation artifact. Do not quote this file's oracle number.

The number that matches Caution 2 is `results_smc3.json` (Beat This, Analyze-SMC harness,
wide grid [30, 215], squeeze bounding, shipped heuristics, λ grid 1–500), n = 217:

| row | F | CMLt | AMLt |
|---|---|---|---|
| peak-pick @0.5 | 0.6270 | 0.5138 | 0.6100 |
| fixed λ=100 | 0.5827 | 0.5091 | 0.6718 |
| R2mix | 0.5878 | 0.5166 | 0.6680 |
| R3_wt | 0.5875 | 0.5174 | 0.6679 |
| R3_lamt | 0.6008 | 0.4960 | 0.5815 |
| ML (label-free per-song λ) | 0.6021 | 0.4886 | 0.5618 |
| **ORACLE per-song λ** | **0.6495** | **0.5801** | 0.6584 |

0.6495 sits between the paper's 0.642 and Analyze-SMC's recomputed 0.653 — same
measurement, minor config drift. Caution 2's premise holds.

### 4.2 The gate answer

**Scale tuning is exhausted in F.** A per-track *oracle* over 13 λ values buys
**+0.0225 F** over doing nothing at all (0.6495 vs peak-pick 0.6270). That is the entire
budget available to any conceivable λ-selection method, oracle or not — and the
best *label-free* selector in the repo (ML, `results_smc3.json`) lands at 0.6021, i.e.
**below** peak-picking.

The mechanism, which is the part that decides Phase 2. Distribution of λ_oracle over the
1–500 grid:

| λ | 1 | 2 | 5 | 10 | 20 | 30 | 50 | 75 | 100 | 150 | 200 | 300 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracks | **73** | 11 | 14 | 11 | 21 | 16 | 25 | 12 | 8 | 11 | 6 | 7 | 2 |

**73 of 217 tracks (33.6 %) pin λ_oracle at the grid floor of 1**; 84 (38.7 %) at λ ≤ 2.
Only 2 tracks are censored at the ceiling of 500. And on those 73 floor-censored tracks:

| subset | n | ORACLE-λ F / CMLt / AMLt | peak-pick F / CMLt / AMLt |
|---|---|---|---|
| λ_oracle == 1 | 73 | 0.6422 / 0.5267 / 0.6198 | **0.6518** / **0.5448** / **0.6482** |
| λ_oracle ≤ 2 | 84 | 0.6403 / 0.5129 / 0.6079 | **0.6498** / **0.5280** / **0.6449** |
| λ_oracle ≥ 100 | 34 | **0.7028** / **0.7848** / **0.8295** | 0.6179 / 0.6136 / 0.6707 |
| all | 217 | 0.6495 / 0.5801 / 0.6584 | 0.6270 / 0.5138 / 0.6100 |

**On the third of the corpus that wants the loosest λ, the DBN at its loosest still loses
to peak-picking on all three metrics** (−0.0096 F, −0.0181 CMLt, −0.0284 AMLt). Those
tracks are asking for the transition prior to be switched *off*, and λ → 0 is a limit the
exponential family approaches but cannot usefully pass — a uniform kernel leaves only the
legal-path topology, which is still more constraint than these tracks tolerate.
**Extending the λ grid downward therefore cannot help.** Caution 2 is confirmed: the win
must come from the kernel's shape.

Two further results that sharpen which shape:

- **The corpus is bimodal in λ, not merely mistuned.** 34 tracks (15.7 %) want λ ≥ 100
  and are *hugely* rewarded for it: 0.7028 F / 0.7848 CMLt / 0.8295 AMLt versus
  peak-picking's 0.6179 / 0.6136 / 0.6707. A third of SMC wants no prior; a sixth wants
  a very tight one. No single global λ can serve both, which is the heterogeneity
  argument, measured.
- **You cannot route on observable tempo statistics.** Spearman(λ_oracle, median BPM)
  ρ = 0.019, p = 0.786; Spearman(λ_oracle, max tempo change) ρ = 0.098, p = 0.15.
  Both null. So "estimate the tempo, pick λ" is closed off, and the repo's existing
  label-free selector (marginal likelihood) is worse than null — VBPM already measured
  Spearman(log λ_ML, log λ_oracle) ρ = **−0.230**, p = 6e-4, i.e. it points the *wrong
  way*. A per-song λ *selector* is therefore not the Phase-2 candidate.

### 4.3 The one place scale tuning is *not* exhausted: continuity

Reporting F alone would overstate the case, in the direction Caution 5 warns about. The
same oracle-λ sweep is worth **+0.0225 F but +0.0663 CMLt and +0.0484 AMLt** over
peak-picking. Three times the effect in continuity. The decoder is doing real work on
metrical coherence even where it costs beat placement — consistent with the paper's
"two independent problems" framing, and it means a Phase-2 kernel should be graded on
CMLt/AMLt first and F second.

For reference, the per-track envelope `max(peak-pick, oracle-λ)` on `results_smc3.json`
is **F 0.666 / CMLt 0.6154** — oracle-λ beats peak-pick on 114 / 217 tracks, peak-pick
beats oracle-λ on 67 (36 ties). Same qualitative structure as Blind Spot 1 in
`docs/findings/FOLLOWUP_FINDINGS.md` (which measured 0.691 on a differently-configured DBN family and
78 / 217 DBN wins).

### 4.4 Phase-2 recommendation

Ranked by what the data above supports:

1. **Heterogeneous scale (Caution 2's option 3)** — best supported. The λ_oracle
   distribution is bimodal (73 at the floor, 34 at ≥ 100), the modes are not predictable
   from tempo, and the tight-λ mode is worth +0.085 F / +0.171 CMLt / +0.159 AMLt over
   peak-picking when it is served. A mixture over scale marginalised exactly does not
   need to *select* the mode, which is the operation that has repeatedly failed here.
2. **The dither-corrected mixture kernel** — untested on SMC and cheap, but note it adds
   ±1-frame *placement* flexibility (RESULTS.md §"MIXTURE'S BARE EDGE IS ALSO PLACEMENT"),
   and it was a dead tie under deployment decode on the 146-song fold. Since
   `results_smc3.json` already runs deployed decode with peak-snap, the redundancy of
   Caution 4 should be expected here. Run it — it is nearly free — but do not build the
   week on it.
3. **A heavy-tailed kernel** — a heavier tail lets one global λ cover both modes, which
   is the right *shape* intuition, but it is strictly weaker than (1) because it is still
   a single homogeneous kernel and 33.6 % of tracks want no kernel at all.

---

## 5. Reproduce

```bash
PY=/disk4/anaconda3/envs/vbpm/bin/python
cd /home/sogang/jaehoon/Analyze-SMC/vbpm/rungs_lab   # cwd matters, see §1.4

$PY smc_diagnose.py     # §2, Beat Transformer branch: R0 vs R1 certificate + config sweep
$PY smc3_certify.py     # §2, Beat This branch: engine parity vs madmom under _harness
$PY hygiene_check.py    # §1.5 — flaky, run it several times
$PY /tmp/act_compare.py # §3.2 — the two activation sets side by side
```
