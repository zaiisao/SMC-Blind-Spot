# Adversarial review: `scripts/experiments/exp_obs_lambda.py`

Reviewed while running; the job **completed during the review** (24/24 cells, 5208 = 24x217 JSONL
rows, validation PASS). Nothing was modified or killed. All checks below were done by execution
with `/home/sogang/mnt/db_2/anaconda3/envs/analyze-smc/bin/python` from `/home/sogang/jaehoon/Analyze-SMC`.

**Overall verdict: SAFE TO TRUST (results valid, mechanism claim confirmed) — with two
presentation-level SERIOUS issues in the per-track oracle block that must be fixed before any of
those numbers go into the paper.** No FATAL bug. No restart needed.

---

## 1. Is `dbn_with(obs=16, lam=100)` behaviourally identical to `_harness.make_dbn(transition_lambda=100)`?

**Ran** (`/tmp/adv1.py`): constructed both objects and compared a full structural descriptor for
each of the two HMMs (`beats_per_bar=[3,4]`): `state_space.num_states`, `num_beats`,
`sum(state_positions)`, `transition_model.transition_lambda`, the full
`observation_model.pointers` array (which is exactly what `observation_lambda` controls), plus
processor-level `threshold`, `correct`, `fps`, `beats_per_bar`.

**Observed:** `IDENTICAL_CONSTRUCTION: True`. The descriptor is discriminative — `dbn_with(6,100)`
and `dbn_with(16,5)` both differ from it (`SENSITIVE: True` for both).

Argument-by-argument: madmom's signature is
`(beats_per_bar, min_bpm=55.0, max_bpm=215.0, num_tempi=60, transition_lambda=100,
observation_lambda=16, threshold=0.05, correct=True, fps=None)`.
`make_dbn` passes `beats_per_bar=[3,4], min_bpm=30.0, max_bpm=215.0, fps=50, transition_lambda=100`
and leaves `num_tempi=60, observation_lambda=16, threshold=0.05, correct=True` at default.
`dbn_with` passes the identical five and explicitly sets `observation_lambda`, whose default is
**16** — so `obs=16` reproduces the harness exactly. `make_dbn`'s clamps
(`max(30, min_bpm)`, `min(300, max_bpm)`, `max_bpm > min_bpm+5`) are all no-ops at (30, 215).

**Verdict: CORRECT.** The validation cell is genuinely comparable to the paper's DBN rows.

## 2. Is the activation preprocessing byte-identical to `_harness.dbn_beats()`?

**Ran:** for `smc_001/002/003`, `run_cell((16,100,[t]))` vs
`ev(load_gt(t), dbn_beats(bthis_beat_prob(t), bthis_downbeat_prob(t), make_dbn(transition_lambda=100)))`.

**Observed:** dicts compare `True` on all 5 metrics for all 3 tracks
(F 0.410714…, 0.336842…, 0.436975…). The inline code is a character-level copy of `dbn_beats`
(same `eps=1e-5`, same `p*(1-eps)+eps/2`, same `np.maximum(bp-dp, eps/2)`, same `[:,0]`).

Also checked the empty-output path: madmom returns `np.empty((0,2))` when everything is below
`threshold`, so `[:, 0]` is safe and `ev` returns zeros — same as the harness.

**Verdict: CORRECT (duplicated code, not divergent code).** MINOR: it should call `dbn_beats`.

## 3. `ProcessPoolExecutor` + madmom; state leakage; oversubscription

**Ran:** `ps -eo pid,ppid,nlwp,...` on the live job.

**Observed:** 1 parent + 6 children, `nlwp=1` for every child — madmom's own
`num_threads` defaults to 1 (`self.map = map`, no internal `mp.Pool`), and no BLAS threads spawn.
So 6 workers = 6 threads, not 6xN. The box's load average was 441 on 64 cores from *other* jobs;
that costs wall time, not correctness. The job completed cleanly (exit 0, all 24 cells present).

State leakage: `run_cell` calls `dbn_with(obs, lam)` as its first statement, so the processor —
including the `observation_lambda`-dependent `pointers` — is rebuilt per cell. Nothing is module-level
or cached; the only shared objects are the read-only `.npz` files. `pool.map` with 24 tasks / 6
workers gives chunksize 1. Confirmed empirically: the 24 logged rows are all distinct
(e.g. obs=2 CMLt 0.4120 vs obs=16 CMLt 0.5102), which is impossible under leakage.

**Verdict: CORRECT.**

## 4. `cells` construction

**Ran:** re-executed the two list comprehensions and de-dup.

**Observed: exactly 24 cells, no duplicates, none dropped:**
```
(2,100) (4,5) (4,20) (4,50) (4,100) (4,200) (6,5) (6,20) (6,50) (6,100) (6,200)
(8,5) (8,20) (8,50) (8,100) (8,200) (12,100) (16,5) (16,20) (16,50) (16,100) (16,200)
(24,100) (32,100)
```
`(16,100)` present: True. All 8 `OBS_GRID` values present at lam=100: True. Duplicates: False.
The `if not (l == 100 and o in OBS_GRID)` guard is what prevents the double-add, and
`dict.fromkeys` is a redundant belt-and-braces. Matches the 24 rows actually logged.

**Verdict: CORRECT. n_cells = 24.**

## 5. `gt_bpm` and tempo bands

**Ran:** `gt_bpm(load_gt(t))` for all 217 tracks; `Counter(band_of(...))`.

**Observed:** `gt_bpm` = `60/median(diff(ref))`, i.e. BPM (not interval, not ratio).
- median BPM = **70.9220**, min 32.708, max 206.897
- **n < 55 BPM = 45**, n < 60 BPM = **61** — both match the known SMC distribution exactly.
- Bands: `0-55: 45, 55-70: 58, 70-90: 52, 90-120: 44, 120-999: 18`. Sum = **217**. Zero `"unk"`.

**Verdict: CORRECT.**

## 6. The per-track oracle — TWO SERIOUS ISSUES

Computed from the finished `/tmp/overnight/obs_lambda_sweep.per_track.jsonl` (217 tracks x 8 obs at
lam=100).

**(a) SERIOUS — independent max per metric, not disclosed.** The script's reported oracle is
`max` over obs taken separately for F, CMLt, AMLt. The honest "one obs per track, chosen by F" row is
materially lower:

| row | F | CMLt | AMLt |
|---|---|---|---|
| script's oracle (independent max per metric) | 0.6461 | **0.5995** | **0.7321** |
| consistent oracle (obs chosen by F per track) | 0.6461 | **0.5754** | **0.6823** |
| best fixed obs (obs=8) | 0.5961 | 0.5177 | 0.6491 |

The independent max inflates **CMLt by +0.0241 and AMLt by +0.0498** — the AMLt inflation is larger
than the entire genuine oracle-over-fixed AMLt gain (0.6823 - 0.6491 = 0.0332). Nothing in the
docstring, the printed line, or the JSON key `per_track_oracle_obs_at_lam100` says the metrics are
maximised independently. Fix: report the consistent row, or label the other one
"per-metric upper bound".

**(b) SERIOUS — the oracle histogram is a tie-breaking artifact.** `max(obs_cells, key=...)` returns
the *first* maximiser, and `obs_cells` is in ascending `OBS_GRID` order, so every tie goes to the
smallest obs. **110 of 217 tracks (50.7%) have a tied F-maximum.**

```
script's histogram:  {2:74, 4:36, 8:24, 6:20, 16:20, 32:17, 24:14, 12:12}
tie-aware:           {TIE:110, 2:32, 32:17, 4:14, 8:12, 6:9, 16:9, 24:7, 12:7}
```
obs=2's apparent 74-track plurality is really 32 unique wins plus 42 ties it won by array order.
Any claim of the form "a third of tracks prefer obs=2" is false.

**(c) Winner's curse.** The DBN is deterministic, so there is no measurement noise to over-fit and a
within-track permutation null is degenerate (it leaves the per-track max unchanged, verified:
null gain = observed gain = 0.0560 exactly, sd 0.0000). The oracle is not statistically biased —
it is simply *unachievable*, since selecting obs per track requires the ground truth. The honest
headline is the consistent row: **F 0.6461 vs 0.5961 at the best fixed obs, +0.0500**. It bounds an
adaptive observation model; it is not an attainable score. This should be said explicitly.

**(d) MINOR — mislabelled variable.** `orc_lam` holds *obs* values, not lambdas (the JSON key
`oracle_obs_choice` is right; only the local name is wrong). Same class of error as the documented
mislabelled-print bug, but here it is cosmetic.

## 7. `summarise()`

**Ran:** band counts (check 5) plus a recount of the finished JSONL.

**Observed:** `"all"` is `np.mean` over `ids` = all **217** tracks, no filtering. `by[bands[t]]`
partitions the same 217 with no `"unk"` bucket, and the band `n` values sum to 217. No track is
dropped or double-counted.

MINOR: the per-band dicts carry only F/CMLt/AMLt (no CMLc/AMLc) while `"all"` carries all five —
asymmetric but harmless. MINOR: `summarise` is recomputed ~9x at the end for printing; wasteful only.

**Verdict: CORRECT.**

## 8. Other findings

**Validation tolerance (SERIOUS as a *check*, though it happened to pass exactly).** `abs(F - 0.5851)
< 0.003` is far too loose to be diagnostic, because **F is nearly flat in obs**: obs=2 gives F
0.5851, obs=12 gives 0.5844, obs=16 gives 0.5851 — all inside the tolerance while CMLt swings from
0.4120 to 0.5102. A pipeline that silently ran the wrong obs would have passed. The assert should
have been on CMLt (or on both). In the event the observed run matched to 4 dp on both
(F 0.5851 / CMLt 0.5102 / AMLt 0.6735), so the outcome is fine — the *check* is weak, not the result.

**A near-collision worth flagging so nobody mistakes it for a bug:** obs=2 and obs=16 both print
F 0.5851. They are *not* the same numbers — per-track F differs on **171/217** tracks, mean
difference -0.000053. Coincidence at 4 dp only.

**λ=2 boundary.** madmom's no-beat log-density is `log((1 - sum(obs)) / (observation_lambda - 1))`.
At obs=2 the divisor is 1, i.e. no normalisation; at obs=1 it would divide by zero. `OBS_GRID`
starts at 2, so it is safe, but obs=2 sits exactly on the degenerate edge of the model and its
numbers should not be over-read.

**No crash-class issue found:** empty-DBN output, missing-GT uppercase fallback, and JSONL flush
ordering are all handled; the run finished at exit 0 with the full 5208 rows.

---

## Mechanism claim — CONFIRMED

`/home/sogang/jaehoon/madmom/madmom/features/beats_hmm.py`,
`RNNDownBeatTrackingObservationModel.__init__` (lines 598-609):
```python
border = 1. / observation_lambda
pointers[state_space.state_positions % 1 < border] = 1   # BEAT
pointers[state_space.state_positions < border] = 2       # DOWNBEAT
```
and `vbpm/rungs_lab/rungs/bar_pointer/state_space.py:92-95` is the identical logic
(`beat_region_width = 1.0 / observation_lambda`, then the same two masks with the comment
"overwrites BEAT at position ~0").

(a) **Confirmed** — `observation_lambda` sets the beat-region width as `1/λ` of a beat.
`state_positions` are in beat units within the bar, so `% 1` is the within-beat fraction.
(b) **Confirmed** — the DOWNBEAT assignment is the second, unmasked write, so it overwrites BEAT
for positions in `[0, border)` of the bar.
(c) **Confirmed and sound** — the region is a fraction of the beat period, so its width in
milliseconds is `60000/(BPM·λ)`. At λ=16: 78.1 ms at 48 BPM, 31.2 ms at 120 BPM. Against a fixed
±70 ms F-measure window the mis-specification is genuinely tempo-dependent, exactly as the docstring
argues. The observed sweep supports it: F peaks at obs=8 (0.5961) not 16 (0.5851), and the
obs x lam grid shows the two interact (at lam=5 the F optimum moves to obs=8 at 0.6054; at lam=200
larger obs is better for CMLt/AMLt) — so the paper's transition-lambda conclusions, all drawn at
obs=16, were indeed conditioned on an unexamined parameter.

**The interpretation is NOT wrong.** The one caveat is that the obs effect on **F** is small
(0.5646-0.6054 across the whole 24-cell grid) while the effect on **CMLt/AMLt** is large
(CMLt 0.4120-0.5177 at lam=100) — the story is a continuity/metrical-level story, not an F story,
and should be written that way.
