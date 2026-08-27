# Loss work: decision tree

All readouts are the **failure mode**, not F. F is reported but never gates a
decision, because both arms train on ~43% of upstream's corpus and cannot make
an absolute claim. The quantities that gate:

    doubled = frac of tracks <55 bpm with est/true beat-count ratio > 1.5
    halved  = frac of tracks >100 bpm with est/true beat-count ratio < 0.7

Baseline to beat (fold-honest released model, SMC n=217):
    doubled 0.356    halved 0.316

Pooled over 12 datasets / 1,501 tracks the same effect is monotonic:
    <55 doubled 0.644 -> >140 halved 0.163, minimum at 100-140 bpm.

---

## TEST 1 — pos_weight sign sweep (pw 8 / 32 / 128, 25 epochs)  ~01:40

Identical data, seed, epochs. ONLY the global beat pos_weight varies. Brackets
the per-item range (10 at 180 bpm .. 68 at 40 bpm), so the slope says whether
per-item reweighting pushes the right way.

### 1A. doubling and halving both move < 0.05 across the 16x range
**The prior is not in the loss normalisation.** It lives in the learned
representation, absorbed from the training tempo distribution -- consistent with
the neutral point tracking corpus beat density (~78 bpm ours, ~120 bpm theirs).
-> KILL the per-item arms (they cannot work; free the GPUs).
-> GO TO TEST 3 (beat-count loss). This is the branch I expect.

### 1B. higher pos_weight -> MORE doubling (slope > +0.05)
**Per-item is wrong-signed.** It raises the weight to 68 on exactly the slow
tracks that already double, and drops it to 10 on the fast tracks that already
halve. It should make both errors worse.
-> KILL the per-item arms.
-> GO TO TEST 3. Optionally also test INVERTED per-item (weight ∝ beat density)
   as a cheap confirmation that the sign is the whole story.

### 1C. higher pos_weight -> LESS doubling (slope < -0.05)
**Per-item is correctly aimed.**
-> LET THE 2x2 FINISH. GO TO TEST 2.

---

## TEST 2 — the 2x2, per-item loss cells (100 epochs)  ~03:00
Only decisive if Test 1 landed on 1C.

                        old loss        per-item loss
    shipped +/-20     0.5818 (done)          C
    wide -60..+20     0.5734 (done)          D

Gate (pre-registered 00:42, cache/phase4_decision.txt):
CONFIRMED if doubled AND halved both shrink by >= 0.10 absolute.

### 2A. CONFIRMED
-> add seeds 1-2 for both cells (~4 h).
-> check the INTERACTION: if D beats C by more than wide beat shipped under the
   old loss (-0.0084), Jaehoon's "the loss blocked the data interventions" claim
   is supported and the wide corpus becomes useful again.
-> then STACK with the count loss (Test 3) and measure whether they compose.

### 2B. NOT CONFIRMED (either fraction moves < 0.10)
-> per-item is not sufficient on its own.
-> GO TO TEST 3, and report per-item as a measured negative.

---

## TEST 3 — beat-count loss   (the symmetric fix)

    L_count = lambda * | sum_t sigmoid(logit_t) - N_true | / N_true

per excerpt, over valid frames only, added to the existing loss.

Why this and not the midpoint hard-negative penalty Jaehoon originally proposed:
the failure is BIDIRECTIONAL. Midpoint penalties attack over-detection only, so
on a 150 bpm track that is already being halved they push the wrong way. The
count term penalises over-counting and under-counting with one mechanism, and
the quantity it constrains -- beats per window -- IS the tempo.

Build order:
1. unit test: reproduce stock loss at lambda=0; verify gradient; verify the term
   is large for a doubled prediction and for a halved one, and ~0 when correct.
2. lambda sweep {0.1, 0.3, 1.0} at 25 epochs, shipped augmentation, one seed.
   Readout: doubled / halved. Pick lambda by the failure mode, not by F.
3. full 100-epoch arm at the chosen lambda, both augmentation ranges.

### 3A. doubled and halved both shrink >= 0.10
-> the mechanism claim is complete: prior identified, fix demonstrated.
-> seeds 1-2, then folds 5 and 7. Then the absolute claim: full-corpus training
   + TTA stacking, measured against 0.6449 on 217 fold-honest tracks.

### 3B. shrink, but asymmetrically (one side only)
-> report honestly; the count term addresses one direction. Consider adding the
   midpoint hard negatives ONLY for the doubling side, as a second arm, since
   by then we would know it is the residual failure.

### 3C. no movement
-> the prior is not reachable from the objective at all. STOP the loss work.
   The characterisation stands on its own (see below) and that is the paper.

---

## TEST 4 (already running) — convergence history  ~01:20
Does the AMLt-CMLt gap improve at ANY point during training? 10 checkpoints,
epoch 9 to 99, both arms.

- gap FLAT while F climbs  -> the objective demonstrably never acted on metrical
  level. Supports every loss-side intervention above; quote it in the paper.
- gap SHRINKING            -> the objective was working on it and simply ran out
  of capacity/data. Weakens the "loss is the bottleneck" framing; the fix may
  need to be architectural (an explicit tempo representation) rather than a loss
  term. Note the DBN result says a tempo model bolted on at DECODE time makes
  things worse (-0.051), so this would have to be in the encoder.

---

## The floor, in every branch

Even if 1A -> 3C (nothing fixable), this stands and does not depend on a fix:

  beat trackers carry an implicit tempo prior inherited from their training
  distribution, and report octave-wrong tempi in BOTH directions as material
  departs from it -- 64% of sub-55 bpm tracks doubled, 16% of >140 bpm tracks
  halved, measured fold-honestly across 1,501 tracks and 12 datasets.

That is a characterisation of how the field's trackers fail, it generalises the
SMC Blind Spot result from one corpus to twelve, and it is publishable without
any intervention working.

## Explicitly NOT funded (from the Aug-24 audit, still binding)
decoder-side work (threshold sweep -0.0055, DBN -0.051, both refuted tonight),
per-track parameter selection, synthetic audio, second feature spaces.
