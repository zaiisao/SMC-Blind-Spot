# Morning report

Everything below ran unattended. Numbers are fold-honest on all 217 SMC
tracks; neither arm saw any SMC track in training.

## 1. The overnight headline (already established before you slept)

```
Pooled across 1,501 tracks / 12 datasets, fold-honest released model:
  band       n    AMLt-CMLt   doubled   halved
  <55       59      0.2435     0.644     0.000
  55-70    147      0.1919     0.401     0.007
  70-100   358      0.0947     0.162     0.020
  100-140  556      0.0282     0.025     0.022
  >140     381      0.1063     0.013     0.163
```
Slow music is tracked at DOUBLE its rate, fast music at HALF, monotonically
in both directions, minimum where the training corpora's tempo mass sits.

## 2. Sign sweep -- does pos_weight cause it?

Identical data/seed/epochs; ONLY the global beat pos_weight varies (8/32/128),
bracketing the per-item range (10 at 180bpm .. 68 at 40bpm).

```
   pos_weight  doubled(<55)  halved(>100)
            8         0.098         0.711
           32         0.867         0.105
          128         0.978         0.079
```

Slope 8 -> 128: doubling +0.880, halving -0.632

**Higher pos_weight => MORE doubling.** The per-item fix is WRONG-SIGNED:
it raises the weight on exactly the slow tracks that already double.
Drop it; go to the beat-count loss.

## 3. The 2x2 (per-item loss cells)

Compared against OUR OWN old-loss arms, not against released checkpoints.
(The released model trained on a different, larger corpus with a different
tempo mass, so its 0.356/0.316 is not the right baseline for these cells.)

```
  cell                         F(217)   doubled   halved
  shipped + old loss           0.5818     0.511    0.316
  wide + old loss              0.5734     0.614    0.316
  shipped + per-item        (pending)     0.578    0.316
  wide + per-item           (pending)     0.556    0.316
```

  shipped: doubled +0.067, halved +0.000  -> **PARTIAL**

  wide: doubled -0.058, halved +0.000  -> **PARTIAL**

Pre-registered gate (cache/phase4_decision.txt, written 00:42 before any
checkpoint existed): CONFIRMED if doubled AND halved both shrink by >=0.10.
KILL if both move <0.05 while per-item weights genuinely vary 7x.

## 3b. Beat-count loss -- lambda sweep (25 epochs)

Calibration on a trained model: the count term reads 1.943 at <55 bpm and
0.479 at >100 -- it orders exactly like the doubling fractions. Mean 1.09,
so lambda=0.3 would perturb a ~1.2 base loss by 27%; swept gentler.

```
    lambda  doubled(<55)  halved(>100)
      0.05         0.644         0.237
      0.15         0.467         0.474
      0.50         0.143         0.649
      none         0.867         0.105   <- stock loss, 25 ep
```

  best lambda=0.5: doubled -0.724, halved +0.543 -> **PARTIAL**
  (compared against pw32 = stock loss at the SAME 25-epoch budget,
   not against the 100-epoch arms -- different training length)

## 4. Also finished overnight

- **conv_shipped** (did the metrical gap EVER improve during training): gap<70 0.0648 at epoch 9 -> 0.1150 at epoch 99, change +0.0502
- **conv_wide** (same, wide-augmentation arm): gap<70 0.0595 at epoch 9 -> 0.1309 at epoch 99, change +0.0713
- **SDEdit guidance sweep**: done (does beat-pathway guidance extend the no-drift noise range)

## 5. What I'd do next, depending on section 2

- prior NOT in the loss  -> build the beat-count loss |sum(sigmoid) - true_count|;
  it constrains tempo directly and penalises doubling AND halving symmetrically.
- prior IS in the loss    -> the 2x2 finishes the story; add seeds 1-2.
- either way, the cross-dataset table in section 1 stands on its own as a
  characterisation, and does not depend on any fix working.
