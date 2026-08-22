# Preliminary: vanilla Beat Transformer vs Beat Transformer + R2 (end-to-end)

2026-07-18 overnight run. Both arms: official Demixed_DilatedTransformerModel FROM SCRATCH on our
4 CV datasets (fold-0 split: 1,020 train / 146 val songs, meter-representable subset), identical
beat-aligned crops (<=16 s), identical optimizer (RAdam+Lookahead, lr 1e-3, clip .5, batch 1, 30
epochs). Arm difference only in loss/decode:
  vanilla = their BCE (widened targets) -> madmom-as-BT-ships-it (obs_lambda=6, num_tempi=None,
            threshold=0.2), transition_lambda=100
  r2 e2e  = BCE + CRF NLL through the exact structured forward (rungs/r2_learned_dbn.py),
            learning the frontend AND transition_lambda jointly -> same decode, learned lambda

## Final table (FULL fold-0 val, 146 songs, best checkpoints)

| activations           | decode lambda | beat F | downbeat F |
|-----------------------|---------------|--------|------------|
| vanilla               | 100           | 0.9506 | 0.9053     |
| vanilla               | 14.2 (learned)| 0.9467 | 0.8808     |
| r2 e2e                | 100           | 0.9446 | 0.8899     |
| r2 e2e                | 14.2 (learned)| 0.9374 | 0.8637     |
| r2 e2e                | 14.2, threshold=0 | 0.9341 | 0.8600 |
| pretrained fold_0 (LEAKY ref: their folds are not ours) | 100 | 0.9622 | 0.9012 |

## Verdict (preliminary)

VANILLA WINS. The e2e CRF arm does not beat BCE + fixed madmom DBN; best r2 cell trails by
-0.006 beat F / -0.015 downbeat F, and r2's own learned lambda makes it worse, not better.

Decomposed:
1. LEARNED LAMBDA IS LIKELIHOOD-OPTIMAL BUT F-SUBOPTIMAL. lambda converged decisively to ~14.2
   (7x looser than madmom's 100) and was rock-stable for 10+ epochs -- the CRF genuinely prefers
   a flexible tempo kernel (consistent with the heavy-tailed tempo-increment finding). But at
   DECODE, lambda=100 beats lambda=14.2 for BOTH frontends (vanilla: 0.9506 vs 0.9467). F-measure
   rewards tempo continuity more than likelihood does. Likelihood != task metric.
2. THE CRF TERM SLIGHTLY HURT THE FRONTEND: r2 acts decoded at lambda=100 still trail vanilla
   acts (0.9446 vs 0.9506). The structure gradient cost a little BCE-fit without buying F.
3. NO DECODE ARTIFACT: bare (threshold=0) ~= shipped for r2 (0.9341 vs 0.9374), so the BCE anchor
   kept activations calibrated; the loss is real, not a thresholding illusion.
4. OUR PIPELINE IS HEALTHY: from-scratch vanilla (0.9506) nearly matches the pretrained
   reference (0.9622) which SAW some of our val songs in its training (leaky).

## Failure archaeology (what it took to get here)

- Run 1: both arms NaN'd (their model does this; their own train.py carries a NaN-skip guard we
  had not replicated). Fixed: skip nonfinite loss AND nonfinite grad-norm; per-epoch weight
  health check; resume support. vanilla resumed from epoch 7.
- Run 2 (r2): PURE CRF SATURATES. With no calibration pressure, logits hit |55|, 100% of frames
  sigmoid-saturated within ONE epoch; CRF gradient dies (flat under saturation), lambda absorbs
  all remaining gradient, decode freezes (bit-identical F across evals). Fix: hybrid
  CRF + BCE -- BCE's gradient is maximal exactly where CRF's is zero. This also sharpened the
  design: arm delta = the exact-forward structure term alone.

## Caveats

Single fold, single seed, 30 epochs, hybrid weight 1:1 untuned, 16-s crops (short-range; CRF's
long-range structure advantage may need full songs), lambda learned jointly (not decode-tuned).
Skipped 139 songs (no meter/grid representation). The 60-song training-time evals ran ~+0.01 high
vs the full fold (subset bias) -- use this table, not the training logs.

## Follow-ups suggested by the data

1. Decode-time lambda sweep on vanilla acts (is 100 even optimal? maybe 150-300).
2. Frozen-frontend R2 (pure ladder R2): learn lambda on vanilla activations by CRF -- decouples
   the two effects cleanly.
3. Full-song (or 60-s) crops for the CRF arm.
4. Meter set (2,3,4) variant (16 songs excluded today).
5. Multi-seed + all 8 folds before any strong claim.

## ADDENDUM (post-review): the deficit was substantially OUR harness, not the idea

Comprehensive review after the first table found three asymmetries; two measured, all fixed:

1. LR ANNEALING NEVER FIRED FOR R2. The plateau scheduler stepped on the COMBINED loss; r2's
   CRF term kept falling, so its lr stayed 1e-3 for all 30 epochs while vanilla fine-tuned at
   2e-4 from epoch 10 -- and most of vanilla's final margin accrued in its post-anneal phase.
   Fixed: both arms now anneal on the BCE component (identical signal).
2. OBSERVATION-MODEL TRAIN/DECODE MISMATCH. The CRF trained against observation_lambda=16
   (chassis default) but decode is BT-shipped 6. Probe on the full fold:
       vanilla obs=6  lam=100 : 0.9506 | obs=16 lam=100 : 0.9017
       r2      obs=6  lam=100 : 0.9446 | obs=16 lam=100 : 0.8911
       r2      obs=6  lam=14  : 0.9374 | obs=16 lam=14  : 0.9295
   Under the TRAIN-CONSISTENT obs=16 decode, the learned lambda=14 BEATS lambda=100 -- the
   learned factor was optimal for the world it was trained in. THE EARLIER "likelihood-optimal
   but F-suboptimal lambda" INTERPRETATION IS THEREFORE CONFOUNDED AND WITHDRAWN pending the
   obs=6-consistent rematch. (obs=6 remains the right deployment decode for both frontends.)
3. Epoch/data edge to vanilla (35 effective epochs incl. resume vs 30 fresh; early vanilla
   epochs overlapped cache build). Fixed: both arms fresh, 30 epochs, same seed.

Rematch launched 2026-07-18 with all three fixes (r2 chassis at observation_lambda=6, BCE-keyed
scheduler, symmetric fresh runs).

---

# 2026-07-20: R2 promotion, the integer-interval dithering artifact, and the mixture-kernel repair

## Ladder renumbering
R1.5 (unsupervised exact EM) PROMOTED to R2 (rungs/r2_em_dbn.py). The CRF-trained variant is
NOT a rung (discriminative conditional objective, off-program): demoted to
experiments/bt_e2e/crf_baseline.py, kept as the comparison estimator.

## R2 verification (all three passes clean on the EM core)
- Adversarial review: E-step counts exact vs brute-force path enumeration (4.8e-7); M-step argmax
  verified; monotonicity traced. ONE conceptual bug found: comparing marginals across
  observation_lambda is ill-posed (the (obs-1) normalization + obs-dependent state partition
  inject a bookkeeping term of hundreds of nats favoring small obs; pure noise selects obs=2 by
  ~1200 nats). --learn-obs verdicts are bookkeeping, not model selection.
- Blind black-box suite (written without reading the code): tests/test_r2_em.py, 16/16 pass;
  independently rediscovered the obs-comparison ill-posedness from the outside.
- Live run: lambda=40.27 (EM, 8 iters) == 39.05 (250 Adam steps, same marginal) -- Fisher's
  identity on the real model. Decode (146-song val fold, shipped-BT decode):
      R0 madmom 0.9506 = R1 0.9506 = CRF(98.6) 0.9506 > R2-generative(40.3) 0.9488.
- learn-obs arm: obs drifted 6->4 (never 2), lambda re-equilibrated ~31. Artifact regime as
  predicted by the review.

## THE FINDING: integer-interval dithering corrupts the generative lambda
The discrete state space represents only INTEGER frames-per-beat. A song at true interval 21.4
frames cannot be represented: the Viterbi path DITHERS between adjacent integer tempi,
manufacturing fictitious tempo transitions on metronomically perfect input. MLE faithfully
bills these to lambda. Three independent confirmations (scripts in this dir):

1. ISOLATION (synthetic_grid_probe.py): metronomically exact synthetic beats.
      interval 21.0 (on-grid) : flat path 21,21,21,...      lambda_MLE = 133.4
      interval 21.4           : dither 21,21,22,... (60/40)  lambda_MLE = 39.7
      interval 21.5           : perfect 22,21,22,21,...      lambda_MLE = 37.6
      real 300-song corpus    :                              lambda_MLE = 40.27
   A perfect metronome reproduces the real-data lambda to 1.5%. The "generative lambda ~= 40"
   is the grid, not the music.
2. SCALING LAW (train_r2_em_fps2.py): same crops, activations 2x-upsampled, chassis at 2x fps:
   lambda 40.27 -> 80.08 (ratio 1.988 vs predicted 2.0). Ratio-deviation per dither step halves,
   lambda doubles. Grid origin confirmed on real data.
3. REPAIR (mixture_kernel_probe.py): two-component kernel
   p(j|i) = w*Dither(|interval diff|<=1, uniform) + (1-w)*exp(-lambda|r-1|), (w,lambda) learned
   by the SAME exact EM (2-D M-step). Metronome: w=0.51, lambda->~96. REAL DATA: w=0.370,
   lambda=93.1 -- the artifact-corrected generative MLE agrees with hand-set 100 and CRF 98.6.
   Three estimators, one answer. Decode (bare Viterbi, apples-to-apples):
      R1 shipped decode      : 0.9506  (deployment heuristics worth ~+0.04 on this frontend)
      R1 bare, lambda=100    : 0.9100
      MIXTURE bare (learned) : 0.9193  (+0.0093 over bare R1)
   The unsupervised artifact-aware kernel BEATS the hand-set single kernel at equal footing.

## RETRACTION
The earlier interpretation "generative lambda~=40 is interpretable, matching heavy-tailed
tempo-increment statistics" is WITHDRAWN: the metronome reproduces it without any musical
timing variation. The tempo-increment-law heavy tails (kurtosis ~13) are themselves suspect --
computed from frame-quantized annotation intervals, which carry the same dithering. Re-derive
on sub-frame-interpolated annotations before reuse.

## Prior-art status (Gemini deep research + primary-source check of Krebs & Boeck PhD theses)
The chain [dithering -> corrupted MLE of transition_lambda -> likelihood/task divergence]
appears UNPUBLISHED. Nearest anticipations (cite generously):
- Krebs thesis 4.3.2: chose Viterbi training over Baum-Welch MLE ("improvements in segmentation
  quality come incidentally") and used tolerance-window (evidential) decoding against "imprecise
  annotations" -- the intuition and a mitigation, without the mechanism. Ch 6: tried Gaussian
  mixtures for the tempo-change kernel (rejected on task performance; not artifact-motivated);
  efficient state space's Round() preserves the artifact.
- Exact mathematics published in another field: bang-bang PLL limit cycles (Levantino 2013).
- Same misspecification->scaling logic: ASR acoustic scale (frame-independence violation).
- Unchecked residue: madmom GitHub issues, Cemgil full texts, non-English/gray literature.

## learn-obs arm final (2026-07-20, predicted artifact confirmed)
EM with the obs coordinate-argmax: obs drifted 6->4 at iter 0 and froze; lambda re-equilibrated
to 30.9. Fisher check holds in this regime too (EM 30.92 == grad 30.30 at obs=4). Decode:
    R0 0.9506 = R1 0.9506 = CRF 0.9506 > R2 fixed-obs (40.3, obs=6) 0.9488
                                       > R2 learn-obs (30.9, obs=4) 0.9459
The obs-argmax LOWERED F by 0.0029 -- the F1 bookkeeping bias (small-obs normalization windfall)
acting as predicted; observation_lambda "learning" through the plug-in marginal is model
selection over an ill-posed comparison and is hereby retired. obs stays swept/hand-set until
R4's proper emission density makes the comparison well-posed.

## Per-frame conditioning ablation (v1/v2, VERDICT PENDING adversarial+blind review)
Generative training of per-frame transition modulation on the corrected mixture kernel:
    global mixture (w=.370, lam=93.1)     : 0.9193
    v1: net on lambda_t -> collapsed (med 15), NLL flat->worse : trained F 0.9191
    v2: net on w_t -> saturated (med 0.93), NLL flat->worse    : trained F 0.9206
0.0015 spread across wildly different parameter configs; printed NLL worsens while F holds/rises.
Candidate readings: (1) per-frame transition surface flat in likelihood AND F post-mixture
(identifiability vacuum -> nothing to learn, nothing to lose); (2) mostly-dither kernels decode
slightly better (another likelihood!=task point); (3) printed minibatch NLL untrustworthy.
Adversarial review + blind test suite running; no conclusion recorded until they report.

## Per-frame ablation VERDICT (adversarial + blind review complete)
CODE CLEAN (both agents): kernel alignment verified by gradient-mass placement; normalization
exact; gradients machine-precision (fd ~1e-8 in f64); 15/15 blind black-box tests pass.
MY EARLIER READING RETRACTED: the printed "NLL flat-to-worse" was a MINIBATCH-COMPOSITION
artifact (fixed-seed batches; same batches at init reproduce the trajectory). FULL-SET NLL
improved in BOTH runs (1.50747 -> 1.50583 v1 / 1.50601 v2). No degeneration, no leak, no bug.
CORRECT READING: a near-flat identifiability ridge -- both runs express the same small, genuine
preference for MORE +-1 (dither) mass than the global w=0.370 (v1 via lambda down, v2 via w up),
worth ~0.0016 nats/frame and ~+0.001 F (v2 decode 0.9206 > 0.9193 init). Per-frame transition
conditioning WORKS but the available signal is tiny -> R4 (emission) owns the headroom; R3
retired as a rung on effect-size grounds, not on failure. Instrumentation lesson: log full-set
or fixed-held-out NLL, never fresh minibatches.

## Mixture kernel under DEPLOYMENT decode (mixture_shipped_decode.py)
Ported heuristics validated: R1 lam=100 through our threshold_crop + peak-snap wrapper =
0.9504/0.9057 vs DBN2016 shipped 0.9506/0.9053 (faithful to +-0.0004). Then:
    R1 shipped                       : 0.9506 / 0.9053
    R1 lam=100 + wrapper             : 0.9504 / 0.9057
    MIXTURE (w=.370,93.1) + wrapper  : 0.9503 / 0.9053
DEAD TIE. The bare-level +0.0093 does NOT stack on the heuristics: peak-snap and the dither
component repair the SAME symptom (within-beat-region placement flexibility) -- one learned,
one hand-crafted. Redundant remedies, not additive ones. Net position: the unsupervised
artifact-corrected kernel EQUALS the fully hand-engineered deployment stack at full config,
and REPLACES the need for hand-tuned lambda; the heuristics' remaining value is the threshold
crop + snap, which are model-independent. Pipeline-level improvement over shipped R1 must come
from the emission/frontend (R4+), not the transition -- transition axis now closed at parity.

## Continuity tiebreaker (tiebreak_continuity.py): CMLt/AMLt, beat + downbeat
                                  beatF   CMLt    AMLt    dbF     dbCMLt  dbAMLt
DEPLOYED lam=100 hand-set         0.9504  0.9191  0.9410  0.9057  0.8842  0.9366
DEPLOYED crf lam=98.6             0.9504  0.9191  0.9410  0.9057  0.8842  0.9366
DEPLOYED MIXTURE w=.37 lam=93.1   0.9503  0.9188  0.9407  0.9053  0.8845  0.9369
DEPLOYED em-single lam=40.3       0.9494  0.9144  0.9340  0.8924  0.8671  0.9154
DEPLOYED learn-obs lam=30.9 o=4   0.9457  0.9034  0.9245  0.8876  0.8610  0.9089
BARE     lam=100 hand-set         0.9114  0.9107  0.9325  0.8677  0.8831  0.9359
BARE     MIXTURE w=.37 lam=93.1   0.9193  0.9145  0.9363  0.8744  0.8843  0.9372
BARE     em-single lam=40.3       0.9180  0.9079  0.9274  0.8616  0.8682  0.9164

1. TOP-3 TIE IS METRIC-ROBUST: hand-set / CRF / mixture within +-0.0004 on ALL SIX metrics.
   The transition-axis parity claim survives the continuity tiebreakers.
2. ARTIFACT CONFIGS STRATIFY HARD where F barely moved: lam=40.3 loses up to -0.021 (dbAMLt),
   learn-obs up to -0.028; damage concentrated in DOWNBEAT CONTINUITY (bar-level lock). F's
   -0.001/-0.005 understated the real cost by an order of magnitude. Loose kernels break bars,
   not beats.
3. DEPLOYMENT HEURISTICS = PLACEMENT REPAIR: bare-vs-deployed costs ~0.04 F but only ~0.008
   CMLt / ~0.001 dbCMLt -- the wrappers polish beat positions, they never fixed tracking.
4. MIXTURE'S BARE EDGE IS ALSO PLACEMENT (+0.008 F, +0.001-0.004 continuity) -- an in-model
   peak-snap, hence the redundancy with correct=True. Notably bare-mixture dbCMLt (0.8843)
   already EQUALS the deployed baseline (0.8842): the mixture model alone achieves
   deployment-grade bar continuity with no heuristics.

## R4.5-v0: rich-feature Gaussian emission (VERIFIED NEGATIVE -- emission domination)

Oracle context first (oracle_ceiling.py, per-song oracle over 12 decode variants + half-bar flip):
    beat  F: shipped 0.9504 / oracle 0.9596  -> decoder-side headroom +0.009 (near ceiling)
    db    F: shipped 0.9057 / oracle 0.9533  -> decoder-side headroom +0.048 (WIDE OPEN;
             mostly half-bar phase + meter decisions -- the evidence supports the right bar)
The remaining in-domain opportunity is a DOWNBEAT/BAR-PHASE problem, decoder-side.

R4.5-v0 (rungs/r4_5_rich_emission.py + train_r4_5.py): class-conditional diagonal Gaussians on
[T,256] BT penultimate features, EXACT Baum-Welch (autograd E-step posteriors, closed-form
M-step), fixed mixture transition, self-supervised init from the [T,2] pipeline's own decode.
EM converged in ~3 iters, monotone. Decode (146-song val):
    R1/R2 ref DEPLOYED: 0.9504 0.9191 0.9410 | 0.9057 0.8842 0.9366
    R4.5   DEPLOYED   : 0.9331 0.8768 0.8955 | 0.8635 0.8347 0.8736   (LOSES everywhere;
    R4.5   BARE       : 0.8859 0.8563 0.8749 | 0.8133 0.8342 0.8738    continuity hit hardest)

VERIFIED DIAGNOSIS (adversarial + blind, both complete):
- CODE CORRECT: E-step identity exact vs independent forward-backward (4.8e-6, meter mixture
  included); M-step algebra 1e-16; feature/activation alignment 1.2e-7; blind suite 9/9 incl.
  parameter recovery from model-generated data and exact on-grid decode.
- MODEL IS THE BUG -- EMISSION DOMINATION (the ASR acoustic-scaling disease, measured in-house):
  per-frame emission gaps ~22,600 nats vs transition scores 0.2-272; UNIFORM-kernel decode ties
  the real kernel (dF 0.004, 91% frame-argmax agreement) -> the transition is INERT; only the
  legal-path topology survives. Mechanism: exact MLE gave the rare homogeneous downbeat class
  the sharpest Gaussian (96/256 dims at the 1e-3 variance floor -> +660-nat log-det bonus),
  so the path chases emission spikes and continuity collapses (below even the teacher init,
  whose paths were tempo-smoothed).
- Scale-dependence confirmed: the blind structure-matters test at D=4-8 shows the transition
  DOES constrain the path when emission magnitudes are sane -- domination is a property of
  D=256 + collapsed variances, not of the machinery.
- Float32 DP noise ~0.5 nat/song at these magnitudes: read EM-trace monotonicity only above
  ~1 nat; near-tied meter selections partly numerical.

v0's honest claim: "a diagonal-Gaussian frame classifier + legal-path constraint loses to the
Bock plug-in" -- NOT a verdict on learned emissions per se; the transition never got a vote.

R4.5-v1 SHORTLIST (to make the emission question posable):
  (a) emission tempering log p/kappa (acoustic-scale analog; directly counters D-scaling),
  (b) TIED covariances across classes (kills the 660-nat log-det asymmetry; likely best single fix),
  (c) discriminative calibrated p(class|f) emission (off-program, diagnostic).
Target unchanged and now quantified: the +0.048 dbF / half-bar-phase prize.

## R4.5 v1/v2: domination fixed, gate opened, and the fair verdict on rich Gaussian emissions

v1 (train_r4_5_v1.py): tied covariance + PCA-16 whitening (both fixes verified: adversarial
CLEAN incl. tied M-step 3.6e-15 + monotone; blind tied suite tests/test_r4_5_tied.py 9/9 incl.
the no-log-det-games certificate). Emission gaps 22,600 -> 4.2 nats; VALIDITY GATE OPEN on the
trustworthy F-delta clause (real-vs-uniform +0.0152) -- the transition regained its vote.
v2 (train_r4_5_v2.py): LDA control (2 discriminant dims from init-label class stats + 14
residual-PCA dims) to control the "unsupervised PCA discarded downbeat directions" confound.
Emission gaps 10.2 nats (sharper class evidence), gate OPEN (+0.0098). Result: IDENTICAL to v1.

                       beatF   CMLt    AMLt    dbF     dbCMLt  dbAMLt
    R1/R2 ref DEPLOYED 0.9504  0.9191  0.9410  0.9057  0.8842  0.9366
    v1 DEPLOYED        0.9392  0.8840  0.9420  0.8826  0.8496  0.9096
    v2 DEPLOYED (LDA)  0.9382  0.8817  0.9281  0.8791  0.8465  0.9044
    bare R1            0.9114  0.9107  0.9325  0.8677  0.8831  0.9359
    v1 BARE            0.9146  0.8776  0.9354  0.8554  0.8490  0.9091

VERDICT (now unconfounded): unsupervised class-conditional Gaussian emissions on rich features
LOSE to the Bock plug-in on distilled [T,2] activations (-0.011 beat / -0.023 dbF deployed),
with the loss concentrated in CMLt (wrong metrical level) and downbeats. Notable positives:
v1 bare BEATS bare-R1 on beats (0.9146 vs 0.9114) and v1's AMLt matches deployed reference --
the learned emission tracks continuously, just at the wrong level more often. LDA control rules
out the projection; near-frozen EM (~85 nats) means the emission is essentially "Gaussians on
the teacher's labels" -- the teacher's 2 channels, distilled by supervised training, carry the
class information better than any unsupervised density over the penultimate layer.

R4.5 line status: the GENERATIVE-GAUSSIAN route to rich emissions is CLOSED (three variants,
fully verified machinery, controlled confound). The +0.048 dbF oracle prize remains untouched
and decoder-side; remaining candidate collectors: (a) calibrated discriminative emission
p(class|f)/p(class) (the ASR hybrid recipe -- strongest known fix, off-program), (b) bar-phase
disambiguation at decode (two-hypothesis posterior -- cheapest, targets the dominant error mode
directly), (c) R5's decoder-as-density (the on-program endgame).
