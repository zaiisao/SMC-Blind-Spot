# Where the audio bar-period estimator loses, 2026-08-09

Commit 712adb9 removed the last oracle (delta came from annotated downbeats) and measured
the price on gtzan: F 0.752 -> 0.737, true CMLt 0.844 -> 0.710, est/ref 1.003 -> 1.161.
It named the cause as octave error and left the fix open. This is the audit of that
error: what the estimator gets wrong, whether the evidence was there, and which repairs
work. All numbers are the deterministic (middle) 45 s window per song, 993 gtzan songs
and 239 val songs, frontend beat_this/final0, tolerance +-5% on the period.

Scripts: `octave_audit.py` (error anatomy), `tempo_dump.py` (freeze activations),
`tempo_lab.py` (bench decision rules), `tempo_ranker.py` (learned reranker).

## 1. The error is halving, and it is not spread evenly

Shipped rule (harmonic-sum ACF, smallest peak clearing 0.75 * max):

    gtzan  exact 79.3%   halved 11.6%   doubled 3.4%   other 5.7%
    val    exact 80.3%   halved  5.9%   doubled 5.4%   other 8.4%

By gtzan genre, exact / halved:

    disco   95% / 2%     hiphop 93% / 7%     pop    88% /  9%     country 84% /  9%
    rock    84% / 9%     reggae 82% / 3%     metal  79% / 12%     jazz  74% / 14%
    blues   69% / 24%    classical 44% / 27%

Val agrees: hjdb 100%, ballroom 94%, beatles 96%, hainsworth 85%, rwc 73%, asap 47%.
The failure is a SOFT-ONSET failure -- classical and asap (solo piano) are half the
problem, and those are the same conditions the SMC mission targets. Percussive genres
are effectively solved.

## 2. The evidence is present; the DECISION throws it away

Of the 206 wrong picks on gtzan: the truth is inside the [1,6]s search range for 188,
is a peak in the score curve for 145, and clears the 0.75*max acceptance floor for 85.
All 85 of those lost purely because the rule takes the SMALLEST accepted peak, and the
winner outscores the truth by a median of only 3.6% -- the evidence is nearly tied and
the tie-break is what fails.

Extending the ACF's peaks by factors {1/2, 1, 2, 3, 4} gives a 22-candidate ladder
(~11 unique) whose ceiling -- can ANY reranker get it right -- is:

    gtzan 96.7%   val 92.5%

against nulls on the same windows: candidates drawn log-uniform over [1,6]s 70.1%,
drawn from the train split's period distribution 78.1%, and each window handed ANOTHER
window's ladder 31.3%. The permutation null is the honest one (it keeps the ladder's
geometric spacing and kills only the audio link): 96.7% vs 31.3%. Nomination works.

So the gap is 79.3% -> 96.7% and it lives entirely in the ranking.

## 3. What does NOT close it

    rule                                       gtzan exact   val exact   gtzan halved
    shipped: smallest peak >= 0.75*max            79.3%        80.3%        11.6%
    strongest peak (argmax)                       81.4%        77.0%         8.1%
    comb contrast, on-vs-background                80.5%        79.9%         9.0%
    comb contrast, isolation (2nd-best comb)      74.3%        74.5%        12.6%
    peak-pick the frontend's own downbeats        74.1%        83.3%        10.4%
    + log-normal tempo prior (train-fit) w=0.5    77.5%        73.6%        10.2%
    + same prior w=1.0                            73.0%        70.3%        13.2%
    + same prior w=2.0                            69.6%        68.2%        14.9%

The tempo prior is a clean negative and monotone in its weight: halving a 2 s bar gives
1 s, which a prior fit on real bar periods still finds ordinary, so the prior cannot
punish the actual error -- it only drags genuinely long bars toward the median. The comb
criteria were verified to work on synthetic input (they pick the true period over its
half AND its double on planted signals, see the check in `tempo_lab.comb_contrast`), so
their failure here is real, not a bug: on soft-onset audio the folded profile at half
the true period is not visibly diluted.

Widening the search range is not the answer either: [0.6, 8]s raises the ceiling
(gtzan 96.5 -> 97.5, val 92.1 -> 97.1) but the shipped rule gets WORSE on gtzan
(79.3 -> 75.9, halving 11.6 -> 14.6) because it hands the smallest-peak rule more short
candidates to prefer.

## 4. What partly does: learn the ranking

A 2-layer MLP over each candidate's rotation-normalised folded downbeat profile, the
beat profile rotated by the SAME offset, log period and the comb statistics, trained by
softmax over the ladder against the TRAIN split's annotated periods (train labels, like
the emission's targets -- the deployed reader sees only folded activations):

    held out BY SONG inside train   91.6%   (ceiling on that pool 95.4%)
    val                             84.5%   (ceiling 92.5%)
    gtzan                           82.1%   (ceiling 96.7%)   halved 11.6% -> 6.3%

It nearly saturates on unseen SONGS of seen datasets (87% of the available headroom) and
captures only ~17% of it on the unseen CORPUS. Per genre it improves nearly everything
and roughly halves the halving (blues 69->81, classical halving 27%->14%, country
halving 9%->2%), regressing only on reggae (82->78). Trained on one 45 s window per song
(1661 examples) it fits the training pool perfectly, so the corpus-transfer gap is not
capacity -- it is what the features generalise.

CAVEAT, and it is the one this project has been burned by before (see the MERT
conditioning verdict): a within-corpus win that shrinks cross-corpus is the signature of
a shortcut. gtzan is the honest number here. SMC is not in this catalog and would be the
sharper test, since the residual is concentrated exactly in its conditions.

## 5. The read

The bar period is currently a PREPROCESSING DECISION: one hand-written rule commits to a
number before the model runs, and the model can never reconsider it. The audit says the
right period is in a ~20-candidate set 96.7% of the time and that no hand rule picks it.
That is the same shape as the phase problem anchor_k already solved -- enumerate the
candidates, let the objective choose -- which suggests making the period a discrete
latent the ELBO marginalises over rather than a number handed in. Untested; the ceiling
for such a design is the 96.7% above, and it needs no new supervision.

Downstream cost is unmeasured here: nothing in this audit was scored through the model.
The current run does not save a checkpoint, so validating any of this end-to-end needs
a retrain.
