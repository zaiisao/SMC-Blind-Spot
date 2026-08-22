# vbpm: measured rationale behind the knobs

Prose moved out of `run.py`'s argparse help strings during the 2026-08-06 cleanup.
Every claim here is a measurement from this codebase, not a belief.

## --drift-bound (structured posterior)
`mu = offset + cumsum(delta + eps*tanh(g))`. With the bound off (free per-frame mu) and
the cosine emission, the encoder produced a SPIKE TRAIN: 0.715 rad/frame advance against
a true 0.065, dipping to phase 0 on downbeats (concentration 0.969). Free-q measurement:
F 0.075 -> 0.911 at eps = 0.01. The true per-frame residual sd is 0.0135 rad, so the
0.01 bound holds real music comfortably except asap (8.7% of frames exceed it).
Provenance note: this is OUR structured-q invention, not the tutorial's 9.9.2 factorised
family — legitimate under §7 (which leaves q's family open), but a recorded decision.

## --emission
`cosine` is the two-scalar `a + b cos(phi)` every number before 2026-08-04 was measured
with — an invention (picked for differentiability over a hard wrap indicator), not the
spec's, and the only phase-aware term in the objective, so its weakness was load-bearing.
`triangle` keeps two parameters but has constant-magnitude phase gradient 2b/pi
everywhere; champion as of 2026-08-05 (gtzan F 0.501 / CMLt 0.864).
`transformer` is the tutorial's §9.6 emission. WARNING: §9.6 belongs to the §9
architecture (encoder sees b, conditional prior deployed, Sohn/EB remedies). Grafted
onto the §7 scaffolding it colludes with the encoder: smooth right-rate trajectory,
chance alignment, downbeats carried in micro-modulations only the co-trained decoder
reads (rule-g 0.03–0.06 = below null, emission-D 0.19–0.66, logs_faithful.txt).

## --emission-positional
OFF by default: with attention, positional encoding lets the emission emit a periodic
pattern from POSITION and ignore the latent — the decoder shortcut Point 1 forbids,
invisible in the loss.

## --pos-weight
The target is 3.2% positive frames, so unweighted, the whole value of knowing the phase
is 0.051 nats/frame against a KL of 1.13; at 30 it is 0.754. Any value != 1 makes the
objective a weighted surrogate, not an ELBO. The true-ELBO arms (pos_weight = 1) match
or beat the weighted surrogate everywhere, so 1 is both faithful and best.

## --emission-sharpness / --sharpness-warmup
A scheduled FLOOR on the emission amplitude b. The broad-emission/imprecise-phase
deadlock keeps b ~2 (peak p 0.14) even for the triangle; a floor removes the broad
hideout (b = 5 halves an exp-regime peak within ~70 ms of a 2 s bar). Audited outcome:
for the TRIANGLE the floor is a pure gradient gain (no localisation) and a small
significant harm (paired delta -0.016, p = 0.003); for the COSINE, whose b*sin(phi)
gradient concentrates with b, it helps (+0.03..0.05 on 5/7 datasets). Rotation sd is
unchanged by either — the anchor floor is not emission-side.

## beta annealing (--beta-start/--beta-end/--beta-warmup)
The KL is not too strong because the prior is wrong — the true per-frame residual is
0.0135 rad sd against the 0.0224 the prior permits (too LOOSE, if anything). It is
strong because an untrained encoder's trajectory is not smooth and the misalignment
term scales as kappa*delta^2/2. Annealing lets reconstruction shape a trajectory first;
a training schedule, not a claim about music.

## --gtzan-checkpoint
gtzan through final0 (the cache default) is an activation space the encoder never sees
in training (final0 serves no CV songs): supervised probe 0.750 through fold7 vs 0.058
through final0. Any fold checkpoint is fold-honest for gtzan.

## training loss normalisation
The optimised loss is `-((recon - beta*kl)/frames).mean()` — per-frame, because crops
differ in length (24–45 s) and a 45 s crop would otherwise carry ~1.6x the gradient of
a 29 s one. This reweights crops by 1/T relative to the corpus sum-ELBO: a recorded
deviation. The REPORTED elbo always uses beta = 1 so epochs stay comparable.

## fp16 wire, fixed buckets
Features ship fp16 (max abs err 0.0039 on [-14, 8.4]) and batch membership is fixed
length-sorted buckets with only visit order shuffled — both recorded in the 2026-08-05
faithfulness audit's disclosure list.
