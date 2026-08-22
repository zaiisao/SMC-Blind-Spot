# The count factor p(N | phi) -- parked 2026-08-15

Parked deliberately, not rejected. Written down so the reasoning survives.

## What it is

The interval emission currently scores `p(t_1..t_N | phi)`: given that there are N
annotated downbeats, where do they fall. N is CONDITIONED ON, not modelled.

But N is informative about phi. A comb running 8% fast fits 13 bars into a window
where 12 are annotated. Conditioning on N throws that information away.

The repair is to score the full observation `(N, t_1..t_N)` -- a marked point
process rather than a fixed-N density -- by adding a factor `p(N | phi)`. No new
latent, no new parameters, no meter. Same class of move as the mean-phase
coordinate: fix what the density is over, not what the model contains.

## Why it came up

Measured on the mean-phase + surgical-lift arm (place_coord=mean,
place_attach=false, place_lift=1.0), 400 ep, acf-init, gated:

  hjdb0  (N=27 annotations, 1.67 s bars)
    s1  6ms/100% F 1.000    s2 34ms/100% F 1.000
    s3 49ms/100% F 1.000    s0 148ms/22% F 0.222 (CMLt 0.852)
    zdec off in all four: 340-423 ms  <- the corrector is load-bearing, finally

  ballroom0 (N=12 annotations, 2.45 s bars)
    s1 533ms/0% F 0.000 CMLt 0.000, n13/12 wraps, ratio 1.00, res 0.995
    zdec off: 203ms/17%  <- the corrector makes it WORSE here

Ballroom's trajectory: at ep 150 it HAD the answer (190ms, CMLt 1.000, n12/12,
res 0.988), then between ep 150-200 walked to ~650 ms and parked there for 200
epochs with the fold still coherent at 0.99. The tell is n13/12: it emits
THIRTEEN wraps for twelve bars. Mean rate reads 1.00; the local rate is wrong in
a way that inserts one bar.

## The hypothesis

An inserted bar shifts the mean-phase coordinate by ~1/N of a bar, so the drift
term's grip on bar-insertion scales with N. At N=27 (hjdb) an insertion is
expensive; at N=12 (ballroom) it is cheap enough that the corrector buys it.
p(N | phi) would price the insertion directly instead of through the coordinate.

UNTESTED. The cheap check, no training required: compute the cost of an
inserted-bar path under the mean coordinate at N=12 vs N=27 and see whether the
ratio matches the song split. Do this BEFORE writing any count term.

## Why it lands on octave, not meter

Meter = beats per bar (the M latent). Already probed: timing-only meter is
~chance except asap/ballroom, and needs a beat-in-bar emission channel. NOT this.

Octave = which period is the bar (T vs 2T vs T/2). Doubling the rate doubles the
wrap count, so a count term is the only factor in this likelihood that separates
T from T/2 DIRECTLY rather than through the interval ruler's log-ratio penalty.
That is why interval.py's own docstring names p(N|phi) as the derived route to
octave identifiability. "Endpoints always harmonics" has been the recurring
failure since the rungs era; this is the principled term aimed at it.

## Risks / open questions before building

- Normalisation. The mean-phase coordinate was safe because its Jacobian is
  provably identical to the first-annotation one (verified numerically:
  log|det J| -24.624688881 for both, diff 0.00e+00). A count factor changes the
  observation space itself; the density must be re-derived, not asserted. The
  docstring already records what happens when this is skipped: scoring a vM at
  every annotation gave an unnormalised Z = 0.013 drifting 3.6 nats with the
  model's own rate.
- What form? Poisson on wraps-in-window is the obvious first guess but the wraps
  are not independent given phi -- they are a deterministic function of it.
- Interaction with the anneal. kappa_place anneals 3 -> 300; a count term with a
  fixed weight would dominate early and vanish late, or vice versa.

## Status of the surrounding work at parking time

WORKING and verified: mean-phase coordinate (drift priced 0.35 -> 44 nats,
Jacobian unchanged), surgical lift (repair gradient to zdec 82,484 with the rate
gradient held at the mainline -109.1, vs +14,906 and sign-flipped under
place_attach). hjdb 3/4 seeds at F 1.000.

NOT addressed by any of it: the residual snapshot flap (seed 0 oscillates
36 <-> 148 ms with CMLt 1.000 throughout), and basin selection on ballroom.

---

# The Bernoulli case, measured 2026-08-15

Written for the "why doesn't the Bernoulli emission work" question. Landscape
measured on an exact ramp (N=20 annotations, 1.667 s bars, T=2250, y widened
+-1 frame as the pipeline builds it), triangle emission, per-event recon.

## The defect: a structural octave asymmetry that tuning cannot remove

  emission b   peak p   cost of x2   cost of x0.5   ratio
      0.97      0.117      0.80         21.78        27x
      1.31      0.156      1.11         29.66        27x
      2.13      0.295      2.02         49.88        25x
      4.02      0.735      6.06        110.71        18x
      8.00      0.993     16.88        306.47        18x
  pos_weight 10 (b=1.31)   7.69        231.45        30x

Doubling the bar rate costs ~1 nat; halving costs ~30. The RATIO survives a 20x
sweep of emission sharpness and a 10x sweep of pos_weight -- sharpening raises
the x2 penalty but raises every other term by the same factor.

MECHANISM: a per-event Bernoulli grades recall and precision on a sparse target.
At 2x rate every annotation is still covered (recall intact) and the only cost is
false positives between them, which are cheap because the per-empty-frame NLL is
small. At 0.5x rate recall is destroyed, which is expensive. So the objective has
a permanent downhill slope toward the 2x harmonic. This is the mechanism behind
the recorded "30 single-song runs, zero retained solutions, endpoints always
harmonics" -- the folklore was right and now it has a number.

For comparison, the interval emission with the mean coordinate: x2 costs 131.64,
x0.5 costs 432.18 -- a 3.3x asymmetry rather than 27x.

## The surprise: the Bernoulli was BETTER on drift

Same probe, mid-window drift of 0.2425 rad:
    BERNOULLI          2.60 nats
    INTERVAL @ t1      0.00 nats   <- the shipped emission, blind
    INTERVAL @ mean    5.92 nats

So the two emissions did not fail as "broken vs working"; they failed on
ORTHOGONAL axes. The interval emission won the octave axis and silently lost the
drift axis, and that second half went unnoticed until 2026-08-15. Only the
mean-phase coordinate makes the interval emission dominate on both (2-100x
stronger than the Bernoulli on every row measured).

Full comparison, cost in nats (higher = more strongly penalised):

  perturbation              BERNOULLI   INTERVAL(mean)
  global rotation 0.05          0.51          0.94
  global rotation 0.2425        4.15         11.49
  global rotation pi/2         28.25        311.09
  mid-window drift 0.2425       2.60          5.92
  drift ramp 0.2425             1.14          2.50
  rate x2 (octave)              1.11        131.64
  rate x0.5 (octave)           29.66        432.18
  rate x1.02                   21.44        203.89

## Is it saveable?

Not by tuning -- the asymmetry is structural. It needs a factor that prices the
NUMBER of downbeats claimed, i.e. exactly the p(N | phi) count term parked above.
Note the connection: that one term addresses the Bernoulli's octave blindness AND
the interval emission's bar-insertion failure on ballroom (n13/12). Same missing
factor, two different emissions -- which is a point in favour of it being the
right missing piece rather than a patch.

## Caveat

This is the OBJECTIVE'S LANDSCAPE on an exact ramp. It explains the 30-run
training record; it does not replace it. Training dynamics add the amortisation
gap, the sampling noise, and the corrugation that the interval.py docstring also
attributes to the Bernoulli (corrugation NOT measured here).
