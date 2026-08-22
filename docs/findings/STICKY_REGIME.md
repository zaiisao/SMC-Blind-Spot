# Sticky two-regime tempo transition kernel on the R1 DBN chassis

Falsification test. Pre-registered criteria fixed before results were inspected (see bottom).
Scripts: `scripts/experiments/exp_sticky_gate0.py`, `scripts/experiments/exp_sticky_gate0c_bis.py`, `scripts/experiments/exp_sticky_regime.py`.
Raw output: `/tmp/sticky/gate0.json`, `/tmp/sticky/gate0c_bis.json`, `/tmp/sticky/report.json`,
`/tmp/sticky/per_track.jsonl`.

## Model

Each physical tempo interval appears twice on the grid, once per regime
(0 = steady, `lambda_stiff`; 1 = yielding, `lambda_loose`). The beat-boundary kernel is

    K_full[(i,a), (j,b)] = log P(s=b | s=a) + log K_{lambda_b}(j | i),
    regime matrix R = [[1-eps, eps], [eta, 1-eta]].

Chassis: `min_bpm=30, max_bpm=215, observation_lambda=6, fps=50`, `beats_per_bar=(3,4)`,
`num_tempi=None`. V = 87 integer intervals (14..100 frames), base `num_states` = 14877 per meter,
duplicated 29754.

### Duplicated state space is structurally valid (all checks True)

Every array in `BarPointerStateSpace.__init__` is built by concatenation over `interval_frames`, so
repeated interval values are structurally fine. Verified explicitly:

| check | result |
|---|---|
| `num_states` doubles (14877 -> 29754) | True |
| interval grid is the base grid tiled twice | True |
| beat-0 regime-0 `first_states` == base `first_states[0]` | True |
| beat-0 regime-0 `last_states` == base `last_states[0]` | True |
| per-beat offsets == `beat * num_states_per_beat` | True |
| every tempo block has exactly `interval` states | True |
| `state_positions` / `position_classes` identical on block 0 | True |

(Per-beat *offsets* necessarily differ from the base space because `num_states_per_beat` doubles;
that is not a violation. The read-out reads `state_positions` / `state_interval_frames`, which are
identical for the two copies of a tempo, so beats decode identically regardless of regime.)

## GATE 0 — is the probability term valid, and can it escape the theorem?

Pure kernel algebra, no decoding. All 14 (pair, eps) cells.

### 0a. Validity — PASS

| quantity | worst value over all 14 cells |
|---|---|
| max abs(row sum - 1) of the [2V, 2V] joint kernel (probability space) | 1.332e-15 |
| max abs(row sum - 1) of `K_stiff` / `K_loose` | <= 8.9e-16 |
| max abs(row sum - 1) of the regime matrix R | 0.0 (exact) |
| total mass of the tiled uniform initial over the 2V grid | 1.0 (exact) |
| stationarity residual, max abs(pi R - pi) | 0.0 (exact) |

No probability mass leaks. The construction is a proper conditional distribution.

### 0b. Single-step marginal — the theorem DOES apply. CONFIRMED, as the coordinator predicted.

Summing the regime out of one beat-boundary transition at the stationary regime distribution
pi = (eta/(eps+eta), eps/(eps+eta)) reproduces the memoryless mixture at w = pi_0 **exactly**:
max abs difference over the whole [V, V] kernel is 8.9e-16 (pair 100/1) and 7.1e-15 (pair 200/5),
i.e. float64 round-off. With eps = eta, pi = (0.5, 0.5) for every cell in the grid.

Effective decay `lam_eff(d) = -d/dd log K(d)` of that single-step marginal (i0 = 50 frames = 60 BPM):

| pair | d=0.02 | d=0.1 | d=0.3 | d=0.6 | min(lambda) |
|---|---|---|---|---|---|
| (100, 1) | 85.9913537218926 | 1.217793545116308 | 1.0000000005838672 | 1.000000000139778 | 1.0 |
| (200, 5) | 59.13259421141959 | 5.000012580413937 | 4.999999999810711 | 5.000000000254801 | 5.0 |

The tail decays at exactly `min(lambda_stiff, lambda_loose)`. **The one-step marginal has no
escape whatsoever** — it is the already-refuted mixture, term for term.

### 0c. Path-level escape — REAL, and it is the only escape there is

Two probes, both holding the total tempo change and the path length fixed.

**Probe 1 (sustained vs isolated, total change D over n boundaries).** log P(sustained) - log P(isolated),
in nats, and the sticky-minus-mixture delta. D = 0.5, i0 = 50 frames, pair (100, 1):

| eps | n=1 | n=2 | n=4 | n=8 |
|---|---|---|---|---|
| mixture (any eps, w=0.5) | 0.0 | -3.564631 | -10.754473 | -23.050026 |
| sticky eps=0.5 | 0.0 | -3.564631 | -10.754473 | -23.050026 |
| sticky eps=0.2 | 0.0 | -2.254211 | -9.328282 | -23.364334 |
| sticky eps=0.1 | 0.0 | -1.558197 | -8.512899 | -22.741195 |
| sticky eps=0.03 | 0.0 | -0.689185 | -7.232270 | -21.529841 |
| sticky eps=0.01 | 0.0 | -0.243484 | -6.114002 | -20.422877 |
| sticky eps=0.003 | 0.0 | -0.031241 | -4.907254 | -19.215431 |
| sticky eps=1e-9 | 0.0 | +0.074476 | +0.157737 | -4.309825 |

At eps = 0.5 the sticky row is **bit-identical** to the mixture row (delta = 0.000000) — an
independent confirmation of the memoryless limit. The delta grows for n = 2 -> 4 at every interior
eps, but is non-monotone at n = 8, because in this probe the *isolated* path also pays only one
entry toll and then sits in the loose regime for free. That contamination motivated probe 2.

**Probe 2 (clustered vs interleaved — the clean price-structure probe).** Same *number* of loose
moves (n moves of `step` frames), same endpoint, same path length (2n states); the only difference
is whether the moves are consecutive or separated by a stay at r = 1. A memoryless model must be
indifferent. pair (100, 1), step = 4 frames, log P(clustered) - log P(interleaved) in nats:

| eps | mixture n=2 | sticky n=2 | delta n=2 | delta n=3 | delta n=4 |
|---|---|---|---|---|---|
| 0.5 | -0.043576 | -0.043576 | -0.000000 | +0.000000 | +0.000000 |
| 0.2 | -0.043576 | +1.020522 | +1.064097 | +3.196829 | +5.241339 |
| 0.1 | -0.043576 | +1.176442 | +1.220018 | +4.039255 | +6.739537 |
| 0.03 | -0.043576 | +0.725472 | +0.769047 | +3.822562 | +7.121235 |
| 0.01 | -0.043576 | +0.311958 | +0.355533 | +2.943675 | +6.451992 |
| 0.003 | -0.043576 | +0.090001 | +0.133576 | +1.900883 | +5.382296 |
| 1e-9 | -0.043576 | -0.025680 | +0.017895 | +0.065170 | +0.142553 |

pair (200, 5), step = 4: delta n=2/3/4 = +0.977511 / +3.082971 / +5.312372 at eps=0.2;
+0.940872 / +3.481910 / +6.327406 at eps=0.1; +0.436945 / +2.659820 / +5.684017 at eps=0.03;
+0.143133 / +1.690129 / +4.623144 at eps=0.01; +0.012429 / +0.805266 / +3.436047 at eps=0.003;
-0.049424 / -0.129113 / -0.218557 at eps=1e-9.

The mixture's clustered-vs-interleaved difference is a constant -0.043576 nats independent of n
(pure integer-grid quantization; the memoryless model is indifferent to arrangement, as it must be).
The sticky delta **grows monotonically in n** at every interior eps, and vanishes exactly at both
endpoints (0.000000 at eps=0.5; +0.14 at eps=1e-9, where the loose regime is unreachable).

**GATE 0 VERDICT: PASS, with the coordinator's correction upheld.** The one-step marginal is
exactly the refuted mixture and its tail decays at min(lambda) (0b). The escape is purely
path-level and it is genuine: correlated (clustered) yielding is up to +7.12 nats cheaper relative
to interleaved yielding than any memoryless model can make it, and the advantage is maximised at
an INTERIOR eps (approximately 0.1 at n=2..3, 0.03 at n=4), not at either endpoint. Proceeding to
Gates A/B and the sweep.

## GATE A / GATE B / SWEEP

(pending)
