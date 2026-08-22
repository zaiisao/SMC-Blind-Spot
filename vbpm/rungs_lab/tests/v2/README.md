# tests/v2 — a suite that passes *iff* the implementation is proper and m is really a latent

    /disk4/anaconda3/envs/chart/bin/python -m pytest tests/v2 -q

144 tests, all green today. Nothing under `vbpm/` is implemented yet — the suite runs
against a reference implementation and against deliberately broken ones, which is what
makes it evidence rather than decoration.

North star is the **professor's tutorial** (`VAEBPM_fin.pdf`, held by the user; working
notes recoverable via `git show 43ecf34:docs/professor_tutorial_notes.md`). The old
`SPEC_v2_stage0.md` is *not* authoritative — it came out of a session that was
hallucinating, and the 197 contract tests derived from it have been deleted.

## The model under test

Tutorial §9/§12, Sohn-standard — three parameter sets, not two:

| set | object | sees | role |
|---|---|---|---|
| θ | emission `p_θ(y \| m)` | m | two scalars; bar offset **marginalised** |
| ψ | prior `p_ψ(m \| h)` | h only | **the deployable path** |
| φ | encoder `q_φ(m \| h, y)` | h and y | training only |

`p(y|m) = (1/m) Σ_r Π_i Bern(y_i; π_i)`, with `π_i = sigmoid(α)` on downbeats and
`sigmoid(β)` elsewhere. K ≤ 4, so the ELBO is an exact enumeration — no sampling.

## Why it is shaped this way

The v1 post-mortem lists five failures. **None was a component failure:**

- a meter count produced correctly and consumed correctly — but as an *index* (a SEAM),
- ψ registered but never called, so 50.88% of parameters sat at zero gradient (WIRING),
- a gradient aligned with truth in 0.000 of crops (a SYSTEM property),
- a metric invariant to the error that mattered (MEASUREMENT),
- an emission term present in code and absent from the spec (MODEL).

A suite of per-function tests fed hand-built perfect inputs is structurally blind to all
five. So every check here takes the implementation as a **parameter** and asserts a
property of the *fitted system*.

## The iff, and how each half is earned

**"proper ⟹ passes"** — `test_properties.py` runs all 22 properties against `subject.oracle()`.
A failure there means the *test* is wrong, not the implementation. Equivalent mutants
(below) guard the same direction: they must **survive**.

**"passes ⟹ proper"** — you cannot get this by writing more assertions, because the set of
wrong programs is not enumerable. What you *can* do is fix a set of wrongnesses and prove
each is caught. `test_mutation_registry.py` runs the full property suite against 16 named
corruptions and requires each to die. A surviving mutant is reported **by name**: a legible
hole, not a silent pass.

Every mutant is either a bug v1 actually shipped or a plausible near-miss on the same seam.
`test_registry_covers_the_v1_post_mortem` fails if any shipped failure loses its mutant.

## Layout

| file | role |
|---|---|
| `reference.py` | numpy oracle; **never imports `vbpm`** |
| `test_reference_selftest.py` | 104 tests making the oracle trustworthy *before* it judges anything |
| `subject.py` | the `Stage0` interface + `oracle()`, the known-correct subject |
| `properties.py` | 22 properties of the fitted system, cheap-to-expensive |
| `test_properties.py` | runs them against the subject under `--impl` |
| `mutants.py` | 16 named corruptions + the provably-`EQUIVALENT` set |
| `test_mutation_registry.py` | the "only if" half |
| `conftest.py` | `--impl=oracle\|vbpm`; fixtures for the oracle self-tests |

## Running against the real package

    pytest tests/v2 --impl=vbpm

Fails loudly until `_vbpm_factory()` in `conftest.py` is written. That is deliberate: a
guessed adapter that quietly smooths over an API mismatch is exactly how a suite starts
passing for the wrong reason.

## What this suite deliberately does NOT assert

- **That the amortization gap is zero.** Tutorial Misconception 6: the gap is genuinely
  *present* in the §9 variant, and §6.8.6–7 make it structural — x-only inference at best
  matches the aggregated posterior, which is definitionally broader. Asserting it away
  would fail correct code. What *is* asserted is §6.8.5: ψ converges to the aggregated
  posterior.
- **That m is informative on real music.** The identifiability check found balanced
  accuracy 0.512 vs 0.333 chance, exactly chance on beatles/gtzan/hainsworth, with real
  recovery only on asap and ballroom. A beat-only Bernoulli emission makes the meter latent
  *faithfully* vacuous on non-expressive material. "m carries information" is asserted only
  on synthetic data, where m is identifiable by construction.

Both are cases where the honest thing and the flattering thing differ.

## Two design details worth knowing

**Disjoint vocabulary.** `subject.DISJOINT_VALUES = (5, 7, 11)`: no integer is both a legal
meter and a legal index into `[0, K)`. v1's count/index confusion cannot land on a
plausible neighbour and hide.

**Marginalising the bar offset** makes downbeat *phase* unobservable by construction — so
`downbeat_off_by_one` is provably equivalent to correct code and is listed in
`mutants.EQUIVALENT`. A suite that "kills" it is over-specifying, asserting an
implementation detail the model quotients out.
