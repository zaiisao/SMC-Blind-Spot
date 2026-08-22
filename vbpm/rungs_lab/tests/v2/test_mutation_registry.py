"""The "only if" half: every named wrongness must be caught by at least one property.

``test_properties.py`` shows the suite passes on a correct implementation. That alone is
worthless -- ``assert True`` also passes on a correct implementation. This file supplies the
missing direction by running the same properties against deliberately broken subjects and
requiring each to die.

A surviving mutant is reported BY NAME. That is a hole in the suite, not a pass.

Equivalent mutants (``mutants.EQUIVALENT``) are the opposite control: they are provably
indistinguishable from correct code, so they must SURVIVE. Killing one means a property is
asserting an implementation detail the model deliberately quotients out, which breaks the
other half of the iff.
"""
from __future__ import annotations

import pytest

import mutants as M
import properties as P


def _first_kill(make):
    """Run the properties cheap-first and stop at the first that fires."""
    for name, fn in P.CHECKS.items():          # dict order == definition order == cheap first
        try:
            fn(make)
        except AssertionError as exc:
            return name, str(exc).split("\n")[0]
        except Exception as exc:               # a crash is detection too
            return name, f"{type(exc).__name__}: {exc}"
    return None, None


def _factory(mutant_name):
    def make(values=None, capacity=None):
        return M.build(mutant_name, values, capacity or "full")
    return make


@pytest.mark.parametrize("mutant", [m for m in M.MUTANTS if m not in M.EQUIVALENT])
def test_mutant_is_killed(mutant, record_property):
    killer, why = _first_kill(_factory(mutant))
    record_property("killed_by", killer or "")
    assert killer is not None, (
        f"MUTANT SURVIVED: {mutant!r} -- {M.describe(mutant)}\n"
        f"Every property passed on an implementation known to be broken, so passing this "
        f"suite does not imply the implementation is correct. Add a property that "
        f"distinguishes it.")


@pytest.mark.parametrize("mutant", sorted(M.EQUIVALENT))
def test_equivalent_mutant_survives(mutant):
    killer, why = _first_kill(_factory(mutant))
    assert killer is None, (
        f"OVER-SPECIFIED: {mutant!r} is provably equivalent to correct code "
        f"({M.describe(mutant)}), but {killer!r} rejected it:\n  {why}\n"
        f"A correct implementation that made this choice would fail the suite.")


def test_registry_covers_the_v1_post_mortem():
    """Each failure v1 actually shipped must have a mutant standing for it."""
    required = {
        "count_as_index",       # meter count consumed as an index
        "psi_frozen",           # 50.88% of parameters at zero gradient
        "q_is_prior",           # latent dead
        "predict_leaks_y",      # deployable path reading annotations
        "emission_inert",       # latent present but decorative
        "predict_constant",     # collapse the metric was invariant to
    }
    missing = required - set(M.MUTANTS)
    assert not missing, f"no mutant represents: {sorted(missing)}"
