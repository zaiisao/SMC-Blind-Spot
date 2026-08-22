"""Every property in ``properties.py``, run against the known-correct oracle.

These MUST all pass. A failure here means the TEST is wrong, not the implementation --
which is the whole point of having an oracle. The mirror image lives in
``test_mutation_registry.py``: the same properties run against known-BROKEN subjects,
where at least one must fail.

Point the suite at the real package once it exists:

    pytest tests/v2 --impl=vbpm
"""
from __future__ import annotations

import pytest

import properties as P

pytestmark = pytest.mark.oracle


@pytest.mark.parametrize("name", list(P.CHECKS))
def test_property_holds(name, subject_factory):
    try:
        P.CHECKS[name](subject_factory)
    except P.CapacityUnsupported as e:
        # `capacity` is a testing affordance, not a model requirement (SPEC Appendix A).
        # A conformant implementation with one encoder must not FAIL for lacking it -- but
        # it must not silently look covered either, so this skips loudly.
        pytest.skip(str(e))


def test_every_check_is_registered():
    """Guard against a check being defined but never run because @check was forgotten."""
    import inspect

    import properties as mod
    defined = {n for n, f in vars(mod).items()
               if n.startswith("check_") and inspect.isfunction(f)}
    assert defined == set(P.CHECKS), (
        f"defined but unregistered: {sorted(defined - set(P.CHECKS))}; "
        f"registered but undefined: {sorted(set(P.CHECKS) - defined)}")
