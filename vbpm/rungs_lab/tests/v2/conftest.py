"""Suite wiring. One knob: which implementation the properties are run against."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import reference as R      # noqa: E402
import subject as S        # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--impl", action="store", default="oracle", choices=("oracle", "vbpm"),
        help="which Stage-0 implementation the properties run against "
             "(default: the reference oracle in subject.py)")


def pytest_configure(config):
    for line in (
        "oracle: runs against the known-correct reference; must always pass",
        "mutation: runs against known-broken subjects; each must be caught",
        "gap: settles a question the spec text gets wrong or leaves open",
    ):
        config.addinivalue_line("markers", line)


def pytest_report_header(config):
    impl = config.getoption("--impl")
    present = importlib.util.find_spec("vbpm") is not None
    return [f"stage-0 subject under test: {impl}   "
            f"(vbpm package: {'present' if present else 'NOT present'})"]


def _vbpm_factory():
    """Adapter from the real package onto the Stage0 interface.

    Thin on purpose (SS9): ``vbpm.Stage0`` implements Appendix A directly. ``capacity``
    is not part of the model (SS4.5) and is ignored; the two checks that need a zero-gap
    q skip via CapacityUnsupported.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
    import vbpm

    def make(values=None, capacity=None):
        return vbpm.Stage0(tuple(values) if values is not None else R.DEFAULT_VALUES)

    return make


@pytest.fixture(scope="session")
def subject_factory(request):
    """``(values=None, capacity=None) -> Stage0`` -- the implementation under test.

    ``capacity`` is a testing affordance only (SPEC SS4.5); an implementation may ignore it.
    """
    if request.config.getoption("--impl") == "vbpm":
        return _vbpm_factory()

    def make(values=None, capacity=None):
        return S.oracle(tuple(values) if values is not None else R.DEFAULT_VALUES,
                        capacity=capacity or "full")
    return make


# ------------------------------------------------------------------------------------
# fixtures for the oracle self-tests (test_reference_selftest.py)
#
# These serve reference.py ONLY -- plain numpy arguments, never a subject. The oracle has
# to be trustworthy before it can judge anything, so its tests must not depend on the
# implementation under test.
# ------------------------------------------------------------------------------------
@pytest.fixture
def values():
    """The Stage-0 meter alphabet. A COUNT vocabulary (C1), not indices."""
    return R.DEFAULT_VALUES


@pytest.fixture
def rng():
    """Seeded per test: a self-test that only passes on a lucky draw is not a self-test."""
    return np.random.default_rng(20260730)


@pytest.fixture
def theta():
    """Emission parameters as kwargs, so tests read ``**theta``.

    Deliberately NOT saturated: alpha/beta near the decision boundary keep the ELBO
    identities being checked here non-degenerate. ``saturated`` behaviour is covered
    separately by ``test_saturated_emission_identifies_the_true_meter``.
    """
    return {"alpha": 1.5, "beta": -1.0}


@pytest.fixture
def clean_song():
    """One noise-free synthetic song, for the alignment guard."""
    return R.make_synth_song(4, 2.0, 6, rng=np.random.default_rng(7))
