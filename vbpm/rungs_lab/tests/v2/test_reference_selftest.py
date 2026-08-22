"""Self-tests for the oracle (tests/v2/reference.py).

These pass with no implementation present. Their job is to make the oracle trustworthy
BEFORE it is used to judge the implementation -- and, in three places, to settle questions
the spec text gets wrong or leaves open (marked ``gap``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import reference as R

pytestmark = pytest.mark.oracle


# --------------------------------------------------------------------------------------
# numerics
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("x", [-60.0, -8.0, -1e-9, 0.0, 1e-9, 8.0, 60.0])
def test_log_sigmoid_matches_torch(x):
    assert R.log_sigmoid(x) == pytest.approx(F.logsigmoid(torch.tensor(x, dtype=torch.float64)).item(), abs=1e-12)
    assert R.log1m_sigmoid(x) == pytest.approx(F.logsigmoid(torch.tensor(-x, dtype=torch.float64)).item(), abs=1e-12)
    assert R.sigmoid(x) == pytest.approx(torch.sigmoid(torch.tensor(x, dtype=torch.float64)).item(), abs=1e-12)


def test_logsumexp_matches_torch(rng):
    v = rng.normal(0, 30, size=9)
    assert R.logsumexp(v) == pytest.approx(torch.logsumexp(torch.tensor(v), 0).item(), abs=1e-9)


# --------------------------------------------------------------------------------------
# SS4.3 emission -- the oracle's own version of T-EM-1/2/3/4
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("m", [2, 3, 4])
def test_emission_is_a_distribution_over_y(n, m):
    """sum over all y in {0,1}^n of p(y|m) == 1 (brute force). This is T-EM-1's oracle."""
    total = sum(math.exp(R.emission_logp(y, m, 0.7, -1.3)) for y in R.all_binary(n))
    assert total == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("m", [2, 3, 4])
@pytest.mark.parametrize("shift", list(range(12)))
def test_offset_marginalisation_gives_full_shift_invariance(m, shift):
    """With n a multiple of m, p(y|m) is invariant to ANY cyclic shift of y.

    This is the real content of marginalising the bar offset r, and it is a strictly
    stronger statement than the spec's T-EM-2 ("shift by m").
    """
    n = 12
    y = np.zeros(n, dtype=np.uint8)
    y[::m] = 1
    base = R.emission_logp(y, m, 2.0, -2.0)
    assert R.emission_logp(R.cyclic_shift(y, shift), m, 2.0, -2.0) == pytest.approx(base, abs=1e-12)


@pytest.mark.gap
def test_TEM2_as_written_is_false_when_m_does_not_divide_n():
    """Cyclic-shift invariance needs the precondition ``m | n``, which SS4.3 states.

    A cyclic shift by m preserves each beat's residue class mod m only when m divides n;
    otherwise the wrap-around remaps residues and the likelihood changes. Here is a
    concrete counterexample, so the spec sentence gets a precondition rather than the
    implementation getting a bug report from a wrong test.
    """
    n, m = 8, 3
    y = np.zeros(n, dtype=np.uint8)
    y[::m] = 1  # downbeats at 0, 3, 6 -- 8 is not a multiple of 3
    a = R.emission_logp(y, m, 1.5, -1.0)
    b = R.emission_logp(R.cyclic_shift(y, m), m, 1.5, -1.0)
    assert abs(a - b) > 1e-3, "expected the wrap-around to break invariance"


def _torch_emission_logp(y, m, alpha, beta):
    """Independent torch transcription of SS4.3, for autograd cross-checking."""
    la, l1a = F.logsigmoid(alpha), F.logsigmoid(-alpha)
    lb, l1b = F.logsigmoid(beta), F.logsigmoid(-beta)
    terms = []
    for r in range(m):
        acc = torch.zeros((), dtype=torch.float64)
        for i, yi in enumerate(y):
            on_down = ((i - r) % m) == 0
            if on_down:
                acc = acc + (la if yi else l1a)
            else:
                acc = acc + (lb if yi else l1b)
        terms.append(acc)
    return torch.logsumexp(torch.stack(terms), 0) - math.log(m)


@pytest.mark.parametrize("trial", range(12))
def test_analytic_gradient_matches_autograd(trial):
    """The oracle's hand-derived d logp/d(alpha,beta) is right. Backs T-EM-3."""
    g = np.random.default_rng(trial)
    n, m = int(g.integers(4, 14)), int(g.choice([2, 3, 4]))
    y = g.integers(0, 2, n).astype(np.uint8)
    al = torch.tensor(float(g.normal(0, 2)), dtype=torch.float64, requires_grad=True)
    be = torch.tensor(float(g.normal(0, 2)), dtype=torch.float64, requires_grad=True)
    _torch_emission_logp(y, m, al, be).backward()
    ga, gb = R.emission_grad(y, m, al.item(), be.item())
    assert ga == pytest.approx(al.grad.item(), abs=1e-9)
    assert gb == pytest.approx(be.grad.item(), abs=1e-9)


@pytest.mark.parametrize("m", [2, 3, 4])
@pytest.mark.parametrize("n_bars", [3, 5, 8])
def test_saturated_emission_identifies_the_true_meter(m, n_bars, values):
    """Backs T-EM-4: alpha -> +inf, beta -> -inf makes argmax_m the true meter."""
    y = np.zeros(n_bars * m, dtype=np.uint8)
    y[::m] = 1
    lp = R.emission_logp_all(y, 20.0, -20.0, values)
    assert values[int(np.argmax(lp))] == m


# --------------------------------------------------------------------------------------
# SS4.6 objective
# --------------------------------------------------------------------------------------
def test_elbo_identity_and_nonnegative_gap(theta, values, rng):
    y = np.zeros(12, dtype=np.uint8)
    y[::3] = 1
    prior = np.log(np.array([0.2, 0.3, 0.5]))
    for _ in range(20):
        q = rng.dirichlet(np.ones(len(values)))
        t = R.elbo_terms(np.log(q), y, prior, **theta, values=values)
        assert t["elbo"] == pytest.approx(t["recon"] - t["kl"], abs=1e-12)
        assert t["gap"] >= -1e-12
        assert t["elbo"] <= t["log_evidence"] + 1e-9  # ELBO is a bound


def test_exact_posterior_zeroes_the_gap_and_attains_the_evidence(theta, values):
    """Backs T-ELBO-3."""
    y = np.zeros(12, dtype=np.uint8)
    y[::3] = 1
    prior = np.log(np.array([0.2, 0.3, 0.5]))
    ex = R.exact_posterior(y, prior, **theta, values=values)
    t = R.elbo_terms(ex, y, prior, **theta, values=values)
    assert t["gap"] == pytest.approx(0.0, abs=1e-12)
    assert t["elbo"] == pytest.approx(t["log_evidence"], abs=1e-12)
    assert R.logsumexp(ex) == pytest.approx(0.0, abs=1e-12)


def test_elbo_slack_is_the_reverse_kl(theta, values, rng):
    """log p(y|h) - ELBO == KL(q || p_exact), the REVERSE KL."""
    y = np.zeros(9, dtype=np.uint8)
    y[::3] = 1
    prior = np.log(np.array([0.3, 0.4, 0.3]))
    for _ in range(20):
        q = np.log(rng.dirichlet(np.ones(3)))
        t = R.elbo_terms(q, y, prior, **theta, values=values)
        assert t["gap_reverse"] == pytest.approx(t["log_evidence"] - t["elbo"], abs=1e-10)
        assert t["gap_reverse"] >= -1e-12


@pytest.mark.gap
def test_spec_gap_is_not_the_elbo_gap(theta, values, rng):
    """The two KL directions are different numbers, and only one is the bound's slack.

    The bound's slack is KL(q || p). Both vanish together and both are legitimate
    amortization diagnostics, but they are different numbers -- here by a factor of ~4 --
    The UNTRUSTED SPEC_v2_stage0.md defined the diagnostic as the forward KL and read "gap
    large" as a threshold; docs/SPEC.md SS4.6 now states the reverse KL correctly. This test
    survives as the pin on that distinction,
    since the mass-covering (forward) and mode-seeking (reverse) directions fail
    differently exactly when q is bad, which is when the diagnostic is consulted.
    """
    y = np.zeros(9, dtype=np.uint8)
    y[::3] = 1
    prior = np.log(np.array([0.3, 0.4, 0.3]))
    q = np.log(np.array([0.80, 0.15, 0.05]))  # confidently wrong
    t = R.elbo_terms(q, y, prior, **theta, values=values)
    assert t["gap"] > 0 and t["gap_reverse"] > 0
    assert abs(t["gap"] - t["gap_reverse"]) > 1.0, "the two directions differ materially"


# --------------------------------------------------------------------------------------
# SS6.2 m_true
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("m", [2, 3, 4])
def test_m_true_recovers_the_generating_meter(m):
    song = R.make_synth_song(m, 2.4, 8, rng=np.random.default_rng(3))
    assert R.derive_m_true(song["beats_s"], song["downs_s"]) == m


def test_m_true_rejects_too_few_bars():
    song = R.make_synth_song(4, 2.0, 8, rng=np.random.default_rng(4))
    # only 3 downbeats -> 2 complete bars < MIN_BARS
    assert R.derive_m_true(song["beats_s"], song["downs_s"][:3]) is None
    assert R.derive_m_true(song["beats_s"], song["downs_s"][:4]) == 4


@pytest.mark.gap
def test_m_true_even_bar_count_tiebreak_is_a_choice_not_a_spec():
    """SS6.2's half-up tie-break: the median over an even number of bars can land on .5.

    Bars of 3 and 4 beats in equal measure -> median 3.5. The spec says "median" and
    C1 says the result must be in MeterSpec.values, which 3.5 is not. This reference
    rounds half up; the implementation must match or the spec must say something else.
    """
    beats, downs, t = [], [], 0.0
    for k, bpb in enumerate([3, 4, 3, 4]):
        downs.append(t)
        for _ in range(bpb):
            beats.append(t)
            t += 0.5
    downs.append(t)
    assert R.derive_m_true(np.array(beats), np.array(downs)) == 4  # 3.5 rounded half up


# --------------------------------------------------------------------------------------
# SS6.4 bench and SS8 baselines
# --------------------------------------------------------------------------------------
def test_synth_downbeats_are_beats_and_y_matches(values):
    for m in values:
        s = R.make_synth_song(m, 2.2, 6, rng=np.random.default_rng(5))
        assert set(np.round(s["downs_s"], 9)).issubset(set(np.round(s["beats_s"], 9)))
        assert len(s["y"]) == len(s["beats_s"])
        assert s["y"].sum() == len(s["downs_s"])


def test_peak_count_baseline_is_perfect_on_noise_free_synth(values):
    """SS6.4 hard requirement: the peak-count baseline must reach balanced accuracy 1.000."""
    songs = R.make_synth_dataset(8, seed=11)
    true = [s["m_true"] for s in songs]
    pred = [R.peak_count_estimate(s["h"], values) for s in songs]
    assert R.balanced_accuracy(true, pred, values) == 1.0


def test_bench_tempo_carries_no_meter_information():
    """SS6.4: the observable tempo is the BEAT rate, and it must not encode m.

    This previously asserted ``corr(m, bar_s) ~ 0``, which is the wrong quantity: nothing
    observes a bar without first finding downbeats. Because ``beat_period = bar_s / m`` is an
    identity, holding ``bar_s`` independent of ``m`` forced ``corr(m, beat_rate) = +0.80``,
    and a downbeat-BLIND classifier could read the meter straight off tempo. The bench now
    draws the beat period independently instead; ``bar_s`` consequently DOES scale with m,
    which is correct and is only readable via downbeat spacing.
    """
    songs = R.make_synth_dataset(40, seed=12)
    m = np.array([s["m_true"] for s in songs], dtype=float)
    beat_rate = np.array([len(s["y"]) / (s["h"].shape[0] / R.DEFAULT_FPS) for s in songs])
    assert abs(float(np.corrcoef(m, beat_rate)[0, 1])) < 0.10


def test_bench_songs_are_all_distinct():
    """v1's bench emitted 160 bit-identical songs; effective n was 1."""
    songs = R.make_synth_dataset(10, seed=13)
    digests = {s["h"].tobytes() for s in songs}
    assert len(digests) == len(songs)


def test_alignment_z_separates_aligned_from_shifted(clean_song):
    """SS6.3 guard: aligned crop passes ALIGN_Z_MIN, a shifted one does not."""
    h, beats = clean_song["h"], clean_song["beats_s"]
    z_ok = R.alignment_z(h, beats)
    z_bad = R.alignment_z(h, beats + 0.37)  # shift by ~a third of a beat
    assert z_ok > R.ALIGN_Z_MIN
    assert z_bad < z_ok


# --------------------------------------------------------------------------------------
# SS8 / SS8 metrics
# --------------------------------------------------------------------------------------
def test_balanced_accuracy_exposes_the_majority_shortcut(values):
    """SS8's motivating case: predicting one class scores high raw, chance balanced."""
    true = [4] * 76 + [3] * 20 + [2] * 4
    pred = [4] * 100
    raw = float(np.mean(np.array(true) == np.array(pred)))
    assert raw == pytest.approx(0.76)
    assert R.balanced_accuracy(true, pred, values) == pytest.approx(1.0 / 3.0)
    assert R.distinct_predicted(pred) == 1


def test_balanced_accuracy_hand_cases(values):
    assert R.balanced_accuracy([2, 3, 4], [2, 3, 4], values) == 1.0
    assert R.balanced_accuracy([2, 3, 4], [3, 4, 2], values) == 0.0
    # ignores classes absent from the truth
    assert R.balanced_accuracy([4, 4], [4, 3], values) == pytest.approx(0.5)


def test_confusion_rows_sum_to_class_counts(values):
    true = [2, 3, 3, 4, 4, 4]
    pred = [3, 3, 4, 4, 4, 2]
    C = R.confusion(true, pred, values)
    assert C.sum() == len(true)
    assert list(C.sum(axis=1)) == [1, 2, 3]


def test_collapse_ratio_flags_a_constant_predictor_and_clears_a_real_one(rng):
    """SS8 offset-to-signal: huge for a crop-independent logit pattern."""
    B, K = 64, 3
    collapsed = np.tile(np.array([0.0, 0.0, 9.0]), (B, 1)) + rng.normal(0, 1e-4, (B, K))
    informative = rng.normal(0, 2.0, (B, K))
    assert R.collapse_ratio(collapsed) > 100.0
    assert R.collapse_ratio(informative) < 1.0


def test_nll_true_is_positive_and_minimised_at_the_truth(values):
    logp = np.log(np.array([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]))
    assert R.nll_true(logp, [4, 2], values) == pytest.approx(-math.log(0.8))
    assert R.nll_true(logp, [2, 4], values) > R.nll_true(logp, [4, 2], values)


def test_majority_predict(values):
    assert R.majority_predict([4] * 10 + [3] * 3, values) == 4
    assert R.majority_predict([3] * 5 + [2], values) == 3
