"""Properties of the FITTED SYSTEM, not of functions given perfect inputs.

Every check takes ``make``, a factory ``(values=...) -> Stage0``, so the same
body runs against the oracle, against each mutant, and later against ``vbpm``. Checks raise
AssertionError; ``test_properties.py`` wraps them for pytest and
``test_mutation_registry.py`` uses them as the kill criterion.

Ordered cheap-to-expensive: the registry short-circuits on the first kill, so putting the
fast structural checks first keeps the mutation sweep to seconds rather than minutes.

Two things this file deliberately does NOT assert
-------------------------------------------------
* **That the amortization gap is zero.** The tutorial's Misconception 6 says the gap is
  genuinely present in the SS9 variant. Asserting it away would fail correct code.
* **That m is informative on real music.** The identifiability check found balanced accuracy
  0.512 vs 0.333 chance, near-chance on beatles/gtzan/hainsworth, with real recovery only on
  asap and ballroom -- a beat-only Bernoulli emission makes the meter latent faithfully
  vacuous on non-expressive material. "m carries information" is asserted ONLY on synthetic
  data where m is identifiable by construction.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import reference as R
import subject as S

CHECKS: dict = {}


def check(fn):
    CHECKS[fn.__name__] = fn
    return fn


# Small on purpose: the mutation sweep runs these many times over.
N_PER_CLASS = 3
N_BEATS = (24, 36)      # target beat count; n_bars is derived per song (see make_synth_dataset)
FIT_STEPS = 500
FIT_LR = 0.5
# NOT free parameters -- 250/0.1 was calibrated to the oracle, whose hand-built per-class
# features already contain the answer so it converges almost immediately. Measured mean
# balanced accuracy over 5 seeds, oracle vs a SPEC-4.4-shaped mean(+)max prior:
#
#     steps/lr   oracle   spec-4.4   sec/fit
#      250/0.1    1.000     0.656      3.1     <- old default: spec-conformant prior FAILS
#      250/0.5    1.000     0.889      2.8         the 0.75 gate while being perfectly
#      500/0.5    1.000     0.956      5.6     <- capable. Undertrained, not underpowered.
#      800/0.3    1.000     1.000      9.4
#
# 500/0.5 buys 0.206 of margin over the gate at 2x fit cost; 800/0.3 would triple it. The
# lesson generalises past these two numbers: a threshold on a TRAINED quantity measures the
# optimiser as much as the model, so any budget tuned against one implementation silently
# becomes part of the contract.


def _data(values, n_per_class=N_PER_CLASS, seed=0, **kw):
    """The bench. ``n_bars`` is NOT fixed -- see ``R.make_synth_dataset``.

    Fixing it makes ``n_beats = n_bars * m``, so beat count alone identifies the meter and a
    downbeat-blind classifier scores 1.000. Every property here would then pass on a model
    that never looks at ``y``'s structure.
    """
    kw.setdefault("n_beats", N_BEATS)
    return R.make_synth_dataset(n_per_class, values=tuple(values), seed=seed, **kw)


def _tv(logp, logq) -> float:
    """Total-variation distance between two distributions given as log-probs."""
    p = np.exp(np.asarray(logp, dtype=np.float64))
    q = np.exp(np.asarray(logq, dtype=np.float64))
    return 0.5 * float(np.abs(p - q).sum())


def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)


def _randomise(s, group: str, seed: int, scale: float = 1.0):
    """Perturb one SS4.7 parameter group off its (degenerate) initialisation.

    Shape-agnostic ON PURPOSE. Properties need a non-degenerate model to probe -- at init
    ``w_prior`` is zero, so a correct prior is constant in h and any dependence check would
    fail on correct code. Meeting that need by writing named tensors of hardcoded shape is
    what bound the suite to ``subject.oracle``: this file's psi is a [3] weight vector over
    per-class features, SPEC SS4.4's is a [K,2D]=[3,4] matrix, and an implementation that
    follows the spec CRASHES rather than fails. Going through ``param_groups()`` and
    ``p.shape`` works for either.

    theta is exempt: SS4.3 fixes it as two named scalars, so tests may set alpha/beta directly.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in s.param_groups()[group].values():
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * scale)


class CapacityUnsupported(Exception):
    """Raised when a check needs an encoder capacity the subject does not provide.

    ``capacity`` is a TESTING affordance, not part of the model (SPEC Appendix A). A real
    implementation exposes ONE encoder; only the oracle can be built with a deliberately
    exact or deliberately crippled q. Checks that genuinely require one of those are
    oracle-coverage, and must announce themselves as not-run rather than fail a conformant
    implementation for lacking a knob the spec does not ask for.
    """


def _require_capacity(s, want: str):
    """Assert the subject honoured a capacity request; raise CapacityUnsupported if not."""
    if getattr(s, "capacity", None) != want:
        raise CapacityUnsupported(
            f"subject does not support capacity={want!r} (got "
            f"{getattr(s, 'capacity', None)!r}); this check is oracle-coverage only")


def _prior_varies_with_h(s, songs) -> float:
    """Largest TV between prior distributions over a set of songs. 0 => constant in h."""
    lp = [_np(s.prior_logp(sg["h"])) for sg in songs]
    return max(_tv(lp[0], q) for q in lp[1:]) if len(lp) > 1 else 0.0


def _predict_values(s, songs) -> list:
    return [s.values[int(s.predict(sg["h"]).argmax())] for sg in songs]


# ======================================================================================
# 1. structural -- cheap, no fitting
# ======================================================================================
@check
def check_meter_count_index_roundtrip(make):
    """C1: a meter is a COUNT. The count<->index map must be a bijection.

    Run on values disjoint from any legal index, so a confusion cannot land on a
    plausible neighbour and hide -- which is exactly how v1's bug survived.
    """
    s = make(values=S.DISJOINT_VALUES)
    for k, m in enumerate(s.values):
        assert s.to_idx(m) == k, (
            f"to_idx({m}) == {s.to_idx(m)}, expected index {k}. A beats-per-bar COUNT is "
            f"being treated as an index.")
        assert s.to_value(k) == m
    idxs = [s.to_idx(m) for m in s.values]
    assert len(set(idxs)) == len(s.values), (
        f"to_idx is not injective: {dict(zip(s.values, idxs))} -- distinct meters collapse "
        f"into one class, which is what merged bpb=3 (20 songs) with bpb=4 (119) in v1.")


@check
def check_emission_normalises_over_y(make):
    """sum over all y in {0,1}^n of p(y|m) == 1, for every m. Brute force, n <= 6.

    A likelihood that does not normalise is not a likelihood, and the ELBO built on it
    bounds nothing.
    """
    s = make()
    for n in range(1, 7):
        totals = np.zeros(len(s.values))
        for y in R.all_binary(n):
            totals += np.exp(_np(s.emission_logp_all(y)))
        for k, m in enumerate(s.values):
            assert abs(totals[k] - 1.0) < 1e-9, (
                f"sum_y p(y|m={m}) == {totals[k]:.6f} != 1 at n={n}")


@check
def check_emission_matches_independent_oracle(make):
    """Agreement with reference.py on RANDOM y -- off the model's own manifold.

    The old suite only ever fed the emission a perfect ``y[::m] = 1`` pattern, so it could
    not distinguish a correct implementation from one that is correct only on data drawn
    from its own generative process.
    """
    g = np.random.default_rng(7)
    for trial in range(12):
        alpha, beta = float(g.normal(0, 2)), float(g.normal(0, 2))
        s = make()
        with torch.no_grad():
            s.alpha.fill_(alpha)
            s.beta.fill_(beta)
        n = int(g.integers(3, 18))
        y = g.integers(0, 2, n).astype(np.uint8)   # arbitrary, not periodic
        got = _np(s.emission_logp_all(y))
        for k, m in enumerate(s.values):
            want = R.emission_logp(y, m, alpha, beta)
            assert abs(got[k] - want) < 1e-8, (
                f"trial {trial}: logp_all[{k}] (m={m}) == {got[k]:.9f}, oracle {want:.9f}")


@check
def check_offset_is_marginalised_not_maximised(make):
    """(1/m) sum_r, not max_r. The two differ by up to log(m)."""
    s = make()
    with torch.no_grad():
        s.alpha.fill_(3.0)
        s.beta.fill_(-3.0)
    y = np.zeros(12, dtype=np.uint8)
    y[::3] = 1
    m = s.values[1]
    got = float(_np(s.emission_logp_all(y))[1])
    offs = R.offset_logps(y, m, 3.0, -3.0)
    assert abs(got - (R.logsumexp(offs) - math.log(m))) < 1e-8, "not the marginal over r"
    assert abs(got - max(offs)) > 1e-3, "looks like max_r, not a marginal over r"


@check
def check_emission_prefers_the_generating_period(make):
    """A y with period m must score m above every other candidate.

    The sensitivity half of the invariance/equivariance pair: insensitive to the
    irrelevant (bar offset), sensitive to the relevant (the period).
    """
    s = make()
    with torch.no_grad():
        s.alpha.fill_(4.0)
        s.beta.fill_(-4.0)
    lcm = 1
    for m in s.values:
        lcm = lcm * m // math.gcd(lcm, m)
    n = lcm * 2
    for k, m in enumerate(s.values):
        y = np.zeros(n, dtype=np.uint8)
        y[::m] = 1
        lp = _np(s.emission_logp_all(y))
        assert int(lp.argmax()) == k, (
            f"y with period {m} scored highest at values[{int(lp.argmax())}]="
            f"{s.values[int(lp.argmax())]}, not {m}. logp_all={lp.round(3)}")


@check
def check_shift_invariance_when_m_divides_n(make):
    """Marginalising the bar offset makes logp invariant to ANY cyclic shift when m | n.

    (The spec's T-EM-2 says "a shift by m", which is false unless m | n; this is the
    stronger true form.)

    Kills ``offset_fixed_at_zero`` (pinning the offset at r=0 makes the emission depend on
    which beat happens to sit at index 0, which is exactly what a cyclic shift moves), but
    is never the FIRST to fire: ``check_emission_matches_independent_oracle`` catches any
    pure-emission defect earlier, since an emission agreeing with the oracle on random y is
    already correct.

    Kept deliberately even so. It states the design property in executable form, and it is
    the check that keeps its teeth if Stage 1 adds emission factors and the direct oracle
    comparison has to be relaxed -- at which point subsumption stops holding.
    """
    s = make()
    with torch.no_grad():
        s.alpha.fill_(2.0)
        s.beta.fill_(-2.0)
    for k, m in enumerate(s.values):
        n = m * 4
        y = np.zeros(n, dtype=np.uint8)
        y[::m] = 1
        base = float(_np(s.emission_logp_all(y))[k])
        for shift in (1, 2, 3, 5, 11):
            got = float(_np(s.emission_logp_all(R.cyclic_shift(y, shift)))[k])
            assert abs(got - base) < 1e-8, (
                f"m={m}: shift {shift} changed logp by {got - base:.3e}")


@check
def check_posterior_is_bayes(make):
    """exact_posterior == normalise(log p(y|m) + log p_psi(m|h)). Prior included."""
    songs = _data(R.DEFAULT_VALUES, n_per_class=1)
    s = make()
    _randomise(s, "psi", seed=21)      # a uniform prior would make "prior included" vacuous
    for sg in songs:
        lik = _np(s.emission_logp_all(sg["y"]))
        prior = _np(s.prior_logp(sg["h"]))
        want = lik + prior
        want = want - R.logsumexp(list(want))
        got = _np(s.exact_posterior(sg["h"], sg["y"]))
        assert np.allclose(got, want, atol=1e-9), (
            f"posterior {np.exp(got).round(4)} != Bayes {np.exp(want).round(4)}; the prior "
            f"is being dropped or double counted")


@check
def check_every_parameter_receives_gradient(make):
    """v1 ran for weeks with 41 tensors / 50.88% of parameters at exactly zero gradient.

    Two traps this has to avoid, both measured:

    ``== 0.0`` is the wrong predicate. A disconnected parameter can still pick up rounding
    noise, and a CONNECTED one can sit at an analytic stationary point: at ``c_lik = 1.0``
    the encoder IS the exact posterior, so ``dELBO/dc_lik = 0`` by the envelope theorem.
    Measured there: ``2.25e-14`` -- the check passed on float dust, not on a live gradient.
    So: perturb off the stationary point first, then compare against a scaled floor.
    """
    s = make()
    songs = _data(s.values, n_per_class=1)
    # move every parameter off any analytic stationary point before reading gradients
    with torch.no_grad():
        for i, p in enumerate(s.named_params().values()):
            p.add_(torch.full_like(p, 0.3 if i % 2 == 0 else -0.2))
    loss = -torch.stack([s.elbo(sg["h"], sg["y"]) for sg in songs]).mean()
    for p in s.named_params().values():
        if p.grad is not None:
            p.grad = None
    loss.backward()
    scale = max((float(p.grad.abs().max()) for p in s.named_params().values()
                 if p.grad is not None), default=0.0)
    floor = max(1e-10 * scale, 1e-12)
    dead, frozen = [], []
    for name, p in s.named_params().items():
        if not p.requires_grad:
            frozen.append(name)
        elif p.grad is None or float(p.grad.abs().max()) <= floor:
            dead.append(name)
    assert not frozen, f"parameters excluded from optimisation entirely: {frozen}"
    assert not dead, (
        f"parameters with no gradient (|grad| <= {floor:.3e}, largest in model {scale:.3e}): "
        f"{dead}")


@check
def check_fit_actually_trains_every_named_parameter(make):
    """SS4.7 governs the objective ``fit()`` DIFFERENTIATES, not the one ``elbo()`` returns.

    ``check_every_parameter_receives_gradient`` builds ``s.elbo(...)`` itself and backprops
    that. Nothing connects it to the training loop, so ``fit()`` can optimise a different
    objective -- or none -- and no property notices. Measured: a ``fit()`` that ascends the
    exact marginal likelihood ``logsumexp(lik + prior)`` instead of the ELBO passes all other
    checks while phi moves EXACTLY 0.0 (``c_lik`` [1.0] -> [1.0]). That deletes variational
    inference and leaves the suite green.

    Works on ANY encoder: phi is displaced before fitting rather than a special capacity
    being selected. Selecting one bound this check to an oracle-only knob for no gain --
    displacement makes the same point and holds for an implementation with a single encoder.

    Only the bit-identity arm is asserted. The natural second arm -- "the amortization gap
    must shrink" -- does not belong on THIS bench: the data is deliberately noise-free and
    periodic, so ``y.mean() = 1/m`` exactly and the gap is ~1e-4 nats whatever phi does. A
    real gap needs hard inference, which is a different data source and a separate question.
    """
    s = make()
    songs = _data(s.values, n_per_class=4, seed=1)
    # Perturb phi OFF its initialisation first. Some encoders start at their own optimum --
    # the oracle's c_lik=1.0 makes q exactly the posterior, where dELBO/dc_lik = 0 by the
    # envelope theorem -- and a frozen encoder is then indistinguishable from a trained one.
    # Displaced, any real fit() must pull it back, so "did not move" means "was not trained".
    _randomise(s, "phi", seed=7)
    before = {n: _np(p).copy() for n, p in s.named_params().items()}
    s.fit(songs, steps=FIT_STEPS, lr=FIT_LR)
    still = [n for n, p in s.named_params().items() if np.array_equal(_np(p), before[n])]
    assert not still, (
        f"parameters bit-identical after fit(): {still}. fit() is not optimising the "
        f"objective these parameters belong to -- phi is untrained scaffolding, and the "
        f"amortization gap is then whatever the initialisation happened to give.")


@check
def check_the_objective_is_exact_not_sampled(make):
    """With K <= 4 the ELBO is a sum over four terms. Sampling it is a defect.

    Estimating an enumerable expectation by Monte Carlo buys nothing and injects gradient
    variance the model does not need. This project has already been burned by that: a whole
    round of seed-to-seed differences turned out to be numerical noise rather than signal,
    which made 17 A/B results vacuous.

    Checked BEHAVIOURALLY -- two evaluations of the same crop must agree to the last bit,
    and two fits from different seeds must land on identical parameters. An AST scan for
    ``torch.randn`` would be weaker: it whitelists spellings, so numpy, a custom RNG or a
    ``.sample()`` behind an alias all slip through. Bit-identity cannot be evaded.
    """
    s = make()
    songs = _data(s.values, n_per_class=1)
    for sg in songs:
        a = float(s.elbo(sg["h"], sg["y"]))
        b = float(s.elbo(sg["h"], sg["y"]))
        assert a == b, (
            f"the ELBO is not a deterministic function of (h, y): {a!r} then {b!r}. "
            f"With |values| = {len(s.values)} it should be an exact enumeration.")

    fits = []
    for seed in (0, 12345):
        f = make()
        f.fit(_data(f.values, n_per_class=1), steps=25, lr=FIT_LR, seed=seed)
        fits.append({k: _np(v).copy() for k, v in f.named_params().items()})
    for name in fits[0]:
        assert np.array_equal(fits[0][name], fits[1][name]), (
            f"fitting is seed-dependent: {name!r} differs between seeds by "
            f"{np.abs(fits[0][name] - fits[1][name]).max():.3e}. Exact enumeration has no "
            f"sampling for a seed to perturb.")


# ======================================================================================
# 2. bound identities -- cheap
# ======================================================================================
@check
def check_elbo_lower_bounds_the_evidence(make):
    """ELBO <= log p(y|h) for ANY q. The defining property of the bound."""
    s = make()      # no capacity: _randomise(phi) makes q suboptimal for ANY encoder
    _randomise(s, "phi", seed=3, scale=2.0)
    _randomise(s, "psi", seed=4)
    songs = _data(s.values, n_per_class=2)
    slack = []
    for sg in songs:
        elbo = float(s.elbo(sg["h"], sg["y"]))
        ev = float(s.log_evidence(sg["h"], sg["y"]))
        assert elbo <= ev + 1e-9, f"ELBO {elbo:.6f} EXCEEDS log evidence {ev:.6f}"
        slack.append(ev - elbo)
    # the bound must actually be SLACK here, or "<=" is being confirmed at equality and the
    # check would pass on an implementation that always returns the evidence
    assert max(slack) > 1e-6, (
        f"q was randomised yet the bound is tight everywhere (max slack {max(slack):.3e}); "
        f"this check is vacuous as configured")


@check
def check_bound_is_tight_exactly_when_q_is_the_posterior(make):
    """With q set to the exact posterior the bound is tight; otherwise it is slack."""
    s_exact = make(capacity="exact")
    _require_capacity(s_exact, "exact")   # needs q to BE the posterior; oracle-only
    for sg in _data(s_exact.values, n_per_class=2):
        elbo = float(s_exact.elbo(sg["h"], sg["y"]))
        ev = float(s_exact.log_evidence(sg["h"], sg["y"]))
        assert abs(elbo - ev) < 1e-8, (
            f"q IS the exact posterior yet the bound is slack by {ev - elbo:.6e}")


@check
def check_slack_equals_the_reverse_kl(make):
    """log p(y|h) - ELBO == KL(q || p(m|y,h)), exactly.

    Note the direction: the spec's SS3.5 defines the diagnostic as the FORWARD KL(p||q),
    which is a different number. This pins the identity that actually holds.
    """
    s = make()      # no capacity: _randomise(phi) makes q suboptimal for ANY encoder
    _randomise(s, "phi", seed=11)
    for sg in _data(s.values, n_per_class=2):
        elbo = float(s.elbo(sg["h"], sg["y"]))
        ev = float(s.log_evidence(sg["h"], sg["y"]))
        q_logp = _np(s.q_logp(sg["h"], sg["y"]))
        post = _np(s.exact_posterior(sg["h"], sg["y"]))
        rev = float((np.exp(q_logp) * (q_logp - post)).sum())
        assert abs((ev - elbo) - rev) < 1e-8, (
            f"slack {ev - elbo:.6f} != KL(q||p) {rev:.6f}")


@check
def check_q_depends_on_both_h_and_y(make):
    """q_phi(m | h, y) must actually be a function of BOTH.

    q that ignores y is a dead latent; q that ignores h is not amortized at all. Both were
    live failure modes in v1.

    Probed at NON-DEGENERATE parameters. At initialisation psi is zero, which makes a correct
    prior constant in h -- a valid starting point, not a bug. Asserting functional dependence
    on an unfitted model would fail correct code.
    """
    s = make()
    songs = _data(s.values, n_per_class=2)
    a, b = songs[0], songs[1]
    _randomise(s, "psi", seed=4)
    # the perturbation must have DONE something: with a random draw rather than hand-picked
    # values, an h-independent prior would silently make the second assertion below vacuous
    assert _prior_varies_with_h(s, songs) > 1e-6, (
        "psi was randomised yet the prior is still constant in h, so the h-dependence probe "
        "below would pass on a q that ignores h")

    same_h_diff_y = _tv(_np(s.q_logp(a["h"], a["y"])), _np(s.q_logp(a["h"], b["y"])))
    assert same_h_diff_y > 1e-6, (
        "holding h fixed and changing y left q unchanged: q ignores the labels, so the "
        "latent is dead")
    diff_h_same_y = _tv(_np(s.q_logp(a["h"], a["y"])), _np(s.q_logp(b["h"], a["y"])))
    assert diff_h_same_y > 1e-6, (
        "holding y fixed and changing h left q unchanged: nothing is amortized")


# ======================================================================================
# 3. leakage -- cheap, and the cheat direction of "m as z"
# ======================================================================================
@check
def check_predict_is_annotation_blind(make):
    """The deployable path is a pure function of h.

    A signature of ``predict(h)`` does NOT prove this: the realistic leak is state stashed
    during training and reused at deployment. So predict is called, the model is then shown
    a completely different y, and predict must return bit-identical output.
    """
    s = make()
    songs = _data(s.values, n_per_class=2)
    h = songs[0]["h"]
    before = _np(s.predict(h)).copy()
    for sg in songs:                       # exercise the training path with many y's
        s.elbo(sg["h"], sg["y"])
    after = _np(s.predict(h))
    assert np.array_equal(before, after), (
        f"predict(h) changed after the model saw annotations: {np.exp(before).round(4)} -> "
        f"{np.exp(after).round(4)}. The deployable path is reading y through hidden state.")

    weird = np.ones_like(songs[0]["y"])
    s.elbo(h, weird)
    assert np.array_equal(before, _np(s.predict(h))), (
        "predict(h) responds to the y last passed to elbo()")


# ======================================================================================
# 4. the latent is real -- requires fitting
# ======================================================================================
@check
def check_posterior_recovers_the_true_meter(make):
    """On clean synthetic data the exact posterior must concentrate on m_true.

    Indexed through ``to_idx``, so a broken count<->index seam shows up here as the model
    being confidently wrong rather than as a silent relabelling.
    """
    s = make()
    songs = _data(s.values)
    s.fit(songs, steps=FIT_STEPS, lr=FIT_LR)
    masses = []
    for sg in songs:
        post = np.exp(_np(s.exact_posterior(sg["h"], sg["y"])))
        masses.append(float(post[s.to_idx(sg["m_true"])]))
    mean_mass = float(np.mean(masses))
    assert mean_mass > 0.85, (
        f"mean posterior mass on the true meter is {mean_mass:.3f} on noise-free synthetic "
        f"data where m is identifiable by construction")


@check
def check_evidence_moves_the_belief_when_h_is_ambiguous(make):
    """I(m ; y | h) = E[ KL( p(m|y,h) || p_psi(m|h) ) ] must be large when h is ambiguous.

    This is the quantitative form of "m is doing work": if observing the labels does not
    move the belief over m, the latent is decorative however well the plumbing is wired.

    It MUST be measured where h is uninformative. On the noise-free bench the deployable
    prior reaches accuracy 1.000 at entropy 0.011 nats, so y is genuinely redundant and a
    correct model scores I ~ 0.002 -- asserting I > 0 there would fail correct code.

    Ambiguity is made STRUCTURAL, not tuned. Every song is given the SAME h, so h carries
    exactly zero information about m and any correct prior must fall back on the marginal --
    uniform, since the bench is balanced. ``y`` is untouched, so the posterior can still
    identify m and the mutual information stays measurable.

    Adding noise to h does NOT work, and the failed attempt is worth recording. A fixed
    ``noise=1.2`` was calibrated to the oracle's peak-count reducer, which noise destroys. A
    SPEC-4.4 mean(+)max reducer averages over T frames, so noise SHIFTS its features rather
    than erasing them: measured prior entropy 0.656 at noise 1.2, and 0.000 at noise 20 --
    more noise made it MORE confident. Escalating cannot fix that, because "noise destroys h"
    is a claim about one reducer. Sharing h is a claim about the data, and holds for any.
    """
    s = make()
    songs = _data(s.values)
    shared = songs[0]["h"]
    songs = [{**sg, "h": shared} for sg in songs]
    s.fit(songs, steps=FIT_STEPS, lr=FIT_LR)
    infos, prior_H = [], []
    for sg in songs:
        post = _np(s.exact_posterior(sg["h"], sg["y"]))
        prior = _np(s.prior_logp(sg["h"]))
        infos.append(float((np.exp(post) * (post - prior)).sum()))
        prior_H.append(-float((np.exp(prior) * prior).sum()))
    mi = float(np.mean(infos))
    assert float(np.mean(prior_H)) > 0.6 * math.log(len(s.values)), (
        f"h is not actually ambiguous here (prior entropy {np.mean(prior_H):.3f}). Every song "
        f"was given an IDENTICAL h, so a correct prior cannot tell them apart and must sit "
        f"near uniform; a confident prior means it is reading something other than h")
    assert mi > 0.5, (
        f"I(m;y|h) = {mi:.4f} nats with an uninformative prior: observing the labels does "
        f"not move the belief over m, so m is not functioning as a latent variable")


@check
def check_severing_the_meter_costs_likelihood(make):
    """Held-out evidence of the fitted model vs a meter-free null (alpha tied to beta).

    The null is nested inside the model, so if m is real the likelihood ratio is positive.
    A model whose m is decorative ties with its own ablation.
    """
    train = _data(R.DEFAULT_VALUES, seed=0)
    held = _data(R.DEFAULT_VALUES, seed=99)

    full = make()
    full.fit(train, steps=FIT_STEPS, lr=FIT_LR)

    null = make()
    null.beta = null.alpha              # meter cannot influence the likelihood
    null.fit(train, steps=FIT_STEPS, lr=FIT_LR)

    def evidence(s):
        return float(np.mean([float(s.log_evidence(sg["h"], sg["y"])) for sg in held]))

    lr_stat = evidence(full) - evidence(null)
    assert lr_stat > 0.5, (
        f"held-out log-evidence gain over the meter-free null is {lr_stat:.4f} nats/crop; "
        f"m is not earning its place in the model")


@check
def check_predictions_are_not_degenerate(make):
    """After fitting on a balanced set the deployed predictor must not answer one class.

    v1's collapse detector read 1.2e7 after training and nobody looked.
    """
    s = make()
    songs = _data(s.values)
    s.fit(songs, steps=FIT_STEPS, lr=FIT_LR)
    preds = _predict_values(s, songs)
    assert R.distinct_predicted(preds) > 1, (
        f"the deployed predictor answered {preds[0]} for all {len(preds)} balanced crops")
    logits = np.stack([_np(s.predict(sg["h"])) for sg in songs])
    ratio = R.collapse_ratio(logits)
    assert ratio < 10.0, (
        f"offset-to-signal ratio {ratio:.3g}: a crop-independent class bias dominates the "
        f"crop-dependent signal")


@check
def check_meter_recovered_with_disjoint_values(make):
    """The whole pipeline on values (5, 7, 11) -- disjoint from every legal index.

    If anything anywhere confuses a count with an index, it cannot silently succeed here.
    """
    s = make(values=S.DISJOINT_VALUES)
    songs = _data(s.values, seed=5)
    s.fit(songs, steps=FIT_STEPS, lr=FIT_LR)
    preds = _predict_values(s, songs)
    true = [sg["m_true"] for sg in songs]
    assert set(preds) <= set(s.values), (
        f"predicted meters {sorted(set(preds))} are not all legal values {s.values}")
    ba = R.balanced_accuracy(true, preds, values=s.values)
    assert ba > 0.6, f"balanced accuracy {ba:.3f} with meter values {s.values}"


# ======================================================================================
# 5. amortization (SS9 / SS6.8.5) -- the expensive tail
# ======================================================================================
@check
def check_psi_matches_the_aggregated_posterior(make):
    """SS6.8.5: psi converges to the AGGREGATED posterior, not to any per-instance one.

    Two crops are given the IDENTICAL h and different y. psi sees only h, so it cannot tell
    them apart; minimising sum_i KL(q_i || p_psi) over p_psi is uniquely solved by the
    arithmetic mean of the q_i. So the fitted deployable predictor must sit BETWEEN the two
    posteriors, near their mean.

    Matching either endpoint instead is the signature of y leaking into the deployable path.
    This is the tutorial's "definitionally broader" argument made into a pass/fail.
    """
    values = R.DEFAULT_VALUES
    a = R.make_synth_song(values[0], 2.0, 6)
    b = R.make_synth_song(values[-1], 2.0, 6)
    shared_h = a["h"]
    pair = [{"h": shared_h, "y": a["y"], "m_true": a["m_true"]},
            {"h": shared_h, "y": b["y"], "m_true": b["m_true"]}]

    s = make(capacity="exact")
    _require_capacity(s, "exact")         # needs a well-defined zero-gap q; oracle-only
    s.fit(pair, steps=600, lr=0.08)

    q_a = _np(s.q_logp(shared_h, a["y"]))
    q_b = _np(s.q_logp(shared_h, b["y"]))
    psi = _np(s.predict(shared_h))
    mean = np.log(0.5 * (np.exp(q_a) + np.exp(q_b)))

    d_mean = _tv(psi, mean)
    d_a, d_b = _tv(psi, q_a), _tv(psi, q_b)
    assert d_mean < 0.5 * min(d_a, d_b), (
        f"psi is at TV {d_mean:.4f} from the aggregated posterior but {d_a:.4f}/{d_b:.4f} "
        f"from the two per-instance posteriors. psi should be the AGGREGATE; being close to "
        f"one endpoint means the deployable path can tell the two crops apart, which only y "
        f"can do.")


@check
def check_shuffled_labels_destroy_deployable_accuracy(make):
    """The negative control. Break the h<->y correspondence and accuracy must fall to chance.

    Run as a PAIR: the intact fit must be well above chance AND the shuffled fit at chance.
    Both high means a shortcut is being measured; both low means the model is simply broken.
    Nothing in the previous suite did this, and it is the cheapest guard against reporting a
    number that owes nothing to the audio.

    Scored on a HELD-OUT split. Scored on the training split the shuffled arm reaches 0.583
    against a chance of 0.333, because with three ratio values and a handful of crops the
    prior can memorise whichever label each ratio happened to be dealt. A control read off
    the training split is not a control.
    """
    values = R.DEFAULT_VALUES
    chance = 1.0 / len(values)
    n_rep = 5

    ba_reals, ba_shufs = [], []
    for rep in range(n_rep):
        train = _data(values, n_per_class=4, seed=1 + 10 * rep)
        held = _data(values, n_per_class=6, seed=77 + 10 * rep)
        held_true = [sg["m_true"] for sg in held]

        intact = make()
        intact.fit(train, steps=FIT_STEPS, lr=FIT_LR)
        ba_reals.append(
            R.balanced_accuracy(held_true, _predict_values(intact, held), values=values))

        # DERANGE the labels by class: every song is re-paired with a y whose meter differs
        # from its own, so ZERO residual h<->m signal survives. A plain permutation of a
        # balanced set leaves 1/K of pairs correctly labelled (measured 25% at the old
        # hard-wired rng(0)), and a competent learner then beats chance *through the genuine
        # relationship* -- the opposite of what this control claims to show.
        rng = np.random.default_rng(1000 + rep)
        deranged = []
        for i, sg in enumerate(train):
            others = [j for j, o in enumerate(train) if o["m_true"] != sg["m_true"]]
            j = int(rng.choice(others))
            deranged.append({"h": sg["h"], "y": train[j]["y"],
                             "m_true": train[j]["m_true"]})
        broken = make()
        broken.fit(deranged, steps=FIT_STEPS, lr=FIT_LR)
        ba_shufs.append(
            R.balanced_accuracy(held_true, _predict_values(broken, held), values=values))

    ba_real, ba_shuf = float(np.mean(ba_reals)), float(np.mean(ba_shufs))

    assert ba_real > 0.75, (
        f"mean balanced accuracy on intact held-out data is only {ba_real:.3f} over {n_rep} "
        f"draws ({[round(v, 3) for v in ba_reals]}); the model does not work, so the "
        f"control below proves nothing")
    assert ba_shuf < chance + 0.20, (
        f"balanced accuracy survives label derangement (mean {ba_shuf:.3f} over {n_rep} "
        f"draws {[round(v, 3) for v in ba_shufs]}, vs chance {chance:.3f}). The deployable "
        f"path is exploiting something other than the h->m relationship.")
