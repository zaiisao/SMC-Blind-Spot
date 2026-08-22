"""Named corruptions of a known-correct subject -- the machinery behind "only if".

You cannot get "the tests pass ONLY IF the implementation is proper" by writing more
assertions, because the set of wrong programs is not enumerable. What you CAN do is fix a
set of wrongnesses you care about, and prove mechanically that each one is caught. That is
what this file is: every entry is either a bug v1 actually shipped or a plausible
near-miss on the same seam.

``test_mutation_registry.py`` runs the whole property suite against each mutant and
asserts at least one test dies. **A surviving mutant is a named hole in the suite** -- it
is reported by name rather than silently tolerated, so coverage of wrongness is legible
instead of assumed.

Adding a bug found in the wild here is how the suite gets stronger over time.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import subject as S


# ------------------------------------------------------------------------------------
# the corruptions
# ------------------------------------------------------------------------------------
def _count_as_index(s, m):
    """v1's ACTUAL bug, transplanted onto the count<->index seam.

    ``pm_common.py:61`` returned ``int(max(2, min(round(bpb), 4)))`` -- a COUNT.
    ``wire_test.py:383`` did ``tr["m"][sel].long().clamp(0, K - 1)`` -- consuming it as an
    INDEX. On the real distribution {2: 8, 3: 20, 4: 119} that merged bpb=3 and bpb=4 into
    a single class of 139 songs. Both functions were individually correct; the SEAM was
    wrong, which is why no per-function test could have caught it.
    """
    return min(max(int(m), 0), len(s.values) - 1)


def _offset_maximised(s, y):
    """max_r instead of (1/m) sum_r -- the marginal replaced by a mode."""
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
    n = len(y_t)
    la, l1a = torch.nn.functional.logsigmoid(s.alpha), torch.nn.functional.logsigmoid(-s.alpha)
    lb, l1b = torch.nn.functional.logsigmoid(s.beta), torch.nn.functional.logsigmoid(-s.beta)
    idx = torch.arange(n, dtype=torch.float64)
    out = []
    for m in s.values:
        per = []
        for r in range(m):
            on = (((idx - r) % m) == 0).to(torch.float64)
            per.append((on * (y_t * la + (1 - y_t) * l1a)
                        + (1 - on) * (y_t * lb + (1 - y_t) * l1b)).sum())
        out.append(torch.stack(per).max())
    return torch.stack(out)


def _offset_unnormalised(s, y):
    """Forgets the -log(m): p(y|m) no longer sums to one over y, biasing toward large m."""
    return S._emission_logp_all(s, y) + torch.tensor(
        [math.log(m) for m in s.values], dtype=torch.float64)


def _downbeat_off_by_one(s, y):
    """((i - r + 1) % m == 0): the downbeat lands one beat late.

    PROVABLY EQUIVALENT to the correct code, and listed in ``EQUIVALENT`` for that reason.
    The offset-r mask is {i : (i - r) mod m == 0}; shifting by one gives
    {i : (i - (r-1)) mod m == 0}. As r ranges over 0..m-1 so does (r-1) mod m, so the two
    implementations enumerate the SAME SET of masks in a different order -- and the
    logsumexp over offsets is symmetric. Marginalising the bar offset makes the model
    invariant to downbeat phase BY DESIGN.

    It is kept as a control: a suite that "kills" this mutant is over-specifying, asserting
    an implementation detail the model deliberately quotients out.
    """
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
    n = len(y_t)
    la, l1a = torch.nn.functional.logsigmoid(s.alpha), torch.nn.functional.logsigmoid(-s.alpha)
    lb, l1b = torch.nn.functional.logsigmoid(s.beta), torch.nn.functional.logsigmoid(-s.beta)
    idx = torch.arange(n, dtype=torch.float64)
    out = []
    for m in s.values:
        per = []
        for r in range(m):
            on = (((idx - r + 1) % m) == 0).to(torch.float64)
            per.append((on * (y_t * la + (1 - y_t) * l1a)
                        + (1 - on) * (y_t * lb + (1 - y_t) * l1b)).sum())
        out.append(torch.logsumexp(torch.stack(per), 0) - math.log(m))
    return torch.stack(out)


def _offset_fixed_at_zero(s, y):
    """Never marginalises: assumes bar one starts at beat one.

    Realistic because it is what you write if you forget the crop can start mid-bar. Still
    a proper distribution over y, so the normalisation check cannot see it.
    """
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
    n = len(y_t)
    lsig = torch.nn.functional.logsigmoid
    on_term = y_t * lsig(s.alpha) + (1 - y_t) * lsig(-s.alpha)
    off_term = y_t * lsig(s.beta) + (1 - y_t) * lsig(-s.beta)
    out = []
    for m in s.values:
        M = S.offset_masks(n, m)[0]
        out.append((M * on_term + (1 - M) * off_term).sum())
    return torch.stack(out)


def _misordered(s, y):
    """logp_all[k] no longer corresponds to values[k]."""
    return S._emission_logp_all(s, y).flip(0)


def _kl_flipped(s, h, y):
    """KL(p_psi || q) in place of KL(q || p_psi) -- v1's diagnostic used the wrong
    direction too (SS4.6 pins the correct one). Here it is put where it does real damage: the bound."""
    q_logp = s.q_logp(h, y)
    q = q_logp.exp()
    lik = s.emission_logp_all(y)
    prior = s.prior_logp(h)
    p = prior.exp()
    recon = (q * lik).sum()
    kl = (p * (prior - q_logp)).sum()
    return recon - kl


def _elbo_sign(s, h, y):
    """recon + kl. The KL stops being a penalty, so nothing restrains q."""
    q_logp = s.q_logp(h, y)
    q = q_logp.exp()
    recon = (q * s.emission_logp_all(y)).sum()
    kl = (q * (q_logp - s.prior_logp(h))).sum()
    return recon + kl


def _posterior_ignores_prior(s, h, y):
    """Bayes without the prior: p(m|y,h) proportional to the likelihood alone."""
    lp = s.emission_logp_all(y)
    return lp - torch.logsumexp(lp, -1)


def _q_is_prior(s, h, y):
    """q ignores y: the latent is dead, q carries no information the prior lacks."""
    return s.prior_logp(h)


def _q_ignores_h(s, h, y):
    """q drops h: nothing is amortized, so the deployable path can learn nothing."""
    lik = s.emission_logp_all(y)
    return lik - torch.logsumexp(lik, -1)


def _predict_leaks_y(s, h):
    """The realistic leak: elbo() stashed the last y, predict() quietly reuses it. A
    signature of predict(h) does NOT prove the deployable path is annotation-free."""
    y = getattr(s, "_stashed_y", None)
    if y is None:
        return s.prior_logp(h)
    return s.exact_posterior(h, y)


def _elbo_stashing(s, h, y):
    s._stashed_y = y
    return S._elbo(s, h, y)


def _elbo_sampled(s, h, y):
    """A one-sample REINFORCE-style estimate of an expectation that has four terms.

    Unbiased, so every identity in the suite still holds IN EXPECTATION -- which is exactly
    why it is dangerous: it looks correct on paper and turns every downstream comparison
    into a coin flip. Spelled with numpy rather than torch so that an AST scan for
    ``torch.randn`` would not see it.
    """
    q_logp = s.q_logp(h, y)
    q = q_logp.exp()
    lik = s.emission_logp_all(y)
    prior = s.prior_logp(h)
    k = int(np.random.default_rng().choice(len(s.values), p=_np_probs(q)))
    recon = lik[k] * q[k] / q[k].detach()
    kl = (q * (q_logp - prior)).sum()
    return {"recon": recon, "kl": kl, "elbo": recon - kl,
            "q_logp": q_logp, "prior_logp": prior, "lik": lik}


def _np_probs(q):
    p = q.detach().numpy().astype(np.float64)
    return p / p.sum()


def _predict_constant(s, h):
    """Total collapse: the same answer for every crop, regardless of h."""
    K = len(s.values)
    return torch.full((K,), -math.log(K), dtype=torch.float64).index_put_(
        (torch.tensor([0]),), torch.tensor([0.0], dtype=torch.float64))


# ------------------------------------------------------------------------------------
# builders
# ------------------------------------------------------------------------------------
def _with(hooks, **kw):
    def build(values, capacity):
        s = S.oracle(values, capacity=capacity)
        s.hooks.update(hooks)
        s.name = "mutant"
        for k, v in kw.items():
            setattr(s, k, v)
        return s
    return build


def _psi_frozen(values, capacity):
    """v1's psi: registered but never CALLED, so 41 tensors / 50.88% of parameters got zero
    gradient for weeks and the prior's scales were effectively constants."""
    s = S.oracle(values, capacity=capacity)
    s.name = "mutant"
    s.w_prior = torch.zeros(3, dtype=torch.float64, requires_grad=False)
    s.b_prior = torch.zeros(len(values), dtype=torch.float64, requires_grad=False)
    return s


def _emission_inert(values, capacity):
    """beta tied to alpha: the labels carry no meter information at all, so m cannot
    matter however well everything else is wired. The 'meter latent is decorative' failure."""
    s = S.oracle(values, capacity=capacity)
    s.name = "mutant"
    s.beta = s.alpha
    return s


MUTANTS = {
    "count_as_index": (_with({"to_idx": _count_as_index}),
                       "beats-per-bar count consumed as an index (v1's shipped bug)"),
    "psi_frozen": (_psi_frozen,
                   "psi gets no gradient: registered but never called (v1, 50.88%)"),
    "q_is_prior": (_with({"q_logp": _q_is_prior}),
                   "q ignores y: the latent is dead"),
    "q_ignores_h": (_with({"q_logp": _q_ignores_h}),
                    "q ignores h: nothing is amortized"),
    "offset_maximised": (_with({"emission_logp_all": _offset_maximised}),
                         "max_r instead of the marginal over bar offset"),
    "offset_unnormalised": (_with({"emission_logp_all": _offset_unnormalised}),
                            "missing -log(m): p(y|m) does not normalise"),
    "downbeat_off_by_one": (_with({"emission_logp_all": _downbeat_off_by_one}),
                            "downbeat position shifted by one beat"),
    "offset_fixed_at_zero": (_with({"emission_logp_all": _offset_fixed_at_zero}),
                             "bar offset never marginalised: assumes crops start on a bar"),
    "misordered_values": (_with({"emission_logp_all": _misordered}),
                          "logp_all[k] does not correspond to values[k]"),
    "kl_flipped": (_with({"elbo": _kl_flipped}),
                   "KL(p||q) instead of KL(q||p) in the bound"),
    "elbo_sign": (_with({"elbo": _elbo_sign}),
                  "recon + kl: the KL stops being a penalty"),
    "elbo_sampled": (_with({"elbo": _elbo_sampled}),
                     "an enumerable K<=4 expectation estimated by one sample"),
    "posterior_ignores_prior": (_with({"exact_posterior": _posterior_ignores_prior}),
                                "Bayes rule without the prior"),
    "emission_inert": (_emission_inert,
                       "alpha == beta: labels carry no meter information"),
    "predict_leaks_y": (_with({"predict": _predict_leaks_y, "elbo": _elbo_stashing}),
                        "deployable predict() reuses a y stashed during training"),
    "predict_constant": (_with({"predict": _predict_constant}),
                         "the deployed predictor collapses to one class"),
}


# Mutants that are provably indistinguishable from the correct implementation. They are
# EXPECTED to survive; killing one means a test asserts an implementation detail the model
# quotients out, which breaks the other half of the iff ("proper => passes").
EQUIVALENT = {"downbeat_off_by_one"}


def build(name: str, values=None, capacity: str = "full"):
    import reference as R
    values = R.DEFAULT_VALUES if values is None else values
    builder, _ = MUTANTS[name]
    return builder(tuple(values), capacity)


def describe(name: str) -> str:
    return MUTANTS[name][1]
