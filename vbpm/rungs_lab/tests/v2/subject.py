"""The Stage-0 implementation *under test*, behind one adapter.

Why this exists
---------------
The v1 post-mortem lists five failures. None of them was a component failure:

  * a meter count produced correctly and consumed correctly -- but as an index (a SEAM),
  * psi registered and handed to the optimiser yet never CALLED, so 50.88% of parameters
    sat at exactly zero gradient (WIRING) -- invisible to "is it in the optimiser?",
  * a gradient aligned with truth in 0.000 of crops (a SYSTEM property),
  * a metric invariant to the error that mattered (MEASUREMENT),
  * an emission term present in code and absent from the spec (MODEL).

A suite of per-function tests with hand-built inputs is structurally blind to all five.
So the tests here take an implementation as a *parameter* and assert properties of the
fitted system. The same test body runs against:

  * ``oracle()``            -- known-correct, so a failure means the TEST is wrong,
  * ``mutants.MUTANTS``     -- known-broken, so a pass means the TEST is too weak,
  * the real ``vbpm`` package once it exists -- the actual verdict.

That triangulation is what turns "these tests pass" into evidence. See
``test_mutation_registry.py``.

Model (professor's tutorial SS9/SS12, Sohn-standard -- three parameter sets)
---------------------------------------------------------------------------
    theta : emission   p_theta(y | m)      two scalars, bar offset marginalised
    psi   : prior      p_psi(m | h)        sees h ONLY  <- the deployable path
    phi   : encoder    q_phi(m | h, y)     sees h AND y

Per the tutorial's Misconception 6 the training-inference gap is genuinely PRESENT in
this variant, so no test may assert it away; SS6.8.5 says psi converges to the AGGREGATED
posterior, which is what ``test_properties.py`` checks instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

import reference as R

# Deliberately disjoint from any valid index: with values (5, 7, 11) there is no integer
# that is both a legal meter and a legal index into [0, K). A count/index confusion --
# v1's actual bug -- cannot land on a plausible neighbour and hide.
DISJOINT_VALUES = (5, 7, 11)


_FEAT_CACHE: dict = {}
_MASK_CACHE: dict = {}


def h_features(h, values, threshold: float = 0.5) -> np.ndarray:
    """Per-class features of the activation, [K, 3]. Deployable: reads h only.

    Beats-per-bar shows up in h as (#beat peaks) / (#downbeat peaks), so each class gets
    a closeness feature to that ratio. A shared weight vector over per-class features (a
    conditional-logit model) can represent the correct classifier without being handed it.
    """
    key = (id(h), tuple(values), threshold)
    hit = _FEAT_CACHE.get(key)
    if hit is not None:
        return hit[1]
    arr = np.asarray(h, dtype=np.float64)
    n_beat = len(R.pick_peaks(arr[:, 0], threshold))
    n_down = len(R.pick_peaks(arr[:, 1], threshold))
    ratio = n_beat / max(n_down, 1)
    F = np.empty((len(values), 3), dtype=np.float64)
    for k, m in enumerate(values):
        F[k] = (1.0, -((ratio - m) ** 2), -abs(ratio - m))
    # Hold the ORIGINAL h, not the dtype-converted copy: the key is id(h), so if h were
    # collected its id could be recycled onto a different song and this cache would start
    # serving one song's features for another. That produced an order-dependent failure
    # that passed in isolation -- the exact shape of bug this whole suite exists to catch.
    _FEAT_CACHE[key] = (h, F)
    return F


def offset_masks(n: int, m: int) -> torch.Tensor:
    """[m, n] float mask: entry (r, i) is 1 when beat i is the downbeat under offset r."""
    key = (n, m)
    hit = _MASK_CACHE.get(key)
    if hit is None:
        idx = torch.arange(n, dtype=torch.float64)
        hit = torch.stack([(((idx - r) % m) == 0).to(torch.float64) for r in range(m)])
        _MASK_CACHE[key] = hit
    return hit


def coarse_y_features(y) -> np.ndarray:
    """A deliberately LOSSY summary of y: density and length, no positional information.

    Used by ``capacity='coarse'`` to induce a genuine amortization gap. It is lossy on
    purpose -- the meter lives in the POSITIONS of the ones, which this discards.
    """
    y = np.asarray(y, dtype=np.float64)
    n = max(len(y), 1)
    return np.array([y.mean(), math.log(n)], dtype=np.float64)


@dataclass
class Stage0:
    """One Stage-0 implementation. All log-probabilities, all natural log (C5)."""

    values: tuple[int, ...] = R.DEFAULT_VALUES
    capacity: str = "full"          # 'exact' | 'full' | 'coarse'
    name: str = "oracle"
    # theta
    alpha: torch.Tensor = field(default=None)
    beta: torch.Tensor = field(default=None)
    # psi (prior, h-only)
    w_prior: torch.Tensor = field(default=None)
    b_prior: torch.Tensor = field(default=None)
    # phi (encoder, h and y)
    c_lik: torch.Tensor = field(default=None)
    w_coarse: torch.Tensor = field(default=None)
    # hooks the mutants replace; keeping them as attributes means a mutant is a small,
    # readable, reversible edit rather than a forked copy of the class.
    hooks: dict = field(default_factory=dict)

    def __post_init__(self):
        K = len(self.values)
        if self.alpha is None:
            self.alpha = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        if self.beta is None:
            self.beta = torch.tensor(-0.5, dtype=torch.float64, requires_grad=True)
        if self.w_prior is None:
            self.w_prior = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        if self.b_prior is None:
            self.b_prior = torch.zeros(K, dtype=torch.float64, requires_grad=True)
        if self.c_lik is None:
            self.c_lik = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        if self.w_coarse is None:
            self.w_coarse = torch.zeros((K, 2), dtype=torch.float64, requires_grad=True)

    # -- introspection -------------------------------------------------------------
    def named_params(self) -> dict:
        p = {}
        for g in self.param_groups().values():
            p.update(g)
        return p

    def param_groups(self) -> dict:
        """The SS4.7 theta/psi/phi split, as {group: {name: tensor}}.

        This is what lets a property say "make the prior non-degenerate" without naming
        ``w_prior``. Only theta has a spec-mandated parameterisation (SS4.3: two scalars
        alpha, beta); psi's and phi's internals are implementation choices -- SS4.4 requires
        the reducer be swappable -- so tests must reach them by GROUP and by ``p.shape``,
        never by field name. A test that hardcodes a psi shape binds the suite to one
        implementation: this file's ``w_prior`` is [3] (weights over 3 per-class features)
        while SPEC SS4.4's W is [K, 2D] = [3,4], and ``copy_`` between them raises.
        """
        theta = {"alpha": self.alpha, "beta": self.beta}
        psi = {"w_prior": self.w_prior, "b_prior": self.b_prior}
        phi = {}
        if self.capacity == "coarse":
            phi["w_coarse"] = self.w_coarse
        elif self.capacity == "full":
            phi["c_lik"] = self.c_lik
        return {"theta": theta, "psi": psi, "phi": phi}

    def _hook(self, name, default):
        return self.hooks.get(name, default)

    def to_idx(self, m: int) -> int:
        """C1: a meter is a COUNT. Converting to an index is explicit and one-way."""
        fn = self._hook("to_idx", lambda s, mm: s.values.index(int(mm)))
        return fn(self, m)

    def to_value(self, k: int) -> int:
        return int(self.values[int(k)])

    # -- theta ---------------------------------------------------------------------
    def emission_logp_all(self, y) -> torch.Tensor:
        """[K] log p_theta(y | m), bar offset marginalised (not maximised)."""
        fn = self._hook("emission_logp_all", _emission_logp_all)
        return fn(self, y)

    # -- psi (DEPLOYABLE: h only) --------------------------------------------------
    def prior_logp(self, h) -> torch.Tensor:
        fn = self._hook("prior_logp", _prior_logp)
        return fn(self, h)

    def predict(self, h) -> torch.Tensor:
        """The deployed predictor. By construction it may not see y."""
        fn = self._hook("predict", lambda s, hh: s.prior_logp(hh))
        return fn(self, h)

    # -- phi -----------------------------------------------------------------------
    def q_logp(self, h, y) -> torch.Tensor:
        fn = self._hook("q_logp", _q_logp)
        return fn(self, h, y)

    # -- inference / objective -----------------------------------------------------
    def exact_posterior(self, h, y) -> torch.Tensor:
        fn = self._hook("exact_posterior", _exact_posterior)
        return fn(self, h, y)

    def log_evidence(self, h, y) -> torch.Tensor:
        return torch.logsumexp(self.emission_logp_all(y) + self.prior_logp(h), dim=-1)

    def elbo(self, h, y) -> torch.Tensor:
        fn = self._hook("elbo", _elbo)
        return fn(self, h, y)

    # -- fitting -------------------------------------------------------------------
    def fit(self, songs, steps: int = 400, lr: float = 0.05, seed: int = 0) -> "Stage0":
        """Maximise the mean ELBO. Exact enumeration, so no sampling and no seed noise."""
        torch.manual_seed(seed)
        seen, ps = set(), []
        for p in self.named_params().values():
            # a tied parameter (beta is alpha in the meter-free null) must be handed to the
            # optimiser once, not twice
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                ps.append(p)
        if not ps:
            return self
        opt = torch.optim.Adam(ps, lr=lr)
        cache = [(s["h"], s["y"]) for s in songs]
        for _ in range(steps):
            opt.zero_grad()
            loss = -torch.stack([self.elbo(h, y) for h, y in cache]).mean()
            loss.backward()
            opt.step()
        return self


# ------------------------------------------------------------------------------------
# the correct implementations of each hook
# ------------------------------------------------------------------------------------
def _emission_logp_all(s: Stage0, y) -> torch.Tensor:
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
    n = len(y_t)
    lsig = torch.nn.functional.logsigmoid
    on_term = y_t * lsig(s.alpha) + (1 - y_t) * lsig(-s.alpha)      # [n] beat IS a downbeat
    off_term = y_t * lsig(s.beta) + (1 - y_t) * lsig(-s.beta)       # [n] beat is not
    out = []
    for m in s.values:
        M = offset_masks(n, m)                                      # [m, n]
        ll = (M * on_term + (1 - M) * off_term).sum(-1)             # [m]
        # (1/m) * sum_r  -- MARGINALISED. max_r would differ by up to log(m).
        out.append(torch.logsumexp(ll, 0) - math.log(m))
    return torch.stack(out)


def _prior_logp(s: Stage0, h) -> torch.Tensor:
    F = torch.as_tensor(h_features(h, s.values), dtype=torch.float64)
    logits = F @ s.w_prior + s.b_prior
    return logits - torch.logsumexp(logits, -1)


def _q_logp(s: Stage0, h, y) -> torch.Tensor:
    # go through the METHOD, not the module function: SS4.5 builds q on top of the prior's
    # logits, so an implementation that swaps its reducer (SS4.4 requires that be possible)
    # must have q follow. Calling _prior_logp directly silently left q on the old prior --
    # the encoder then looked h-independent while prior_logp was demonstrably not.
    prior = s.prior_logp(h)
    if s.capacity == "exact":
        logits = prior + s.emission_logp_all(y)
    elif s.capacity == "full":
        logits = prior + s.c_lik * s.emission_logp_all(y)
    elif s.capacity == "coarse":
        cy = torch.as_tensor(coarse_y_features(y), dtype=torch.float64)
        logits = prior + s.w_coarse @ cy
    else:
        raise ValueError(f"unknown capacity {s.capacity!r}")
    return logits - torch.logsumexp(logits, -1)


def _exact_posterior(s: Stage0, h, y) -> torch.Tensor:
    lp = s.emission_logp_all(y) + s.prior_logp(h)
    return lp - torch.logsumexp(lp, -1)


def _elbo(s: Stage0, h, y) -> torch.Tensor:
    """E_q[log p(y|m)] - KL(q || p_psi). A SCALAR.

    It used to return a dict of the decomposition, justified as "the suite checks the parts".
    No check ever read ``recon``, ``kl``, ``lik`` or ``prior_logp`` -- the KL direction is
    pinned behaviourally by the bound-tightness identity instead -- and ``q_logp`` is already
    its own method. A return shape nothing consumes is contract for free.
    """
    q_logp = s.q_logp(h, y)
    q = q_logp.exp()
    recon = (q * s.emission_logp_all(y)).sum()
    kl = (q * (q_logp - s.prior_logp(h))).sum()   # KL(q || p_psi), the ELBO's own direction
    return recon - kl


# ------------------------------------------------------------------------------------
# factories
# ------------------------------------------------------------------------------------
def oracle(values=R.DEFAULT_VALUES, capacity: str = "full", **kw) -> Stage0:
    """The known-correct subject. Every property test MUST pass against this."""
    return Stage0(values=tuple(values), capacity=capacity, name="oracle", **kw)


def saturated(values=R.DEFAULT_VALUES, capacity: str = "exact") -> Stage0:
    """Oracle with a converged-looking emission: sigmoid(alpha)~1, sigmoid(beta)~0."""
    s = oracle(values, capacity=capacity)
    with torch.no_grad():
        s.alpha.fill_(8.0)
        s.beta.fill_(-8.0)
    return s
