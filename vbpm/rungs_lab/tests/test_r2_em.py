"""Black-box tests for rungs.r2_em_dbn.R2GenerativeLambda (exact Baum-Welch EM on the
Krebs bar-pointer + Böck 2016 observation model).

STRICTLY interface-contract tests: the implementation file is never read. The only whitebox
knowledge used is the CERTIFIED R1 state space (rungs/bar_pointer/state_space.py), which the
contract explicitly allows, and which the synthetic data generator needs to place beats on
the model's own tempo grid.

Run: /home/sogang/mnt/db_2/anaconda3/envs/jaehoon_bt/bin/python -m pytest tests/test_r2_em.py -x -q
"""
import math
import sys

import numpy as np
import pytest
import torch

from rungs.r2_em_dbn import R2GenerativeLambda

FPS = 50.0
DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else (
    "cuda" if torch.cuda.is_available() else "cpu")


def make_model(**kw):
    kw.setdefault("fps", FPS)
    kw.setdefault("device", DEVICE)
    return R2GenerativeLambda(**kw)


# ----------------------------------------------------------------------------------------------
# Synthetic data FROM THE GENERATIVE MODEL ITSELF
# ----------------------------------------------------------------------------------------------

def sample_crop(model, transition_lambda, num_frames, rng, beats_per_bar=4,
                obs_lambda=None, noise=0.02):
    """Sample a state path from the bar-pointer model at a known transition_lambda and emit
    activations consistent with the Böck classes.

    Tempo grid = the model's OWN interval grid (frames per beat). Within a beat the pointer
    advances +1 frame deterministically; at each beat boundary the next interval is sampled
    from the exponential kernel exp(-lam * |ratio - 1|), row-normalized.
    """
    space = None
    for s in model.chassis.state_spaces:
        if s.beats_per_bar == beats_per_bar:
            space = s
    assert space is not None, "chassis lacks the requested meter"
    intervals = space.interval_frames.astype(float)                      # [V]
    if obs_lambda is None:
        obs_lambda = model.observation_lambda

    ratio = intervals[None, :] / intervals[:, None]
    kernel = np.exp(-transition_lambda * np.abs(ratio - 1.0))
    kernel /= kernel.sum(axis=1, keepdims=True)

    beat_act = np.full(num_frames, 0.04)
    down_act = np.full(num_frames, 0.01)

    v = rng.integers(len(intervals))          # initial tempo uniform
    frame, beat_in_bar = 0, 0
    while frame < num_frames:
        interval = int(intervals[v])
        # Böck beat region: first ceil(interval / obs_lambda) states of the beat... precisely,
        # states with position-within-beat < 1/obs_lambda. With interval states per beat that is
        # frames f with f / interval < 1 / obs_lambda.
        region = [f for f in range(interval) if f / interval < 1.0 / obs_lambda]
        for f in region:
            t = frame + f
            if t >= num_frames:
                break
            if beat_in_bar == 0:
                down_act[t] = 0.90
                beat_act[t] = 0.93     # frontend beat channel fires on downbeats too
            else:
                beat_act[t] = 0.90
        frame += interval
        beat_in_bar = (beat_in_bar + 1) % beats_per_bar
        v = rng.choice(len(intervals), p=kernel[v])

    beat_act = np.clip(beat_act + rng.normal(0, noise, num_frames), 0.005, 0.985)
    down_act = np.clip(down_act + rng.normal(0, noise, num_frames), 0.005, 0.97)
    down_act = np.minimum(down_act, beat_act - 0.005)   # keep beat >= downbeat (decorrelation-safe)
    down_act = np.clip(down_act, 0.005, None)
    acts = np.stack([beat_act, down_act], axis=1)
    return torch.from_numpy(acts).to(dtype=torch.float64, device=DEVICE)


def make_dataset(model, transition_lambda, num_crops, num_frames=300, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [sample_crop(model, transition_lambda, num_frames, rng, **kw)
            for _ in range(num_crops)]


def total_marginal(model, crops):
    lk = model.log_kernel(model.transition_lambda)
    return sum(float(model.marginal_log_likelihood(c, lk)) for c in crops)


def run_em(model, crops, max_steps=40, tol=1e-3, **em_kw):
    prev = model.transition_lambda
    for _ in range(max_steps):
        lam = model.em_step(crops, **em_kw)
        if abs(lam - prev) < tol * max(1.0, abs(prev)):
            break
        prev = lam
    return model.transition_lambda


# ----------------------------------------------------------------------------------------------
# 1. STRUCTURAL
# ----------------------------------------------------------------------------------------------

class TestStructural:
    def test_log_kernel_rows_normalized(self):
        m = make_model()
        for lam in (0.0, 1.0, 30.0, 100.0, 300.0):
            lk = m.log_kernel(lam)
            assert lk.dim() == 2 and lk.shape[0] == lk.shape[1]
            rowsum = torch.logsumexp(lk, dim=1)
            assert torch.allclose(rowsum, torch.zeros_like(rowsum), atol=1e-5), \
                f"lam={lam}: max |logsumexp(row)| = {rowsum.abs().max().item()}"

    def test_log_kernel_lam0_uniform(self):
        m = make_model()
        lk = m.log_kernel(0.0)
        V = lk.shape[0]
        assert torch.allclose(lk, torch.full_like(lk, -math.log(V)), atol=1e-5), \
            "lam=0 must give uniform rows"

    def test_log_kernel_concentrates_with_lam(self):
        m = make_model()
        diag_small = torch.diagonal(m.log_kernel(10.0).exp()).min().item()
        diag_large = torch.diagonal(m.log_kernel(200.0).exp()).min().item()
        assert diag_large > diag_small, \
            f"diagonal mass must grow with lam: lam=10 min={diag_small}, lam=200 min={diag_large}"
        assert diag_large > 0.5, f"lam=200 should be near-diagonal, min diag mass {diag_large}"

    def test_log_kernel_matches_certified_r1_transition(self):
        """The kernel IS madmom's exponential_transition -- cross-check against the certified R1
        state space's own tempo_transition_probabilities."""
        m = make_model()
        lam = 100.0
        ref = m.chassis.state_spaces[0].tempo_transition_probabilities(lam)
        got = m.log_kernel(lam).exp().detach().cpu().numpy()
        assert got.shape == ref.shape
        assert np.allclose(got, ref, atol=1e-5), \
            f"max abs diff vs certified R1 kernel: {np.abs(got - ref).max()}"

    def test_log_class_densities_shape_and_finite(self):
        m = make_model()
        rng = np.random.default_rng(1)
        acts = rng.uniform(0.01, 0.6, size=(200, 2))
        acts[:, 1] = np.minimum(acts[:, 1], acts[:, 0] * 0.5)   # keep sum < 1, beat > downbeat
        acts_t = torch.from_numpy(acts).to(dtype=torch.float64, device=DEVICE)
        lcd = m.log_class_densities(acts_t)
        assert lcd.shape == (200, 3)
        assert torch.isfinite(lcd).all(), "densities must be finite for activations in (0,1), sum<1"

    def test_marginal_is_finite_scalar_and_differentiable(self):
        m = make_model()
        crop = make_dataset(m, 30.0, 1, num_frames=200, seed=2)[0]
        lk = m.log_kernel(m.transition_lambda).clone().requires_grad_(True)
        ll = m.marginal_log_likelihood(crop, lk)
        assert ll.dim() == 0 and torch.isfinite(ll)
        ll.backward()
        assert lk.grad is not None and torch.isfinite(lk.grad).all() \
            and lk.grad.abs().sum() > 0, "marginal must be differentiable w.r.t. log_kernel"

    def test_marginal_meter_mixture_upper_bounds_components(self):
        """Mixture over meters: total marginal must be >= each single-meter marginal + log prior
        weight is unknown, but it must at least be >= min component and <= logsumexp of components
        ... cheapest sharp check: marginal of (3,4) model lies between min and max single-meter
        marginals + log 2 slack."""
        crop = make_dataset(make_model(), 30.0, 1, num_frames=200, seed=3)[0]
        m34 = make_model(beats_per_bar=(3, 4))
        m3 = make_model(beats_per_bar=(3,))
        m4 = make_model(beats_per_bar=(4,))
        ll34 = float(m34.marginal_log_likelihood(crop, m34.log_kernel(100.0)))
        ll3 = float(m3.marginal_log_likelihood(crop, m3.log_kernel(100.0)))
        ll4 = float(m4.marginal_log_likelihood(crop, m4.log_kernel(100.0)))
        lo, hi = min(ll3, ll4), max(ll3, ll4)
        assert lo - math.log(2) - 1e-6 <= ll34 <= hi + math.log(2) + 1e-6, \
            f"mixture ll {ll34} outside [{lo - math.log(2)}, {hi + math.log(2)}] from components ({ll3}, {ll4})"


# ----------------------------------------------------------------------------------------------
# 2. MONOTONICITY (the EM theorem)
# ----------------------------------------------------------------------------------------------

class TestMonotonicity:
    def _check_monotone(self, learn_obs):
        m = make_model(init_transition_lambda=5.0)     # start far from anything reasonable
        crops = make_dataset(m, 60.0, 6, num_frames=250, seed=4)
        lls = [total_marginal(m, crops)]
        for _ in range(6):
            m.em_step(crops, learn_observation_lambda=learn_obs)
            lls.append(total_marginal(m, crops))
        for i in range(1, len(lls)):
            slack = 1e-3 * abs(lls[i - 1])
            assert lls[i] >= lls[i - 1] - slack, \
                f"EM step {i} DECREASED the marginal: {lls[i - 1]:.4f} -> {lls[i]:.4f} " \
                f"(learn_obs={learn_obs}); full trace {lls}"

    def test_em_monotone(self):
        self._check_monotone(learn_obs=False)

    def test_em_monotone_with_observation_coordinate(self):
        self._check_monotone(learn_obs=True)


# ----------------------------------------------------------------------------------------------
# 3. PARAMETER RECOVERY (decisive)
# ----------------------------------------------------------------------------------------------

class TestRecovery:
    def _recover(self, true_lam, init_lam, seed):
        m = make_model(init_transition_lambda=init_lam)
        crops = make_dataset(m, true_lam, 50, num_frames=300, seed=seed)
        lam = run_em(m, crops, max_steps=40)
        assert true_lam / 1.5 <= lam <= true_lam * 1.5, \
            f"EM from init {init_lam} on data with true lambda {true_lam} " \
            f"converged to {lam} (outside factor-1.5 band)"
        return lam

    def test_recover_low_from_high_init(self):
        self._recover(true_lam=30.0, init_lam=100.0, seed=10)

    def test_recover_high_from_low_init(self):
        self._recover(true_lam=100.0, init_lam=30.0, seed=11)

    def test_em_step_updates_and_returns_attribute(self):
        m = make_model(init_transition_lambda=100.0)
        crops = make_dataset(m, 30.0, 5, num_frames=250, seed=12)
        out = m.em_step(crops)
        assert isinstance(out, float)
        assert out == pytest.approx(m.transition_lambda), \
            "em_step return value must equal the updated .transition_lambda"


# ----------------------------------------------------------------------------------------------
# 4. FIXED POINT
# ----------------------------------------------------------------------------------------------

class TestFixedPoint:
    def test_converged_lambda_stays_put(self):
        m = make_model(init_transition_lambda=100.0)
        crops = make_dataset(m, 40.0, 30, num_frames=300, seed=13)
        lam = run_em(m, crops, max_steps=40, tol=1e-4)
        after = m.em_step(crops)
        assert abs(after - lam) <= 1e-2 * max(1.0, abs(lam)), \
            f"converged lambda {lam} moved to {after} on an extra em_step"


# ----------------------------------------------------------------------------------------------
# 5. OBSERVATION LAMBDA RECOVERY / argmax invariant
# ----------------------------------------------------------------------------------------------

class TestObservationStep:
    CANDIDATES = (2, 4, 6, 8, 16, 32)

    def _marginal_at_obs_lambda(self, obs_lambda, transition_lambda, crops):
        m = make_model(observation_lambda=obs_lambda,
                       init_transition_lambda=transition_lambda)
        return total_marginal(m, crops)

    def test_observation_step_contract_and_argmax(self):
        gen = make_model(observation_lambda=16)
        crops = make_dataset(gen, 60.0, 12, num_frames=250, seed=14, obs_lambda=16)

        m = make_model(observation_lambda=6, init_transition_lambda=60.0)
        selected = m.observation_step(crops)
        assert selected in self.CANDIDATES, f"selected {selected} not in candidate set"
        assert m.observation_lambda == selected, \
            f".observation_lambda ({m.observation_lambda}) != returned value ({selected})"

        # Sharp invariant: the marginal at the selected value beats every other candidate.
        best = self._marginal_at_obs_lambda(selected, m.transition_lambda, crops)
        for c in self.CANDIDATES:
            if c == selected:
                continue
            other = self._marginal_at_obs_lambda(c, m.transition_lambda, crops)
            assert best >= other - 1e-6 * abs(other), \
                f"observation_step picked {selected} (ll {best:.4f}) but candidate {c} " \
                f"has higher marginal ({other:.4f})"

    def test_observation_step_is_exact_argmax(self):
        """NOTE: true obs-lambda "recovery" is NOT a valid black-box test here: the Böck plug-in
        observation model is not a normalized emission one can sample from, so data emitted with a
        1/16-wide beat region is legitimately better explained by a smaller lambda (fewer no-beat
        classes -> higher no-beat density on the majority frames). Verified empirically: brute-force
        marginals over the candidate set prefer 4 on such data, and observation_step agrees. The
        valid sharp test is exact agreement with the brute-force argmax."""
        gen = make_model(observation_lambda=16)
        crops = make_dataset(gen, 60.0, 12, num_frames=250, seed=15, obs_lambda=16)
        m = make_model(observation_lambda=6, init_transition_lambda=60.0)
        selected = m.observation_step(crops)
        marginals = {c: self._marginal_at_obs_lambda(c, m.transition_lambda, crops)
                     for c in self.CANDIDATES}
        brute = max(marginals, key=marginals.get)
        assert selected == brute, \
            f"observation_step chose {selected} but brute-force argmax is {brute}: {marginals}"


# ----------------------------------------------------------------------------------------------
# 6. INIT INDEPENDENCE
# ----------------------------------------------------------------------------------------------

class TestInitIndependence:
    def test_two_inits_same_fixed_point(self):
        m_a = make_model(init_transition_lambda=10.0)
        m_b = make_model(init_transition_lambda=300.0)
        crops = make_dataset(m_a, 50.0, 40, num_frames=300, seed=16)
        lam_a = run_em(m_a, crops, max_steps=40, tol=1e-4)
        lam_b = run_em(m_b, crops, max_steps=40, tol=1e-4)
        ratio = max(lam_a, lam_b) / min(lam_a, lam_b)
        assert ratio <= 1.25, \
            f"EM from inits 10 and 300 disagreed: {lam_a} vs {lam_b} (ratio {ratio:.3f})"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))
