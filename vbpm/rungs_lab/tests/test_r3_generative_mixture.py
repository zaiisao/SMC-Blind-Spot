"""Black-box tests for the per-frame mixture-kernel machinery.

Targets (treated as opaque; only the interface contract is assumed):
  * experiments/bt_e2e/mixture_kernel_probe.MixtureLambda   (global (w, lambda) mixture)
  * experiments/bt_e2e/r3_generative_mixture_v2.R3GenerativeMixture (per-frame w_t)

Trusted reference code: rungs/r1_2016_dbn.py, rungs/r2_em_dbn.py, rungs/bar_pointer/*.

Run:
  /home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python -m pytest tests/test_r3_generative_mixture.py -x -q
or standalone:
  /home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python tests/test_r3_generative_mixture.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "bt_e2e"))

from mixture_kernel_probe import MixtureLambda            # noqa: E402
from r3_generative_mixture_v2 import R3GenerativeMixture  # noqa: E402

# NEVER cuda:0/2/3 (live runs).
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
FPS = 43.06640625          # Beat Transformer frontend fps
W0, LAM0 = 0.370, 93.1     # contract: v2 zero-init constants

torch.manual_seed(0)
np.random.seed(0)


# ------------------------------------------------------------------ helpers
def make_mixture(**kw):
    return MixtureLambda(fps=FPS, device=DEVICE, **kw)


def make_v2(**kw):
    kw.setdefault("lambda_base", LAM0)  # class default is 100.0; experiment uses K_93.1
    net = R3GenerativeMixture(fps=FPS, device=DEVICE, **kw)
    return net.to(DEVICE)


def perturb_net(model, scale=0.05, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.net.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g).to(p.device))
    return model


def metronome(T, period, fps=FPS, peak=0.95, down_every=4, base=0.02, noise=0.0, seed=0):
    """[T,2] activations with beat peaks at round(k*period); every down_every-th is a downbeat."""
    rng = np.random.default_rng(seed)
    beat = np.full(T, base); down = np.full(T, base / 2)
    k = 0
    while round(k * period) < T:
        t = round(k * period)
        beat[t] = peak
        if k % down_every == 0:
            down[t] = peak - 0.05
        k += 1
    if noise:
        beat = np.clip(beat + noise * rng.standard_normal(T), 1e-3, 0.99)
        down = np.clip(down, 1e-3, beat - 1e-3)
    acts = np.stack([beat, down], axis=1).astype(np.float32)
    return torch.from_numpy(acts).to(DEVICE)


def reference_D(m):
    """Row-uniform dither kernel over |interval_i - interval_j| <= 1, built from the
    certified state space only."""
    iv = m.chassis.state_spaces[0].interval_frames.astype(np.int64)
    adj = (np.abs(iv[None, :] - iv[:, None]) <= 1).astype(np.float64)
    return torch.from_numpy(adj / adj.sum(axis=1, keepdims=True)).to(DEVICE).float()


def total_marginal(m, crops, log_kernel):
    with torch.no_grad():
        return sum(float(m.marginal_log_likelihood(a, log_kernel)) for a in crops)


# ------------------------------------------------------------------ 1. normalization
@pytest.mark.parametrize("w,lam", [(0.05, 20.0), (0.37, 93.1), (0.8, 300.0), (0.5, 5.0)])
def test_global_kernel_rows_normalized(w, lam):
    m = make_mixture()
    rows = m.log_mixture_kernel(w, lam).exp().sum(dim=1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5), \
        f"row sums off: [{rows.min()}, {rows.max()}]"


def test_perframe_kernel_rows_normalized_zero_init_and_perturbed():
    acts = metronome(200, 21.5, noise=0.05)
    for seed in (None, 2, 3):
        net = make_v2()
        if seed is not None:
            perturb_net(net, scale=0.3, seed=seed)
        with torch.no_grad():
            pk = net.per_frame_mixture_kernel(acts)
        rows = pk.exp().sum(dim=2)
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-4), \
            f"seed={seed}: row sums [{rows.min()}, {rows.max()}]"


# ------------------------------------------------------------------ 2. init equivalence
def test_zero_init_matches_global_kernel():
    m = make_mixture()
    net = make_v2()
    acts = metronome(150, 21.5, noise=0.05)
    with torch.no_grad():
        w = net.per_frame_w(acts)
        pk = net.per_frame_mixture_kernel(acts)
    assert torch.allclose(w, torch.full_like(w, W0), atol=1e-6), \
        f"per_frame_w at init: [{w.min()}, {w.max()}] != {W0}"
    ref = m.log_mixture_kernel(W0, LAM0)
    dev = (pk - ref[None]).abs().max().item()
    assert dev < 1e-4, f"zero-init per-frame kernel deviates from global by {dev}"


def test_zero_init_marginal_matches_r2_forward():
    m = make_mixture()
    net = make_v2()
    acts = metronome(250, 21.5, noise=0.05)
    with torch.no_grad():
        ll_net = float(net.marginal_log_likelihood(acts))
        ll_ref = float(m.marginal_log_likelihood(acts, m.log_mixture_kernel(W0, LAM0)))
    assert abs(ll_net - ll_ref) < 1e-2 * max(1.0, abs(ll_ref)) and abs(ll_net - ll_ref) < 0.5, \
        f"net {ll_net} vs R2-forward {ll_ref}"


# ------------------------------------------------------------------ 3. limits
def test_w_to_zero_recovers_exponential_kernel():
    m = make_mixture()
    for lam in (30.0, 93.1, 250.0):
        mix = m.log_mixture_kernel(1e-9, lam).exp()
        pure = m.log_kernel(lam).exp()
        dev = (mix - pure).abs().max().item()
        assert dev < 1e-6, f"lam={lam}: w->0 limit deviates by {dev}"


def test_w_to_one_recovers_dither_kernel():
    m = make_mixture()
    D = reference_D(m)
    mix = m.log_mixture_kernel(1.0 - 1e-9, 93.1).exp()
    dev = (mix - D).abs().max().item()
    assert dev < 1e-5, f"w->1 limit deviates from adjacent-uniform D by {dev}"


# ------------------------------------------------------------------ 4. gradient flow
def test_gradient_flow_nonzero_finite():
    net = perturb_net(make_v2(), scale=0.1, seed=4)
    acts = metronome(200, 21.5, noise=0.05)
    ll = net.marginal_log_likelihood(acts)
    ll.backward()
    grads = [p.grad for p in net.net.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads), "non-finite/missing grads"
    total = sum(float(g.abs().sum()) for g in grads)
    assert total > 1e-8, f"all net grads are zero (total abs {total})"


def test_gradient_finite_difference():
    # float32 log-likelihoods (~1e2 nats) only resolve ~1e-5, which is too coarse for a small
    # central difference while a large eps hits curvature; run the check in float64 instead.
    net = perturb_net(make_v2(), scale=0.1, seed=5).double()
    acts = metronome(150, 21.5, noise=0.05).double()
    ll = net.marginal_log_likelihood(acts)
    ll.backward()
    params = [p for p in net.net.parameters() if p.grad is not None and p.grad.abs().max() > 1e-4]
    assert params, "no parameter with usable gradient magnitude"
    checked = 0
    for p in params[:4]:
        flat_g = p.grad.flatten()
        idx = int(flat_g.abs().argmax())
        analytic = float(flat_g[idx])
        eps = 1e-5
        with torch.no_grad():
            p.flatten()[idx] += eps
            lp = float(net.marginal_log_likelihood(acts))
            p.flatten()[idx] -= 2 * eps
            lm = float(net.marginal_log_likelihood(acts))
            p.flatten()[idx] += eps
        fd = (lp - lm) / (2 * eps)
        rel = abs(fd - analytic) / max(abs(analytic), abs(fd), 1e-6)
        assert rel < 1e-4, f"finite-diff mismatch: analytic {analytic} vs fd {fd} (rel {rel})"
        checked += 1
        if checked >= 2:
            break
    assert checked >= 2, "fewer than 2 parameters finite-diff-checked"


# ------------------------------------------------------------------ 5. monotone-benefit
def test_dither_component_worth_something_on_dithered_data():
    m = make_mixture()
    dith = metronome(300, 21.5, noise=0.0)
    integ = metronome(300, 21.0, noise=0.0)
    def ll(acts, w):
        return float(m.marginal_log_likelihood(acts, m.log_mixture_kernel(w, LAM0)))
    margin_dither = ll(dith, W0) - ll(dith, 0.01)
    margin_int = ll(integ, W0) - ll(integ, 0.01)
    assert margin_dither > 0, f"w={W0} does not beat w=0.01 on dithered data (margin {margin_dither})"
    assert margin_int < margin_dither, \
        f"integer-interval data should weaken/reverse the ordering: {margin_int} !< {margin_dither}"


# ------------------------------------------------------------------ 6. EM sanity
def test_em_step_mixture_monotone_and_m_step_argmax():
    m = make_mixture()
    crops = [metronome(200, 21.5, noise=0.05, seed=s) for s in range(3)]
    w, lam = float(m.mixture_weight), float(m.transition_lambda)
    ll_prev = total_marginal(m, crops, m.log_mixture_kernel(w, lam))
    for it in range(2):
        w, lam = m.em_step_mixture(crops)
        assert 0.0 < w < 1.0, f"iter {it}: w={w} outside (0,1)"
        assert 1.0 <= lam <= 1000.0, f"iter {it}: lambda={lam} outside plausible range"
        ll_new = total_marginal(m, crops, m.log_mixture_kernel(w, lam))
        assert ll_new >= ll_prev - 0.5, f"iter {it}: marginal decreased {ll_prev} -> {ll_new}"
        ll_prev = ll_new
    # m_step_2d: returned (w, lam) is a local argmax of sum(counts * log_mixture)
    V = m._abs_ratio_dev.shape[0]
    g = torch.Generator(device="cpu").manual_seed(7)
    counts = (torch.rand(V, V, generator=g) *
              m.log_mixture_kernel(0.3, 80.0).exp().cpu()).to(DEVICE) * 100
    w2, lam2 = m.m_step_2d(counts)
    assert 0.0 < w2 < 1.0 and lam2 > 0
    def score(w_, lam_):
        return float((counts * m.log_mixture_kernel(w_, lam_)).sum())
    best = score(w2, lam2)
    for dw in (-0.03, 0.03):
        for f in (0.85, 1.18):
            w_alt = min(max(w2 + dw, 1e-4), 1 - 1e-4)
            alt = score(w_alt, lam2 * f)
            assert best >= alt - 1e-3, \
                f"m_step_2d not a local argmax: ({w2},{lam2})={best} < ({w_alt},{lam2*f})={alt}"


# ------------------------------------------------------------------ 7. decode contract
def _check_events(events, T):
    assert isinstance(events, dict) and "beats" in events, f"bad events dict: {type(events)}"
    beats = np.asarray(events["beats"], dtype=float)
    assert beats.ndim == 1 and len(beats) >= 2
    assert np.all(np.diff(beats) > 0), "beats not strictly increasing"
    assert beats.min() >= 0 and beats.max() <= T / FPS + 1e-6, \
        f"beats outside [0, {T/FPS}]: [{beats.min()}, {beats.max()}]"
    if "downbeats" in events and len(events["downbeats"]) >= 2:
        db = np.asarray(events["downbeats"], dtype=float)
        assert np.all(np.diff(db) > 0) and db.min() >= 0 and db.max() <= T / FPS + 1e-6
    return beats


@pytest.mark.parametrize("which", ["mixture_global", "v2_perframe"])
def test_decode_metronome(which):
    T = 300
    acts = metronome(T, 21.0, noise=0.0)
    if which == "mixture_global":
        events = make_mixture().decode(acts)
    else:
        with torch.no_grad():
            events = make_v2().decode_mixture(acts)
    beats = _check_events(events, T)
    diffs = np.diff(beats)
    vals, cnts = np.unique(np.round(diffs * FPS).astype(int), return_counts=True)
    mode = int(vals[cnts.argmax()])
    assert abs(mode - 21) <= 1, f"{which}: modal inter-beat interval {mode} frames != 21 +/- 1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
