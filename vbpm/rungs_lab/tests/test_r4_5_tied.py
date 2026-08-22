"""BLIND black-box tests for R45RichEmission(tied_covariance=True).

Written strictly from the interface contract; the implementation was never read. Contract under
test: with tied_covariance=True, after initialize_from_labels or em_step all three classes share
ONE diagonal covariance equal to the posterior-weighted POOLED within-class variance (floored at
1e-3); .log_var rows identical; .mu per-class; the M-step is the exact closed-form maximizer of
the expected complete-data log-likelihood under the tied-diagonal family.

Run: /home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python -m pytest tests/test_r4_5_tied.py -x -q
"""
import numpy as np
import pytest
import torch

from rungs.r4_5_rich_emission import R45RichEmission
import importlib.util as _ilu
import os as _os
_spec = _ilu.spec_from_file_location(
    "test_r4_5_emission", _os.path.join(_os.path.dirname(__file__), "test_r4_5_emission.py"))
_helpers = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
D = _helpers.D
FPS = _helpers.FPS
DEVICE = _helpers.DEVICE
TRUE_MU = _helpers.TRUE_MU
make_model = _helpers.make_model
set_params = _helpers.set_params
feat_tensor = _helpers.feat_tensor
interval_grid = _helpers.interval_grid
mixture_kernel = _helpers.mixture_kernel
classes_from_beat_intervals = _helpers.classes_from_beat_intervals
features_from_classes = _helpers.features_from_classes
steady_crop = _helpers.steady_crop

EQUAL_VAR = np.full((3, D), 0.25)  # genuinely tied truth for the tied-model tests


def make_tied(**kw):
    kw.setdefault("tied_covariance", True)
    return make_model(**kw)


def rows_identical(model):
    lv = model.log_var.detach().cpu().numpy()
    return np.array_equal(lv[0], lv[1]) and np.array_equal(lv[1], lv[2]), lv


def total_marginal(model, crops):
    return sum(float(model.marginal_log_likelihood(c)) for c in crops)


def separated_crops(model, rng, n=3, var=EQUAL_VAR):
    crops = []
    for k in range(n):
        classes = classes_from_beat_intervals([22 + 3 * k] * 10)
        crops.append(feat_tensor(model, features_from_classes(classes, rng, var=var)))
    return crops


# ----------------------------------------------------------------------------------------------
# 1. TIED INVARIANT after init and after each em_step
# ----------------------------------------------------------------------------------------------

def test_tied_invariant_init_and_em():
    rng = np.random.default_rng(10)
    model = make_tied()
    classes = classes_from_beat_intervals([25] * 12)
    feats = features_from_classes(classes, rng, var=EQUAL_VAR)
    model.initialize_from_labels([feat_tensor(model, feats)],
                                 [torch.as_tensor(classes, device=model.device)])
    same, lv = rows_identical(model)
    assert same, f"log_var rows differ after initialize_from_labels:\n{lv}"
    mu = model.mu.detach().cpu().numpy()
    assert np.abs(mu[0] - mu[1]).max() > 1.0 and np.abs(mu[1] - mu[2]).max() > 1.0, \
        "mu rows collapsed together on well-separated data"

    crops = separated_crops(model, rng)
    for it in range(3):
        model.em_step(crops)
        same, lv = rows_identical(model)
        assert same, f"log_var rows differ after em_step {it}:\n{lv}"
        mu = model.mu.detach().cpu().numpy()
        assert np.abs(mu[0] - mu[2]).max() > 1.0, "mu became tied too"


# ----------------------------------------------------------------------------------------------
# 2. POOLED FORMULA on hard labels
# ----------------------------------------------------------------------------------------------

def test_pooled_variance_formula():
    rng = np.random.default_rng(11)
    model = make_tied()
    feats = [rng.normal(size=(150, D)) * rng.uniform(0.5, 2.0, size=D) + 2.0,
             rng.normal(size=(90, D)) - 1.0]
    labels = [rng.integers(0, 3, size=150), rng.integers(0, 3, size=90)]
    model.initialize_from_labels([feat_tensor(model, f) for f in feats],
                                 [torch.as_tensor(l, device=model.device) for l in labels])

    all_f = np.concatenate(feats)
    all_l = np.concatenate(labels)
    N = len(all_l)
    ss = np.zeros(D)
    for c in range(3):
        sel = all_f[all_l == c]
        np.testing.assert_allclose(model.mu.detach().cpu().numpy()[c], sel.mean(axis=0),
                                   atol=1e-6)
        ss += ((sel - sel.mean(axis=0)) ** 2).sum(axis=0)
    pooled_biased = np.maximum(ss / N, 1e-3)
    pooled_dof = np.maximum(ss / (N - 3), 1e-3)

    var = np.exp(model.log_var.detach().cpu().numpy())
    for c in range(3):
        ok = (np.allclose(var[c], pooled_biased, rtol=1e-5)
              or np.allclose(var[c], pooled_dof, rtol=1e-5))
        assert ok, (f"row {c} var {var[c]} matches neither pooled/N {pooled_biased} "
                    f"nor pooled/(N-3) {pooled_dof}")


def test_pooled_variance_floor():
    rng = np.random.default_rng(12)
    model = make_tied()
    T = 90
    labels = rng.integers(0, 3, size=T)
    feats = np.tile(np.arange(3, dtype=float)[:, None], (1, D))[labels]  # constant per class
    model.initialize_from_labels([feat_tensor(model, feats)],
                                 [torch.as_tensor(labels, device=model.device)])
    var = np.exp(model.log_var.detach().cpu().numpy())
    same, _ = rows_identical(model)
    assert same
    np.testing.assert_allclose(var[0], 1e-3, rtol=1e-6)


# ----------------------------------------------------------------------------------------------
# 3. EM MONOTONICITY under tied M-step (sharpest test of the correct restricted argmax)
# ----------------------------------------------------------------------------------------------

def test_tied_em_monotone():
    rng = np.random.default_rng(13)
    model = make_tied()
    crops = separated_crops(model, rng, n=3)
    set_params(model, TRUE_MU + rng.normal(size=(3, D)) * 1.5,
               np.tile(rng.uniform(0.1, 1.0, size=D), (3, 1)))

    lls = [float(model.em_step(crops)) for _ in range(6)]
    diffs = np.diff(lls)
    assert np.all(diffs >= -1e-4 * np.maximum(1.0, np.abs(np.array(lls[:-1])))), \
        f"tied EM marginal decreased: {lls}"
    final = total_marginal(model, crops)
    assert final >= lls[-1] - 1e-4 * max(1.0, abs(lls[-1])), \
        f"post-final marginal {final} < last EM return {lls[-1]}"


# ----------------------------------------------------------------------------------------------
# 4. TIED-VS-FREE SANITY (both orderings)
# ----------------------------------------------------------------------------------------------

def _run_em(model, crops, n_iter=8):
    for _ in range(n_iter):
        model.em_step(crops)
    return total_marginal(model, crops)


def test_tied_matches_free_when_truth_is_tied():
    rng = np.random.default_rng(14)
    init_mu = TRUE_MU + np.random.default_rng(99).normal(size=(3, D)) * 0.8
    init_var = np.tile(np.full(D, 0.5), (3, 1))

    tied = make_tied()
    free = make_model(tied_covariance=False)
    crops_np = []
    for k in range(3):
        classes = classes_from_beat_intervals([22 + 3 * k] * 10)
        crops_np.append(features_from_classes(classes, rng, var=EQUAL_VAR))
    tied_crops = [feat_tensor(tied, f) for f in crops_np]
    free_crops = [feat_tensor(free, f) for f in crops_np]
    set_params(tied, init_mu, init_var)
    set_params(free, init_mu, init_var)

    ll_tied = _run_em(tied, tied_crops)
    ll_free = _run_em(free, free_crops)
    T = sum(len(f) for f in crops_np)

    mu_t = tied.mu.detach().cpu().numpy()
    mu_f = free.mu.detach().cpu().numpy()
    assert np.abs(mu_t - mu_f).max() < 0.5, \
        f"tied vs free mu diverge on equal-cov data: max diff {np.abs(mu_t - mu_f).max():.3f}"
    assert np.abs(mu_t - TRUE_MU).max() < 0.7, \
        f"tied mu off truth by {np.abs(mu_t - TRUE_MU).max():.3f}"
    gap = (ll_free - ll_tied) / T
    assert abs(gap) < 0.5, f"tied vs free likelihood gap {gap:.3f} nats/frame on tied-truth data"


def test_free_beats_tied_when_truth_is_unequal():
    rng = np.random.default_rng(15)
    uneq_var = np.stack([np.full(D, 4.0), np.full(D, 0.01), np.full(D, 0.5)])
    init_mu = TRUE_MU + np.random.default_rng(98).normal(size=(3, D)) * 0.5
    init_var = np.tile(np.full(D, 1.0), (3, 1))

    tied = make_tied()
    free = make_model(tied_covariance=False)
    crops_np = []
    for k in range(3):
        classes = classes_from_beat_intervals([22 + 3 * k] * 10)
        crops_np.append(features_from_classes(classes, rng, var=uneq_var))
    tied_crops = [feat_tensor(tied, f) for f in crops_np]
    free_crops = [feat_tensor(free, f) for f in crops_np]
    set_params(tied, init_mu, init_var)
    set_params(free, init_mu, init_var)

    ll_tied = _run_em(tied, tied_crops)
    ll_free = _run_em(free, free_crops)
    T = sum(len(f) for f in crops_np)
    assert ll_free > ll_tied, \
        (f"free family failed to exceed tied on wildly-unequal-variance data: "
         f"free {ll_free/T:.3f} vs tied {ll_tied/T:.3f} nats/frame")


# ----------------------------------------------------------------------------------------------
# 5. PARAMETER RECOVERY under tied EM on model-generated data
# ----------------------------------------------------------------------------------------------

def test_tied_em_parameter_recovery():
    rng = np.random.default_rng(16)
    intervals = interval_grid()
    kernel = mixture_kernel(intervals)

    def sample_crop(num_beats=12):
        idx = int(np.nonzero(intervals == 25)[0][0])
        beat_ivs = []
        for _ in range(num_beats):
            beat_ivs.append(int(intervals[idx]))
            idx = rng.choice(len(intervals), p=kernel[idx])
        classes = classes_from_beat_intervals(beat_ivs)
        return features_from_classes(classes, rng, var=EQUAL_VAR)

    model = make_tied()
    crops = [feat_tensor(model, sample_crop()) for _ in range(4)]
    set_params(model, TRUE_MU + rng.normal(size=(3, D)) * 1.0,
               np.tile(rng.uniform(0.1, 1.2, size=D), (3, 1)))
    for _ in range(8):
        model.em_step(crops)

    mu = model.mu.detach().cpu().numpy()
    err = np.abs(mu - TRUE_MU).max(axis=1)
    assert np.all(err < 0.5), f"tied EM mu off by {err} per class (tol 0.5)"
    var = np.exp(model.log_var.detach().cpu().numpy())
    same, _ = rows_identical(model)
    assert same
    ratio = var[0] / EQUAL_VAR[0]
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0), \
        f"shared variance {var[0]} vs true {EQUAL_VAR[0]}: ratio {ratio}"


# ----------------------------------------------------------------------------------------------
# 6. NO LOG-DET GAMES: rare tight class cannot sharpen its covariance; share stays honest
# ----------------------------------------------------------------------------------------------

def test_rare_tight_class_cannot_sharpen_or_collapse():
    rng = np.random.default_rng(17)
    model = make_tied()
    # Steady 4/4, interval 25: downbeat frames are a few % of all frames -> naturally rare class.
    classes = np.concatenate([classes_from_beat_intervals([25] * 12) for _ in range(2)])
    # rare class 2: TIGHT true spread (0.02); everything else broad (1.0)
    var = np.stack([np.full(D, 1.0), np.full(D, 1.0), np.full(D, 0.02)])
    feats = features_from_classes(classes, rng, var=var)
    true_rate = (classes == 2).mean()
    assert true_rate < 0.06, f"setup: downbeat class not rare ({true_rate:.3f})"

    crops = [feat_tensor(model, feats)]
    model.initialize_from_labels(crops, [torch.as_tensor(classes, device=model.device)])
    shares = []
    for _ in range(5):
        model.em_step(crops)
        same, lv = rows_identical(model)
        assert same, f"rare class obtained its own covariance:\n{lv}"
        post = model.class_posteriors(crops[0]).cpu().numpy()
        shares.append(post[:, 2].mean())
    shares = np.array(shares)
    # tied covariance forbids the log-det race: the rare class can neither vanish nor explode
    assert np.all(shares > 0.25 * true_rate), \
        f"rare class collapsed: shares {np.round(shares, 4)} vs true rate {true_rate:.4f}"
    assert np.all(shares < 4.0 * true_rate), \
        f"rare class exploded: shares {np.round(shares, 4)} vs true rate {true_rate:.4f}"


# ----------------------------------------------------------------------------------------------
# 7. DECODE STILL EXACT under tied covariance
# ----------------------------------------------------------------------------------------------

def test_tied_decode_recovers_exact_grid():
    rng = np.random.default_rng(18)
    model = make_tied()
    classes = classes_from_beat_intervals([25] * 12)
    feats = features_from_classes(classes, rng, var=EQUAL_VAR)
    model.initialize_from_labels([feat_tensor(model, feats)],
                                 [torch.as_tensor(classes, device=model.device)])
    same, _ = rows_identical(model)
    assert same

    out = model.decode(feat_tensor(model, feats))
    beats = np.asarray(out["beats"])
    downbeats = np.asarray(out["downbeats"])
    assert len(beats) >= 9, f"only {len(beats)} beats decoded"
    frame_diffs = np.round(np.diff(beats) * FPS).astype(int)
    values, counts = np.unique(frame_diffs, return_counts=True)
    mode = values[np.argmax(counts)]
    assert mode == 25, f"mode IBI {mode} frames != true 25"
    assert len(downbeats) >= 2
    for db in downbeats:
        assert np.min(np.abs(beats - db)) < 1e-9, "downbeat not in beat list"
    db_gaps = np.round(np.diff(downbeats) * FPS).astype(int)
    assert np.all(db_gaps == 100), f"downbeat gaps {db_gaps} != 100 frames (4 beats)"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-x", "-q"]))
