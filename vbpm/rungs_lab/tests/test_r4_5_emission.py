"""BLIND black-box tests for rungs.r4_5_rich_emission.R45RichEmission.

Written strictly from the interface contract -- the implementation file was never read. The only
whitebox knowledge used is the CERTIFIED bar-pointer state space (rungs/bar_pointer/state_space.py,
allowed), which the synthetic generators need to place beats on the model's own tempo grid and to
map states to the {no-beat, beat, downbeat} classes exactly as the model partitions them.

Run: /home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python -m pytest tests/test_r4_5_emission.py -x -q
"""
import numpy as np
import pytest
import torch
from scipy.stats import multivariate_normal

from rungs.r4_5_rich_emission import R45RichEmission

FPS = 50.0
D = 6
# cuda:3 if the machine has it, else cpu. NEVER cuda:0/1/2 (live jobs).
DEVICE = "cuda:3" if torch.cuda.is_available() and torch.cuda.device_count() > 3 else "cpu"

# Well-separated true Gaussians for the generative tests: per-class means far apart in every dim.
TRUE_MU = np.stack([np.full(D, -5.0), np.full(D, 0.0), np.full(D, 5.0)])
TRUE_MU += np.arange(D) * 0.1  # break symmetry across dims
TRUE_VAR = np.stack([np.full(D, 0.30), np.full(D, 0.20), np.full(D, 0.25)])

MIXTURE_W = 0.370
TRANSITION_LAMBDA = 93.1
OBS_LAMBDA = 6


def make_model(**kw):
    kw.setdefault("fps", FPS)
    kw.setdefault("feature_dim", D)
    kw.setdefault("device", DEVICE)
    return R45RichEmission(**kw)


def set_params(model, mu, var):
    """Overwrite the model's Gaussians with known values (contract exposes .mu / .log_var)."""
    mu_t = torch.as_tensor(mu, dtype=torch.float64, device=model.mu.device)
    lv_t = torch.log(torch.as_tensor(var, dtype=torch.float64, device=model.mu.device))
    with torch.no_grad():
        try:
            model.mu.copy_(mu_t)
            model.log_var.copy_(lv_t)
        except (AttributeError, RuntimeError):
            model.mu = mu_t
            model.log_var = lv_t


def feat_tensor(model, features):
    return torch.as_tensor(features, dtype=torch.float64, device=model.device)


# ----------------------------------------------------------------------------------------------
# Independent construction of the documented pieces (from the contract, NOT the implementation)
# ----------------------------------------------------------------------------------------------

def interval_grid(fps=FPS, min_bpm=55.0, max_bpm=215.0):
    lo = round(60.0 * fps / max_bpm)
    hi = round(60.0 * fps / min_bpm)
    return np.arange(lo, hi + 1)


def mixture_kernel(intervals, w=MIXTURE_W, lam=TRANSITION_LAMBDA):
    """Documented FIXED two-component mixture, built independently from its formula:
    w * row-uniform-over-|interval diff|<=1  +  (1-w) * row-softmax(-lambda * |ratio - 1|)."""
    iv = intervals.astype(float)
    near = (np.abs(iv[None, :] - iv[:, None]) <= 1).astype(float)
    uniform = near / near.sum(axis=1, keepdims=True)
    scores = -lam * np.abs(iv[None, :] / iv[:, None] - 1.0)
    scores -= scores.max(axis=1, keepdims=True)
    softmax = np.exp(scores)
    softmax /= softmax.sum(axis=1, keepdims=True)
    return w * uniform + (1 - w) * softmax


def classes_from_beat_intervals(beat_intervals, beats_per_bar=4, obs_lambda=OBS_LAMBDA,
                                first_beat_in_bar=0):
    """Frame-level class sequence of a legal bar-pointer path: within a beat the pointer advances
    deterministically through its interval's states; the certified partition puts a state in the
    beat class iff its within-beat position < 1/obs_lambda, downbeat iff additionally on beat 0."""
    classes = []
    beat_idx = first_beat_in_bar
    for L in beat_intervals:
        pos = np.arange(L) / L
        cls = np.zeros(L, dtype=np.int64)
        cls[pos < 1.0 / obs_lambda] = 2 if beat_idx % beats_per_bar == 0 else 1
        classes.append(cls)
        beat_idx += 1
    return np.concatenate(classes)


def features_from_classes(classes, rng, mu=TRUE_MU, var=TRUE_VAR):
    noise = rng.standard_normal((len(classes), D))
    return mu[classes] + np.sqrt(var[classes]) * noise


def steady_crop(rng, interval=25, num_beats=12, beats_per_bar=4):
    """Fixed-tempo 4/4 crop (interval frames per beat) generated from the model class itself."""
    classes = classes_from_beat_intervals([interval] * num_beats, beats_per_bar)
    return features_from_classes(classes, rng), classes


# ----------------------------------------------------------------------------------------------
# 1. DENSITY CORRECTNESS vs scipy
# ----------------------------------------------------------------------------------------------

def test_log_class_densities_match_scipy():
    rng = np.random.default_rng(0)
    model = make_model()
    mu = rng.normal(size=(3, D)) * 2.0
    var = rng.uniform(0.05, 3.0, size=(3, D))
    set_params(model, mu, var)

    features = rng.normal(size=(40, D)) * 3.0
    out = model.log_class_densities(feat_tensor(model, features)).cpu().numpy()
    assert out.shape == (40, 3)

    expected = np.stack([multivariate_normal(mean=mu[c], cov=np.diag(var[c])).logpdf(features)
                         for c in range(3)], axis=1)
    np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-9)


# ----------------------------------------------------------------------------------------------
# 2. GAUSSIAN MLE from hard labels + variance floor
# ----------------------------------------------------------------------------------------------

def test_initialize_from_labels_matches_numpy_mle():
    rng = np.random.default_rng(1)
    model = make_model()
    feats = [rng.normal(size=(120, D)) + 2.0, rng.normal(size=(80, D)) - 1.0]
    labels = [rng.integers(0, 3, size=120), rng.integers(0, 3, size=80)]
    model.initialize_from_labels(
        [feat_tensor(model, f) for f in feats],
        [torch.as_tensor(l, device=model.device) for l in labels])

    all_f = np.concatenate(feats)
    all_l = np.concatenate(labels)
    mu = model.mu.detach().cpu().numpy()
    var = np.exp(model.log_var.detach().cpu().numpy())
    for c in range(3):
        sel = all_f[all_l == c]
        np.testing.assert_allclose(mu[c], sel.mean(axis=0), rtol=0, atol=1e-6)
        biased = sel.var(axis=0, ddof=0)
        unbiased = sel.var(axis=0, ddof=1)
        ok = (np.allclose(var[c], np.maximum(biased, 1e-3), atol=1e-6)
              or np.allclose(var[c], np.maximum(unbiased, 1e-3), atol=1e-6))
        assert ok, f"class {c}: var {var[c]} matches neither biased {biased} nor unbiased {unbiased}"


def test_initialize_from_labels_variance_floor():
    rng = np.random.default_rng(2)
    model = make_model()
    T = 90
    labels = rng.integers(0, 3, size=T)
    feats = rng.normal(size=(T, D))
    feats[labels == 1] = 3.14  # constant-feature class -> zero empirical variance
    model.initialize_from_labels([feat_tensor(model, feats)],
                                 [torch.as_tensor(labels, device=model.device)])
    var1 = np.exp(model.log_var.detach().cpu().numpy())[1]
    mu1 = model.mu.detach().cpu().numpy()[1]
    np.testing.assert_allclose(mu1, 3.14, atol=1e-6)
    assert np.all(var1 >= 1e-3 * (1 - 1e-9)), f"variance floor violated: {var1}"


# ----------------------------------------------------------------------------------------------
# 3 + 4. POSTERIORS: distribution axioms and sanity on model-generated data
# ----------------------------------------------------------------------------------------------

def test_class_posteriors_are_distributions():
    """Tolerance note (measured): the forward-backward runs in float32 internally, so on far
    off-model features (log-densities thousands of nats) the shared logZ scalar carries ~2e-3
    relative rounding -- row sums came out 1.0017 CONSTANT across frames, while in-model data
    sums to 1 within 1e-4. A STRUCTURAL normalization bug (e.g. meter-union double counting)
    would be off by ~2x or a state-count ratio, far outside the 5e-3 band asserted here."""
    rng = np.random.default_rng(3)
    model = make_model()
    set_params(model, TRUE_MU, TRUE_VAR)
    features = rng.normal(size=(200, D)) * 4.0  # arbitrary, far off-model
    post = model.class_posteriors(feat_tensor(model, features)).cpu().numpy()
    assert post.shape == (200, 3)
    assert np.all(post >= -1e-6) and np.all(post <= 1 + 5e-3)
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=5e-3)

    # in-model data: tight normalization
    f2, _ = steady_crop(rng)
    post2 = model.class_posteriors(feat_tensor(model, f2)).cpu().numpy()
    np.testing.assert_allclose(post2.sum(axis=1), 1.0, atol=1e-3)
    assert np.all(post2 >= -1e-6)


def test_class_posteriors_sanity_on_periodic_beats():
    rng = np.random.default_rng(4)
    model = make_model()
    set_params(model, TRUE_MU, TRUE_VAR)
    features, classes = steady_crop(rng, interval=25, num_beats=12)  # T=300
    post = model.class_posteriors(feat_tensor(model, features)).cpu().numpy()

    beat_frames = np.nonzero(classes == 1)[0]
    nobeat_frames = np.nonzero(classes == 0)[0]
    assert post[beat_frames, 1].mean() > 0.5, \
        f"mean beat posterior at beat-favoring frames = {post[beat_frames, 1].mean():.3f}"
    assert post[nobeat_frames, 0].mean() > 0.9, \
        f"mean no-beat posterior off the beat = {post[nobeat_frames, 0].mean():.3f}"
    agreement = (post.argmax(axis=1) == classes).mean()
    assert agreement > 0.9, f"posterior argmax agreement with true classes = {agreement:.3f}"


# ----------------------------------------------------------------------------------------------
# 5. EM MONOTONICITY (returned old-parameter marginals non-decreasing; true objective improves)
# ----------------------------------------------------------------------------------------------

def test_em_monotone_and_improves_true_objective():
    rng = np.random.default_rng(5)
    model = make_model()
    crops = []
    for k in range(3):
        f, _ = steady_crop(rng, interval=22 + 3 * k, num_beats=10)
        crops.append(feat_tensor(model, f))
    set_params(model, TRUE_MU + rng.normal(size=(3, D)) * 1.5,
               TRUE_VAR * rng.uniform(0.5, 2.0, size=(3, D)))

    lls = [float(model.em_step(crops)) for _ in range(5)]
    diffs = np.diff(lls)
    assert np.all(diffs >= -1e-4 * np.maximum(1.0, np.abs(np.array(lls[:-1])))), \
        f"EM marginal decreased: {lls}"

    final = sum(float(model.marginal_log_likelihood(c)) for c in crops)
    assert final >= lls[-1] - 1e-4 * max(1.0, abs(lls[-1])), \
        f"final true marginal {final} < last EM return {lls[-1]}"


# ----------------------------------------------------------------------------------------------
# 6. PARAMETER RECOVERY on data sampled FROM the model class
# ----------------------------------------------------------------------------------------------

def test_em_parameter_recovery():
    rng = np.random.default_rng(6)
    intervals = interval_grid()
    kernel = mixture_kernel(intervals)

    def sample_crop(seed_interval=25, num_beats=12, beats_per_bar=4):
        idx = int(np.nonzero(intervals == seed_interval)[0][0])
        beat_ivs = []
        for _ in range(num_beats):
            beat_ivs.append(int(intervals[idx]))
            idx = rng.choice(len(intervals), p=kernel[idx])
        classes = classes_from_beat_intervals(beat_ivs, beats_per_bar)
        return features_from_classes(classes, rng), classes

    crops, all_classes = [], []
    for _ in range(4):
        f, c = sample_crop()
        crops.append(f)
        all_classes.append(c)
    all_classes = np.concatenate(all_classes)

    model = make_model()
    set_params(model, TRUE_MU + rng.normal(size=(3, D)) * 1.0,
               TRUE_VAR * rng.uniform(0.5, 2.0, size=(3, D)))
    tcrops = [feat_tensor(model, f) for f in crops]
    for _ in range(8):
        model.em_step(tcrops)

    mu = model.mu.detach().cpu().numpy()
    err = np.abs(mu - TRUE_MU).max(axis=1)
    assert np.all(err < 0.5), f"recovered mu off by {err} per class (tol 0.5)"

    post_mass = np.zeros(3)
    total = 0
    for t in tcrops:
        p = model.class_posteriors(t).cpu().numpy()
        post_mass += p.sum(axis=0)
        total += len(p)
    post_prop = post_mass / total
    true_prop = np.bincount(all_classes, minlength=3) / len(all_classes)
    assert np.all(np.abs(post_prop - true_prop) < 0.05), \
        f"class proportions: model {post_prop} vs path {true_prop}"


# ----------------------------------------------------------------------------------------------
# 7. DECODE on model-generated steady data
# ----------------------------------------------------------------------------------------------

def test_decode_recovers_true_beat_grid():
    rng = np.random.default_rng(7)
    model = make_model()
    set_params(model, TRUE_MU, TRUE_VAR)
    interval = 25  # 120 BPM at fps 50, on the model's grid
    features, _ = steady_crop(rng, interval=interval, num_beats=12)  # 300 frames = 6 s
    out = model.decode(feat_tensor(model, features))
    assert "beats" in out and "downbeats" in out and "path" in out and "space" in out

    beats = np.asarray(out["beats"])
    downbeats = np.asarray(out["downbeats"])
    assert len(beats) >= 9, f"only {len(beats)} beats decoded (expected ~11)"
    frame_diffs = np.round(np.diff(beats) * FPS).astype(int)
    values, counts = np.unique(frame_diffs, return_counts=True)
    mode = values[np.argmax(counts)]
    assert mode == interval, f"mode inter-beat interval {mode} frames != true {interval}"
    assert 2 <= len(downbeats) <= 4, f"{len(downbeats)} downbeats for 12 beats (expect ~3)"
    for db in downbeats:
        assert np.min(np.abs(beats - db)) < 1e-9, "downbeat not in the beat list"
    if len(downbeats) >= 2:
        db_gaps = np.round(np.diff(downbeats) * FPS).astype(int)
        assert np.all(db_gaps == 4 * interval), f"downbeat gaps {db_gaps} != {4 * interval} frames"


# ----------------------------------------------------------------------------------------------
# 8. STRUCTURE MATTERS: corrupted off-beat frames must not break tempo continuity
# ----------------------------------------------------------------------------------------------

def test_decode_transition_resists_corrupted_frames():
    rng = np.random.default_rng(8)
    model = make_model()
    set_params(model, TRUE_MU, TRUE_VAR)
    interval = 25
    features, classes = steady_crop(rng, interval=interval, num_beats=12)
    clean_beats = np.asarray(model.decode(feat_tensor(model, features))["beats"])

    # Corrupt ISOLATED mid-beat frames (far from any true beat region) toward the beat Gaussian.
    corrupted = features.copy()
    mid_beat = np.nonzero((np.arange(len(classes)) % interval) == interval // 2)[0]
    corrupt_frames = mid_beat[2:8:2]  # a few isolated single frames
    corrupted[corrupt_frames] = TRUE_MU[1] + 0.3 * rng.standard_normal((len(corrupt_frames), D))
    dirty_beats = np.asarray(model.decode(feat_tensor(model, corrupted))["beats"])

    assert len(dirty_beats) >= 8, f"corruption destroyed decoding: {len(dirty_beats)} beats"
    dirty_diffs = np.diff(dirty_beats)
    assert np.all(dirty_diffs > 0.42) and np.all(dirty_diffs < 0.58), \
        f"beats no longer near-periodic after corruption: diffs {np.round(dirty_diffs, 3)}"
    matched = sum(np.min(np.abs(clean_beats - b)) <= 0.04 for b in dirty_beats)
    assert matched / len(dirty_beats) > 0.85, \
        f"only {matched}/{len(dirty_beats)} dirty beats within 40 ms of the clean grid"
    for cf in corrupt_frames:
        t = cf / FPS
        assert np.min(np.abs(dirty_beats - t)) > 3 / FPS, \
            f"decoder flipped a beat onto corrupted mid-beat frame {cf}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-x", "-q"]))
