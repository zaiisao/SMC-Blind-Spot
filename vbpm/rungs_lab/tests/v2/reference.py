"""Independent reference implementation of docs/SPEC.md SS4 (model), SS6 (data), SS8 (eval).

This file deliberately does NOT import ``vbpm``. It is the oracle the implementation is
checked against, so it must be derivable from the spec text alone. Everything here is
brute force and slow; correctness is the only goal.

Self-tested by ``test_reference_selftest.py`` -- those tests pass with no implementation
present, which is what makes this file usable as an oracle rather than a second guess.

Section references point at docs/SPEC.md. They previously pointed at SPEC_v2_stage0.md,
which docs/SPEC.md SS0 marks NOT TRUSTED -- so every SS in this tree resolved to a
different section than it named. Remapped 2026-07-30.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

# Constants SS5/SS6.2 name but do not give a home to.
MIN_BARS = 3
DOWNBEAT_TOL_S = 0.02
ALIGN_Z_MIN = 1.0
DEFAULT_VALUES = (2, 3, 4)
DEFAULT_FPS = 50.0


# --------------------------------------------------------------------------------------
# numerics
# --------------------------------------------------------------------------------------
def log_sigmoid(x: float) -> float:
    """log(1 / (1 + exp(-x))), stable for |x| large."""
    return -math.log1p(math.exp(-abs(x))) + (0.0 if x >= 0 else x)


def log1m_sigmoid(x: float) -> float:
    """log(1 - sigmoid(x)) = log_sigmoid(-x)."""
    return log_sigmoid(-x)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def logsumexp(vals) -> float:
    vals = list(vals)
    mx = max(vals)
    if mx == -math.inf:
        return -math.inf
    return mx + math.log(sum(math.exp(v - mx) for v in vals))


# --------------------------------------------------------------------------------------
# SS4.3 emission  p_theta(y | m), bar offset r marginalised
# --------------------------------------------------------------------------------------
def offset_logps(y, m: int, alpha: float, beta: float) -> list[float]:
    """log prod_i Bern(y_i; pi_i(r)) for each offset r in 0..m-1. SS4.3 inner product."""
    la, l1a = log_sigmoid(alpha), log1m_sigmoid(alpha)
    lb, l1b = log_sigmoid(beta), log1m_sigmoid(beta)
    out = []
    for r in range(m):
        acc = 0.0
        for i, yi in enumerate(y):
            on_downbeat = ((i - r) % m) == 0
            if on_downbeat:
                acc += la if yi else l1a
            else:
                acc += lb if yi else l1b
        out.append(acc)
    return out


def emission_logp(y, m: int, alpha: float, beta: float) -> float:
    """log p_theta(y | m) = log[ (1/m) sum_r prod_i Bern(y_i; pi_i(r)) ].  SS4.3"""
    return logsumexp(offset_logps(y, m, alpha, beta)) - math.log(m)


def emission_logp_all(y, alpha: float, beta: float, values=DEFAULT_VALUES) -> np.ndarray:
    """Appendix A ``Emission.logp_all`` -- every meter at once, ordered by ``values``."""
    return np.array([emission_logp(y, m, alpha, beta) for m in values], dtype=np.float64)


def emission_grad(y, m: int, alpha: float, beta: float) -> tuple[float, float]:
    """Analytic d log p_theta(y|m) / d(alpha, beta).  Used by T-EM-3.

    With L_r the offset-r log-likelihood and w_r = softmax_r(L_r) the posterior over the
    marginalised offset, and using d/dx log Bern(y; sigmoid(x)) = y - sigmoid(x):

        d logp / d alpha = sum_r w_r * sum_{i : (i-r) mod m == 0} (y_i - sigmoid(alpha))
        d logp / d beta  = sum_r w_r * sum_{i : (i-r) mod m != 0} (y_i - sigmoid(beta))
    """
    L = offset_logps(y, m, alpha, beta)
    Z = logsumexp(L)
    w = [math.exp(v - Z) for v in L]
    pa, pb = sigmoid(alpha), sigmoid(beta)
    ga = gb = 0.0
    for r in range(m):
        sa = sum((yi - pa) for i, yi in enumerate(y) if ((i - r) % m) == 0)
        sb = sum((yi - pb) for i, yi in enumerate(y) if ((i - r) % m) != 0)
        ga += w[r] * sa
        gb += w[r] * sb
    return ga, gb


# --------------------------------------------------------------------------------------
# SS4.6 objective and exact posterior
# --------------------------------------------------------------------------------------
def log_evidence(y, prior_logp, alpha: float, beta: float, values=DEFAULT_VALUES) -> float:
    """log p(y | h) = logsumexp_m [ log p_theta(y|m) + log p_psi(m|h) ]."""
    lp = emission_logp_all(y, alpha, beta, values)
    return logsumexp(lp + np.asarray(prior_logp, dtype=np.float64))


def exact_posterior(y, prior_logp, alpha: float, beta: float,
                    values=DEFAULT_VALUES) -> np.ndarray:
    """SS4.6 p(m | y, h), returned as LOG probs (C5)."""
    lp = emission_logp_all(y, alpha, beta, values) + np.asarray(prior_logp, np.float64)
    return lp - logsumexp(lp)


def elbo_terms(q_logp, y, prior_logp, alpha: float, beta: float,
               values=DEFAULT_VALUES) -> dict:
    """SS4.6 exact-enumeration ELBO plus the amortization gap. All nats, natural log."""
    q_logp = np.asarray(q_logp, dtype=np.float64)
    prior_logp = np.asarray(prior_logp, dtype=np.float64)
    q = np.exp(q_logp)
    lik = emission_logp_all(y, alpha, beta, values)
    recon = float((q * lik).sum())
    kl = float((q * (q_logp - prior_logp)).sum())
    ex = exact_posterior(y, prior_logp, alpha, beta, values)
    p = np.exp(ex)
    # FORWARD KL(p || q) -- reported for continuity, but see gap_reverse below.
    gap = float((p * (ex - q_logp)).sum())
    # The ELBO's actual slack is the REVERSE KL(q || p). These differ; SS4.6 says so.
    gap_reverse = float((q * (q_logp - ex)).sum())
    return {
        "recon": recon,
        "kl": kl,
        "elbo": recon - kl,
        "q_logp": q_logp,
        "exact_logp": ex,
        "gap": gap,
        "gap_reverse": gap_reverse,
        "log_evidence": log_evidence(y, prior_logp, alpha, beta, values),
    }


# --------------------------------------------------------------------------------------
# SS6.2 m_true derivation
# --------------------------------------------------------------------------------------
def derive_m_true(beats_s, downs_s, min_bars: int = MIN_BARS,
                  tol_s: float = DOWNBEAT_TOL_S):
    """SS6.2: median over complete bars of the beat count in [d_k, d_{k+1}).

    Returns ``None`` when there are fewer than ``min_bars`` complete bars, which the spec
    says must reject the crop. The .5 tie-break on an even bar count is NOT specified by
    SS6.2 pins it as half-up; this reference implements that and the test holds it.
    """
    beats_s = np.asarray(beats_s, dtype=np.float64)
    downs_s = np.asarray(downs_s, dtype=np.float64)
    n_bars = len(downs_s) - 1
    if n_bars < min_bars:
        return None
    counts = []
    for k in range(n_bars):
        lo, hi = downs_s[k], downs_s[k + 1]
        counts.append(int(np.sum((beats_s >= lo - tol_s) & (beats_s < hi - tol_s))))
    med = float(np.median(counts))
    return int(math.floor(med + 0.5))


# --------------------------------------------------------------------------------------
# SS6.4 synthetic bench
# --------------------------------------------------------------------------------------
def make_synth_song(m: int, bar_s: float, n_bars: int, *, fps: float = DEFAULT_FPS,
                    sigma_s: float = 0.06, rng=None, noise: float = 0.0,
                    jitter_s: float = 0.0, miss_prob: float = 0.0,
                    extra_prob: float = 0.0) -> dict:
    """One synthetic song per SS6.4.

    Beats equidistant at bar_s/m, downbeats every bar_s; downbeats are a
    subset of beats (SS6.2).
    h channel 0 = Gaussian bumps at all beats, channel 1 = bumps at downbeats only.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    beat_period = bar_s / m
    n_beats = n_bars * m
    beats = np.arange(n_beats, dtype=np.float64) * beat_period + 0.5
    downs = beats[::m].copy()
    if jitter_s > 0:
        shift = rng.normal(0.0, jitter_s, size=beats.shape)
        # keep downbeats a subset of beats (SS6.2): jitter beats, then re-read downbeats
        beats = beats + shift
        beats = np.sort(beats)
        downs = beats[::m].copy()

    duration = float(beats[-1] + 2.0 * beat_period)
    n_frames = int(math.ceil(duration * fps))
    t = (np.arange(n_frames, dtype=np.float64) + 0.5) / fps  # C4 frame centres
    h = np.zeros((n_frames, 2), dtype=np.float32)

    def bumps(times):
        acc = np.zeros(n_frames, dtype=np.float64)
        for tc in times:
            acc += np.exp(-0.5 * ((t - tc) / sigma_s) ** 2)
        return acc

    beat_times = list(beats)
    down_times = list(downs)
    if miss_prob > 0:
        beat_times = [b for b in beat_times if rng.random() >= miss_prob]
    if extra_prob > 0:
        n_extra = rng.binomial(len(beats), extra_prob)
        beat_times += list(rng.uniform(t[0], t[-1], size=n_extra))
    h[:, 0] = bumps(beat_times)
    h[:, 1] = bumps(down_times)
    if noise > 0:
        h += rng.normal(0.0, noise, size=h.shape).astype(np.float32)

    y = np.zeros(len(beats), dtype=np.uint8)
    y[::m] = 1
    return {
        "h": h, "beats_s": beats, "downs_s": downs, "y": y, "m_true": int(m),
        "bar_s": float(bar_s), "n_frames": n_frames, "fps": float(fps),
    }


def make_synth_dataset(n_per_class: int, *, values=DEFAULT_VALUES, seed: int = 0,
                       bar_min_s: float = 1.6, bar_max_s: float = 3.2,
                       n_beats: tuple[int, int] = (24, 36), n_bars: int | None = None,
                       stem_prefix: str = "s", **song_kw) -> list[dict]:
    """Balanced-by-construction bench (SS6.4). The data is deliberately trivial; the one thing
    it must not be is *bypassable*.

    Two independent leaks are closed here, both of which let a model score perfectly without
    ever consulting the downbeat channel:

    The **beat** period is drawn independently of ``m``, so tempo carries no meter
    information. Note this is the opposite of drawing the BAR period independently, which an
    earlier version did: ``beat_period = bar_s / m`` is an identity, so exactly one of the two
    can be independent of ``m``. Fixing ``bar_s`` leaves ``corr(m, beat_rate) = +0.80`` and a
    downbeat-blind classifier can read the meter off tempo alone. Fixing the beat period
    instead pushes the correlation onto ``bar_s`` -- which is *only* observable by finding
    downbeats, i.e. the mechanism under test. It is also the physically honest choice: real
    music holds beat rate in a human range and lets bar duration vary with meter.

    ``n_bars`` is derived per song from a target BEAT count rather than held fixed, so the
    number of beats carries none either. Held fixed, ``n_beats = n_bars * m`` makes
    ``corr(m, n_beats) == 1.0`` exactly, and a downbeat-BLIND classifier that counts
    beat-channel peaks scores balanced accuracy 1.000 -- the emission never has to work.
    Drawing ``n_beats`` first and setting ``n_bars = round(n_beats / m)`` puts every meter on
    the same beat-count support, so ``m`` is readable only from downbeat SPACING, which is
    the mechanism under test.

    The data stays noise-free and exactly periodic: this is the "simplest data imaginable"
    sanity bench, not a benchmark. Pass ``n_bars=`` to pin the old fixed-bar behaviour.
    """
    rng = np.random.default_rng(seed)
    songs = []
    # beat period range, carried over from the old bar range at the middle meter so the
    # bench keeps roughly the same durations as before
    beat_min_s, beat_max_s = bar_min_s / 3.0, bar_max_s / 3.0
    for j in range(n_per_class):
        for m in values:
            beat_s = float(rng.uniform(beat_min_s, beat_max_s))   # independent of m
            bar_s = beat_s * m                                    # identity, so bar_s does vary
            if n_bars is None:
                target = int(rng.integers(n_beats[0], n_beats[1] + 1))
                bars = max(MIN_BARS, int(round(target / m)))
            else:
                bars = n_bars
            song = make_synth_song(m, bar_s, bars, rng=rng, **song_kw)
            song["stem"] = f"{stem_prefix}{j:04d}_m{m}"
            songs.append(song)
    return songs


# --------------------------------------------------------------------------------------
# SS8 baselines
# --------------------------------------------------------------------------------------
def pick_peaks(x, threshold: float = 0.5, min_distance: int = 2) -> np.ndarray:
    """Strict local maxima above ``threshold``, greedily thinned to ``min_distance``."""
    x = np.asarray(x, dtype=np.float64)
    cand = [i for i in range(1, len(x) - 1)
            if x[i] > threshold and x[i] >= x[i - 1] and x[i] > x[i + 1]]
    out = []
    for i in cand:
        if not out or i - out[-1] >= min_distance:
            out.append(i)
        elif x[i] > x[out[-1]]:
            out[-1] = i
    return np.asarray(out, dtype=int)


def peak_count_estimate(h, values=DEFAULT_VALUES, threshold: float = 0.5) -> int:
    """SS8 peak-count baseline: DEPLOYABLE, uses only ``h``.

    beats-per-bar ~= (#peaks in the beat channel) / (#peaks in the downbeat channel).
    """
    n_beat = len(pick_peaks(h[:, 0], threshold))
    n_down = len(pick_peaks(h[:, 1], threshold))
    if n_down == 0:
        return max(values)
    est = n_beat / n_down
    return min(values, key=lambda m: (abs(m - est), m))


def majority_predict(train_m, values=DEFAULT_VALUES) -> int:
    counts = {m: 0 for m in values}
    for m in train_m:
        counts[int(m)] += 1
    return max(values, key=lambda m: (counts[m], -m))


# --------------------------------------------------------------------------------------
# SS8 / SS8 metrics and collapse detectors
# --------------------------------------------------------------------------------------
def balanced_accuracy(true_m, pred_m, values=DEFAULT_VALUES) -> float:
    """Mean over PRESENT classes of per-class recall (SS8 primary metric)."""
    true_m = np.asarray(true_m)
    pred_m = np.asarray(pred_m)
    recalls = []
    for m in values:
        sel = true_m == m
        if sel.sum() == 0:
            continue
        recalls.append(float((pred_m[sel] == m).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def confusion(true_m, pred_m, values=DEFAULT_VALUES) -> np.ndarray:
    idx = {m: i for i, m in enumerate(values)}
    C = np.zeros((len(values), len(values)), dtype=int)
    for t, p in zip(np.asarray(true_m), np.asarray(pred_m)):
        C[idx[int(t)], idx[int(p)]] += 1
    return C


def nll_true(logp, true_m, values=DEFAULT_VALUES) -> float:
    idx = {m: i for i, m in enumerate(values)}
    logp = np.asarray(logp, dtype=np.float64)
    return float(-np.mean([logp[b, idx[int(m)]] for b, m in enumerate(true_m)]))


def collapse_ratio(logits) -> float:
    """SS8 offset-to-signal ratio, PINNED definition (spec is ambiguous -- gaps G-09).

    Decompose logits[B, K] into a crop-independent class bias and a crop-dependent
    residual, both with the per-crop (class-independent) offset removed since softmax is
    invariant to it:

        centred[b, k] = logits[b, k] - mean_k logits[b, k]
        bias[k]       = mean_b centred[b, k]
        residual      = centred - bias
        ratio         = ||bias||_rms / ||residual||_rms

    Large ratio = the model prefers the same class for every crop with no per-crop
    signal, i.e. the v1 collapse (167x at init, 1.2e7 after training).
    """
    z = np.asarray(logits, dtype=np.float64)
    centred = z - z.mean(axis=1, keepdims=True)
    bias = centred.mean(axis=0)
    residual = centred - bias
    num = float(np.sqrt((bias ** 2).mean()))
    den = float(np.sqrt((residual ** 2).mean()))
    return num / den if den > 0 else math.inf


def distinct_predicted(pred_m) -> int:
    return int(len(set(int(p) for p in np.asarray(pred_m).ravel())))


# --------------------------------------------------------------------------------------
# SS6.3 alignment guard
# --------------------------------------------------------------------------------------
def alignment_z(h, beats_s, fps: float = DEFAULT_FPS, n_null: int = 64,
                seed: int = 0) -> float:
    """z-score of beat-channel activation AT annotated beats vs a shifted null.

    SS6.3 mandates an alignment statistic but never defines z (still open). This reference
    pins one definition so the guard is testable: mean activation sampled at the beat
    frames, standardised against the same statistic under random circular shifts.
    """
    h = np.asarray(h, dtype=np.float64)
    ch = h[:, 0]
    T = len(ch)
    idx = np.clip(np.round(np.asarray(beats_s) * fps - 0.5).astype(int), 0, T - 1)
    obs = float(ch[idx].mean())
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_null):
        s = int(rng.integers(1, T))
        null.append(float(np.roll(ch, s)[idx].mean()))
    mu, sd = float(np.mean(null)), float(np.std(null))
    return (obs - mu) / sd if sd > 0 else math.inf


# --------------------------------------------------------------------------------------
# helpers for brute-force tests
# --------------------------------------------------------------------------------------
def all_binary(n: int):
    """All y in {0,1}^n -- T-EM-1 normalisation check."""
    return [np.array(bits, dtype=np.uint8) for bits in itertools.product((0, 1), repeat=n)]


def cyclic_shift(y, k: int) -> np.ndarray:
    return np.roll(np.asarray(y), k)
