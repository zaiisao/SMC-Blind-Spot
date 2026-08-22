"""Paired stats shared by the overnight batch: Wilcoxon + 10k paired bootstrap CI."""
import numpy as np
from scipy.stats import wilcoxon

METRICS = ["F", "CMLt", "AMLt", "CMLc", "AMLc"]


def paired(a, b, n_boot=10000, seed=0):
    """a, b: equal-length arrays of per-track scores. Tests a - b."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        p = 1.0
    else:
        try:
            p = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
        except Exception:
            p = float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "p_wilcoxon": p,
            "ci95_lo": float(lo), "ci95_hi": float(hi), "n": int(len(d))}


def summarize(per_track):
    """per_track: list of metric dicts -> mean row."""
    return {k: float(np.mean([x[k] for x in per_track])) for k in METRICS}


def compare_row(name_a, per_a, name_b, per_b, metrics=METRICS):
    out = {"a": name_a, "b": name_b}
    for m in metrics:
        out[m] = paired([x[m] for x in per_a], [x[m] for x in per_b])
    return out
