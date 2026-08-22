"""GATE 0 for the sticky two-regime tempo kernel: is the construction VALID, and can it escape
the min-lambda tail theorem? Pure kernel algebra, no decoding.

0a validity, 0b single-step marginal + effective decay, 0c path-level sustained-vs-isolated
log-probability ratios under sticky vs memoryless mixture, 0d stationary regime distribution and
entry/exit tolls.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))

from rungs.bar_pointer.state_space import BarPointerStateSpace

FPS = 50.0
PAIRS = [(100.0, 1.0), (200.0, 5.0)]
EPS_GRID = [0.5, 0.2, 0.1, 0.03, 0.01, 0.003, 1e-9]
OUT = Path("/tmp/sticky/gate0.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


def log_kernel(intervals, lam):
    """Row-log-softmax of -lam*|r-1| -- exactly R2GenerativeLambda.log_kernel, in float64."""
    r = intervals[None, :].astype(np.float64) / intervals[:, None].astype(np.float64)
    s = -lam * np.abs(r - 1.0)
    return s - np.log(np.exp(s - s.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)) \
        - s.max(axis=1, keepdims=True)


def logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def sticky_kernel(K0, K1, eps, eta):
    V = K0.shape[0]
    R = np.log(np.array([[1.0 - eps, eps], [eta, 1.0 - eta]]))
    out = np.empty((2 * V, 2 * V))
    for a in range(2):
        for b in range(2):
            out[a * V:(a + 1) * V, b * V:(b + 1) * V] = R[a, b] + (K0 if b == 0 else K1)
    return out


def main():
    space = BarPointerStateSpace(FPS, 30.0, 215.0, 4, 6, num_tempi=None)
    iv = space.interval_frames.astype(np.int64)
    V = len(iv)
    rep = {"V": V, "interval_min": int(iv[0]), "interval_max": int(iv[-1]),
           "num_states_base": int(space.num_states)}
    print(f"V={V} intervals {iv[0]}..{iv[-1]}", flush=True)

    rep["cells"] = []
    for lam_s, lam_l in PAIRS:
        K0, K1 = log_kernel(iv, lam_s), log_kernel(iv, lam_l)
        for eps in EPS_GRID:
            eta = eps
            Kf = sticky_kernel(K0, K1, eps, eta)
            pi0 = eta / (eps + eta)
            pi = np.array([pi0, 1.0 - pi0])
            cell = {"lam_stiff": lam_s, "lam_loose": lam_l, "eps": eps, "eta": eta,
                    "pi": pi.tolist(),
                    "entry_toll_nats": float(-np.log(eps)), "exit_toll_nats": float(-np.log(eta))}

            # ---- 0a validity ----
            cell["max_abs_rowsum_minus_1_joint"] = float(
                np.abs(np.exp(logsumexp(Kf, axis=1)) - 1.0).max())
            cell["max_abs_rowsum_minus_1_K_stiff"] = float(
                np.abs(np.exp(logsumexp(K0, axis=1)) - 1.0).max())
            cell["max_abs_rowsum_minus_1_K_loose"] = float(
                np.abs(np.exp(logsumexp(K1, axis=1)) - 1.0).max())
            R = np.array([[1.0 - eps, eps], [eta, 1.0 - eta]])
            cell["max_abs_rowsum_minus_1_regime"] = float(np.abs(R.sum(1) - 1.0).max())
            # tiled uniform initial over 2V grid: mass per duplicated state = 1/(2*num_states)
            n_dup = 2 * int(space.num_states)
            cell["initial_total_mass"] = float(n_dup * (1.0 / n_dup))
            # stationarity of pi under R
            cell["pi_stationarity_residual"] = float(np.abs(pi @ R - pi).max())

            # ---- 0b single-step marginal under stationary regime ----
            # sum_a pi_a sum_b R[a,b] K_b  ==  pi0' K0 + pi1' K1 with pi' = pi R = pi
            marg = logsumexp(np.stack([
                logsumexp(np.stack([np.log(pi[a]) + Kf[a * V:(a + 1) * V, b * V:(b + 1) * V]
                                    for a in range(2)]), axis=0)
                for b in range(2)]), axis=0)
            mix = logsumexp(np.stack([np.log(pi[0]) + K0, np.log(pi[1]) + K1]), axis=0)
            cell["single_step_marginal_max_abs_diff"] = float(np.abs(marg - mix).max())
            cell["mixture_weight_w_equals_pi0"] = float(pi[0])

            # effective decay of the single-step marginal, as a function of d = |r-1|
            # (analytic continuation of the same functional form; Z per component from row i0)
            i0 = 50  # 60 bpm at 50 fps
            row = int(np.where(iv == i0)[0][0])
            Z0 = np.exp(-lam_s * np.abs(iv / i0 - 1.0)).sum()
            Z1 = np.exp(-lam_l * np.abs(iv / i0 - 1.0)).sum()

            def logK(d):
                return np.log(pi[0] * np.exp(-lam_s * d) / Z0 + pi[1] * np.exp(-lam_l * d) / Z1)

            h = 1e-6
            cell["lam_eff"] = {f"{d}": float(-(logK(d + h) - logK(d - h)) / (2 * h))
                               for d in (0.02, 0.1, 0.3, 0.6)}
            cell["min_lambda"] = float(min(lam_s, lam_l))

            # ---- 0c path-level: sustained vs isolated ----
            # total tempo change D over n beat boundaries; intervals snapped to the integer grid.
            paths = {}
            for D in (0.24, 0.5):
                for n in (1, 2, 4, 8):
                    sustained = [i0] + [int(round(i0 * (1.0 + D) ** ((k + 1) / n)))
                                        for k in range(n)]
                    isolated = [i0, int(round(i0 * (1.0 + D)))] + \
                               [int(round(i0 * (1.0 + D)))] * (n - 1)
                    if sustained[-1] != isolated[-1] or max(sustained) > iv[-1]:
                        continue
                    idx = {int(v): int(np.where(iv == v)[0][0]) for v in set(sustained + isolated)}

                    def mix_lp(path):
                        return float(sum(mix[idx[path[k]], idx[path[k + 1]]]
                                         for k in range(len(path) - 1)))

                    def sticky_lp(path):
                        alpha = np.log(pi)                       # over regimes
                        for k in range(len(path) - 1):
                            i, j = idx[path[k]], idx[path[k + 1]]
                            emit = np.array([K0[i, j], K1[i, j]])
                            alpha = logsumexp(alpha[:, None] + np.log(R), axis=0) + emit
                        return float(logsumexp(alpha))

                    paths[f"D{D}_n{n}"] = {
                        "sustained_intervals": sustained, "isolated_intervals": isolated,
                        "mix_sustained": mix_lp(sustained), "mix_isolated": mix_lp(isolated),
                        "mix_ratio_nats": mix_lp(sustained) - mix_lp(isolated),
                        "sticky_sustained": sticky_lp(sustained),
                        "sticky_isolated": sticky_lp(isolated),
                        "sticky_ratio_nats": sticky_lp(sustained) - sticky_lp(isolated),
                    }
                    p = paths[f"D{D}_n{n}"]
                    p["sticky_minus_mix_ratio_nats"] = p["sticky_ratio_nats"] - p["mix_ratio_nats"]
            cell["paths"] = paths
            rep["cells"].append(cell)
            print(f"pair({lam_s},{lam_l}) eps={eps}: pi0={pi[0]:.6f} "
                  f"rowsum_err={cell['max_abs_rowsum_minus_1_joint']:.3e} "
                  f"marg_diff={cell['single_step_marginal_max_abs_diff']:.3e} "
                  f"lam_eff={cell['lam_eff']}", flush=True)
            for k, p in paths.items():
                print(f"   {k}: mix ratio {p['mix_ratio_nats']:+.6f}  "
                      f"sticky ratio {p['sticky_ratio_nats']:+.6f}  "
                      f"delta {p['sticky_minus_mix_ratio_nats']:+.6f}", flush=True)

    OUT.write_text(json.dumps(rep, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
