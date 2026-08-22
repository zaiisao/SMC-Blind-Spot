"""GATE 0c-bis: the decisive price-structure probe.

Same NUMBER of loose moves and the same total tempo change, arranged two ways:
  clustered    -- n consecutive moves of `step` frames, then stays
  interleaved  -- the same n moves, each separated by a stay at r = 1
Both paths have the same length (2n states) and the same endpoint, so any difference is purely
the price of CORRELATION. Under a memoryless mixture the two must cost the same (up to integer
grid quantization); under a sticky regime the clustered arrangement pays the entry toll once.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))

from exp_sticky_gate0 import log_kernel, logsumexp
from rungs.bar_pointer.state_space import BarPointerStateSpace

PAIRS = [(100.0, 1.0), (200.0, 5.0)]
EPS_GRID = [0.5, 0.2, 0.1, 0.03, 0.01, 0.003, 1e-9]
OUT = Path("/tmp/sticky/gate0c_bis.json")


def main():
    sp = BarPointerStateSpace(50.0, 30.0, 215.0, 4, 6, num_tempi=None)
    iv = sp.interval_frames.astype(np.int64)
    idx = {int(v): i for i, v in enumerate(iv)}
    out = {}
    for lam_s, lam_l in PAIRS:
        K0, K1 = log_kernel(iv, lam_s), log_kernel(iv, lam_l)
        for eps in EPS_GRID:
            pi = np.array([0.5, 0.5])
            R = np.array([[1 - eps, eps], [eps, 1 - eps]])
            mix = logsumexp(np.stack([np.log(.5) + K0, np.log(.5) + K1]), axis=0)

            def mlp(p):
                return sum(mix[idx[p[k]], idx[p[k + 1]]] for k in range(len(p) - 1))

            def slp(p):
                a = np.log(pi)
                for k in range(len(p) - 1):
                    i, j = idx[p[k]], idx[p[k + 1]]
                    a = logsumexp(a[:, None] + np.log(R), axis=0) + np.array([K0[i, j], K1[i, j]])
                return logsumexp(a)

            for n in (2, 3, 4):
                for step in (4, 6):
                    i0 = 50
                    clustered = ([i0 + step * k for k in range(n + 1)]
                                 + [i0 + step * n] * (n - 1))[:2 * n]
                    interleaved = [i0]
                    for k in range(n):
                        interleaved += [i0 + step * (k + 1), i0 + step * (k + 1)]
                    interleaved = interleaved[:2 * n]
                    assert clustered[-1] == interleaved[-1] and len(clustered) == len(interleaved)
                    if max(clustered) > iv[-1]:
                        continue
                    d_mix = mlp(clustered) - mlp(interleaved)
                    d_st = slp(clustered) - slp(interleaved)
                    out[f"{lam_s}/{lam_l}|eps{eps}|n{n}|s{step}"] = {
                        "mix_clustered_minus_interleaved": d_mix,
                        "sticky_clustered_minus_interleaved": d_st,
                        "delta": d_st - d_mix}
                    print(f"pair({lam_s:.0f},{lam_l:.0f}) eps={eps:<7} n={n} step={step}: "
                          f"mix {d_mix:+.6f}  sticky {d_st:+.6f}  delta {d_st - d_mix:+.6f}",
                          flush=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
