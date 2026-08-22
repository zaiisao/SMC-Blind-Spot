"""GATE 4: the batched forward must equal the certified unbatched forward, exactly.

structured_dp.forward_log_likelihood carries a certificate chain (dense DP -> hmmlearn /
torch-struct; R1 == madmom event-for-event). The batched version inherits none of that on its own
word, so it is pinned to the original here rather than trusted: same tracks, same emissions, batched
vs one-at-a-time, in float64.

Three things are checked, because equal log Z alone is not enough for a TRAINING path:
  1. log Z agreement, batched vs unbatched.
  2. GRADIENT agreement -- the whole point is backprop, and a reduction over the wrong axis can
     leave log Z right while scrambling d logZ / d emission across batch members.
  3. Variable-length handling. Padding with zeros is NOT harmless (log p = 0 means probability 1, so
     the pointer keeps advancing and accumulates transition mass), so ragged batches are compared
     against per-track unbatched calls on the unpadded emissions.
Plus the speedup, since that is the only reason this exists.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from batched_dp import forward_log_likelihood_batched

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:3"


def main():
    m = make_beatonly_dbn(30.0, 16, 100, DEV)
    dp0, s2c = m.dynamic_programs[0], m.state_to_classes[0]
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition

    ems = []
    for tid in track_ids():
        got = emissions_beatonly(m, np.asarray(bthis_beat_prob(tid), np.float64))
        if got is not None:
            ems.append(got[0])
        if len(ems) == 12:
            break
    Tmin = min(e.shape[0] for e in ems)

    # ---- 1 & 2: rectangular batch, log Z and gradient ----
    stack = torch.stack([e[:Tmin] for e in ems], 0).clone().requires_grad_(True)
    zb = forward_log_likelihood_batched(dp0, logpi, logA, stack, state_to_class=s2c)
    gb, = torch.autograd.grad(zb.sum(), stack)

    zu, gu = [], []
    for e in ems:
        x = e[:Tmin].clone().requires_grad_(True)
        z = dp0.forward_log_likelihood(logpi, logA, x, state_to_class=s2c)
        g, = torch.autograd.grad(z, x)
        zu.append(z.detach())
        gu.append(g)
    zu, gu = torch.stack(zu), torch.stack(gu)

    dz = float((zb.detach() - zu).abs().max())
    dg = float((gb - gu).abs().max())
    print(f"rectangular batch B={len(ems)} T={Tmin}")
    print(f"  max |log Z_batched - log Z_unbatched|      : {dz:.3e}")
    print(f"  max |dlogZ/dem  batched - unbatched|       : {dg:.3e}")
    print(f"  log Z magnitude (for scale)                : {float(zu.abs().max()):.1f}")

    # ---- 3: ragged batch ----
    lens = torch.tensor([Tmin - 137 * i for i in range(len(ems))], device=DEV)
    Tmax = int(lens.max())
    pad = torch.zeros(len(ems), Tmax, 3, dtype=ems[0].dtype, device=DEV)
    for i, e in enumerate(ems):
        pad[i, :lens[i]] = e[:lens[i]]
    zr = forward_log_likelihood_batched(dp0, logpi, logA, pad, state_to_class=s2c, lengths=lens)
    zref = torch.stack([dp0.forward_log_likelihood(logpi, logA, ems[i][:lens[i]],
                                                   state_to_class=s2c)
                        for i in range(len(ems))])
    dr = float((zr - zref).abs().max())
    print(f"ragged batch lengths {lens.tolist()[:4]}...")
    print(f"  max |log Z_ragged - per-track unbatched|   : {dr:.3e}")

    # ---- speedup ----
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(3):
        forward_log_likelihood_batched(dp0, logpi, logA, stack.detach(), state_to_class=s2c)
    torch.cuda.synchronize(); tb = (time.time() - t0) / 3
    t0 = time.time()
    for _ in range(3):
        for e in ems:
            dp0.forward_log_likelihood(logpi, logA, e[:Tmin], state_to_class=s2c)
    torch.cuda.synchronize(); tu = (time.time() - t0) / 3
    print(f"  batched {tb:.3f} s   unbatched {tu:.3f} s   speedup {tu/tb:.1f}x  (B={len(ems)})")

    ok = dz < 1e-8 and dg < 1e-10 and dr < 1e-8
    print(f"\nGATE 4: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
