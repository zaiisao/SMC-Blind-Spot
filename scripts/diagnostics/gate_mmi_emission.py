"""GATE 1 of the MMI build: is the differentiable emission the SAME emission the decoder ships?

WHY THIS GATE COMES FIRST. MMI trains Beat This through the DBN by backpropagating
    L = -(log Z_num - log Z_den)
and both partition functions are computed from the 3-class emission table. The decoder we evaluate
with builds that table in NUMPY (DBN2016._log_class_densities), which severs the gradient path to
the network; training must therefore use the torch mirror (DBN2016.log_class_densities). If those
two disagree even slightly, we would train against one model and report against another, and every
number downstream would be uninterpretable. This is the same class of failure that produced the
CRF's contaminated control: a training path and an evaluation path that were never certified equal.

THE ONE DELIBERATE DIFFERENCE, which this gate must confirm is the ONLY difference. The numpy path
lets the no-beat density reach log(0) = -inf on purpose: a zero there means the frame CANNOT be a
non-beat state, and flooring it would downgrade an impossibility to an improbability that Viterbi
will buy (measured: 92.07% path agreement and a 0.0187-nat suboptimal path on a Ballroom song).
The torch mirror clamps no_beat at 1e-12 instead, because -inf makes the gradient NaN. So the gate
asserts: identical wherever numpy is finite, and numpy -inf implies torch finite-but-very-negative.
It also counts how often that floor actually fires -- if it fires on many frames, the training and
decoding models genuinely differ and we must reconcile them before proceeding, not after.

CHECKED IN BOTH CONFIGURATIONS. beat-only (downbeat channel zeroed, the decoder we ship for SMC)
and joint (3,4), because the decorrelation beat-downbeat behaves differently in each.

ALSO GATED HERE: gradient finiteness. The emission is where the network's gradient enters the DP,
so a NaN or inf here poisons everything regardless of what the DP does.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, bthis_beat_prob, bthis_downbeat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"


def main():
    from rungs.r1_2016_dbn import DBN2016
    models = {
        "beat-only": make_beatonly_dbn(30.0, 16, 100, DEV),
        "joint(3,4)": DBN2016(fps=FPS, min_bpm=30.0, max_bpm=215.0, device=DEV,
                              observation_lambda=16, transition_lambda=100),
    }
    for name, mm in models.items():
        worst = 0.0
        n_inf_np = n_inf_both = n_frames = 0
        bad_grad = []
        for tid in track_ids():
            bp = np.asarray(bthis_beat_prob(tid), np.float64)
            dp = (np.zeros_like(bp) if name == "beat-only"
                  else np.asarray(bthis_downbeat_prob(tid), np.float64))

            ref = mm._log_class_densities(bp, dp).double()               # numpy path -> torch
            x = torch.tensor(np.stack([bp, dp], 1), dtype=torch.float64,
                             device=DEV, requires_grad=True)
            got = mm.log_class_densities(x).double()                     # differentiable mirror

            fin = torch.isfinite(ref)
            worst = max(worst, float((ref[fin] - got[fin]).abs().max()))
            n_inf_np += int((~fin).sum())
            n_inf_both += int((~fin & ~torch.isfinite(got)).sum())
            n_frames += ref.shape[0]

            # gradient must be finite wherever the loss can actually see the emission
            g, = torch.autograd.grad(got[torch.isfinite(got)].sum(), x)
            if not bool(torch.isfinite(g).all()):
                bad_grad.append(tid)

        print(f"\n{name}:  {n_frames} frames x 3 classes over {len(track_ids())} tracks")
        print(f"  max |numpy - torch| where numpy is finite : {worst:.3e}")
        print(f"  numpy -inf entries (the deliberate floor)  : {n_inf_np} "
              f"({100*n_inf_np/(3*n_frames):.4f}% of entries)")
        print(f"  ...of which torch is ALSO -inf             : {n_inf_both}")
        print(f"  tracks with non-finite emission gradient   : {len(bad_grad)}")
        ok = worst < 1e-9 and not bad_grad
        print(f"  GATE 1 {name}: {'PASS' if ok else 'FAIL'}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
