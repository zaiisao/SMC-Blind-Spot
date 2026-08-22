"""Can the MMI objective be driven DOWN at all? Single track, per-step, seconds not epochs.

WHY THIS EXISTS. The fold-0 sweep logged only per-EPOCH means, so a wandering loss (5.88 -> 6.49 ->
6.15) was indistinguishable from a converging one, from a broken one, and from a model that had
barely moved. Per-step logging on a single track answers the prior question -- IS THIS OBJECTIVE
OPTIMISABLE -- in seconds, and it should have been the first thing run.

THE TEST. Take one track, one fixed crop, and take repeated optimiser steps on that same crop.
Overfitting a single example is not a result about generalisation and is not meant to be; it is the
minimal check that the loss surface is descendable and the gradient chain is connected. If MMI/frame
does not fall on ONE track it cannot possibly help on 138, and every downstream number is noise.

READING IT:
  falls steadily              -> the objective is fine; the sweep's problem was step size / schedule
  falls only with BCE present -> the MMI gradient alone cannot make progress
  flat or rising              -> broken chain, or a loss surface with no descent direction here
  falls then explodes         -> step size too large for a term whose gradient is bounded by 1/L

Reported alongside is what the decoded output does, because a falling loss that does not improve
decoding would itself be informative -- it would mean the numerator is being satisfied in a way the
read-out does not reward.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, load_gt, ev
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events
from mmi import annotation_frames, numerator_mask, mmi_loss

SMC_AUD = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="smc_001")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--mmi-w", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--crop", type=int, default=1500)
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()
    dev = torch.device(args.device)

    from beat_this.inference import Audio2Frames
    import beat_this.model.loss as btloss
    a2f = Audio2Frames(checkpoint_path="fold0", device=args.device)
    model = a2f.model
    bce = btloss.ShiftTolerantBCELoss(pos_weight=1).to(dev)

    m = make_beatonly_dbn(30.0, 16, 100, args.device)
    space, dp0 = m.state_spaces[0], m.dynamic_programs[0]
    s2c, iv = m.state_to_classes[0], space.interval_frames
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition
    edtype = logpi.dtype

    y, sr = sf.read(str(SMC_AUD / f"{args.track.upper()}.wav"))
    sp = a2f.signal2spect(y, sr)[:args.crop].to(dev)
    L = sp.shape[0]
    gt = np.asarray(load_gt(args.track), float)
    fr = annotation_frames(gt, FPS, 0, L)
    tgt = torch.zeros(L, device=dev)
    tgt[torch.from_numpy(fr).to(dev)] = 1.0
    msk = torch.ones(1, L, dtype=torch.bool, device=dev)
    mask, n_anch, n_bad = numerator_mask(fr, L, iv, dev, edtype)
    print(f"{args.track}: {L} frames, {len(fr)} annotations, {n_anch} anchored, {n_bad} bad spans")
    print(f"mmi_w={args.mmi_w} alpha={args.alpha} lr={args.lr} clip={args.clip}\n", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    model.train()
    print(f"{'step':>5}{'MMI/frame':>12}{'BCE':>9}{'|g|':>10}{'F':>8}{'CMLt':>8}")
    for i in range(args.steps + 1):
        logits = model(sp.unsqueeze(0))["beat"][0]
        prob = torch.sigmoid(logits).to(edtype)
        em = m.log_class_densities(torch.stack([prob, torch.zeros_like(prob)], 1))
        lmmi, _, _ = mmi_loss(dp0, logpi, logA, em, s2c, mask)
        lbce = bce(logits[None], tgt[None], msk)
        loss = args.mmi_w * (lmmi / L) + args.alpha * lbce

        if i % 5 == 0:
            with torch.no_grad():
                p = np.asarray(torch.sigmoid(logits).double().cpu().numpy(), np.float64)
            got = emissions_beatonly(m, p)
            f = c = float("nan")
            if got is not None:
                lcd, dec, ff = got
                pa, _ = dp0.viterbi(logpi, logA, lcd, state_to_class=s2c, return_log_score=True)
                b = state_path_to_events(pa.cpu().numpy(), space, FPS, snap_to_activations=dec,
                                         first_frame=ff)["beats"]
                r = ev(gt[gt < L / FPS], b)
                f, c = r["F"], r["CMLt"]
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)) if i else 0.0
            print(f"{i:>5}{float(lmmi)/L:>12.4f}{float(lbce):>9.4f}{gn:>10.3f}"
                  f"{f:>8.3f}{c:>8.3f}", flush=True)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
