"""Why does the gated arm sit higher at the off-beats? Chain rule, measured.

For a NEGATIVE frame (target 0), binary_cross_entropy_with_logits with weight w is

    L_i = w_i * softplus(z_i)        =>      dL/dz_i = w_i * sigmoid(z_i)

so the downward push on a non-beat frame is exactly (its weight) x (its own
probability). Nothing else enters. Therefore any difference in where the two arms
settle at phi=0.5 must come from the weight w_i they place there.

    ungated   w = look_at * (1 + gamma*bump)
    gated     w = look_at * (1 + gamma*gate*bump) * k/s     [normalised]

Two multiplicative effects push the gated weight DOWN at midpoints:
    gate < 1              cuts the bump term
    k/s normalisation     rescales so the MEAN negative weight is 1, but the mean
                          is dominated by the vast majority of frames far from a
                          midpoint, so it cannot restore the peak

This script runs REAL batches through a REAL checkpoint, takes the true gradient
at the logits under each loss, and buckets it by phase -- so the decomposition is
measured, not assumed.
"""
import argparse, glob, sys
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
from beat_this.model.loss import (ShiftTolerantBCELoss,
                                  PhaseWeightedShiftTolerantBCELoss,
                                  GatedPhaseShiftTolerantBCELoss)
from beat_this.dataset.dataset import BeatDataModule


def phase_buckets(m, targets):
    ph = m._phase(targets)
    d = (ph - 0.5).abs(); d = torch.minimum(d, 1 - d)
    return ph, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="default: latest gph13")
    ap.add_argument("--data", default="/disk1/jaehoon/bt_wide/data")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"

    ck = a.ckpt or sorted(glob.glob("/disk1/jaehoon/bt_wide/phase5/gph13/*epoch=*.ckpt"))[-1]
    print(f"\n  checkpoint: {ck.split('/')[-1][:60]}")
    sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts/sa3")
    from smc_eval import load_net
    net = load_net(ck, dev).eval()

    dm = BeatDataModule(a.data, batch_size=8, num_workers=4, fold=0)
    dm.setup(stage="fit")
    dl = dm.train_dataloader()

    losses = {
        "stock":   ShiftTolerantBCELoss(pos_weight=32, tolerance=3),
        "ungated": PhaseWeightedShiftTolerantBCELoss(pos_weight=32, tolerance=3,
                                                     phase_gamma=1.3),
        "gated":   GatedPhaseShiftTolerantBCELoss(pos_weight=32, tolerance=3,
                                                  phase_gamma=1.3),
    }
    m = losses["gated"]

    acc = {k: {"mid": [], "near": [], "far": [], "beat": []} for k in losses}
    wacc = {k: {"mid": [], "far": []} for k in losses}
    gate_at_mid = []

    n = 0
    for batch in dl:
        if n >= a.batches:
            break
        n += 1
        spect = batch["spect"].to(dev).float()
        truth = batch["truth_beat"].float().to(dev)
        mask = batch["padding_mask"].to(dev)
        with torch.no_grad():
            z0 = net(spect)["beat"].float()

        ph, d = phase_buckets(m, truth)
        cropped = m.crop(truth, factor=2)
        off = (truth.shape[-1] - cropped.shape[-1]) // 2
        dc = m.crop(d, factor=2)
        neg = cropped <= 0
        look = cropped + (1 - m.spread(truth, factor=2))
        counted = neg & (look > 0)
        mid = counted & (dc < 0.05)
        near = counted & (dc >= 0.05) & (dc < 0.15)
        far = counted & (dc >= 0.30)
        beat = cropped > 0

        g = m._gate(z0, truth)
        gate_at_mid.append(float(m.crop(g, factor=2)[mid].mean()))

        for name, fn in losses.items():
            z = z0.clone().requires_grad_(True)
            L = fn(z, truth, mask)
            gr, = torch.autograd.grad(L, z)
            gr = gr[:, off:off + cropped.shape[-1]] if gr.shape[-1] != cropped.shape[-1] else gr
            for key, sel in (("mid", mid), ("near", near), ("far", far), ("beat", beat)):
                if sel.any():
                    acc[name][key].append(float(gr[sel].mean()))
            # recover the implied weight w = (dL/dz) / sigmoid(z)  on negatives
            with torch.no_grad():
                s = torch.sigmoid(z0[:, off:off + cropped.shape[-1]]
                                  if z0.shape[-1] != cropped.shape[-1] else z0)
            for key, sel in (("mid", mid), ("far", far)):
                if sel.any():
                    wacc[name][key].append(float((gr[sel] / s[sel].clamp(min=1e-6)).mean()))

    print(f"  batches: {n}   mean gate at phi=0.5: {np.mean(gate_at_mid):.4f}\n")

    print("  MEAN GRADIENT dL/dz  (positive = pushed DOWN)")
    print(f"  {'loss':<10}{'phi~0.5':>12}{'phi 0.05-.15':>14}{'phi>0.30':>11}{'at beats':>11}")
    for name in losses:
        r = acc[name]
        print(f"  {name:<10}{np.mean(r['mid']):12.3e}{np.mean(r['near']):14.3e}"
              f"{np.mean(r['far']):11.3e}{np.mean(r['beat']):11.3e}")

    print("\n  IMPLIED WEIGHT w = (dL/dz) / sigmoid(z)   [the only term that differs]")
    print(f"  {'loss':<10}{'w at phi~0.5':>14}{'w at phi>0.30':>15}{'ratio mid/far':>15}")
    for name in losses:
        wm, wf = np.mean(wacc[name]['mid']), np.mean(wacc[name]['far'])
        print(f"  {name:<10}{wm:14.4e}{wf:15.4e}{wm/max(wf,1e-12):15.3f}")

    wm_u, wm_g = np.mean(wacc['ungated']['mid']), np.mean(wacc['gated']['mid'])
    print(f"\n  midpoint weight, gated / ungated = {wm_g/max(wm_u,1e-12):.3f}")
    print(f"  -> the gated loss applies {100*(1-wm_g/max(wm_u,1e-12)):.0f}% LESS "
          f"downward pressure at phi=0.5 than the ungated loss")


if __name__ == "__main__":
    main()
