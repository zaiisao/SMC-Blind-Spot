"""Per-frame loss penalty for stock vs phase-weighted BCE, on a random-init model.

Runs a randomly initialised conv stack over a synthetic spectrogram to get logits
with no learned structure, builds targets from an isochronous beat grid, and plots
the per-frame contribution to the loss under both losses.

Row 1  logits and where the beats are
Row 2  the negative-side weight c_t that the phase term applies
Row 3  per-frame loss contribution, stock vs phase
Row 4  their difference -- where the phase term actually changes the objective

Repeated at 3 tempi so the tempo-dependence of the DOSE is visible: the shape of
c_t is identical in phase units at every tempo, but the collar eats a growing
share of the interval as tempo rises, so the mean applied weight climbs.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAU = 3
POS_W = 32.0
GAMMA = 1.3
SIGMA = 0.06
FPS = 50


def spread(x, factor=1):
    return F.max_pool1d(x, 1 + 2 * factor * TAU, 1)


def crop(x, factor=1):
    return x[..., factor * TAU: -factor * TAU or None]


def phase(targets):
    B, T = targets.shape
    ph = torch.full((B, T), float("nan"))
    idx = torch.arange(T, dtype=torch.float32)
    for b in range(B):
        pos = torch.nonzero(targets[b] > 0).flatten().float()
        if len(pos) < 2:
            continue
        j = torch.searchsorted(pos, idx, right=True) - 1
        ok = (j >= 0) & (j < len(pos) - 1)
        jj = j.clamp(0, len(pos) - 2)
        lo = pos[jj]
        ibi = pos[jj + 1] - lo
        good = ok & (ibi > 0)
        ph[b][good] = (idx[good] - lo[good]) / ibi[good]
    return ph


def phase_weight(targets):
    ph = phase(targets)
    d = (ph - 0.5).abs()
    d = torch.minimum(d, 1 - d)
    bump = torch.exp(-(d ** 2) / (2 * SIGMA ** 2))
    bump = torch.where(d <= 4 * SIGMA, bump, torch.zeros_like(bump))
    return 1.0 + GAMMA * torch.nan_to_num(bump, nan=0.0)


def per_frame_terms(logits, targets):
    """Return (frames, stock, phase, c, mask) -- per-frame loss contributions."""
    zc = crop(spread(logits))
    yc = crop(targets, factor=2)
    m = yc + (1 - spread(targets, factor=2))
    c = crop(phase_weight(targets), factor=2)

    pos = POS_W * yc * F.softplus(-zc)
    neg = (1 - yc) * F.softplus(zc)
    stock = m * (pos + neg)
    phased = m * (pos + c * neg)
    off = TAU * 2
    frames = np.arange(yc.shape[-1]) + off
    return frames, stock[0].numpy(), phased[0].numpy(), c[0].numpy(), m[0].numpy(), yc[0].numpy()


def random_logits(T, seed=0):
    """Logits from an untrained conv stack over noise -- no learned structure."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Conv1d(128, 64, 5, padding=2), torch.nn.GELU(),
        torch.nn.Conv1d(64, 32, 5, padding=2), torch.nn.GELU(),
        torch.nn.Conv1d(32, 1, 1),
    )
    spect = torch.randn(1, 128, T)
    with torch.no_grad():
        return net(spect).squeeze(1)


def main():
    bpms = [40, 70, 140]
    T = 900
    fig, axes = plt.subplots(4, len(bpms), figsize=(6.2 * len(bpms), 11), sharex="col")

    for col, bpm in enumerate(bpms):
        ibi = int(round(FPS * 60 / bpm))
        targets = torch.zeros(1, T)
        beats = list(range(ibi, T - ibi, ibi))
        targets[0, beats] = 1.0
        logits = random_logits(T, seed=0)

        fr, stock, phased, c, m, yc = per_frame_terms(logits, targets)
        lo, hi = beats[2], beats[2] + 6 * ibi          # window: 6 beats
        sel = (fr >= lo) & (fr <= hi)
        t = fr[sel] / FPS

        ax = axes[0, col]
        ax.plot(t, crop(spread(logits))[0].numpy()[sel], lw=1.1, color="#444")
        for b in beats:
            if lo <= b <= hi:
                ax.axvline(b / FPS, color="k", lw=1.0, alpha=.5)
        for i in range(len(beats) - 1):
            mid = (beats[i] + beats[i + 1]) / 2
            if lo <= mid <= hi:
                ax.axvline(mid / FPS, color="green", ls=":", lw=1.0, alpha=.5)
        ax.set_title(f"{bpm} bpm   (IBI {ibi} frames, collar {2*(2*TAU)+1} frames "
                     f"= {100*(2*(2*TAU)+1)/ibi:.0f}% of the interval)", fontsize=10)
        ax.set_ylabel("logit (max-pooled)")

        ax = axes[1, col]
        ax.plot(t, np.ones_like(t), lw=1.8, color="#1f77b4", label="stock  (c = 1)")
        ax.plot(t, c[sel], lw=1.4, color="#d62728", label=f"phase  c(phi), gamma={GAMMA}")
        ax.fill_between(t, 1.0, c[sel], where=c[sel] > 1, color="#d62728", alpha=.15)
        ax.set_ylim(0.9, 1 + GAMMA + 0.15)
        ax.set_ylabel("negative-side weight")
        ax.legend(fontsize=8, loc="upper right")
        cnt = (m > 0) & (yc <= 0)
        ax.text(.02, .86, f"mean c on counted negatives = {c[cnt].mean():.3f}",
                transform=ax.transAxes, fontsize=9)

        ax = axes[2, col]
        ax.plot(t, stock[sel], lw=2.6, color="#1f77b4", alpha=.85,
                label=f"stock  (total {stock.sum():.1f})")
        ax.plot(t, phased[sel], lw=1.2, color="#d62728",
                label=f"phase  (total {phased.sum():.1f})")
        ax.set_ylabel("per-frame loss")
        ax.legend(fontsize=8, loc="upper right")

        ax = axes[3, col]
        diff = phased - stock
        ax.plot(t, diff[sel], lw=1.2, color="#7f2704")
        ax.fill_between(t, 0, diff[sel], color="#7f2704", alpha=.2)
        ax.axhline(0, color="k", lw=.6, alpha=.4)
        ax.set_ylabel("phase - stock")
        ax.set_xlabel("seconds")
        ax.text(.02, .86, f"extra loss = {diff.sum():.2f}  ({100*diff.sum()/stock.sum():+.1f}%)",
                transform=ax.transAxes, fontsize=9)

        for r in range(4):
            for b in beats:
                if lo <= b <= hi and r > 0:
                    axes[r, col].axvline(b / FPS, color="k", lw=.8, alpha=.35)
            for i in range(len(beats) - 1):
                mid = (beats[i] + beats[i + 1]) / 2
                if lo <= mid <= hi and r > 0:
                    axes[r, col].axvline(mid / FPS, color="green", ls=":", lw=.8, alpha=.35)

    fig.suptitle("Per-frame loss penalty, random-init logits.  "
                 "black = annotated beats, dotted green = phi=0.5 midpoints",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out = "/home/sogang/jaehoon/Analyze-SMC/figures/loss_penalty_per_frame.png"
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
