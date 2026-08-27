"""Tempo coverage and rubato: Beat This training datasets vs SMC.

Per-track BPM is derived from the mean inter-beat interval (60 / mean_ibi) and
rubato from the IBI coefficient of variation, both from data/ibi_variability.csv,
so every row of both panels uses the same per-track quantities.

Writes figures/tempo_rubato_smc_vs_training.{png,pdf}.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "ibi_variability.csv"
OUT = ROOT / "figures"

TRAIN_BLUE = "#2a78d6"
SMC_ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"

# Beat This trains with +/-20% tempo augmentation; the effective range quoted in
# the paper is ~64-168 BPM.
AUG_LO, AUG_HI = 64.0, 168.0

ORDER = ["Ballroom", "Beatles", "GTZAN", "Hainsworth", "SMC"]


def load():
    d = pd.read_csv(CSV)
    d["bpm"] = 60.0 / d["mean_ibi"]
    return d


def ridge(ax, groups, col, bins, xlim, xlabel, shade=None):
    """One filled histogram row per dataset, sharing an x axis."""
    n = len(ORDER)
    row_h = 1.0
    for i, name in enumerate(ORDER):
        v = groups[name][col].to_numpy()
        v = v[np.isfinite(v)]
        base = (n - 1 - i) * row_h
        counts, edges = np.histogram(np.clip(v, xlim[0], xlim[1]), bins=bins)
        dens = counts / counts.max() * 0.62 * row_h
        colour = SMC_ORANGE if name == "SMC" else TRAIN_BLUE
        ax.stairs(dens + base, edges, baseline=base, fill=True,
                  color=colour, alpha=0.85, lw=0)
        ax.stairs(dens + base, edges, baseline=base, color=colour, lw=1.0)
        med = np.median(v)
        ax.plot([med, med], [base, base + 0.62], color="white", lw=1.6, zorder=4)
        ax.text(xlim[1], base + 0.70, f"n={len(v)}   med {med:.3g}",
                ha="right", va="bottom", fontsize=7.5, color=MUTED)
        ax.text(xlim[0], base + 0.70, f"  {name}", ha="left", va="bottom",
                fontsize=9, color=INK, weight="bold" if name == "SMC" else "normal")

    if shade is not None:
        ax.axvspan(shade[0], shade[1], color=TRAIN_BLUE, alpha=0.07, zorder=0)
        for x in shade:
            ax.axvline(x, color=TRAIN_BLUE, lw=1.0, ls=(0, (4, 3)), alpha=0.6, zorder=1)

    ax.set_xlim(*xlim)
    ax.set_ylim(-0.05, n * row_h)
    ax.text(1.0, -0.20, "outermost bin includes clipped values",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=MUTED, style="italic")
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def main():
    d = load()
    groups = {k: g for k, g in d.groupby("dataset")}
    missing = [n for n in ORDER if n not in groups]
    if missing:
        raise SystemExit(f"missing datasets in {CSV}: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")

    ridge(axes[0], groups, "bpm", np.linspace(20, 220, 51), (20, 220),
          "Tempo (BPM, from mean inter-beat interval)", shade=(AUG_LO, AUG_HI))
    axes[0].set_title("Tempo coverage", fontsize=11, color=INK, loc="left", pad=10)
    axes[0].text(AUG_HI - 2, len(ORDER) - 0.10,
                 f"Beat This training range ~{AUG_LO:.0f}–{AUG_HI:.0f} BPM",
                 ha="right", va="top", fontsize=7.5, color=TRAIN_BLUE)

    ridge(axes[1], groups, "cv", np.linspace(0, 0.35, 51), (0, 0.35),
          "Rubato: coefficient of variation of inter-beat intervals")
    axes[1].set_title("Rubato (timing variability)", fontsize=11, color=INK,
                      loc="left", pad=10)

    smc = groups["SMC"]
    below = (smc["bpm"] < AUG_LO).mean() * 100
    fig.suptitle(
        "SMC sits outside Beat This's training distribution on both tempo and rubato",
        fontsize=12.5, color=INK, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.915,
             f"{below:.1f}% of SMC tracks fall below {AUG_LO:.0f} BPM "
             f"(Ballroom 0.0%, Beatles 0.0%, Hainsworth 4.5%, GTZAN 5.0%); "
             f"SMC's median IBI CV is {smc['cv'].median():.3f} vs "
             f"{max(groups[n]['cv'].median() for n in ORDER if n != 'SMC'):.3f} "
             "for the most variable training set.",
             fontsize=8.5, color=MUTED, ha="left")

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"tempo_rubato_smc_vs_training.{ext}", dpi=200,
                    facecolor=fig.get_facecolor())
    print("wrote", OUT / "tempo_rubato_smc_vs_training.png")


if __name__ == "__main__":
    main()
