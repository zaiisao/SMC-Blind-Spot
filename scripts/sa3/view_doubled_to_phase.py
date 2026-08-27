"""Draw what happens when a DOUBLED track becomes PHASE.

At epoch 44, sig20 (sigma 0.20) took 5 of ph13's 9 DOUBLED tracks and moved them
to PHASE. The count ratio improved -- the extra peaks really were removed -- but
the alignment did not, so F barely moved or fell. This shows the frames.

For each track: the two activation curves, the annotated beats, the midpoints the
loss penalises, and which emitted peaks matched a true beat within 70 ms. If the
wide window is damaging the grid rather than only the subdivisions, it should be
visible as true-beat peaks losing height or the surviving peaks landing off the
black lines.
"""
import argparse, glob, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts/sa3")
from smc_eval import load_net, fold_songs
FPS, TOL = 50, 0.070


def peaks(logit, thr=0.0, width=7):
    pad = width // 2
    x = np.pad(logit, pad, mode="constant", constant_values=-1e9)
    mx = np.max(np.lib.stride_tricks.sliding_window_view(x, width), axis=-1)
    return np.nonzero((logit == mx) & (logit > thr))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="+",
                    default=["smc_274", "smc_088", "smc_168", "smc_005"])
    ap.add_argument("--a", default="ph13")
    ap.add_argument("--b", default="sig20")
    ap.add_argument("--epoch", default="044")
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out", default="/home/sogang/jaehoon/Analyze-SMC/cache/doubled_to_phase.png")
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"

    import soundfile as sf, soxr
    from beat_this.preprocessing import LogMelSpect
    from beat_this.inference import split_predict_aggregate
    spect = LogMelSpect(device=dev)
    songs = {s: (w, t) for s, w, t in fold_songs(0)}

    curves = {}
    for arm in (a.a, a.b):
        ck = glob.glob(f"/disk1/jaehoon/bt_wide/phase5/{arm}/*epoch={a.epoch}.ckpt")[0]
        net = load_net(ck, dev).eval()
        d = {}
        with torch.inference_mode():
            for s in a.stems:
                wav, truth = songs[s]
                y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
                y = y.mean(axis=1)
                if sr != 22050:
                    y = soxr.resample(y, sr, 22050)
                S = spect(torch.from_numpy(y).to(dev))
                pr = split_predict_aggregate(S, 1500, 6, "keep_first", net)
                d[s] = pr["beat"].float().cpu().numpy().ravel()
        curves[arm] = d

    fig, axes = plt.subplots(len(a.stems), 1, figsize=(16, 3.1 * len(a.stems)))
    for ax, s in zip(np.atleast_1d(axes), a.stems):
        truth = songs[s][1]
        bpm = 60 / np.median(np.diff(truth))
        # window starting at a quarter through, long enough to see ~10 beats
        t0 = float(truth[len(truth) // 4])
        t1 = t0 + a.seconds
        for arm, col in ((a.a, "#1f77b4"), (a.b, "#d62728")):
            lg = curves[arm][s]
            p = 1 / (1 + np.exp(-lg))
            tt = np.arange(len(p)) / FPS
            m = (tt >= t0) & (tt <= t1)
            ax.plot(tt[m], p[m], lw=1.3, color=col, label=arm, alpha=0.85)
            pk = peaks(lg) / FPS
            pk = pk[(pk >= t0) & (pk <= t1)]
            hit = np.array([np.min(np.abs(truth - x)) <= TOL for x in pk]) if len(pk) else np.array([], bool)
            ax.scatter(pk[hit], [1.06 if arm == a.a else 1.13] * int(hit.sum()),
                       marker="v", s=46, color=col, edgecolor="k", linewidth=.4, zorder=5)
            ax.scatter(pk[~hit], [1.06 if arm == a.a else 1.13] * int((~hit).sum()),
                       marker="x", s=46, color=col, zorder=5)
        tb = truth[(truth >= t0) & (truth <= t1)]
        for b in tb:
            ax.axvline(b, color="k", lw=1.1, alpha=.55)
        allb = truth[(truth >= t0 - 3) & (truth <= t1 + 3)]
        for i in range(len(allb) - 1):
            mid = (allb[i] + allb[i + 1]) / 2
            if t0 <= mid <= t1:
                ax.axvline(mid, color="green", lw=1.0, alpha=.35, ls=":")
        ax.set_ylim(0, 1.2); ax.set_xlim(t0, t1)
        ax.set_title(f"{s}   {bpm:.0f} bpm    black = true beats, dotted green = "
                     f"penalised midpoints,  v = peak matched a beat,  x = spurious",
                     fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(a.out, dpi=110)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
