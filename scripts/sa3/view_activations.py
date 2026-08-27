"""Look at the actual activation shape: ungated vs gated, against ground truth.

The aggregate columns (F, level_ok, doubled) can move for reasons unrelated to
what the loss shapes. What the phase term actually manipulates is the activation
at phi=0.5 relative to phi=0, so measure that directly and draw it.

Outputs
  - a per-track numeric summary: mean activation at the true beats (phi~0), at the
    midpoints (phi~0.5), and their ratio. The phase term should DEPRESS the
    midpoint relative to the beat; the gated version should do so selectively,
    only where the beats are already confident.
  - a figure of the worst/best tracks so the shape is visible.
"""
import argparse, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts/sa3")
from smc_eval import load_net, fold_songs
FPS = 50


def activations(ckpt, songs, dev):
    import soundfile as sf, soxr
    from beat_this.preprocessing import LogMelSpect
    from beat_this.inference import split_predict_aggregate
    net = load_net(ckpt, dev); net.eval()
    spect = LogMelSpect(device=dev)
    out = {}
    with torch.inference_mode():
        for stem, wav, truth in songs:
            y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != 22050:
                y = soxr.resample(y, sr, 22050)
            S = spect(torch.from_numpy(y).to(dev))
            pr = split_predict_aggregate(S, 1500, 6, "keep_first", net)
            out[stem] = (torch.sigmoid(pr["beat"].float()).cpu().numpy(), truth)
    return out


def phase_of(n, truth):
    """phase of every frame within its enclosing pair of annotated beats"""
    f = np.rint(np.asarray(truth) * FPS).astype(int)
    f = f[(f >= 0) & (f < n)]
    ph = np.full(n, np.nan)
    if len(f) < 2:
        return ph, f
    idx = np.arange(n)
    j = np.searchsorted(f, idx, side="right") - 1
    ok = (j >= 0) & (j < len(f) - 1)
    jj = np.clip(j, 0, len(f) - 2)
    ibi = (f[jj + 1] - f[jj]).astype(float)
    good = ok & (ibi > 0)
    ph[good] = (idx[good] - f[jj[good]]) / ibi[good]
    return ph, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", default="024")
    ap.add_argument("--base", default="/disk1/jaehoon/bt_wide/phase5")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out", default="/home/sogang/jaehoon/Analyze-SMC/cache/activation_shape.png")
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    songs = fold_songs(0)

    import glob
    arms = {}
    for arm in ("ph13", "gph13"):
        c = glob.glob(f"{a.base}/{arm}/*epoch={a.epoch}.ckpt")
        if c:
            arms[arm] = activations(c[0], songs, dev)
    if len(arms) < 2:
        print("need both arms"); return

    rows = []
    for stem, _, truth in songs:
        r = {"stem": stem}
        for arm, d in arms.items():
            p, tr = d[stem]
            ph, f = phase_of(len(p), tr)
            at_beat = np.zeros(len(p), bool); at_beat[f] = True
            # widen the beat window by +/-3 frames (the loss tolerance)
            for k in range(1, 4):
                at_beat[np.clip(f + k, 0, len(p) - 1)] = True
                at_beat[np.clip(f - k, 0, len(p) - 1)] = True
            mid = np.abs(ph - 0.5) < 0.05
            r[f"{arm}_beat"] = float(p[at_beat].mean())
            r[f"{arm}_mid"] = float(p[mid].mean()) if mid.any() else np.nan
            r[f"{arm}_ratio"] = r[f"{arm}_mid"] / max(r[f"{arm}_beat"], 1e-9)
        rows.append(r)

    print(f"\n  activation at the MIDPOINT relative to the BEAT, epoch {a.epoch}")
    print("  (lower ratio = subdivision better suppressed = what the loss is for)\n")
    print(f"  {'arm':<8}{'mean p@beat':>13}{'mean p@mid':>12}{'mean ratio':>12}"
          f"{'median ratio':>14}")
    for arm in arms:
        b = np.array([r[f"{arm}_beat"] for r in rows])
        mm = np.array([r[f"{arm}_mid"] for r in rows])
        rr = np.array([r[f"{arm}_ratio"] for r in rows])
        print(f"  {arm:<8}{b.mean():13.4f}{np.nanmean(mm):12.4f}"
              f"{np.nanmean(rr):12.4f}{np.nanmedian(rr):14.4f}")

    d = np.array([r["gph13_ratio"] - r["ph13_ratio"] for r in rows])
    print(f"\n  gated - ungated midpoint ratio: mean {np.nanmean(d):+.4f}   "
          f"gated lower on {int((d < 0).sum())}/{len(d)} tracks")
    print("  (negative = the GATE suppresses subdivisions MORE than ungated)")

    # draw the tracks where the two differ most
    order = np.argsort(np.nan_to_num(d))
    pick = [rows[i]["stem"] for i in list(order[:2]) + list(order[-2:])]
    fig, axes = plt.subplots(len(pick), 1, figsize=(15, 2.6 * len(pick)))
    for ax, stem in zip(np.atleast_1d(axes), pick):
        p_u, tr = arms["ph13"][stem]
        p_g, _ = arms["gph13"][stem]
        n = min(len(p_u), len(p_g), 1200)
        t = np.arange(n) / FPS
        ax.plot(t, p_u[:n], lw=1.0, label="ungated ph13", color="#1f77b4")
        ax.plot(t, p_g[:n], lw=1.0, label="gated gph13", color="#d62728", alpha=0.8)
        for b in np.asarray(tr):
            if b < n / FPS:
                ax.axvline(b, color="k", lw=0.8, alpha=0.35)
            mid_prev = None
        # mark the midpoints between consecutive true beats
        tb = np.asarray(tr); tb = tb[tb < n / FPS]
        for i in range(len(tb) - 1):
            ax.axvline((tb[i] + tb[i + 1]) / 2, color="green", lw=0.7,
                       alpha=0.30, ls=":")
        ax.set_title(f"{stem}   ratio ungated "
                     f"{[r for r in rows if r['stem']==stem][0]['ph13_ratio']:.3f}"
                     f"  gated "
                     f"{[r for r in rows if r['stem']==stem][0]['gph13_ratio']:.3f}"
                     f"   (black = true beats, dotted green = midpoints)", fontsize=9)
        ax.set_ylim(0, 1); ax.set_xlim(0, n / FPS)
        ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(a.out, dpi=110)
    print(f"\n  figure -> {a.out}")


if __name__ == "__main__":
    main()
