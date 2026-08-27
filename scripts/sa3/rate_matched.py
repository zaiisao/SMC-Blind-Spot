"""Compare arms at MATCHED output rate, so the gate is judged on activation shape.

gph13 emits ~24% more above-threshold frames than ph13 (mean est/true ratio 1.38
vs 1.22). That is a global rate difference coming from normalise=True holding the
mean negative weight at 1 while ph13 runs ~10% extra negative pressure -- not from
the gate. At different rates the F comparison is uninterpretable: on SMC, where
doubling dominates, simply emitting fewer beats helps.

So: sweep each arm's decision threshold, and compare them at the SAME operating
point. Two matchings, because neither alone is decisive:

  (a) equal mean est/true ratio r  -- same number of beats emitted
  (b) each arm at ITS OWN best threshold -- the fairest "what is this model
      worth if you tuned it" comparison, which also shows whether the default
      logit>0 threshold was simply better suited to one arm

If the gate shapes activations better, gph13 should win once rate is controlled.
If it only ever won/lost by emitting more peaks, the curves will coincide.
"""
import argparse, glob, sys
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts/sa3")
from smc_eval import load_net, fold_songs
FPS = 50


def peak_pick(logit, thr, width=7):
    """the postp_minimal rule with an explicit threshold"""
    n = len(logit)
    pad = width // 2
    x = np.pad(logit, pad, mode="constant", constant_values=-1e9)
    mx = np.max(np.lib.stride_tricks.sliding_window_view(x, width), axis=-1)
    return np.nonzero((logit == mx) & (logit > thr))[0]


def collect(ckpt, songs, dev):
    import soundfile as sf, soxr
    from beat_this.preprocessing import LogMelSpect
    from beat_this.inference import split_predict_aggregate
    net = load_net(ckpt, dev); net.eval()
    spect = LogMelSpect(device=dev)
    out = []
    with torch.inference_mode():
        for stem, wav, truth in songs:
            y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != 22050:
                y = soxr.resample(y, sr, 22050)
            S = spect(torch.from_numpy(y).to(dev))
            pr = split_predict_aggregate(S, 1500, 6, "keep_first", net)
            out.append((stem, pr["beat"].float().cpu().numpy().ravel(), truth))
    return out


def evaluate(data, thr):
    from training.evaluation import beat_f_measure
    fs, rs = [], []
    for stem, logit, truth in data:
        est = peak_pick(logit, thr) / FPS
        fs.append(beat_f_measure(truth, est))
        lo, hi = truth.min(), truth.max()
        rs.append(((est >= lo) & (est <= hi)).sum() / max(len(truth), 1))
    return float(np.mean(fs)), float(np.mean(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", default="024")
    ap.add_argument("--arms", nargs=2, default=["ph13", "gph13"])
    ap.add_argument("--base", default="/disk1/jaehoon/bt_wide/phase5")
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    songs = fold_songs(0)

    data = {}
    for arm in a.arms:
        c = glob.glob(f"{a.base}/{arm}/*epoch={a.epoch}.ckpt")
        if not c:
            print(f"  no ckpt for {arm}@{a.epoch}"); return
        data[arm] = collect(c[0], songs, dev)

    thrs = np.arange(-2.0, 3.01, 0.125)
    curves = {arm: [evaluate(data[arm], t) for t in thrs] for arm in a.arms}

    A, B = a.arms
    print(f"\n  threshold sweep, epoch {a.epoch}   (default is logit>0)\n")
    print(f"  {'thr':>6}" + "".join(f"{arm+' F':>11}{arm+' r':>9}" for arm in a.arms))
    for i, t in enumerate(thrs):
        if abs(t * 4 - round(t * 4)) > 1e-9 or (i % 4):
            continue
        print(f"  {t:6.2f}" + "".join(
            f"{curves[arm][i][0]:11.4f}{curves[arm][i][1]:9.3f}" for arm in a.arms))

    print(f"\n  (a) MATCHED RATE -- both arms forced to the same mean r")
    print(f"  {'target r':>9}{A+' thr':>10}{A+' F':>10}{B+' thr':>10}{B+' F':>10}{'diff':>9}")
    for target in (1.00, 1.05, 1.10, 1.15, 1.22, 1.30):
        row = []
        for arm in a.arms:
            i = int(np.argmin([abs(c[1] - target) for c in curves[arm]]))
            row.append((thrs[i], curves[arm][i][0], curves[arm][i][1]))
        if max(abs(row[0][2] - target), abs(row[1][2] - target)) > 0.06:
            continue
        print(f"  {target:9.2f}{row[0][0]:10.2f}{row[0][1]:10.4f}"
              f"{row[1][0]:10.2f}{row[1][1]:10.4f}{row[1][1]-row[0][1]:+9.4f}")

    print(f"\n  (b) EACH ARM AT ITS OWN BEST THRESHOLD")
    best = {}
    for arm in a.arms:
        i = int(np.argmax([c[0] for c in curves[arm]]))
        best[arm] = (thrs[i], curves[arm][i][0], curves[arm][i][1])
        print(f"    {arm:<8} thr {thrs[i]:+5.2f}   F {curves[arm][i][0]:.4f}   "
              f"r {curves[arm][i][1]:.3f}")
    d = best[B][1] - best[A][1]
    print(f"    difference at own-best: {d:+.4f}")

    i0 = int(np.argmin(np.abs(thrs)))
    print(f"\n  for reference, default thr=0:  "
          + "   ".join(f"{arm} F={curves[arm][i0][0]:.4f}" for arm in a.arms))


if __name__ == "__main__":
    main()
