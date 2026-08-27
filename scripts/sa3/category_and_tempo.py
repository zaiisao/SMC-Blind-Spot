"""Per-tempo scores and the GOOD/PHASE/DOUBLED/HALVED transition table.

Categories follow the plan-of-record definitions, computed from the est/true beat
count ratio r and the F score:
    r > 1.5            DOUBLED
    r < 0.7            HALVED
    otherwise F >= cut GOOD, else PHASE   (right rate, wrong alignment)

Reports, for two arms at a matched epoch:
  - F by tempo tercile
  - the category counts for each arm
  - the full transition matrix (which arm-A category each arm-B track came from),
    which is the pre-registered instrument: it separates "repaired a doubled
    track" from "shattered it into PHASE", outcomes that cancel in the net counts.
"""
import argparse, glob, sys
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts/sa3")
from smc_eval import load_net, fold_songs
FPS = 50


def score(ckpt, songs, dev, cut):
    import soundfile as sf, soxr
    from beat_this.preprocessing import LogMelSpect
    from beat_this.inference import split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from training.evaluation import beat_f_measure
    net = load_net(ckpt, dev); net.eval()
    spect = LogMelSpect(device=dev)
    post = Postprocessor(type="minimal", fps=FPS)
    out = {}
    with torch.inference_mode():
        for stem, wav, truth in songs:
            y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != 22050:
                y = soxr.resample(y, sr, 22050)
            S = spect(torch.from_numpy(y).to(dev))
            pr = split_predict_aggregate(S, 1500, 6, "keep_first", net)
            bt, _ = post(pr["beat"].float(), pr["downbeat"].float())
            est = np.asarray(bt[0] if isinstance(bt, (list, tuple)) else bt).ravel()
            f = beat_f_measure(truth, est)
            # count ratio measured over the annotated span only
            lo, hi = truth.min(), truth.max()
            n_est = int(((est >= lo) & (est <= hi)).sum())
            r = n_est / max(len(truth), 1)
            cat = ("DOUBLED" if r > 1.5 else
                   "HALVED" if r < 0.7 else
                   ("GOOD" if f >= cut else "PHASE"))
            out[stem] = {"F": float(f), "r": float(r), "cat": cat,
                         "bpm": 60.0 / np.median(np.diff(truth))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", default="024")
    ap.add_argument("--arms", nargs=2, default=["ph13", "gph13"])
    ap.add_argument("--base", default="/disk1/jaehoon/bt_wide/phase5")
    ap.add_argument("--cut", type=float, default=0.5)
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    songs = fold_songs(0)

    res = {}
    for arm in a.arms:
        c = glob.glob(f"{a.base}/{arm}/*epoch={a.epoch}.ckpt")
        if not c:
            print(f"  no checkpoint for {arm} at epoch {a.epoch}"); return
        res[arm] = score(c[0], songs, dev, a.cut)
    A, B = a.arms
    stems = sorted(res[A])

    print(f"\n  epoch {a.epoch}   A={A}   B={B}   GOOD cut F>={a.cut}\n")

    bpm = np.array([res[A][s]["bpm"] for s in stems])
    q = np.quantile(bpm, [1 / 3, 2 / 3])
    print(f"  {'tempo group':<12}{'n':>3}{'BPM':>13}{A:>10}{B:>10}{'diff':>9}")
    for nm, msk in (("slowest", bpm <= q[0]), ("middle", (bpm > q[0]) & (bpm <= q[1])),
                    ("fastest", bpm > q[1])):
        fa = np.array([res[A][s]["F"] for s in stems])[msk]
        fb = np.array([res[B][s]["F"] for s in stems])[msk]
        print(f"  {nm:<12}{msk.sum():3d}{f'{bpm[msk].min():.0f}-{bpm[msk].max():.0f}':>13}"
              f"{fa.mean():10.4f}{fb.mean():10.4f}{fb.mean()-fa.mean():+9.4f}")
    fa = np.array([res[A][s]["F"] for s in stems])
    fb = np.array([res[B][s]["F"] for s in stems])
    print(f"  {'ALL':<12}{len(stems):3d}{'':>13}{fa.mean():10.4f}{fb.mean():10.4f}"
          f"{fb.mean()-fa.mean():+9.4f}")

    CATS = ["GOOD", "PHASE", "DOUBLED", "HALVED"]
    ca = Counter(res[A][s]["cat"] for s in stems)
    cb = Counter(res[B][s]["cat"] for s in stems)
    print(f"\n  category counts        {'':<8}" + "".join(f"{c:>9}" for c in CATS))
    print(f"    {A:<20}{'':<8}" + "".join(f"{ca.get(c,0):9d}" for c in CATS))
    print(f"    {B:<20}{'':<8}" + "".join(f"{cb.get(c,0):9d}" for c in CATS))
    print(f"    {'change':<20}{'':<8}" +
          "".join(f"{cb.get(c,0)-ca.get(c,0):+9d}" for c in CATS))

    print(f"\n  transition matrix   rows = {A}   cols = {B}")
    print(f"  {'':<10}" + "".join(f"{c:>9}" for c in CATS) + f"{'total':>8}")
    for ra in CATS:
        row = [sum(1 for s in stems if res[A][s]["cat"] == ra and res[B][s]["cat"] == rb)
               for rb in CATS]
        print(f"  {ra:<10}" + "".join(f"{v:9d}" for v in row) + f"{sum(row):8d}")

    moved = [(s, res[A][s]["cat"], res[B][s]["cat"],
              res[B][s]["F"] - res[A][s]["F"]) for s in stems
             if res[A][s]["cat"] != res[B][s]["cat"]]
    if moved:
        print(f"\n  tracks that changed category ({len(moved)}):")
        for s, x, y, df in sorted(moved, key=lambda z: z[3]):
            print(f"    {s:<12} {x:>8} -> {y:<8} dF {df:+.3f}  "
                  f"({res[A][s]['bpm']:.0f} bpm)")


if __name__ == "__main__":
    main()
