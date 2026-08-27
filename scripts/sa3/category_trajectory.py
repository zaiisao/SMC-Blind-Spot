"""Is DOUBLED a waypoint on the way to GOOD, or a dead end?

Hypothesis worth testing: a track may go PHASE -> DOUBLED -> GOOD during training.
If so, "the gate converts PHASE into DOUBLED" is progress, not damage, and judging
the arm early counts a mid-journey state as a failure.

If instead DOUBLED tracks stay DOUBLED, then converting PHASE into DOUBLED is a
real cost and the gate has to be judged on it.

Scores every checkpoint of an arm and prints, per track, its category at each
epoch -- then counts which paths actually occur.
"""
import argparse, glob, sys
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts/sa3")
from smc_eval import load_net, fold_songs
FPS = 50
SHORT = {"GOOD": "G", "PHASE": "P", "DOUBLED": "D", "HALVED": "H"}


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
            f = float(beat_f_measure(truth, est))
            lo, hi = truth.min(), truth.max()
            r = ((est >= lo) & (est <= hi)).sum() / max(len(truth), 1)
            out[stem] = ("DOUBLED" if r > 1.5 else "HALVED" if r < 0.7
                         else ("GOOD" if f >= cut else "PHASE"), f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ph13")
    ap.add_argument("--base", default="/disk1/jaehoon/bt_wide/phase5")
    ap.add_argument("--cut", type=float, default=0.5)
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    songs = fold_songs(0)

    cks = sorted(glob.glob(f"{a.base}/{a.arm}/*epoch=*.ckpt"))
    eps = [int(c.split("epoch=")[1][:3]) for c in cks]
    hist = [score(c, songs, dev, a.cut) for c in cks]

    stems = sorted(hist[0])
    print(f"\n  {a.arm}: category per track across {len(eps)} epochs "
          f"(G=good P=phase D=doubled H=halved)\n")
    print(f"  {'track':<12}{'bpm':>6}  " + "".join(f"{e:>4}" for e in eps) + "   final F")
    for s in stems:
        path = [h[s][0] for h in hist]
        bpm = 60.0 / np.median(np.diff([t for st, _, t in songs if st == s][0]))
        print(f"  {s:<12}{bpm:6.0f}  " + "".join(f"{SHORT[p]:>4}" for p in path)
              + f"   {hist[-1][s][1]:7.3f}")

    # which paths actually occur?
    print(f"\n  did any track pass THROUGH doubled on its way to good?")
    via, direct, stuck = [], [], []
    for s in stems:
        path = [h[s][0] for h in hist]
        if path[-1] == "GOOD":
            if "DOUBLED" in path[:-1]:
                via.append(s)
            else:
                direct.append(s)
        elif path[-1] == "DOUBLED":
            stuck.append(s)
    print(f"    ended GOOD, was DOUBLED earlier : {len(via):2d}   {via}")
    print(f"    ended GOOD, never DOUBLED       : {len(direct):2d}")
    print(f"    ended DOUBLED                   : {len(stuck):2d}   {stuck}")

    # per-epoch transition census
    print(f"\n  transitions between consecutive checkpoints (all tracks pooled):")
    c = Counter()
    for i in range(len(eps) - 1):
        for s in stems:
            x, y = hist[i][s][0], hist[i + 1][s][0]
            if x != y:
                c[f"{SHORT[x]}->{SHORT[y]}"] += 1
    for k, v in sorted(c.items(), key=lambda z: -z[1]):
        print(f"    {k}  {v}")

    print(f"\n  category counts by epoch:")
    print(f"  {'epoch':>6}" + "".join(f"{k:>9}" for k in ("GOOD", "PHASE", "DOUBLED", "HALVED")))
    for e, h in zip(eps, hist):
        cc = Counter(v[0] for v in h.values())
        print(f"  {e:6d}" + "".join(f"{cc.get(k,0):9d}"
                                    for k in ("GOOD", "PHASE", "DOUBLED", "HALVED")))


if __name__ == "__main__":
    main()
