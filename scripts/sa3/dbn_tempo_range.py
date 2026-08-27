"""Does the DBN help once its tempo floor can actually reach SMC's tempi?

madmom's DBNDownBeatTrackingProcessor is used by Beat This with min_bpm=55. SMC
has 45 of 217 tracks below 55 bpm (20.7%, minimum 32.7), and 9 of the 27 in
fold 0. For those the DBN has no state at the true tempo and must track at double
rate, so any earlier "the DBN does not help on SMC" result was measuring that
constraint rather than the DBN.

Compares, per checkpoint: minimal peak-picking, DBN at the stock floor, and DBN at
floors low enough to cover the corpus. Reports overall F plus F on the sub-55
tracks specifically, since that is where the floor binds.
"""
import argparse, glob, sys
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts/sa3")
from smc_eval import load_net, fold_songs
FPS = 50


def run(ckpt, songs, dev, postp, min_bpm, max_bpm):
    import soundfile as sf, soxr
    from beat_this.preprocessing import LogMelSpect
    from beat_this.inference import split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from training.evaluation import beat_f_measure
    net = load_net(ckpt, dev); net.eval()
    spect = LogMelSpect(device=dev)
    post = Postprocessor(type=postp, fps=FPS, min_bpm=min_bpm, max_bpm=max_bpm)
    rows = []
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
            lo, hi = truth.min(), truth.max()
            rows.append(dict(
                stem=stem, F=float(beat_f_measure(truth, est)),
                bpm=60.0 / np.median(np.diff(truth)),
                r=((est >= lo) & (est <= hi)).sum() / max(len(truth), 1)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=["fold0"])
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--floors", type=float, nargs="+", default=[55.0, 40.0, 30.0])
    ap.add_argument("--max-bpm", type=float, default=215.0)
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    songs = fold_songs(0)
    labels = a.labels or a.ckpts

    print(f"\n  SMC fold 0, n={len(songs)}   sub-55bpm tracks: "
          f"{sum(1 for _,_,t in songs if 60/np.median(np.diff(t)) < 55)}\n")
    print(f"  {'checkpoint':<12}{'decoder':<16}{'F all':>9}{'F sub-55':>10}"
          f"{'F 55+':>8}{'mean r':>9}")
    for ck, lab in zip(a.ckpts, labels):
        combos = [("minimal", None)] + [("dbn", f) for f in a.floors]
        for postp, floor in combos:
            rows = run(ck, songs, dev, postp,
                       floor if floor else 55.0, a.max_bpm)
            F = np.array([r["F"] for r in rows])
            b = np.array([r["bpm"] for r in rows])
            rr = np.array([r["r"] for r in rows])
            name = "minimal" if postp == "minimal" else f"dbn min={floor:g}"
            slow = b < 55
            print(f"  {lab:<12}{name:<16}{F.mean():9.4f}"
                  f"{F[slow].mean():10.4f}{F[~slow].mean():8.4f}{rr.mean():9.3f}")
        print()


if __name__ == "__main__":
    main()
