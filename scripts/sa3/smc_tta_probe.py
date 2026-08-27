"""Can averaging activations help SMC at all? Two cheap tests before spending a day.

The oracle decomposition showed 92.4% of SMC's error survives perfect level and
phase correction, so the bottleneck is the frame activations themselves. The
proposed fix is averaging (seed ensembling). Averaging only works on INDEPENDENT
NOISE, so two things have to be true, and both are checkable without training:

1. DIAGNOSTIC -- is the model diffusely uncertain, or confidently wrong?
   Flat, low-peaked activations mean there is signal to recover by averaging.
   Sharp peaks in the wrong places mean every copy agrees on the wrong answer and
   averaging changes nothing. These look identical in frameAP.

2. TTA -- one model, K pitch-shifted copies of the same audio, activations
   averaged. Pitch shift leaves timing exactly intact so the curves align
   frame-for-frame, and Beat This trains with +/-5 semitones so it is robust to
   the transform by construction. This is the same mechanism as seed ensembling
   at zero training cost: if it does nothing, ensembling almost certainly will
   not either.

Fold-honest throughout: each fold scored with the released checkpoint that held
it out.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shifts", type=int, nargs="+", default=[-4, -2, 0, 2, 4])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--easy-ref", default="ballroom",
                    help="easy dataset for the confidence comparison")
    a = ap.parse_args()

    import torch, soundfile as sf, soxr
    import mir_eval.beat as mb
    from sklearn.metrics import average_precision_score
    from beat_this.inference import load_model, split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect
    from smc_eval import fold_songs, FPS

    dev = f"cuda:{a.gpu}"
    spect = LogMelSpect(device=dev)
    post = Postprocessor(type="minimal", fps=FPS)

    def activations(y22, net):
        with torch.inference_mode():
            pr = split_predict_aggregate(
                spect(torch.from_numpy(y22).to(dev)), 1500, 6, "keep_first", net)
        return pr["beat"].float(), pr["downbeat"].float()

    def widen(truth, n):
        """frame targets with +/-1 frame tolerance, as the released eval uses"""
        w = np.zeros(n, dtype=bool)
        for t in truth:
            i = int(round(t * FPS))
            w[max(0, i - 1): i + 2] = True
        return w

    base_rows, tta_rows, conf = [], [], []
    per_track = []
    for fold in a.folds:
        net = load_model(f"fold{fold}", device=dev).eval()
        for stem, wav, truth in fold_songs(fold):
            y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            y22 = soxr.resample(y, sr, 22050) if sr != 22050 else y
            if len(truth) < 8:
                continue

            acc_b = acc_d = None
            allb = []
            for k, s in enumerate(a.shifts):
                ys = y22 if s == 0 else _pitch(y22, 22050, s)
                b, d = activations(ys, net)
                n = min(len(b), len(acc_b)) if acc_b is not None else len(b)
                if acc_b is None:
                    acc_b, acc_d = b.clone(), d.clone()
                else:
                    acc_b = acc_b[:n] + b[:n]; acc_d = acc_d[:n] + d[:n]
                allb.append(torch.sigmoid(b).cpu().numpy())
                if s == 0:
                    b0, d0 = b, d
            acc_b /= len(a.shifts); acc_d /= len(a.shifts)

            rt = mb.trim_beats(np.asarray(truth))
            for tag, (bb, dd), sink in (("base", (b0, d0), base_rows),
                                        ("tta", (acc_b, acc_d), tta_rows)):
                est, _ = post(bb, dd)
                est = np.asarray(est)
                p = torch.sigmoid(bb).cpu().numpy()
                w = widen(np.asarray(truth), len(p))
                ap_ = average_precision_score(w, p) if w.any() else 0.0
                f = mb.f_measure(rt, mb.trim_beats(est)) if len(est) >= 4 else 0.0
                sink.append((f, ap_))
            p0 = torch.sigmoid(b0).cpu().numpy()
            m = min(len(x) for x in allb)
            stack = np.stack([x[:m] for x in allb])
            disagree = float(stack.std(axis=0).mean())      # cross-copy variance
            peak = float(p0.max()); frac = float(np.mean(p0 > 0.5))
            conf.append((peak, frac, float(p0.mean())))
            per_track.append((stem, base_rows[-1][0], tta_rows[-1][0],
                              base_rows[-1][1], tta_rows[-1][1],
                              disagree, peak, frac, float(p0.mean())))
        print(f"[fold {fold}] done", flush=True)

    B, T = np.array(base_rows), np.array(tta_rows)
    C = np.array(conf)
    print(f"\nn={len(B)} SMC tracks, shifts={a.shifts}\n")
    print(f"{'':<14}{'F':>9}{'frameAP':>10}")
    print(f"{'baseline':<14}{B[:,0].mean():9.4f}{B[:,1].mean():10.4f}")
    print(f"{'TTA (avg)':<14}{T[:,0].mean():9.4f}{T[:,1].mean():10.4f}")
    print(f"{'delta':<14}{T[:,0].mean()-B[:,0].mean():+9.4f}"
          f"{T[:,1].mean()-B[:,1].mean():+10.4f}")
    d = T[:, 0] - B[:, 0]
    import math
    from scipy import stats
    print(f"\npaired: se={d.std(ddof=1)/math.sqrt(len(d)):.4f}  "
          f"p={stats.ttest_rel(T[:,0],B[:,0]).pvalue:.4f}  "
          f"TTA better on {int((d>0).sum())}/{len(d)}")
    import csv as _csv
    with open("/disk1/jaehoon/tta_pertrack.csv", "w") as fh:
        w=_csv.writer(fh); w.writerow(["stem","F_base","F_tta","ap_base","ap_tta",
                                       "disagree","peak","frac_over_half","mean_act"])
        w.writerows(per_track)
    print("[dump] /disk1/jaehoon/tta_pertrack.csv")
    print(f"\nactivation confidence on SMC: peak={C[:,0].mean():.3f}  "
          f"frac>0.5={100*C[:,1].mean():.2f}%  mean={C[:,2].mean():.4f}")


def _pitch(y, sr, semitones):
    import librosa
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitones)


if __name__ == "__main__":
    main()
