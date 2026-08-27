"""Fold-honest SMC baseline: each track scored with the released checkpoint that
held ITS fold out. This is the pre-registered comparison target.

Yesterday's plan set the readout as "+0.03 on all 217 fold-honest tracks" and
explicitly retargeted AWAY from fold-0 0.70, because the fold spread (0.096) is
no tighter than random 8-way splits (p=0.797) and so bounds nothing -- a single
fold cannot carry the claim.

Scoring a single released checkpoint (e.g. fold0) against all 217 tracks is NOT
this number: that checkpoint trained on folds 1-7, so 190 of the 217 are in its
training set. Only the held-out fold of each checkpoint is honest, which is why
this walks all 8.
"""
import argparse, glob, json, os
import numpy as np

ANN = "/disk1/jaehoon/dataset_store/beat_this_annotations/smc"
AUDIO = "/disk1/jaehoon/dataset_store/audio_by_stem/smc"
BANDS = [(0, 55, "<55"), (55, 70, "55-70"), (70, 100, "70-100"),
         (100, 1e9, ">100")]   # 55 is the DBN's hard floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dbn", action="store_true",
                    help="use madmom DBN (explicit tempo model, min_bpm=55) "
                         "instead of framewise peak-picking")
    ap.add_argument("--tta", action="store_true",
                    help="mean-fusion time-stretch TTA over {0.85, 1.0, 1.2}, "
                         "the +0.0179 result this plan re-based onto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch, soxr, soundfile as sf
    import mir_eval.beat as mb
    from beat_this.inference import load_model, split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect

    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    spect = LogMelSpect(device=dev)
    post = Postprocessor(type="dbn" if a.dbn else "minimal", fps=50)
    nets = {}

    folds = {}
    for line in open(f"{ANN}/8-folds.split"):
        p = line.split()
        if len(p) >= 2:
            folds[p[0]] = int(p[1])

    rates = [0.85, 1.0, 1.2] if a.tta else [1.0]
    rows = []
    for bt in sorted(glob.glob(f"{ANN}/annotations/beats/*.beats")):
        stem = os.path.basename(bt)[:-6]
        if stem not in folds:
            continue
        wav = None
        for e in (".wav", ".flac", ".mp3"):
            p = os.path.join(AUDIO, stem + e)
            if os.path.exists(p):
                wav = p
                break
        if wav is None:
            continue
        truth = np.loadtxt(bt, ndmin=2)[:, 0]
        if len(truth) < 8:
            continue
        k = folds[stem]
        if k not in nets:                       # the checkpoint that held k out
            nets[k] = load_model(f"fold{k}", device=dev).eval()
        y, sr = sf.read(wav, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        y22 = soxr.resample(y, sr, 22050) if sr != 22050 else y

        acc = None
        for r in rates:
            yr = y22 if r == 1.0 else soxr.resample(y22, 22050, int(22050 / r))
            with torch.inference_mode():
                pr = split_predict_aggregate(
                    spect(torch.from_numpy(np.ascontiguousarray(yr)).to(dev)),
                    1500, 6, "keep_first", nets[k])
                b = pr["beat"].float()
                d = pr["downbeat"].float()
            if r != 1.0:                        # resample logits back to base rate
                n = int(round(b.shape[-1] * r))
                b = torch.nn.functional.interpolate(
                    b[None, None], size=n, mode="linear", align_corners=False)[0, 0]
                d = torch.nn.functional.interpolate(
                    d[None, None], size=n, mode="linear", align_corners=False)[0, 0]
            if acc is None:
                acc = [b, d]
            else:
                n = min(acc[0].shape[-1], b.shape[-1])
                acc = [acc[0][:n] + b[:n], acc[1][:n] + d[:n]]
        est, _ = post(acc[0] / len(rates), acc[1] / len(rates))
        est = np.asarray(est)
        bpm = 60.0 / float(np.median(np.diff(truth)))
        if len(est) >= 4:
            rt, et = mb.trim_beats(truth), mb.trim_beats(est)
            F = mb.f_measure(rt, et)
            c = mb.continuity(rt, et)
            cmlt, amlt = c[1], c[3]
        else:
            F = cmlt = amlt = 0.0
        rows.append(dict(stem=stem, fold=k, bpm=bpm, F=F, CMLt=cmlt, AMLt=amlt))

    json.dump(rows, open(a.out, "w"))
    F = np.array([r["F"] for r in rows])
    bpm = np.array([r["bpm"] for r in rows])
    print(f"\n  FOLD-HONEST baseline{' + TTA' if a.tta else ''}   n={len(rows)}")
    print(f"  ALL-SMC F = {F.mean():.4f}")
    print(f"\n  {'band':>8}{'n':>6}{'F':>9}")
    for lo, hi, tag in BANDS:
        m = (bpm >= lo) & (bpm < hi)
        if m.sum():
            print(f"  {tag:>8}{m.sum():6d}{F[m].mean():9.4f}")


if __name__ == "__main__":
    main()
