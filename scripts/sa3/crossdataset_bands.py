"""CMLt vs AMLt across every dataset -- is metrical-level confusion specific to
SMC, or does it track TEMPO wherever you look?

AMLt accepts double/half/offbeat variants of the annotated pulse; CMLt does not.
So (AMLt - CMLt) is the share of the failure that is "found the pulse, chose the
wrong metrical level". If that gap is a property of SMC, the story is about that
corpus. If it scales with how far a dataset sits from the loss's implicit ~77 bpm
calibration point, the story is about the objective -- and generalises.

Fold-honest throughout: each track is scored with the released checkpoint that
held ITS fold out. GTZAN has no fold file because upstream holds it out entirely
as the test set, so any checkpoint is honest for it; final0 is used there and the
row is marked.
"""
import argparse, glob, json, os
import numpy as np

ANN = "/disk1/jaehoon/dataset_store/beat_this_annotations"
AUDIO = "/disk1/jaehoon/dataset_store/audio_by_stem"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=[
        "smc", "asap", "ballroom", "hainsworth", "rwc", "beatles",
        "simac", "guitarset", "hjdb", "candombe", "tapcorrect", "gtzan"])
    ap.add_argument("--max-per-dataset", type=int, default=140,
                    help="cap per corpus so one big dataset does not dominate runtime")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch, soxr, soundfile as sf
    import mir_eval.beat as mb
    from beat_this.inference import load_model, split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect

    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    spect = LogMelSpect(device=dev)
    post = Postprocessor(type="minimal", fps=50)
    nets = {}

    def net_for(fold):
        key = "final0" if fold is None else f"fold{fold}"
        if key not in nets:
            nets[key] = load_model(key, device=dev).eval()
        return nets[key]

    rows = []
    for dsn in a.datasets:
        folds = {}
        fp = f"{ANN}/{dsn}/8-folds.split"
        if os.path.exists(fp):
            for line in open(fp):
                p = line.split()
                if len(p) >= 2:
                    folds[p[0]] = int(p[1])
        cands = []
        for bt in sorted(glob.glob(f"{ANN}/{dsn}/annotations/beats/*.beats")):
            stem = os.path.basename(bt)[:-6]
            wav = None
            for e in (".wav", ".flac", ".mp3"):
                p = os.path.join(AUDIO, dsn, stem + e)
                if os.path.exists(p):
                    wav = p
                    break
            if wav:
                cands.append((stem, wav, bt))
        if not cands:
            print(f"  [skip] {dsn}: no audio", flush=True)
            continue
        rng = np.random.RandomState(0)
        if len(cands) > a.max_per_dataset:
            idx = rng.choice(len(cands), a.max_per_dataset, replace=False)
            cands = [cands[i] for i in idx]

        n_ok = 0
        for stem, wav, bt in cands:
            try:
                truth = np.loadtxt(bt, ndmin=2)
            except Exception:
                continue
            if truth.size == 0 or len(truth) < 8:
                continue
            truth = truth[:, 0]
            try:
                y, sr = sf.read(wav, dtype="float32", always_2d=True)
            except Exception:
                continue
            y = y.mean(axis=1)
            y22 = soxr.resample(y, sr, 22050) if sr != 22050 else y
            net = net_for(folds.get(stem))
            with torch.inference_mode():
                pr = split_predict_aggregate(
                    spect(torch.from_numpy(np.ascontiguousarray(y22)).to(dev)),
                    1500, 6, "keep_first", net)
                est, _ = post(pr["beat"].float(), pr["downbeat"].float())
            est = np.asarray(est)
            bpm = 60.0 / float(np.median(np.diff(truth)))
            if len(est) >= 4:
                rt, et = mb.trim_beats(truth), mb.trim_beats(est)
                F = mb.f_measure(rt, et)
                c = mb.continuity(rt, et)
                cmlt, amlt = c[1], c[3]
                # estimated/true beat-rate ratio: >1.5 means roughly double tempo
                ratio = (float(np.median(np.diff(truth)))
                         / max(float(np.median(np.diff(est))), 1e-9))
            else:
                # ratio=nan, NOT 0.0: a decode failure is not "correct metrical
                # level", and pooling it as 0.0 silently counted it as one.
                F = cmlt = amlt = 0.0
                ratio = float("nan")
            rows.append(dict(ds=dsn, stem=stem, bpm=bpm, F=F, CMLt=cmlt,
                             AMLt=amlt, rate_ratio=ratio,
                             honest=folds.get(stem) is not None or dsn == "gtzan"))
            n_ok += 1
        print(f"  {dsn:<12} {n_ok:4d} tracks done", flush=True)

    json.dump(rows, open(a.out, "w"))
    print(f"\n  {'dataset':<12}{'n':>5}{'med bpm':>9}{'F':>8}{'CMLt':>8}"
          f"{'AMLt':>8}{'AMLt-CMLt':>11}{'2x errors':>11}")
    n_dis = sum(1 for x in rows if not x["honest"])
    if n_dis:
        print(f"\n  WARNING: {n_dis} rows are NOT fold-honest (stem missing from a "
              f"split file, so final0 was used and may have trained on them).")
        print("  These are included below; filter on the 'honest' field to exclude.")
    for dsn in a.datasets:
        r = [x for x in rows if x["ds"] == dsn]
        if not r:
            continue
        b = np.array([x["bpm"] for x in r])
        c = np.array([x["CMLt"] for x in r])
        m = np.array([x["AMLt"] for x in r])
        f = np.array([x["F"] for x in r])
        rr = np.array([x["rate_ratio"] for x in r])
        dbl = float(np.nanmean(((rr > 1.6) & (rr < 2.6)).astype(float)
                               [~np.isnan(rr)])) if (~np.isnan(rr)).any() else float("nan")
        print(f"  {dsn:<12}{len(r):5d}{np.median(b):9.1f}{f.mean():8.4f}"
              f"{c.mean():8.4f}{m.mean():8.4f}{m.mean()-c.mean():11.4f}{dbl:11.3f}")


if __name__ == "__main__":
    main()
