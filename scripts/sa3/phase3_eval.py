"""Phase-3 readout: does the wide tempo range recover SMC F?

Evaluates a trained checkpoint on SMC and reports F/CMLt/AMLt BANDED BY TEMPO,
which is the pre-registered readout -- the hypothesis is specifically about slow
music, so a headline average would dilute the effect we are testing.

WHY ALL 217 SMC TRACKS ARE FAIR GAME HERE. The wide corpus staged SMC with
beats-only annotations, and the datamodule drops tracks lacking downbeats, so
neither arm saw a single SMC track in training. That removes the fold-leak
hazard that voided earlier numbers on this project, and it means we can score
the whole dataset instead of one fold's 25 tracks. Both the fold-0 subset (for
comparability with published fold-honest numbers) and the full set are printed.

Writes per-track rows so the two arms can be compared with a PAIRED test -- the
arms differ by a small amount and per-track pairing is what gives the contrast
any power.
"""
import argparse, glob, json, os, sys
import numpy as np

ANN = "/disk1/jaehoon/dataset_store/beat_this_annotations/smc"
AUDIO = "/disk1/jaehoon/dataset_store/audio_by_stem/smc"
BANDS = [(0, 70, "<70"), (70, 100, "70-100"), (100, 1e9, ">100")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--released", default=None,
                    help="score a RELEASED beat_this checkpoint (e.g. 'fold0') "
                         "through this identical harness, so our from-scratch "
                         "arms can be compared against upstream apples-to-apples "
                         "rather than against a remembered number")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dbn", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch, soxr, soundfile as sf
    import mir_eval.beat as mb
    from beat_this.inference import load_checkpoint, split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect

    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    if a.released:
        from beat_this.inference import load_model
        net = load_model(a.released, device=dev).eval()
        label = f"RELEASED {a.released}"
    else:
        # torch>=2.6 defaults weights_only=True, which rejects the numpy scalars
        # lightning stores in its checkpoints. These are checkpoints we produced
        # ourselves minutes ago, so loading them fully is safe here.
        ckpt = torch.load(a.checkpoint, map_location=dev, weights_only=False)
        from beat_this.model.pl_module import PLBeatThis
        hp = dict(ckpt.get("hyper_parameters", {}))
        if not hp:
            raise SystemExit("checkpoint has no hyper_parameters; cannot rebuild")
        pl = PLBeatThis(**hp)
        pl.load_state_dict(ckpt["state_dict"])
        net = pl.model.to(dev).eval()
        label = os.path.basename(a.checkpoint)[:60]

    spect = LogMelSpect(device=dev)
    post = Postprocessor(type="dbn" if a.dbn else "minimal", fps=50)

    folds = {}
    for line in open(f"{ANN}/8-folds.split"):
        p = line.split()
        if len(p) >= 2:
            folds[p[0]] = int(p[1])

    rows = []
    for bt in sorted(glob.glob(f"{ANN}/annotations/beats/*.beats")):
        stem = os.path.basename(bt)[:-6]
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
        y, sr = sf.read(wav, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        y22 = soxr.resample(y, sr, 22050) if sr != 22050 else y
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
        else:
            F = cmlt = amlt = 0.0
        rows.append(dict(stem=stem, fold=folds.get(stem, -1), bpm=bpm,
                         F=F, CMLt=cmlt, AMLt=amlt))

    json.dump(rows, open(a.out, "w"))
    F = np.array([r["F"] for r in rows])
    bpm = np.array([r["bpm"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    print(f"\n  model: {label}")
    if a.released and a.released.startswith("fold"):
        # A released fold-k checkpoint TRAINED on the other 7 folds, i.e. on 190
        # of these 217 tracks. Its ALL-SMC mean is leak-contaminated and must not
        # be printed next to leak-free arms -- only its own held-out fold is
        # honest. Use foldhonest_baseline.py for an honest all-217 number.
        k = int(a.released[4:])
        m = fold == k
        print(f"  LEAK WARNING: {label} trained on folds != {k}, so "
              f"{int((~m).sum())} of {len(rows)} tracks are training data.")
        print(f"  honest (held-out fold {k} only) F={F[m].mean():.4f} (n={int(m.sum())})")
        print(f"  [suppressed: contaminated ALL-SMC mean]")
    else:
        print(f"  n={len(rows)}  ALL-SMC F={F.mean():.4f}   "
              f"fold0-only F={F[fold == 0].mean():.4f} (n={(fold == 0).sum()})")
    print(f"\n  {'band':>8}{'n':>6}{'F':>9}{'CMLt':>9}{'AMLt':>9}")
    for lo, hi, tag in BANDS:
        m = (bpm >= lo) & (bpm < hi)
        if m.sum():
            print(f"  {tag:>8}{m.sum():6d}{F[m].mean():9.4f}"
                  f"{np.mean([r['CMLt'] for r, k in zip(rows, m) if k]):9.4f}"
                  f"{np.mean([r['AMLt'] for r, k in zip(rows, m) if k]):9.4f}")


if __name__ == "__main__":
    main()
