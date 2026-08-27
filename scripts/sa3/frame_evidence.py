"""Is slow music harder at the FRAME level, or only after decoding?

The loss normalizes by frames (reduction='mean' over 1500), so a track's pull on
the objective scales with how many beats it packs into 30 s -- a 180 bpm track
counts ~4.5x a 40 bpm one. Evaluation normalizes by TRACK. If that mismatch is
what hurts slow music, the damage must already be visible in the raw framewise
activations, BEFORE any peak-picking or metrical decision.

Two measurements per track, both decoder-free:
  frame AP   average precision of the beat logits against widened targets
             (+/-70 ms, matching F's tolerance). Pure evidence quality.
  peak SNR   median logit AT true beat frames minus median logit at midpoints
             between beats. How far the evidence separates beats from non-beats.

If frame AP is flat across tempo bands while F is not, the deficit is a decoding
/ metrical-level problem and loss reweighting will not help. If frame AP falls
with tempo, the evidence itself is worse for slow music and the normalization
hypothesis survives its first test.
"""
import argparse, glob, json, os
import numpy as np

ANN = "/disk1/jaehoon/dataset_store/beat_this_annotations/smc"
AUDIO = "/disk1/jaehoon/dataset_store/audio_by_stem/smc"
BANDS = [(0, 70, "<70"), (70, 100, "70-100"), (100, 1e9, ">100")]
FPS = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--released", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch, soxr, soundfile as sf
    from sklearn.metrics import average_precision_score
    from beat_this.inference import split_predict_aggregate
    from beat_this.preprocessing import LogMelSpect

    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    if a.released:
        from beat_this.inference import load_model
        net = load_model(a.released, device=dev).eval()
        label = f"RELEASED {a.released}"
    else:
        ckpt = torch.load(a.checkpoint, map_location=dev, weights_only=False)
        from beat_this.model.pl_module import PLBeatThis
        pl = PLBeatThis(**dict(ckpt["hyper_parameters"]))
        pl.load_state_dict(ckpt["state_dict"])
        net = pl.model.to(dev).eval()
        label = os.path.basename(a.checkpoint)[:55]

    spect = LogMelSpect(device=dev)
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
            logit = pr["beat"].float().cpu().numpy().ravel()
        n = len(logit)
        # targets widened to +/-3 frames (60 ms), inside F's 70 ms tolerance
        tgt = np.zeros(n, dtype=np.int8)
        bi = np.clip(np.rint(truth * FPS).astype(int), 0, n - 1)
        for k in range(-3, 4):
            tgt[np.clip(bi + k, 0, n - 1)] = 1
        if tgt.sum() == 0 or tgt.sum() == n:
            continue
        apr = average_precision_score(tgt, logit)
        # separation: logits at beats vs at midpoints between consecutive beats
        mids = (truth[:-1] + truth[1:]) / 2.0
        mi = np.clip(np.rint(mids * FPS).astype(int), 0, n - 1)
        snr = float(np.median(logit[bi]) - np.median(logit[mi]))
        bpm = 60.0 / float(np.median(np.diff(truth)))
        rows.append(dict(stem=stem, bpm=bpm, frame_ap=float(apr), peak_snr=snr,
                         pos_frac=float(tgt.mean())))

    json.dump(rows, open(a.out, "w"))
    bpm = np.array([r["bpm"] for r in rows])
    apv = np.array([r["frame_ap"] for r in rows])
    snr = np.array([r["peak_snr"] for r in rows])
    pf = np.array([r["pos_frac"] for r in rows])
    print(f"\n  {label}   n={len(rows)}")
    # RAW AP is not comparable across bands: average_precision_score's chance
    # floor equals the positive rate, and pos_frac (7 frames per beat / length)
    # is mechanically proportional to tempo. Raw AP therefore ORDERS BY TEMPO
    # regardless of evidence quality -- it resolves this script's own decision
    # rule backwards. AP/pos_frac is the prevalence-free lift; peak SNR is
    # prevalence-free by construction. Both are printed; read the lift.
    print(f"  {'band':>8}{'n':>5}{'frame AP':>11}{'AP/chance':>11}"
          f"{'peak SNR':>11}{'pos frac':>10}")
    for lo, hi, tag in BANDS:
        m = (bpm >= lo) & (bpm < hi)
        if m.sum():
            print(f"  {tag:>8}{m.sum():5d}{apv[m].mean():11.4f}"
                  f"{(apv[m] / pf[m]).mean():11.2f}"
                  f"{snr[m].mean():11.3f}{pf[m].mean():10.3f}")
    print("\n  read AP/chance, not raw AP -- raw AP's floor rises with tempo")


if __name__ == "__main__":
    main()
