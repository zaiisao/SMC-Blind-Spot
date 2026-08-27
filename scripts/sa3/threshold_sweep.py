"""Does slow music need a LOWER peak-picking threshold?

Measured on arm A (frame_evidence.py): slow tracks have the HIGHEST
prevalence-corrected ranking lift (3.64 vs 2.33 above 100 bpm) but the LOWEST
absolute logit margin (peak SNR 4.27 vs 5.35). The model ranks beat frames
correctly on slow music, it is just less confident about them.

That matters because `postp_minimal` hard-codes the decision at logit > 0
(probability > 0.5). Correctly-ranked peaks that sit below that bar are simply
discarded, so a confidence deficit turns into missed beats.

This caches each track's beat logits ONCE, then re-decodes at many thresholds --
so the sweep costs one forward pass per track, not one per threshold. The
max-pool peak-picking is reproduced exactly as in postprocessor.postp_minimal
(width 7, stride 1, pad 3), with ONLY the threshold varied.

A tempo-dependent threshold is deployable in a way the per-track threshold
oracle already in our notes (0.627 -> 0.670) is not: tempo can be estimated from
the prediction itself, no ground truth needed. Reported both ways -- the honest
deployable number uses the ESTIMATED tempo from the default decode.
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
    ap.add_argument("--released", default=None,
                    help="fold-honest: pass 'honest' to use each track's own fold checkpoint")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import soxr, soundfile as sf
    import mir_eval.beat as mb
    from beat_this.inference import split_predict_aggregate
    from beat_this.preprocessing import LogMelSpect

    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    spect = LogMelSpect(device=dev)

    folds = {}
    for line in open(f"{ANN}/8-folds.split"):
        p = line.split()
        if len(p) >= 2:
            folds[p[0]] = int(p[1])

    nets = {}

    def get_net(fold):
        if a.released == "honest":
            if fold not in nets:
                from beat_this.inference import load_model
                nets[fold] = load_model(f"fold{fold}", device=dev).eval()
            return nets[fold]
        if "net" not in nets:
            ckpt = torch.load(a.checkpoint, map_location=dev, weights_only=False)
            from beat_this.model.pl_module import PLBeatThis
            pl = PLBeatThis(**dict(ckpt["hyper_parameters"]))
            pl.load_state_dict(ckpt["state_dict"])
            nets["net"] = pl.model.to(dev).eval()
        return nets["net"]

    def decode(logit_t, thr):
        """postp_minimal peak-picking, threshold varied."""
        x = logit_t[None, None]
        peaks = (x == F.max_pool1d(x, 7, 1, 3)) & (x > thr)
        idx = torch.nonzero(peaks[0, 0]).flatten().cpu().numpy()
        return idx / FPS

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
        y, sr = sf.read(wav, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        y22 = soxr.resample(y, sr, 22050) if sr != 22050 else y
        net = get_net(folds[stem])
        with torch.inference_mode():
            pr = split_predict_aggregate(
                spect(torch.from_numpy(np.ascontiguousarray(y22)).to(dev)),
                1500, 6, "keep_first", net)
            logit = pr["beat"].float().ravel()

        rt = mb.trim_beats(truth)
        per_thr = {}
        for thr in a.thresholds:
            est = decode(logit, thr)
            if len(est) >= 4:
                per_thr[thr] = float(mb.f_measure(rt, mb.trim_beats(est)))
            else:
                per_thr[thr] = 0.0
        # tempo ESTIMATED from the default decode -- no ground truth used
        est0 = decode(logit, 0.0)
        est_bpm = (60.0 / float(np.median(np.diff(est0)))
                   if len(est0) >= 4 else 0.0)
        rows.append(dict(stem=stem, bpm=60.0 / float(np.median(np.diff(truth))),
                         est_bpm=est_bpm, F=per_thr))

    json.dump(rows, open(a.out, "w"))

    bpm = np.array([r["bpm"] for r in rows])
    print(f"\n  n={len(rows)}   F by threshold and TRUE tempo band")
    print(f"  {'thr':>7}" + "".join(f"{t:>10}" for _, _, t in BANDS) + f"{'ALL':>10}")
    for thr in a.thresholds:
        Fv = np.array([r["F"][thr] for r in rows])
        line = f"  {thr:7.2f}"
        for lo, hi, _ in BANDS:
            m = (bpm >= lo) & (bpm < hi)
            line += f"{Fv[m].mean():10.4f}"
        print(line + f"{Fv.mean():10.4f}")

    est = np.array([r["est_bpm"] for r in rows])
    base = np.array([r["F"][0.0] for r in rows])
    print(f"\n  default (thr=0.0) ALL = {base.mean():.4f}")

    def best_by_band_on(idx, key_bpm):
        """argmax threshold per band, fitted ONLY on the tracks in idx."""
        out = {}
        for lo, hi, tag in BANDS:
            m = np.zeros(len(rows), bool)
            m[idx] = True
            m &= (key_bpm >= lo) & (key_bpm < hi)
            if not m.sum():
                out[tag] = 0.0
                continue
            sc = {t: np.array([r["F"][t] for r in rows])[m].mean()
                  for t in a.thresholds}
            out[tag] = max(sc, key=sc.get)
        return out

    # IN-SAMPLE reference (reported, but it is NOT a deployable estimate):
    # banding by TRUE bpm and picking argmax over the same 217 tracks that are
    # then scored uses ground truth twice and cannot be believed if positive.
    ins = best_by_band_on(np.arange(len(rows)), bpm)
    for lo, hi, tag in BANDS:
        m = (bpm >= lo) & (bpm < hi)
        sc = {t: np.array([r["F"][t] for r in rows])[m].mean() for t in a.thresholds}
        print(f"    best thr for TRUE {tag:>7}: {ins[tag]:+.2f} "
              f"(F {sc[ins[tag]]:.4f} vs {sc[0.0]:.4f} at default)  [IN-SAMPLE]")

    # HONEST estimate: two-fold cross-fit. Thresholds are fitted on one half and
    # applied to the other, and banding uses the ESTIMATED tempo (available at
    # inference) rather than the true one.
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(rows))
    folds = [perm[: len(rows) // 2], perm[len(rows) // 2:]]
    dep = np.empty(len(rows))
    for fit_idx, app_idx in ((folds[0], folds[1]), (folds[1], folds[0])):
        tbl = best_by_band_on(fit_idx, est)          # fit on est bpm, held-out half
        for i in app_idx:
            e = est[i]
            tag = "<70" if e < 70 else ("70-100" if e < 100 else ">100")
            dep[i] = rows[i]["F"][tbl[tag]]
    print(f"\n  CROSS-FIT (thresholds fitted on the other half, banded by ESTIMATED "
          f"tempo): {dep.mean():.4f}   delta {dep.mean()-base.mean():+.4f}")
    from scipy import stats
    d = dep - base
    if np.any(d):
        print(f"    paired p={stats.wilcoxon(dep, base).pvalue:.2e} "
              f"(n_nonzero={int((d != 0).sum())} of {len(d)})")
    else:
        print("    identical to default on every track")


if __name__ == "__main__":
    main()
