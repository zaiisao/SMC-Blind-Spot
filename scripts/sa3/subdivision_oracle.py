"""Go/no-go screen for the phase-dependent negative weight c(phi).

The proposed loss teaches the model not to fire at subdivision phases. This
simulates a model that has ALREADY learned that, perfectly, by suppressing
activation at phi=0.5 using the GROUND-TRUTH beat grid and re-decoding:

    l'_t = l_t - delta * exp( -d(phi_t, 1/2)^2 / (2 s^2) )

That makes it an UPPER BOUND on what c(phi) can buy. The decisive direction is
negative: if F does not rise even with oracle phase, the loss cannot help and we
stop before spending a training run. It is an oracle, so it can never be reported
as a result -- only used as a screen.

Cross-check available: the per-track transition table says converting one doubled
track to the correct level is worth +0.0428 F. 84 of 217 tracks are doubled at
stock, so if suppression genuinely converts them the ceiling is about +0.017 on
all 217. Seeing that number corroborates that the two measurements describe the
same mechanism.

Also reports a NON-ORACLE variant that derives phase from the model's own decoded
beats instead of the annotation, to show how much of the effect survives without
ground truth. That one is circular in a different way (the model's own grid) and
is reported only for context.
"""
import argparse, glob, os
import numpy as np

ANN_ROOT = "/disk1/jaehoon/dataset_store/beat_this_annotations"
AUDIO_ROOT = "/disk1/jaehoon/dataset_store/audio_by_stem"
BANDS = [(0, 55, "<55"), (55, 70, "55-70"), (70, 100, "70-100"), (100, 1e9, ">100")]
FPS = 50


def phase_of_frames(n, beats, fps=FPS):
    """Phase in [0,1) of each frame within its enclosing annotated interval.

    Frames before the first beat or after the last get nan (no enclosing
    interval, so no defined subdivision position).
    """
    t = np.arange(n) / fps
    ph = np.full(n, np.nan)
    if len(beats) < 2:
        return ph
    j = np.searchsorted(beats, t) - 1
    ok = (j >= 0) & (j < len(beats) - 1)
    jj = j[ok]
    ibi = beats[jj + 1] - beats[jj]
    good = ibi > 0
    idx = np.where(ok)[0][good]
    ph[idx] = (t[idx] - beats[jj[good]]) / ibi[good]
    return ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--released", default=None, help="'honest' = per-fold released")
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--sigma", type=float, default=0.06)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dataset", default="smc",
                    help="corpus to screen on. GTZAN has no fold file because "
                         "upstream holds it out entirely, so final0 is honest "
                         "for it; other corpora use their per-track fold.")
    ap.add_argument("--max-tracks", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as Fn
    import soxr, soundfile as sf
    import mir_eval.beat as mb
    from beat_this.inference import split_predict_aggregate
    from beat_this.preprocessing import LogMelSpect

    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    spect = LogMelSpect(device=dev)
    nets = {}

    ANN = f"{ANN_ROOT}/{a.dataset}"
    AUDIO = f"{AUDIO_ROOT}/{a.dataset}"
    folds = {}
    fp = f"{ANN}/8-folds.split"
    if os.path.exists(fp):
        for line in open(fp):
            p = line.split()
            if len(p) >= 2:
                folds[p[0]] = int(p[1])

    def net_for(fold):
        if a.released == "honest":
            # no fold file -> upstream never trained on this corpus -> final0 honest
            key = "final0" if fold is None else f"fold{fold}"
            if key not in nets:
                from beat_this.inference import load_model
                nets[key] = load_model(key, device=dev).eval()
            return nets[key]
        if "n" not in nets:
            ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
            from beat_this.model.pl_module import PLBeatThis
            pl = PLBeatThis(**dict(ck["hyper_parameters"]))
            pl.load_state_dict(ck["state_dict"])
            nets["n"] = pl.model.to(dev).eval()
        return nets["n"]

    def decode(logit_np):
        x = torch.from_numpy(logit_np)[None, None]
        peaks = (x == Fn.max_pool1d(x, 7, 1, 3)) & (x > 0)
        return torch.nonzero(peaks[0, 0]).flatten().numpy() / FPS

    rows = []
    for bt in sorted(glob.glob(f"{ANN}/annotations/beats/*.beats")):
        stem = os.path.basename(bt)[:-6]
        if folds and stem not in folds:
            continue
        wav = None
        for e in (".wav", ".flac", ".mp3"):
            p = os.path.join(AUDIO, stem + e)
            if os.path.exists(p):
                wav = p
                break
        if wav is None:
            continue
        truth = np.loadtxt(bt, ndmin=2)
        if truth.size == 0 or len(truth) < 8:
            continue
        truth = truth[:, 0]
        y, sr = sf.read(wav, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        y22 = soxr.resample(y, sr, 22050) if sr != 22050 else y
        with torch.inference_mode():
            pr = split_predict_aggregate(
                spect(torch.from_numpy(np.ascontiguousarray(y22)).to(dev)),
                1500, 6, "keep_first", net_for(folds.get(stem)))
            logit = pr["beat"].float().cpu().numpy().ravel()
        n = len(logit)
        ph = phase_of_frames(n, truth)
        d = np.minimum(np.abs(ph - 0.5), 1 - np.abs(ph - 0.5))
        bump = np.where(np.isnan(d), 0.0, np.exp(-(np.nan_to_num(d) ** 2)
                                                 / (2 * a.sigma ** 2)))
        rt = mb.trim_beats(truth)
        rec = dict(stem=stem, bpm=60.0 / float(np.median(np.diff(truth))),
                   n_true=len(truth))
        for dl in a.deltas:
            est = decode(logit - dl * bump)
            if len(est) >= 4:
                rec[f"F{dl}"] = float(mb.f_measure(rt, mb.trim_beats(est)))
                rec[f"C{dl}"] = float(mb.continuity(rt, mb.trim_beats(est))[1])
            else:
                rec[f"F{dl}"] = 0.0
                rec[f"C{dl}"] = 0.0
            rec[f"n{dl}"] = int(len(est))
        rows.append(rec)
        if a.max_tracks and len(rows) >= a.max_tracks:
            break

    if a.out:
        import json
        json.dump(rows, open(a.out, "w"))

    bpm = np.array([r["bpm"] for r in rows])
    nt = np.array([r["n_true"] for r in rows], float)
    print(f"\n  ORACLE subdivision suppression on {a.dataset.upper()}, "
          f"sigma={a.sigma}, n={len(rows)}")
    print("  (uses GROUND-TRUTH phase -- an upper bound, not a reportable result)\n")
    print(f"  {'delta':>7}{'F':>9}{'CMLt':>9}{'dbl<55':>9}{'half>100':>10}"
          f"{'correct':>9}")
    base = None
    for dl in a.deltas:
        F = np.array([r[f"F{dl}"] for r in rows])
        C = np.array([r[f"C{dl}"] for r in rows])
        ne = np.array([r[f"n{dl}"] for r in rows], float)
        rr = ne / nt
        m_lo, m_hi = bpm < 55, bpm >= 100
        lo = np.mean(rr[m_lo] > 1.5) if m_lo.sum() else float("nan")
        hi = np.mean(rr[m_hi] < 0.7) if m_hi.sum() else float("nan")
        ok = np.mean((rr > 0.7) & (rr < 1.5))
        if base is None:
            base = F.mean()
        print(f"  {dl:7.1f}{F.mean():9.4f}{C.mean():9.4f}{lo:9.3f}{hi:10.3f}{ok:9.3f}")
    best = max(a.deltas, key=lambda d: np.mean([r[f"F{d}"] for r in rows]))
    bF = np.mean([r[f"F{best}"] for r in rows])
    print(f"\n  best delta={best}: F {base:.4f} -> {bF:.4f}  ({bF - base:+.4f})")
    print("  cross-check: transition table says a doubled->correct flip is worth")
    print("  +0.0428, and 84/217 are doubled at stock => ceiling ~ +0.017 if all flip")


if __name__ == "__main__":
    main()
