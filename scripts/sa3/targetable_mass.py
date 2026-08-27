"""How much of SMC's error is actually reachable by a phase-0.5 penalty?

Uses cache/ts_tta_cache/<stem>__r1.0.npz -- the fold-honest released activations
over all 217 SMC tracks, whose r=1.0 identity reproduces F 0.6270 exactly. That is
20x the sample of a single fold and it is the CONVERGED model, so it answers the
question the training arms cannot yet: at the operating point the paper reports,
what share of the false positives sit where our loss can see them?

Reports
  1. the phase distribution of every spurious peak in the corpus
  2. what fraction falls inside the penalty's effective window at various sigma
  3. an ORACLE ceiling: delete spurious peaks inside the window using ground-truth
     phase and re-decode. That is the best any loss of this shape could achieve --
     it assumes perfect learning and costs nothing elsewhere.
  4. the same restricted to DOUBLED tracks, which is what the term is aimed at
"""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts/sa3")
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM/rungs_lab")
CACHE = "/home/sogang/jaehoon/Analyze-SMC/cache/ts_tta_cache"
ANN = "/disk1/jaehoon/dataset_store/beat_this_annotations/smc/annotations/beats"
FPS, TOL = 50, 0.070


def peak_pick(logit, thr=0.0, width=7):
    pad = width // 2
    x = np.pad(logit, pad, mode="constant", constant_values=-1e9)
    mx = np.max(np.lib.stride_tricks.sliding_window_view(x, width), axis=-1)
    return np.nonzero((logit == mx) & (logit > thr))[0]


def phase_of(frames, bf):
    ph = np.full(len(frames), np.nan)
    if len(bf) < 2:
        return ph
    j = np.searchsorted(bf, frames, side="right") - 1
    ok = (j >= 0) & (j < len(bf) - 1)
    jj = np.clip(j, 0, len(bf) - 2)
    ibi = (bf[jj + 1] - bf[jj]).astype(float)
    good = ok & (ibi > 0)
    ph[good] = (frames[good] - bf[jj[good]]) / ibi[good]
    return ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.25])
    a = ap.parse_args()
    from training.evaluation import beat_f_measure

    tracks = []
    for f in sorted(glob.glob(f"{CACHE}/*__r1.0.npz")):
        stem = os.path.basename(f).split("__")[0]
        bp = f"{ANN}/{stem}.beats"
        if not os.path.exists(bp):
            continue
        truth = np.loadtxt(bp, ndmin=2)[:, 0]
        if len(truth) < 8:
            continue
        lg = np.load(f)["logits"]
        n = len(lg)
        bf = np.clip(np.rint(truth * FPS).astype(int), 0, n - 1)
        pk = peak_pick(lg)
        if len(pk) < 4:
            continue
        est = pk / FPS
        lo, hi = truth.min(), truth.max()
        ins = (est >= lo) & (est <= hi)
        matched = np.array([np.min(np.abs(truth - e)) <= TOL for e in est])
        ph = phase_of(pk.astype(float), bf.astype(float))
        d = np.abs(ph - 0.5); d = np.minimum(d, 1 - d)
        tracks.append(dict(stem=stem, lg=lg, truth=truth, pk=pk, est=est,
                           matched=matched, d=d, ins=ins,
                           r=ins.sum() / len(truth),
                           F=float(beat_f_measure(truth, est)),
                           bpm=60 / np.median(np.diff(truth))))

    print(f"\n  {len(tracks)} SMC tracks from the fold-honest cache")
    base = float(np.mean([t["F"] for t in tracks]))
    print(f"  baseline F = {base:.4f}   (published scale 0.6270)\n")

    D = np.concatenate([t["d"][~t["matched"] & t["ins"]] for t in tracks])
    D = D[~np.isnan(D)]
    P = np.concatenate([(0.5 - t["d"])[~t["matched"] & t["ins"]] for t in tracks])
    print(f"  spurious peaks in the corpus: {len(D)}")
    print(f"  distance from phi=0.5  (0 = exactly at the midpoint):")
    h, e = np.histogram(D, bins=10, range=(0, 0.5))
    for i in range(10):
        print(f"    {e[i]:.2f}-{e[i+1]:.2f} {h[i]:6d} "
              + "#" * int(60 * h[i] / max(h.max(), 1)))

    print(f"\n  share of spurious peaks inside a window of half-width w:")
    for w in a.widths:
        print(f"    w={w:.2f}   {float((D < w).mean()):.3f}")

    # oracle: delete spurious peaks inside the window, re-decode, rescore
    print(f"\n  ORACLE ceiling -- delete spurious peaks inside w using GT phase")
    print(f"  {'w':>6}{'F all':>10}{'gain':>9}{'F doubled-only':>17}{'n doubled':>11}")
    dbl = [t for t in tracks if t["r"] > 1.5]
    for w in a.widths:
        Fs, Fd = [], []
        for t in tracks:
            keep = ~((t["d"] < w) & ~t["matched"])
            keep |= np.isnan(t["d"])
            f = float(beat_f_measure(t["truth"], t["est"][keep]))
            Fs.append(f)
            if t["r"] > 1.5:
                Fd.append(f)
        print(f"  {w:6.2f}{np.mean(Fs):10.4f}{np.mean(Fs)-base:+9.4f}"
              f"{np.mean(Fd):17.4f}{len(dbl):11d}")

    print(f"\n  where the corpus actually sits:")
    r = np.array([t["r"] for t in tracks])
    F = np.array([t["F"] for t in tracks])
    print(f"    DOUBLED (r>1.5)  {int((r>1.5).sum()):3d} tracks   mean F {F[r>1.5].mean():.4f}")
    print(f"    HALVED  (r<0.7)  {int((r<0.7).sum()):3d} tracks   mean F {F[r<0.7].mean():.4f}")
    print(f"    in-range         {int(((r>=0.7)&(r<=1.5)).sum()):3d} tracks   "
          f"mean F {F[(r>=0.7)&(r<=1.5)].mean():.4f}")
    good = (r >= 0.7) & (r <= 1.5) & (F >= 0.5)
    phase = (r >= 0.7) & (r <= 1.5) & (F < 0.5)
    print(f"      of which GOOD  {int(good.sum()):3d}   PHASE {int(phase.sum()):3d}"
          f"  (PHASE mean F {F[phase].mean():.4f})")

    # per-track: how many tracks have a majority of their errors at the midpoint?
    frac = np.array([float((t["d"][~t["matched"] & t["ins"]] < 0.15).mean())
                     if (~t["matched"] & t["ins"]).any() else np.nan
                     for t in tracks])
    print(f"\n  tracks whose spurious peaks are MOSTLY (>50%) within 0.15 of the "
          f"midpoint: {int(np.nansum(frac > 0.5))}/{len(tracks)}")
    print(f"    among DOUBLED tracks: "
          f"{int(np.nansum([f > 0.5 for f, t in zip(frac, tracks) if t['r'] > 1.5]))}"
          f"/{len(dbl)}")


if __name__ == "__main__":
    main()
