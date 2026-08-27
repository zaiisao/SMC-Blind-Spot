"""End-to-end check that the wide corpus is labelled correctly.

The dataloader maps a precomputed variant back to its labels by name:
    beat_time_augmented = beat_time / (1 + pct/100)
so for ts-60 the beats must sit 2.5x later in a 2.5x longer spectrogram. If that
mapping and the rendered audio ever disagree, training silently learns wrong
timing -- and the NEW range (-60..-24) is exactly the part no upstream code path
has ever exercised.

Measures onset salience: mel-flux energy at the mapped beat positions versus at
random positions in the same file. A correctly labelled variant scores well above
1.0; a mislabelled one collapses toward 1.0 because the beats point nowhere in
particular.
"""
import os, random
import numpy as np

SPECT = "/disk1/jaehoon/bt_wide/data/audio/spectrograms/wide"
BEATS = "/disk1/jaehoon/bt_wide/data/annotations/wide/annotations/beats"
FPS = 50.0


def salience(spec, beats_sec, rng):
    """Onset-flux energy at beat frames vs random frames."""
    flux = np.maximum(np.diff(spec.astype(np.float32), axis=0), 0).sum(axis=1)
    flux = np.maximum(flux - np.median(flux), 0)
    if flux.max() <= 0:
        return np.nan
    n = len(flux)
    tol = 2  # +/-2 frames = +/-40 ms, inside F's 70 ms tolerance

    def E(ts):
        idx = np.clip((np.asarray(ts) * FPS).astype(int), 0, n - 1)
        return np.mean([flux[max(0, i - tol): i + tol + 1].max() for i in idx])

    b = np.asarray(beats_sec)
    b = b[(b >= 0) & (b < n / FPS)]
    if len(b) < 6:
        return np.nan
    r = rng.uniform(b[0], b[-1], len(b))
    return E(b) / (E(r) + 1e-9)


def main():
    rng = np.random.RandomState(0)
    tracks = sorted(os.listdir(SPECT))
    random.seed(0)
    sample = random.sample(tracks, 40)

    # shipped range vs the NEW range that no upstream code has exercised
    variants = [("track.npy", 0), ("track_ts-20.npy", -20), ("track_ts20.npy", 20),
                ("track_ts-32.npy", -32), ("track_ts-44.npy", -44),
                ("track_ts-60.npy", -60)]

    print(f"  {'variant':<18}{'factor':>8}{'len ratio':>11}{'salience':>10}"
          f"{'n':>5}   (salience ~1.0 = labels point nowhere)")
    for name, pct in variants:
        factor = 1.0 + pct / 100.0
        sals, ratios = [], []
        for t in sample:
            base_p = os.path.join(SPECT, t, "track.npy")
            p = os.path.join(SPECT, t, name)
            bt = os.path.join(BEATS, t + ".beats")
            if not (os.path.exists(p) and os.path.exists(bt)):
                continue
            try:
                spec = np.load(p)
                base = np.load(base_p)
                arr = np.loadtxt(bt, ndmin=2)
            except Exception:
                continue
            if arr.size == 0:
                continue
            beats = arr[:, 0] / factor          # the dataloader's own mapping
            s = salience(spec, beats, rng)
            if not np.isnan(s):
                sals.append(s)
                ratios.append(spec.shape[0] / base.shape[0])
        if sals:
            print(f"  {name:<18}{factor:8.2f}{np.mean(ratios):11.3f}"
                  f"{np.median(sals):10.2f}{len(sals):5d}")

    # negative control: deliberately WRONG mapping on the new range
    print("\n  CONTROL -- ts-60 scored with the WRONG factor (labels unstretched):")
    sals = []
    for t in sample:
        p = os.path.join(SPECT, t, "track_ts-60.npy")
        bt = os.path.join(BEATS, t + ".beats")
        if not (os.path.exists(p) and os.path.exists(bt)):
            continue
        try:
            spec = np.load(p); arr = np.loadtxt(bt, ndmin=2)
        except Exception:
            continue
        if arr.size == 0:
            continue
        s = salience(spec, arr[:, 0], rng)   # NO division by factor
        if not np.isnan(s):
            sals.append(s)
    print(f"  {'track_ts-60 (bad)':<18}{'--':>8}{'--':>11}{np.median(sals):10.2f}"
          f"{len(sals):5d}   <- should be MUCH lower than the correct row")


if __name__ == "__main__":
    main()
