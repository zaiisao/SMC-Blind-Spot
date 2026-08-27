"""Build the two matched arms for the SDEdit augmentation experiment.

The comparison has to isolate the TRANSFORMATION, not the source material. Both
arms add the same 840-odd windows from the same ASAP/SMC tracks with the same
labels; only one set has been through SDEdit. Comparing augmented clips against
plain "real training set" instead would confound the transformation with extra
ASAP/SMC exposure.

    control/   the ORIGINAL audio window, untouched
    aug/       the SDEdit version of that same window

Also applies the quality filter measured on 2026-08-24 (400 paired clips): 16.2%
of augmented clips have onset salience below 1.05 or lag beyond F's 70 ms
tolerance, against 10.2% of the source windows. Clips failing on EITHER side are
dropped from BOTH arms, so the two datasets stay paired clip for clip.
"""
import argparse, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def one(args):
    jp, control_dir, min_sal, max_lag = args
    from sa3_sdedit_fidelity import lag_ms, salience
    import soxr

    meta = json.load(open(jp))
    base = jp[:-5]
    stem = os.path.basename(base)
    aug_wav, btp = base + ".wav", base + ".beats"
    if not (os.path.exists(aug_wav) and os.path.exists(btp)):
        return None
    beats = np.loadtxt(btp, ndmin=2)
    if len(beats) < 8:
        return None
    bt = beats[:, 0]

    ya, sr = sf.read(aug_wav, dtype="float32", always_2d=True)
    ya = ya.mean(axis=1)
    try:
        yo, sro = sf.read(meta["source_wav"], dtype="float32", always_2d=True)
    except Exception:
        return None
    yo = yo.mean(axis=1)
    if sro != sr:
        yo = soxr.resample(yo, sro, sr)
    s = int(meta["window_start"] * sr)
    yo = yo[s: s + len(ya)]
    if len(yo) < len(ya) * 0.9:
        return None
    yo = yo[:len(ya)]

    sal_a = salience(ya, sr, bt)
    sal_o = salience(yo, sr, bt)
    lag = lag_ms(yo, ya, sr)
    keep = (sal_a >= min_sal) and (sal_o >= min_sal) and (abs(lag) <= max_lag)

    # write the control window regardless; the keep-list decides what is used
    sf.write(os.path.join(control_dir, stem + ".wav"), yo, sr, format="WAV")
    with open(os.path.join(control_dir, stem + ".beats"), "w") as fh:
        for row in beats:
            mark = int(row[1]) if beats.shape[1] > 1 else 2
            fh.write(f"{row[0]:.6f}\t{mark}\n")
    json.dump(meta, open(os.path.join(control_dir, stem + ".json"), "w"))
    return stem, keep, sal_o, sal_a, lag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-dir", default="/disk1/jaehoon/sd_aug1000")
    ap.add_argument("--control-dir", default="/disk1/jaehoon/sd_control1000")
    ap.add_argument("--min-salience", type=float, default=1.05)
    ap.add_argument("--max-lag", type=float, default=70.0)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()

    os.makedirs(a.control_dir, exist_ok=True)
    js = sorted(glob.glob(os.path.join(a.aug_dir, "*.json")))
    jobs = [(j, a.control_dir, a.min_salience, a.max_lag) for j in js]
    with ProcessPoolExecutor(a.workers) as ex:
        res = [r for r in ex.map(one, jobs) if r is not None]

    keep = [r[0] for r in res if r[1]]
    sal_o = np.array([r[2] for r in res])
    sal_a = np.array([r[3] for r in res])
    lag = np.array([r[4] for r in res])

    with open(os.path.join(a.aug_dir, "keep.txt"), "w") as fh:
        fh.write("\n".join(keep) + "\n")
    with open(os.path.join(a.control_dir, "keep.txt"), "w") as fh:
        fh.write("\n".join(keep) + "\n")

    print(f"clips examined      {len(res)}")
    print(f"kept (both arms)    {len(keep)}   ({100*len(keep)/max(1,len(res)):.1f}%)")
    print(f"dropped, aug low    {int((sal_a < a.min_salience).sum())}")
    print(f"dropped, src low    {int((sal_o < a.min_salience).sum())}")
    print(f"dropped, lag>{a.max_lag:.0f}ms   {int((np.abs(lag) > a.max_lag).sum())}")
    print(f"salience  source {np.median(sal_o):.3f}  augmented {np.median(sal_a):.3f}")
    print(f"lag       median {np.median(lag):+.1f} ms")


if __name__ == "__main__":
    main()
