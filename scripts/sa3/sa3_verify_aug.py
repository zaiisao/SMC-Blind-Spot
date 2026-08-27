"""Paired before/after check on the SDEdit-augmented corpus: are we adding garbage?

Every augmented clip records its source file and window start, so the ORIGINAL
audio for the identical window can be reconstructed and scored against the
identical labels. That makes this a paired test rather than a comparison of two
different samples.

What each column means for the decision:

  lag_ms      cross-correlation lag, original vs augmented. Must stay near zero;
              F's tolerance is 70 ms, so drift silently invalidates labels.
  salience    onset energy at the labelled beats vs random times. A collapse means
              the beats stopped being physically marked -- that IS garbage.
  F / CMLt    released tracker against the true labels. A DROP IS EXPECTED and is
              the point (harder training data), but only if lag and salience hold.
              A drop with drifting lag means broken labels, not useful difficulty.
"""
import argparse, glob, json, os, sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa3_sdedit_fidelity import lag_ms, salience


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-dir", default="/disk1/jaehoon/sd_aug1000")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    import torch, soxr
    import mir_eval.beat as mb
    from beat_this.inference import load_model, split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect

    dev = f"cuda:{a.gpu}"
    net = load_model("final0", device=dev).eval()
    spect = LogMelSpect(device=dev)
    post = Postprocessor(type="minimal", fps=50)

    def track(y22):
        with torch.inference_mode():
            pr = split_predict_aggregate(spect(torch.from_numpy(y22).to(dev)),
                                         1500, 6, "keep_first", net)
            est, _ = post(pr["beat"].float(), pr["downbeat"].float())
        return np.asarray(est)

    def score(y, sr, beats):
        est = track(soxr.resample(y, sr, 22050) if sr != 22050 else y)
        if len(est) < 4:
            return 0.0, 0.0, 0.0
        rt, et = mb.trim_beats(beats), mb.trim_beats(est)
        c = mb.continuity(rt, et)
        return mb.f_measure(rt, et), c[1], c[3]

    js = sorted(glob.glob(os.path.join(a.aug_dir, "*.json")))
    rng = np.random.RandomState(0)
    if len(js) > a.n:
        js = [js[i] for i in sorted(rng.choice(len(js), a.n, replace=False))]

    rows = []
    for k, jp in enumerate(js):
        meta = json.load(open(jp))
        base = jp[:-5]
        aug_wav, btp = base + ".wav", base + ".beats"
        if not (os.path.exists(aug_wav) and os.path.exists(btp)):
            continue
        beats = np.loadtxt(btp, ndmin=2)[:, 0]
        if len(beats) < 8:
            continue
        ya, sr = sf.read(aug_wav, dtype="float32", always_2d=True)
        ya = ya.mean(axis=1)

        # reconstruct the ORIGINAL window
        try:
            yo, sro = sf.read(meta["source_wav"], dtype="float32", always_2d=True)
        except Exception:
            continue
        yo = yo.mean(axis=1)
        if sro != sr:
            yo = soxr.resample(yo, sro, sr)
        s = int(meta["window_start"] * sr)
        yo = yo[s: s + len(ya)]
        if len(yo) < len(ya) * 0.9:
            continue
        yo = yo[:len(ya)]

        Fo, Co, Ao = score(yo, sr, beats)
        Fa, Ca, Aa = score(ya, sr, beats)
        rows.append((lag_ms(yo, ya, sr), salience(yo, sr, beats), salience(ya, sr, beats),
                     Fo, Co, Ao, Fa, Ca, Aa))
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(js)}", flush=True)

    r = np.array(rows)
    n = len(r)
    print(f"\npaired on n={n} clips (identical window, identical labels)\n")
    print(f"{'':<12}{'ORIGINAL':>12}{'AUGMENTED':>12}{'delta':>10}")
    for name, io, ia in (("salience", 1, 2), ("F", 3, 6), ("CMLt", 4, 7), ("AMLt", 5, 8)):
        d = r[:, ia] - r[:, io]
        print(f"{name:<12}{r[:,io].mean():12.4f}{r[:,ia].mean():12.4f}{d.mean():+10.4f}")
    lag = r[:, 0]
    print(f"\nlag: median {np.median(lag):+.1f} ms   |lag|>20ms: {100*np.mean(np.abs(lag)>20):.1f}%"
          f"   |lag|>70ms: {100*np.mean(np.abs(lag)>70):.1f}%")
    # garbage = beats stopped being marked, or the audio moved
    bad = (r[:, 2] < 1.05) | (np.abs(lag) > 70)
    print(f"garbage (salience<1.05 or lag>70ms): {100*bad.mean():.1f}%  ({int(bad.sum())}/{n})")
    orig_bad = (r[:, 1] < 1.05)
    print(f"  for reference, originals with salience<1.05: {100*orig_bad.mean():.1f}%")


if __name__ == "__main__":
    main()
