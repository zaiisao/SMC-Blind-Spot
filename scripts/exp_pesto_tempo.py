"""Run the actual pretrained PESTO model as a tempo estimator on SMC.

RECIPE (Gagnere, Essid & Peeters, ICASSP 2024, "Adapting Pitch-Based Self Supervised Learning
Models for Tempo Estimation"): represent the signal by the constant-Q transform of its
ONSET-STRENGTH FUNCTION rather than of the audio, and generate views by time-stretching rather than
pitch-shifting. Tempo is then literally the "pitch" of the onset-strength function.

PESTO's public API takes audio and computes the CQT internally, so the substitution is done by
handing it the onset-strength function AS IF it were audio. One rescaling is required: a beat at
120 BPM is 2 Hz, far below the pitch range PESTO was trained on, so the OSF's nominal sample rate is
multiplied by SCALE to shift tempo frequencies into that range. This is a relabelling of the time
axis only -- it does not alter the signal -- and it is inverted exactly when converting back:

    BPM = 60 * f_pred / SCALE

HONESTY ABOUT WHAT THIS IS. This is ZERO-SHOT. The released PESTO weights were trained on singing
voice (mir-1k), so a CQT of an onset-strength function is out of distribution for them. Gagnere et
al. TRAINED PESTO's architecture on OSF-CQT with the SSL objective; we are not reproducing that,
we are asking what the released model does on this input with no adaptation. A poor result here
does NOT refute their method -- it only bounds what comes for free.

Compare against, same n=217 SMC, track-level tempo (median IBI as truth):
    our head (baseline)                 52.1% correct, 30.0% octave error
    our head + PESTO equivariance loss  57.6% correct, 27.6% octave error
    Beat Transformer's own tempo head   56.7% correct, 20.7% octave error
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _harness import track_ids, load_gt

SMC_AUD = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=200.0,
                    help="multiplier on the OSF's nominal sample rate, to move tempo frequencies "
                         "(0.5-3.6 Hz for 30-215 BPM) into PESTO's trained pitch range. PESTO's CQT "
                         "tops out at 3322.4 Hz and therefore REQUIRES a claimed sample rate above "
                         "~6645 Hz; at scale=100 the OSF's 43.07 Hz becomes 4307 Hz and every track "
                         "fails with 'top bin has exceeded the Nyquist frequency'. scale=200 gives "
                         "8614 Hz, and maps 30-215 BPM onto 100-717 Hz, inside PESTO's span "
                         "(roughly 26-3322 Hz).")
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--model", default="mir-1k_g7")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", default="/tmp/overnight/pesto_tempo.json")
    args = ap.parse_args()

    import pesto
    osf_sr = args.sr / args.hop                      # onset-strength frame rate, Hz
    print(f"OSF frame rate {osf_sr:.2f} Hz; presenting to PESTO as sr={osf_sr*args.scale:.0f}; "
          f"a {osf_sr*args.scale/2:.0f} Hz Nyquist covers up to "
          f"{60*osf_sr*args.scale/2/args.scale:.0f} BPM", flush=True)

    ids = track_ids()
    if args.limit:
        ids = ids[:args.limit]
    rows = {}
    for n, t in enumerate(ids):
        w = SMC_AUD / f"{t.upper()}.wav"
        if not w.exists():
            continue
        gt = np.asarray(load_gt(t), float)
        if len(gt) < 4:
            continue
        y, sr = librosa.load(str(w), sr=args.sr, mono=True)
        osf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=args.hop)
        osf = (osf - osf.mean()) / (osf.std() + 1e-8)
        x = torch.from_numpy(osf.astype(np.float32))
        try:
            out = pesto.predict(x, int(round(osf_sr * args.scale)), model_name=args.model,
                                step_size=10.0, convert_to_freq=True)
        except Exception as e:
            print(f"  {t}: {type(e).__name__}: {e}", flush=True)
            continue
        # pesto.predict -> (timesteps, preds, confidence, activations)
        preds = out[1] if not torch.is_tensor(out) else out
        conf = out[2] if len(out) > 2 else None
        f = np.asarray(preds.detach().cpu().numpy(), dtype=np.float64).ravel()
        c = (np.asarray(conf.detach().cpu().numpy(), dtype=np.float64).ravel()
             if conf is not None else np.ones_like(f))
        ok = np.isfinite(f) & (f > 0)
        if ok.sum() < 5:
            continue
        # confidence-weighted median in log space, then undo the time-axis relabelling
        order = np.argsort(f[ok])
        fs, cs = f[ok][order], c[ok][order]
        cum = np.cumsum(cs) / cs.sum()
        f_med = float(fs[np.searchsorted(cum, 0.5)])
        bpm = 60.0 * f_med / args.scale
        true_bpm = 60.0 / float(np.median(np.diff(gt)))
        rows[t] = {"pesto_bpm": bpm, "true_bpm": true_bpm}
        if (n + 1) % 40 == 0:
            print(f"  {n+1}/{len(ids)}", flush=True)

    lr = np.array([np.log2(v["pesto_bpm"] / v["true_bpm"]) for v in rows.values()])
    nn_ = len(lr)

    def cls(x):
        for v, nm in ((0, "correct"), (1, "2x"), (-1, "0.5x"),
                      (np.log2(3), "3x"), (-np.log2(3), "1/3x")):
            if abs(x - v) < 0.15:
                return nm
        return "other"

    import collections
    c = collections.Counter(cls(x) for x in lr)
    oct_ = c["2x"] + c["0.5x"] + c["3x"] + c["1/3x"]
    print(f"\nPESTO (zero-shot, {args.model}) as tempo estimator, SMC n={nn_}")
    print(f"  correct {c['correct']}/{nn_} ({c['correct']/nn_:.1%})   "
          f"octave {oct_} ({oct_/nn_:.1%})   other {c['other']}")
    print(f"  median log2 bias {np.median(lr):+.3f}   mean {lr.mean():+.3f}")
    print(f"  breakdown {dict(c)}")
    print("\nreference (same n, same truth definition):")
    print("  our head baseline           52.1% correct, 30.0% octave")
    print("  our head + PESTO equiv loss 57.6% correct, 27.6% octave")
    print("  Beat Transformer tempo head 56.7% correct, 20.7% octave")
    json.dump({"n": nn_, "scale": args.scale, "model": args.model, "rows": rows},
              open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
