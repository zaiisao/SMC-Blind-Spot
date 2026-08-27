"""Phase-1a falsification: is display tempo ALONE sufficient to cause an SMC-sized
deficit on audio with none of SMC's other properties?

Take easy, steady, intact-drum clips (ballroom + hainsworth, fold-honest F ~0.82-0.94)
and score three arms, each clip with the released checkpoint that held its fold out:

    A  original
    B  pedalboard.time_stretch to a per-clip rate landing it at 45-70 BPM
       (the SMC tempo region), beats mapped analytically
    C  ARTIFACT CONTROL: stretched by that same rate and stretched BACK, so it
       carries two vocoder passes but the original display tempo

pedalboard is used because it is the exact engine Beat This's own augmentation
pipeline uses (launch_scripts/preprocess_audio.py), so arm B's artifact statistics
match what a wide-range training corpus would contain.

Pre-registered decision rule (cache/tempo_transplant_decision.txt is written first):
    PROCEED  dB >= 0.15  and  dC <= 0.05  and  (dB - dC) >= 0.10, wilcoxon p<0.01
    KILL the tempo programme  if dC >= 0.6 * dB  (the effect is vocoder artifacts,
        and the +0.0179 TTA result must be re-described, not reported as tempo)
    AMBIGUOUS otherwise -> repeat arm B with resampling (no vocoder) as the
        second discriminant.
"""
import argparse, glob, json, os, sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ANN = "/disk1/jaehoon/dataset_store/beat_this_annotations"
AUDIO = "/disk1/jaehoon/dataset_store/audio_by_stem"


def load_folds(dsname):
    out = {}
    p = f"{ANN}/{dsname}/8-folds.split"
    for line in open(p):
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = int(parts[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ballroom", "hainsworth"])
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--target-lo", type=float, default=45.0)
    ap.add_argument("--target-hi", type=float, default=70.0)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--resample", action="store_true",
                    help="arm B via RESAMPLING instead of the vocoder: playback-rate "
                         "change, zero vocoder artifacts, pitch drops with tempo. "
                         "The second discriminant for the AMBIGUOUS branch -- dead "
                         "end #4 measured pitch shift alone at +0.0068 F, so pitch "
                         "cannot explain an SMC-sized deficit. Arm C is then the "
                         "resample there-and-back (a near-perfect identity).")
    ap.add_argument("--out", default="cache/tempo_transplant.json")
    a = ap.parse_args()

    import torch, soxr
    import mir_eval.beat as mb
    from pedalboard import time_stretch
    from beat_this.inference import load_model, split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect

    dev = f"cuda:{a.gpu}"
    spect = LogMelSpect(device=dev)
    post = Postprocessor(type="minimal", fps=50)
    nets = {}

    def track(y22, fold):
        if fold not in nets:
            nets[fold] = load_model(f"fold{fold}", device=dev).eval()
        with torch.inference_mode():
            pr = split_predict_aggregate(
                spect(torch.from_numpy(np.ascontiguousarray(y22)).to(dev)),
                1500, 6, "keep_first", nets[fold])
            est, _ = post(pr["beat"].float(), pr["downbeat"].float())
        return np.asarray(est)

    def score(y22, truth, fold):
        est = track(y22, fold)
        if len(est) < 4:
            return 0.0
        return mb.f_measure(mb.trim_beats(truth), mb.trim_beats(est))

    # collect clips
    items = []
    for dsn in a.datasets:
        folds = load_folds(dsn)
        for bt in sorted(glob.glob(f"{ANN}/{dsn}/annotations/beats/*.beats")):
            stem = os.path.basename(bt)[:-6]
            if stem not in folds:
                continue
            wav = None
            for ext in (".wav", ".flac", ".mp3"):
                w = os.path.join(AUDIO, dsn, stem + ext)
                if os.path.exists(w):
                    wav = w
                    break
            if wav:
                items.append((dsn, stem, wav, bt, folds[stem]))
    rng = np.random.RandomState(0)
    items = [items[i] for i in rng.choice(len(items), min(a.n, len(items)),
                                          replace=False)]
    print(f"[items] {len(items)} clips from {a.datasets}", flush=True)

    rows = []
    for k, (dsn, stem, wav, bt, fold) in enumerate(items):
        truth = np.loadtxt(bt, ndmin=2)[:, 0]
        if len(truth) < 8:
            continue
        y, sr = sf.read(wav, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        bpm = 60.0 / float(np.median(np.diff(truth)))
        # deterministic per-clip target inside the SMC band
        tgt = a.target_lo + (a.target_hi - a.target_lo) * ((k * 7919) % 1000) / 1000.0
        rate = bpm / tgt                      # >1 = slow it down by this factor
        if not (1.05 <= rate <= 4.0):         # already slow, or absurd stretch
            continue

        if a.resample:
            # playback-rate change: no vocoder anywhere, pitch drops with tempo.
            # Upsampling sr -> sr*rate and then PLAYING AT sr stretches duration
            # by `rate` (more samples, same clock), i.e. slows it down.
            yb = soxr.resample(y, sr, int(round(sr * rate)))
            tb = truth * rate
            yc = soxr.resample(yb, int(round(sr * rate)), sr)
        else:
            # pedalboard.time_stretch(audio, sr, stretch_factor): factor>1 SPEEDS UP.
            # We want slower, so pass 1/rate; durations scale by rate, beats too.
            yb = time_stretch(y[None, :], sr, 1.0 / rate)[0]
            tb = truth * rate
            yc = time_stretch(yb[None, :], sr, rate)[0]   # back to original tempo

        y22 = soxr.resample(y, sr, 22050) if sr != 22050 else y
        b22 = soxr.resample(yb, sr, 22050) if sr != 22050 else yb
        c22 = soxr.resample(yc, sr, 22050) if sr != 22050 else yc
        FA = score(y22, truth, fold)
        FB = score(b22, tb, fold)
        FC = score(c22, truth, fold)
        rows.append(dict(ds=dsn, stem=stem, fold=fold, bpm=bpm, target=tgt,
                         rate=rate, FA=FA, FB=FB, FC=FC))
        if (len(rows)) % 25 == 0:
            r = rows
            print(f"  {len(rows)} done  A={np.mean([x['FA'] for x in r]):.3f} "
                  f"B={np.mean([x['FB'] for x in r]):.3f} "
                  f"C={np.mean([x['FC'] for x in r]):.3f}", flush=True)

    json.dump(rows, open(a.out, "w"))
    from scipy import stats
    FA = np.array([x["FA"] for x in rows])
    FB = np.array([x["FB"] for x in rows])
    FC = np.array([x["FC"] for x in rows])
    dB, dC = FA - FB, FA - FC
    wB = stats.wilcoxon(FA, FB).pvalue if np.any(dB) else 1.0
    wC = stats.wilcoxon(FA, FC).pvalue if np.any(dC) else 1.0
    print(f"\nn={len(rows)}  median source bpm {np.median([x['bpm'] for x in rows]):.0f}"
          f" -> target {np.median([x['target'] for x in rows]):.0f}"
          f"  median rate {np.median([x['rate'] for x in rows]):.2f}")
    print(f"  A original          F={FA.mean():.4f}")
    print(f"  B slowed to 45-70   F={FB.mean():.4f}   dB={dB.mean():+.4f}  p={wB:.2e}")
    print(f"  C artifact control  F={FC.mean():.4f}   dC={dC.mean():+.4f}  p={wC:.2e}")
    print(f"  dB - dC = {dB.mean()-dC.mean():+.4f}    dC/dB = "
          f"{dC.mean()/max(1e-9,dB.mean()):.2f}")
    verdict = ("PROCEED" if (dB.mean() >= 0.15 and dC.mean() <= 0.05
                             and dB.mean() - dC.mean() >= 0.10 and wB < 0.01)
               else "KILL" if dC.mean() >= 0.6 * dB.mean()
               else "AMBIGUOUS")
    print(f"\n  PRE-REGISTERED VERDICT: {verdict}")


if __name__ == "__main__":
    main()
