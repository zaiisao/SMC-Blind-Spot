"""Go/no-go for drumless pretraining: does Spleeter output live in its own domain?

The archived percussion work measured Spleeter-drumless training at +0.108
in-distribution and ZERO transfer to naturally drumless music, with the two
acoustically distinct (salience 1.81 vs SMC's 2.12). If that gap is caused by
separation artifacts, a student pretrained on Spleeter output learns a skill that
does not apply to SMC, and the whole direction is dead before training starts.

The confound in the archived comparison is that Spleeter-drumless tracks and SMC
are DIFFERENT MUSIC, so the 1.81-vs-2.12 difference could be content, not
artifact. This isolates the artifact:

    take music that is ALREADY drumless (SMC, ASAP piano)
    pass it through Spleeter and keep vocals+bass+other
    there were no drums to remove, so any change is pure artifact
    score the released tracker on both versions, paired, same labels

    F unchanged     -> Spleeter is transparent on drumless audio; a student
                       trained on its output should transfer. GO.
    F drops         -> separation mangles this material and teaches a private
                       domain. NO-GO, and we saved a night.

Fold-honest: each SMC track is scored with the released checkpoint that held its
fold out.
"""
import argparse, glob, os, sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AUDIO = "/disk1/jaehoon/dataset_store/audio_by_stem"
ANN = "/disk1/jaehoon/dataset_store/beat_this_annotations"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-smc", type=int, default=32)
    ap.add_argument("--n-asap", type=int, default=32)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    import torch, soxr, librosa
    import mir_eval.beat as mb
    from spleeter.separator import Separator
    from beat_this.inference import load_model, split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect
    from smc_eval import fold_songs, FPS

    dev = f"cuda:{a.gpu}"
    spect = LogMelSpect(device=dev)
    post = Postprocessor(type="minimal", fps=FPS)
    sep = Separator("spleeter:4stems")
    nets = {}

    def track(y22, net):
        with torch.inference_mode():
            pr = split_predict_aggregate(
                spect(torch.from_numpy(np.ascontiguousarray(y22)).to(dev)),
                1500, 6, "keep_first", net)
            est, _ = post(pr["beat"].float(), pr["downbeat"].float())
        return np.asarray(est)

    def salience(y, sr, beats):
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=128)
        env = np.maximum(env - np.median(env), 0.0)
        te = np.arange(len(env)) * 128 / sr
        ibi = float(np.median(np.diff(beats)))
        k = max(1, int(round(0.10 * ibi / (te[1] - te[0]))))
        def E(t):
            i = np.clip(np.searchsorted(te, t), 0, len(env) - 1)
            return float(np.mean([env[max(0, j - k): j + k + 1].max() for j in i]))
        rng = np.random.RandomState(0)
        return E(beats) / (E(rng.uniform(beats[0], beats[-1], len(beats))) + 1e-9)

    items = []
    for fold in range(8):
        for stem, wav, truth in fold_songs(fold):
            items.append(("smc", fold, stem, str(wav), np.asarray(truth)))
    rng = np.random.RandomState(0)
    smc = [items[i] for i in rng.choice(len(items), min(a.n_smc, len(items)), replace=False)]

    asap = []
    for bt in sorted(glob.glob(f"{ANN}/asap/annotations/beats/*.beats")):
        name = os.path.basename(bt)[:-6]
        w = os.path.join(AUDIO, "asap", name + ".wav")
        if os.path.exists(w):
            asap.append(("asap", None, name, w, np.loadtxt(bt, ndmin=2)[:, 0]))
    asap = [asap[i] for i in rng.choice(len(asap), min(a.n_asap, len(asap)), replace=False)]

    rows = []
    for corpus, fold, stem, wav, truth in smc + asap:
        if len(truth) < 8:
            continue
        key = f"fold{fold}" if corpus == "smc" else "final0"
        if key not in nets:
            nets[key] = load_model(key, device=dev).eval()
        net = nets[key]

        y, sr = sf.read(wav, always_2d=True, dtype="float32")
        if y.shape[1] == 1:
            y = np.repeat(y, 2, axis=1)
        out = sep.separate(y)
        # there are no drums in this material, so this is the artifact-only version
        rest = out["vocals"] + out["bass"] + out["other"]
        drum_leak = float(np.mean(np.abs(out["drums"])))

        o = y.mean(axis=1); r = rest.mean(axis=1)
        o22 = soxr.resample(o, sr, 22050) if sr != 22050 else o
        r22 = soxr.resample(r, sr, 22050) if sr != 22050 else r
        rt = mb.trim_beats(truth)
        res = {}
        for tag, yy in (("orig", o22), ("spl", r22)):
            est = track(yy, net)
            res[tag] = (mb.f_measure(rt, mb.trim_beats(est)) if len(est) >= 4 else 0.0,
                        salience(yy, 22050, truth))
        rows.append((corpus, res["orig"][0], res["spl"][0],
                     res["orig"][1], res["spl"][1], drum_leak))
        if len(rows) % 10 == 0:
            print(f"  {len(rows)}/{len(smc)+len(asap)}", flush=True)

    import math
    from scipy import stats
    print()
    for c in ("smc", "asap", "all"):
        sel = [r for r in rows if c == "all" or r[0] == c]
        if not sel:
            continue
        A = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in sel])
        d = A[:, 1] - A[:, 0]
        p = stats.ttest_rel(A[:, 1], A[:, 0]).pvalue if len(A) > 2 else float("nan")
        print(f"{c.upper():<5} n={len(A):3d}   F orig={A[:,0].mean():.4f} -> "
              f"spleeter={A[:,1].mean():.4f}   delta={d.mean():+.4f} "
              f"se={d.std(ddof=1)/math.sqrt(len(d)):.4f} p={p:.4f}")
        print(f"{'':<5}          salience {A[:,2].mean():.3f} -> {A[:,3].mean():.3f}"
              f"   spurious 'drums' extracted: {A[:,4].mean():.5f}")


if __name__ == "__main__":
    main()
