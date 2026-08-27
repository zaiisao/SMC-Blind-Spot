"""Diagnostic D: did the metrical-level error EVER improve during training?

Jaehoon's claim is that the loss under-penalises double-tempo firing, so the
model never had a gradient pushing it toward the correct metrical level. If that
is right, the AMLt-CMLt gap should be present early and FLAT across training,
even while F keeps climbing -- the objective improving on everything except the
thing it does not measure.

If instead the gap shrinks over epochs, the objective was working on it and the
"loss is the bottleneck" story is much weaker.

Costs no new training: every arm kept a checkpoint every 5 epochs.
"""
import argparse, glob, json, os, re
import numpy as np

ANN = "/disk1/jaehoon/dataset_store/beat_this_annotations/smc"
AUDIO = "/disk1/jaehoon/dataset_store/audio_by_stem/smc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--every", type=int, default=10, help="evaluate every Nth epoch")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch, soxr, soundfile as sf
    import mir_eval.beat as mb
    from beat_this.inference import split_predict_aggregate
    from beat_this.model.pl_module import PLBeatThis
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect

    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    spect = LogMelSpect(device=dev)
    post = Postprocessor(type="minimal", fps=50)

    items = []
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
        t = np.loadtxt(bt, ndmin=2)
        if t.size == 0 or len(t) < 8:
            continue
        items.append((stem, wav, t[:, 0]))

    # cache resampled audio once; the checkpoints change, the audio does not
    cache = []
    for stem, wav, truth in items:
        y, sr = sf.read(wav, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        y22 = soxr.resample(y, sr, 22050) if sr != 22050 else y
        cache.append((stem, np.ascontiguousarray(y22), truth,
                      60.0 / float(np.median(np.diff(truth)))))

    ckpts = {}
    for f in glob.glob(os.path.join(a.ckpt_dir, "*.ckpt")):
        m = re.search(r"epoch=(\d+)", f)
        if m:
            ckpts[int(m.group(1))] = f
    eps = sorted(e for e in ckpts if (e + 1) % a.every == 0 or e == max(ckpts))

    out = []
    for ep in eps:
        ck = torch.load(ckpts[ep], map_location=dev, weights_only=False)
        pl = PLBeatThis(**dict(ck["hyper_parameters"]))
        pl.load_state_dict(ck["state_dict"])
        net = pl.model.to(dev).eval()
        F, C, A, B = [], [], [], []
        for stem, y22, truth, bpm in cache:
            with torch.inference_mode():
                pr = split_predict_aggregate(
                    spect(torch.from_numpy(y22).to(dev)), 1500, 6, "keep_first", net)
                est, _ = post(pr["beat"].float(), pr["downbeat"].float())
            est = np.asarray(est)
            if len(est) >= 4:
                rt, et = mb.trim_beats(truth), mb.trim_beats(est)
                c = mb.continuity(rt, et)
                F.append(mb.f_measure(rt, et)); C.append(c[1]); A.append(c[3])
            else:
                F.append(0.0); C.append(0.0); A.append(0.0)
            B.append(bpm)
        F, C, A, B = map(np.array, (F, C, A, B))
        slow = B < 70
        out.append(dict(epoch=ep, F=F.mean(), CMLt=C.mean(), AMLt=A.mean(),
                        gap=float(A.mean() - C.mean()),
                        F_slow=float(F[slow].mean()),
                        gap_slow=float(A[slow].mean() - C[slow].mean())))
        r = out[-1]
        print(f"  epoch {ep:3d}   F={r['F']:.4f}  CMLt={r['CMLt']:.4f}  "
              f"AMLt={r['AMLt']:.4f}  gap={r['gap']:.4f}   "
              f"[<70: F={r['F_slow']:.4f} gap={r['gap_slow']:.4f}]", flush=True)
        del net, pl, ck
        torch.cuda.empty_cache()

    json.dump(out, open(a.out, "w"))
    g = np.array([r["gap_slow"] for r in out])
    e = np.array([r["epoch"] for r in out], dtype=float)
    if len(g) > 2:
        sl = np.polyfit(e, g, 1)[0]
        print(f"\n  <70 AMLt-CMLt gap trend: {sl*100:+.4f} per 100 epochs")
        print(f"    first {g[0]:.4f} -> last {g[-1]:.4f}   change {g[-1]-g[0]:+.4f}")
        print("    FLAT => the objective never acted on metrical level")


if __name__ == "__main__":
    main()
