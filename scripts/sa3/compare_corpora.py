"""Score generated corpora with the released tracker: F / CMLt / AMLt vs their own labels.

The quality gate this project lacked. Reference points measured 2026-08-24:
  warp (real audio, time-warped)  F=0.8407   -- labels are true by construction
  real SMC fold 0                 F=0.6510
  v2 corpus (sharp prompts)       F=0.4944
  v1 corpus (soft prompts)        F=0.4502
A corpus the released tracker cannot follow cannot teach a tracker to follow it.
"""
import argparse, glob, os, sys
import numpy as np

ROOT = "/home/sogang/jaehoon/Analyze-SMC"
sys.path.insert(0, ROOT + "/scripts/sa3"); sys.path.insert(0, "/home/sogang/jaehoon/VBPM/rungs_lab")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", required=True, help="corpus dirs (or name=dir)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dump-dir", default=None,
                    help="write per-clip F/CMLt/AMLt here, for significance tests")
    a = ap.parse_args()

    import torch, soundfile as sf, soxr, mir_eval.beat as mb
    from smc_eval import load_net
    from beat_this.preprocessing import LogMelSpect
    from beat_this.inference import split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    dev = f"cuda:{a.gpu}"
    net = load_net("fold0", dev); net.eval()
    spect = LogMelSpect(device=dev); post = Postprocessor(type="minimal", fps=50)
    print(f"{'corpus':22s} {'n':>4s} {'F':>8s} {'CMLt':>8s} {'AMLt':>8s}")
    for spec in a.dirs:
        name, _, d = spec.partition("=")
        if not d:
            d = name; name = os.path.basename(d.rstrip("/"))
        fs = [b for b in sorted(glob.glob(d + "/*.beats")) if os.path.getsize(b) > 0]
        if not fs:
            print(f"{name:22s}  (no clips)"); continue
        rng = np.random.RandomState(0)
        sel = [fs[i] for i in rng.choice(len(fs), min(a.n, len(fs)), replace=False)]
        F, C, A, NAMES = [], [], [], []
        with torch.inference_mode():
            for b in sel:
                t = np.loadtxt(b, ndmin=2)
                if t.size == 0 or len(t) < 8:
                    continue
                t = t[:, 0]
                y, sr = sf.read(b[:-6] + ".wav", dtype="float32", always_2d=True)
                y = y.mean(axis=1)
                if sr != 22050:
                    y = soxr.resample(y, sr, 22050)
                pr = split_predict_aggregate(spect(torch.from_numpy(y).to(dev)), 1500, 6,
                                             "keep_first", net)
                bt, _ = post(pr["beat"].float(), pr["downbeat"].float())
                est = np.asarray(bt)
                rt, et = mb.trim_beats(t), mb.trim_beats(est)
                NAMES.append(os.path.basename(b)[:-6])
                if not len(et) or not len(rt):
                    F.append(0.0); C.append(0.0); A.append(0.0); continue
                F.append(mb.f_measure(rt, et))
                cmlc, cmlt, amlc, amlt = mb.continuity(rt, et)
                C.append(cmlt); A.append(amlt)
        print(f"{name:22s} {len(F):4d} {np.mean(F):8.4f} {np.mean(C):8.4f} {np.mean(A):8.4f}")
        # Means alone cannot say whether a difference survives n=100 sampling noise:
        # the v2 baseline itself moved CMLt 0.2619 -> 0.2322 between two samples.
        if a.dump_dir:
            os.makedirs(a.dump_dir, exist_ok=True)
            with open(os.path.join(a.dump_dir, f"{name}.csv"), "w") as fh:
                fh.write("clip,F,CMLt,AMLt\n")
                for i, nm in enumerate(NAMES):
                    fh.write(f"{nm},{F[i]:.6f},{C[i]:.6f},{A[i]:.6f}\n")


if __name__ == "__main__":
    main()
