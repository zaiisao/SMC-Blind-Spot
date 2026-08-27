"""On tracks that are still DOUBLED, is the penalty even landing on the extra peaks?

The phase term can only suppress a false positive that sits near phi=0.5 of the
ANNOTATED interval. Three ways it can fail to attack doubling:

  1. the extra peaks are not at phi=0.5   -> the term cannot see them
  2. they are at phi=0.5 but the gate is shut -> the term is switched off there
  3. the term fires but is too weak to remove the peak

This distinguishes them: for every DOUBLED track it takes the peaks the decoder
actually emitted, splits them into MATCHED (within 70 ms of a true beat) and
SPURIOUS, and reports where the spurious ones live in phase, what gate they see,
and what weight the loss puts on them.
"""
import argparse, glob, sys
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts/sa3")
from smc_eval import load_net, fold_songs
from beat_this.model.loss import GatedPhaseShiftTolerantBCELoss as GL
FPS, TOL = 50, 0.070


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gph13")
    ap.add_argument("--epoch", default="014")
    ap.add_argument("--base", default="/disk1/jaehoon/bt_wide/phase5")
    ap.add_argument("--ckpt", default=None, help="explicit path or a released shortname like fold0")
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    m = GL(pos_weight=32, tolerance=3, phase_gamma=1.3, normalise=False)

    import soundfile as sf, soxr
    from beat_this.preprocessing import LogMelSpect
    from beat_this.inference import split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from training.evaluation import beat_f_measure
    ck = a.ckpt or glob.glob(f"{a.base}/{a.arm}/*epoch={a.epoch}.ckpt")[0]
    net = load_net(ck, dev).eval()
    spect = LogMelSpect(device=dev)
    post = Postprocessor(type="minimal", fps=FPS)

    allsp, rows = [], []
    with torch.inference_mode():
        for stem, wav, truth in fold_songs(0):
            y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != 22050:
                y = soxr.resample(y, sr, 22050)
            S = spect(torch.from_numpy(y).to(dev))
            pr = split_predict_aggregate(S, 1500, 6, "keep_first", net)
            lg = pr["beat"].float().cpu()
            n = lg.shape[-1]
            bt, _ = post(pr["beat"].float(), pr["downbeat"].float())
            est = np.asarray(bt[0] if isinstance(bt, (list, tuple)) else bt).ravel()
            lo, hi = truth.min(), truth.max()
            inside = est[(est >= lo) & (est <= hi)]
            r = len(inside) / max(len(truth), 1)
            if r <= 1.5:
                continue                                    # DOUBLED only
            t = torch.zeros(1, n)
            f = np.clip(np.rint(truth * FPS).astype(int), 0, n - 1)
            t[0, f] = 1.0
            ph = m._phase(t)[0].numpy()
            gate = m._gate(lg.view(1, -1), t)[0].numpy()
            fr = np.clip(np.rint(inside * FPS).astype(int), 0, n - 1)
            matched = np.array([np.min(np.abs(truth - e)) <= TOL for e in inside])
            sp = fr[~matched]
            if len(sp) == 0:
                continue
            p = ph[sp]
            d = np.abs(p - 0.5); d = np.minimum(d, 1 - d)
            bump = np.exp(-(d ** 2) / (2 * 0.06 ** 2))
            w = 1 + 1.3 * gate[sp] * bump
            allsp.append(p[~np.isnan(p)])
            rows.append(dict(stem=stem, r=r, n_sp=len(sp),
                             near_mid=float(np.nanmean(d < 0.05)),
                             gate=float(np.nanmean(gate[sp])),
                             w=float(np.nanmean(w)),
                             F=float(beat_f_measure(truth, est)),
                             bpm=60 / np.median(np.diff(truth))))

    label = a.ckpt if a.ckpt else f"{a.arm} epoch {a.epoch}"
    print(f"\n  {label}: {len(rows)} DOUBLED tracks\n")
    print(f"  {'track':<12}{'bpm':>5}{'r':>6}{'spurious':>10}{'frac@phi.5':>12}"
          f"{'mean gate':>11}{'mean w':>9}{'F':>7}")
    for x in sorted(rows, key=lambda z: -z["n_sp"]):
        print(f"  {x['stem']:<12}{x['bpm']:5.0f}{x['r']:6.2f}{x['n_sp']:10d}"
              f"{x['near_mid']:12.2f}{x['gate']:11.3f}{x['w']:9.3f}{x['F']:7.3f}")

    P = np.concatenate(allsp) if allsp else np.array([])
    if len(P):
        print(f"\n  phase histogram of ALL spurious peaks (n={len(P)}):")
        h, edges = np.histogram(P, bins=10, range=(0, 1))
        for i in range(10):
            bar = "#" * int(60 * h[i] / max(h.max(), 1))
            mark = "  <- penalty peak" if 0.45 <= edges[i] < 0.55 else ""
            print(f"    {edges[i]:.1f}-{edges[i+1]:.1f} {h[i]:5d} {bar}{mark}")
        d = np.abs(P - 0.5); d = np.minimum(d, 1 - d)
        print(f"\n  fraction of spurious peaks within 0.05 of phi=0.5: "
              f"{float((d < 0.05).mean()):.3f}")
        print(f"  fraction within 0.15                              : "
              f"{float((d < 0.15).mean()):.3f}")
        print(f"  mean penalty weight they receive                  : "
              f"{float(np.nanmean(1 + 1.3 * np.exp(-(d**2)/(2*0.06**2)))):.3f}"
              f"   (max possible 2.30)")


if __name__ == "__main__":
    main()
