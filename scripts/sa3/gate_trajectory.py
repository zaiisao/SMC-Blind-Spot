"""Is the gate actually opening as training proceeds?

The gate is a_j * a_{j+1}, the model's own (detached) confidence at the two
annotated beats bounding each midpoint. Its design premise is that it is near
zero early -- no grid to be confident about -- and opens as the model learns,
so the anti-doubling penalty only switches on once there is a real pulse to
protect.

That premise is testable, not just assertable: compute the mean gate at phi=0.5
on SMC fold 0 from each checkpoint. If it does not rise with epoch, the gate
never activates and "it will show later" is false.

Reference from the cached released activations: mean gate 0.636 (sd 0.257).
That is roughly where a converged model sits, so it is the level the arm has to
reach for the mechanism to be doing anything.
"""
import argparse, glob, os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts/sa3")
from beat_this.model.loss import GatedPhaseShiftTolerantBCELoss as GL
from smc_eval import load_net, fold_songs

FPS = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["gph13", "ph13"])
    ap.add_argument("--base", default="/disk1/jaehoon/bt_wide/phase5")
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    m = GL(pos_weight=32, tolerance=3, phase_gamma=1.3)

    songs = fold_songs(0)
    import soundfile as sf, soxr
    from beat_this.preprocessing import LogMelSpect
    from beat_this.inference import split_predict_aggregate
    spect = LogMelSpect(device=dev)

    print(f"\n  mean gate at phi=0.5, SMC fold 0 (n={len(songs)})")
    print(f"  released-model reference: 0.636\n")
    print(f"  {'arm':<8}{'epoch':>7}{'mean gate':>12}{'sd':>8}{'frac>0.5':>10}")
    for arm in a.arms:
        for c in sorted(glob.glob(f"{a.base}/{arm}/*epoch=*.ckpt")):
            ep = int(c.split("epoch=")[1][:3])
            net = load_net(c, dev)
            net.eval()
            gs = []
            with torch.inference_mode():
                for stem, wav, truth in songs:
                    y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
                    y = y.mean(axis=1)
                    if sr != 22050:
                        y = soxr.resample(y, sr, 22050)
                    S = spect(torch.from_numpy(y).to(dev))
                    pr = split_predict_aggregate(S, 1500, 6, "keep_first", net)
                    lg = pr["beat"].float().cpu()
                    n = lg.shape[-1]
                    t = torch.zeros(1, n)
                    f = np.clip(np.rint(truth * FPS).astype(int), 0, n - 1)
                    t[0, f] = 1.0
                    g = m._gate(lg.view(1, -1), t)[0].numpy()
                    ph = m._phase(t)[0].numpy()
                    sel = np.abs(ph - 0.5) < 0.05
                    if sel.any():
                        gs.append(float(np.nanmean(g[sel])))
            gs = np.array(gs)
            print(f"  {arm:<8}{ep:7d}{gs.mean():12.3f}{gs.std():8.3f}"
                  f"{(gs > 0.5).mean():10.2f}")


if __name__ == "__main__":
    main()
