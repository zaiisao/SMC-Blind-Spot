"""Does test-time rate averaging transfer to OUR checkpoints?

Everything we know about TTA comes from the released model's cached activations
on SMC (+0.018 overall, +0.039 on slow music, adversarially verified). Nobody has
checked it works on a model we trained. This does that on the held-out fold-0 set.

Rate change is applied by resampling the spectrogram along time by factor r,
running the model, then resampling the logits back by 1/r so every rate lands on
a common time base. Averaging is done in the logit domain and, separately, in the
probability domain -- the adversarial pass showed the two are not interchangeable
and that the honest headline number is the matched-scale (logit) one.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

BT = "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"


def resample_time(x, factor):
    """x: (T, C) or (T,) -> stretched along time by `factor`."""
    t = torch.as_tensor(x, dtype=torch.float32)
    squeeze = t.ndim == 1
    if squeeze:
        t = t[:, None]
    n = max(2, int(round(t.shape[0] * factor)))
    out = Fn.interpolate(t.T[None], size=n, mode="linear", align_corners=False)[0].T
    return out[:, 0] if squeeze else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.85, 1.0, 1.2])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.chdir(BT)
    sys.path.insert(0, BT)
    import mir_eval
    from beat_this.dataset import BeatDataModule
    from beat_this.model.pl_module import PLBeatThis

    dev = f"cuda:{a.gpu}"
    m = PLBeatThis(
        spect_dim=128, fps=50, transformer_dim=512, ff_mult=4, n_layers=6,
        stem_dim=32, dropout={"frontend": 0.1, "transformer": 0.2}, lr=8e-4,
        weight_decay=0.01, pos_weights={"beat": 20, "downbeat": 86},
        head_dim=32, loss_type="shift_tolerant_weighted_bce",
        warmup_steps=1000, max_epochs=100,
    )
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu",
                                 weights_only=False)["state_dict"], strict=False)
    m = m.eval().to(dev)

    dm = BeatDataModule(Path("data"), batch_size=1, train_length=1500, spect_fps=50,
                        num_workers=4, test_dataset="gtzan",
                        length_based_oversampling_factor=0.65, augmentations={},
                        hung_data=False, no_val=False, fold=0)
    dm.setup(stage="fit")

    fh = open(a.out, "w")
    fh.write("path,corpus,bpm," + ",".join(f"F_r{r}" for r in a.rates)
             + ",F_logitavg,F_probavg,CMLt_base,CMLt_logitavg,AMLt_base,AMLt_logitavg\n")

    with torch.no_grad():
        for batch in dm.val_dataloader():
            spect = batch["spect"][0].cpu().numpy()
            truth = np.frombuffer(batch["truth_orig_beat"][0])
            if len(truth) < 3:
                continue
            T = spect.shape[0]
            logits = {}
            for r in a.rates:
                sp = resample_time(spect, r).to(dev)[None]
                pad = torch.ones(1, sp.shape[1], dtype=torch.bool, device=dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    p = m.model(sp)
                lb = p["beat"].float()[0].cpu()
                db = p["downbeat"].float()[0].cpu()
                logits[r] = (resample_time(lb, T / lb.shape[0]),
                             resample_time(db, T / db.shape[0]))

            def score(lb, db):
                pad = torch.ones(1, lb.shape[0], dtype=torch.bool)
                pb, _ = m.postprocessor(lb[None].to(dev), db[None].to(dev), pad.to(dev))
                est = np.asarray(pb if not isinstance(pb, tuple) else pb[0])
                return m.metrics(truth, est, step="test")

            per = {r: score(*logits[r]) for r in a.rates}
            la = torch.stack([logits[r][0] for r in a.rates]).mean(0)
            da = torch.stack([logits[r][1] for r in a.rates]).mean(0)
            mlog = score(la, da)
            pa = torch.stack([torch.sigmoid(logits[r][0]) for r in a.rates]).mean(0)
            pd = torch.stack([torch.sigmoid(logits[r][1]) for r in a.rates]).mean(0)
            eps = 1e-6
            mprob = score(torch.log(pa.clamp(eps, 1 - eps) / (1 - pa).clamp(eps, 1)),
                          torch.log(pd.clamp(eps, 1 - eps) / (1 - pd).clamp(eps, 1)))

            path = str(batch["spect_path"][0])
            base = per[1.0] if 1.0 in per else per[a.rates[0]]
            fh.write(f'"{path}",{path.split("/",1)[0]},{60.0/np.median(np.diff(truth)):.3f},'
                     + ",".join(f'{per[r]["F-measure"]:.6f}' for r in a.rates)
                     + f',{mlog["F-measure"]:.6f},{mprob["F-measure"]:.6f},'
                     f'{base["CMLt"]:.6f},{mlog["CMLt"]:.6f},'
                     f'{base["AMLt"]:.6f},{mlog["AMLt"]:.6f}\n')
    fh.close()
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
