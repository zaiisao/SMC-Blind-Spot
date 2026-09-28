"""Run a trained Beat This checkpoint over the SDEdit corpus and score it
against the inherited beat grids.

Two things this can catch:
  1. the generated audio has no trackable pulse (F near zero everywhere), in
     which case training on it adds noise;
  2. the audio is fine but the inherited labels drifted, which shows up as the
     model tracking a consistent grid that the annotation disagrees with --
     distinguishable because F stays low while CMLt/AMLt or the doubled/halved
     variants score high.

Uses the spectrograms already packed into sdedit.npz, so no audio decoding.
"""

import argparse
import os
from pathlib import Path

import mir_eval
import numpy as np
import torch

BT = "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"


def f1(est, ref):
    if len(est) < 2 or len(ref) < 2:
        return 0.0
    e = mir_eval.beat.trim_beats(est)
    r = mir_eval.beat.trim_beats(ref)
    if len(e) < 1 or len(r) < 1:
        return 0.0
    return mir_eval.beat.f_measure(r, e, f_measure_threshold=0.07)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.chdir(BT)
    import sys
    sys.path.insert(0, BT)
    from beat_this.dataset.mmnpz import MemmappedNpzFile
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

    npz = MemmappedNpzFile(Path("data/audio/spectrograms/sdedit.npz"))
    annd = Path("data/annotations/sdedit/annotations/beats")
    stems = sorted({k.split("/")[0] for k in npz.files})
    if a.limit:
        stems = stems[: a.limit]
    print(f"  {len(stems)} sdedit clips", flush=True)

    fh = open(a.out, "w")
    fh.write("stem,n_frames,bpm,F,F_doubled,F_halved,CMLt,AMLt,n_est,n_ref\n")
    with torch.no_grad():
        for i, stem in enumerate(stems):
            ref = np.loadtxt(annd / f"{stem}.beats", ndmin=2)[:, 0]
            if len(ref) < 3:
                continue
            spect = np.asarray(npz[f"{stem}/track"], dtype=np.float32)
            x = torch.from_numpy(spect)[None].to(dev)
            pad = torch.ones(1, x.shape[1], dtype=torch.bool, device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = m.model(x)
            pred = {k: v.float() for k, v in pred.items()}
            pb, _ = m.postprocessor(pred["beat"], pred["downbeat"], pad)
            est = np.asarray(pb if not isinstance(pb, tuple) else pb[0])
            dbl = np.sort(np.concatenate([ref, (ref[:-1] + ref[1:]) / 2]))
            try:
                c = mir_eval.beat.continuity(
                    mir_eval.beat.trim_beats(ref), mir_eval.beat.trim_beats(est))
                cml, aml = c[1], c[3]
            except Exception:
                cml = aml = 0.0
            fh.write(f"{stem},{spect.shape[0]},{60.0/np.median(np.diff(ref)):.3f},"
                     f"{f1(est,ref):.6f},{f1(est,dbl):.6f},{f1(est,ref[::2]):.6f},"
                     f"{cml:.6f},{aml:.6f},{len(est)},{len(ref)}\n")
            if i % 200 == 0:
                fh.flush()
                print(f"    {i}/{len(stems)}", flush=True)
    fh.close()
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
