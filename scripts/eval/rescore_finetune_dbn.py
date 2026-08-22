"""Re-score the fine-tuned fold-0 checkpoints with DBN decoding instead of peak-picking.

WHY. The fine-tune pilot was scored with peakpick(0.5), a threshold on raw activation amplitude.
Beat This's own ablation reports that the positive class weight is "crucial when NOT using a DBN"
(92.6 -> 79.5 F without it, peak-picked), the implication being that structured decoding is far
less sensitive to the overall SCALE of the activations -- it cares about where the peaks are, not
whether they clear 0.5. Fine-tuning perturbs activation calibration first and peak LOCATION second,
so a loss that is entirely calibration will vanish under the DBN and a loss that is genuine
localisation damage will survive it. That is the discriminating measurement.

ARMS (all fold 0, all fine-tuned from the released fold-0 checkpoint with correct pos_weight):
    base        released checkpoint, no fine-tuning
    nostretch   fine-tuned with NO stretching        -> isolates the fine-tuning procedure
    narrow      fine-tuned at 0.85-1.2 (= shipped)   -> procedure + mel-zoom artefact
    wide        fine-tuned at 0.7-1.8                -> procedure + zoom + the range under test

Activations are computed ONCE per checkpoint and decoded both ways, so peak-pick and DBN rows differ
only in the decoder. Evaluation is on the 27 SMC tracks held out of fold 0.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third-party" / "beat_this"))
from beat_this.inference import Audio2Frames
from beat_this.preprocessing import load_audio
from _harness import FPS, track_ids, load_gt, ev, peakpick, make_dbn, dbn_beats

SMC_AUD = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
STORE = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
FOLD = 0
ARMS = {"nostretch": "/tmp/overnight/ft_nostretch/model_fold0.pt",
        "narrow": "/tmp/overnight/ft_narrow/model_fold0.pt",
        "wide": "/tmp/overnight/ft_wide2/model_fold0.pt"}
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")


def main():
    dev = "cuda:1"
    split = {l.split("\t")[0]: int(l.split("\t")[1])
             for l in open(STORE / "smc" / "8-folds.split") if l.strip()}
    ids = [t for t in track_ids() if split.get(t, -1) == FOLD]
    ids = [t for t in ids if (SMC_AUD / f"{t.upper()}.wav").exists()]
    print(f"fold {FOLD}: {len(ids)} SMC tracks", flush=True)

    a2f = Audio2Frames(checkpoint_path=f"fold{FOLD}", device=dev)
    states = {"base": {k: v.clone() for k, v in a2f.model.state_dict().items()}}
    for nm, p in ARMS.items():
        states[nm] = torch.load(p, map_location=dev)

    audio = {t: load_audio(str(SMC_AUD / f"{t.upper()}.wav")) for t in ids}
    dbn = make_dbn(min_bpm=55.0, transition_lambda=100)
    res = {nm: {"peak": {}, "dbn": {}} for nm in states}

    for nm, st in states.items():
        a2f.model.load_state_dict(st); a2f.model.eval()
        for t in ids:
            sig, sr = audio[t]
            b, d = a2f(sig, sr)
            bp = torch.sigmoid(b.float()).cpu().numpy()
            dp = torch.sigmoid(d.float()).cpu().numpy()
            gt = load_gt(t)
            res[nm]["peak"][t] = ev(gt, peakpick(bp, 0.5, FPS))
            res[nm]["dbn"][t] = ev(gt, dbn_beats(bp, dp, dbn))
        print(f"  {nm} done", flush=True)

    A = lambda r: {m: float(np.mean([r[t][m] for t in ids])) for m in METRICS}
    print()
    for dec in ("peak", "dbn"):
        print(f"=== {dec} decoding (n={len(ids)}) ===")
        base = A(res["base"][dec])
        print(f"  base       " + "  ".join(f"{m} {base[m]:.4f}" for m in METRICS))
        for nm in ARMS:
            a = A(res[nm][dec])
            d = np.array([res[nm][dec][t]["F"] - res["base"][dec][t]["F"] for t in ids])
            print(f"  {nm:<10} " + "  ".join(f"{m} {a[m]:.4f}" for m in METRICS)
                  + f"   dF {d.mean():+.4f} w/l {int((d>0).sum())}/{int((d<0).sum())}")
        print()

    json.dump({"n": len(ids), "ids": ids,
               "summary": {dec: {nm: A(res[nm][dec]) for nm in res} for dec in ("peak", "dbn")},
               "per_track": res},
              open("/tmp/overnight/rescore_dbn.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
