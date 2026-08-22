"""Final preliminary table: both trained arms x both lambdas on the FULL fold-0 val set,
plus r2 bare-threshold check and the pretrained fold_0 reference (leaky on ballroom/hainsworth --
their folds are not ours -- so it is an upper reference, not a comparable arm)."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "beat_transformer" / "code"))

import mir_eval                                          # noqa: E402
from rungs.r1_2016_dbn import DBN2016                    # noqa: E402
from crf_baseline import CRFLearnedFactors        # noqa: E402
from DilatedTransformer import Demixed_DilatedTransformerModel  # noqa: E402
from train_bt import FPS, MODEL_KWARGS, BT_SHIPPED_DECODE, load_songs  # noqa: E402

DEVICE = "cuda:1"
OUT = Path(__file__).resolve().parent


def load_model(path):
    model = Demixed_DilatedTransformerModel(**MODEL_KWARGS).to(DEVICE).eval()
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return model


@torch.no_grad()
def activations_for(model, entries):
    acts = {}
    for e in entries:
        x = np.load(e["mel_path"])["x"]
        pred, _ = model(torch.from_numpy(x).unsqueeze(0).to(DEVICE))
        acts[e["stem"]] = torch.sigmoid(pred[0, :, :2]).double().cpu().numpy()
    return acts


def score(entries, acts, **rung_kwargs):
    rung = DBN2016(fps=FPS, device=DEVICE, dtype=torch.float32, bounding="none", **rung_kwargs)
    beat_fs, downbeat_fs = [], []
    for e in entries:
        events = rung.predict(acts[e["stem"]])
        ref_b = mir_eval.beat.trim_beats(e["beat_times"])
        est_b = mir_eval.beat.trim_beats(events["beats"])
        beat_fs.append(mir_eval.beat.f_measure(ref_b, est_b) if len(est_b) else 0.0)
        ref_d = mir_eval.beat.trim_beats(e["downbeat_times"])
        est_d = mir_eval.beat.trim_beats(events["downbeats"])
        downbeat_fs.append(mir_eval.beat.f_measure(ref_d, est_d) if len(est_d) else 0.0)
    return float(np.mean(beat_fs)), float(np.mean(downbeat_fs))


def main():
    r2_probe = CRFLearnedFactors(fps=FPS, device=DEVICE)
    _, val_entries, _ = load_songs(r2_probe)
    print(f"full val fold: {len(val_entries)} songs", flush=True)

    lam = float(torch.load(OUT / "r2_best.pt", map_location="cpu")["r2"]
                ["log_transition_lambda"].exp())
    print(f"learned lambda = {lam:.2f}", flush=True)

    models = {"vanilla": load_model(OUT / "vanilla_best.pt"),
              "r2": load_model(OUT / "r2_best.pt")}
    pre = ROOT / "external" / "beat_transformer" / "checkpoint" / "fold_0_trf_param.pt"
    pre_model = Demixed_DilatedTransformerModel(**MODEL_KWARGS).to(DEVICE).eval()
    pre_model.load_state_dict(torch.load(pre, map_location="cpu")["state_dict"])
    models["pretrained_fold0 (LEAKY ref)"] = pre_model

    for name, model in models.items():
        acts = activations_for(model, val_entries)
        for lam_name, lam_value in (("lam=100", 100.0), (f"lam={lam:.1f}", lam)):
            b, d = score(val_entries, acts, transition_lambda=lam_value, **BT_SHIPPED_DECODE)
            print(f"{name:28s} {lam_name:10s} shipped-decode : beatF {b:.4f}  dbF {d:.4f}",
                  flush=True)
        if name == "r2":   # pre-registered calibration check: bare decode (no threshold crop)
            b, d = score(val_entries, acts, transition_lambda=lam,
                         observation_lambda=6, num_tempi=None, threshold=0.0, correct=True)
            print(f"{name:28s} lam={lam:.1f}  threshold=0    : beatF {b:.4f}  dbF {d:.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
