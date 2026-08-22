"""Fill the R0 row: actual madmom (MadmomDBN, shipped-BT decode) scored on all six metrics."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import final_eval; final_eval.DEVICE = "cuda:1"
import mir_eval
from train_bt import FPS, BT_SHIPPED_DECODE, load_songs
from rungs.r0_madmom_dbn import MadmomDBN
from crf_baseline import CRFLearnedFactors

probe = CRFLearnedFactors(fps=FPS, device="cuda:1",
                          observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
_, val, _ = load_songs(probe)
model = final_eval.load_model(Path(__file__).resolve().parent / "vanilla_best_prelim.pt")
val_acts = final_eval.activations_for(model, val)

r0 = MadmomDBN(fps=FPS, bounding="none", transition_lambda=100, **BT_SHIPPED_DECODE)
bf, cml, aml, df, dcml, daml = [], [], [], [], [], []
for e in val:
    ev = r0.predict(val_acts[e["stem"]])
    ref = mir_eval.beat.trim_beats(e["beat_times"])
    est = mir_eval.beat.trim_beats(ev["beats"])
    if len(est):
        bf.append(mir_eval.beat.f_measure(ref, est))
        _, c, _, a = mir_eval.beat.continuity(ref, est)
        cml.append(c); aml.append(a)
    else:
        bf.append(0.0); cml.append(0.0); aml.append(0.0)
    ref_d = mir_eval.beat.trim_beats(e["downbeat_times"])
    est_d = mir_eval.beat.trim_beats(ev["downbeats"])
    if len(est_d) and len(ref_d) > 1:
        df.append(mir_eval.beat.f_measure(ref_d, est_d))
        _, c, _, a = mir_eval.beat.continuity(ref_d, est_d)
        dcml.append(c); daml.append(a)
    else:
        df.append(0.0); dcml.append(0.0); daml.append(0.0)
print(f"R0 madmom shipped: beatF {np.mean(bf):.4f}  CMLt {np.mean(cml):.4f}  "
      f"AMLt {np.mean(aml):.4f}  dbF {np.mean(df):.4f}  dbCMLt {np.mean(dcml):.4f}  "
      f"dbAMLt {np.mean(daml):.4f}", flush=True)
