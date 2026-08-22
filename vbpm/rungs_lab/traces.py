"""Qualitative traces: per-segment (10 equal chunks) mean w_t and lambda_t for the 3 most
volatile and 2 steadiest val-fold-0 songs, beside annotated per-segment tempo volatility."""
import json
import numpy as np, torch
from training import frontend, data
from rungs.r1_2016_dbn import DBN2016
from r3_model import R3Mixture

device = "cuda:0"
mix = torch.load("runs/r2mix_seed0.pt"); W0, LAM0 = float(mix["w"]), float(mix["lambda"])
chassis = DBN2016(fps=data.FPS, device=device, dtype=torch.float32, observation_lambda=6,
                  num_tempi=None, threshold=0.0, correct=False)
_, val_e, _ = data.load_songs(chassis.annotated_state_path)
model = frontend.load_frozen_model("checkpoints/bt_fold0_repacked.pt", device)
acts = frontend.activations_for(model, val_e, device)

models = {}
for arm in ("wt", "lamt"):
    ck = torch.load(f"runs/r3_{arm}_seed0.pt")
    r3 = R3Mixture(arm=arm, w0=W0, fps=data.FPS, device=device,
                   observation_lambda=6, lambda_base=LAM0)
    r3.net.load_state_dict(ck["net"]); models[arm] = r3

def vol(bt):
    ibi = np.diff(bt)
    return float(np.std(np.diff(ibi)/ibi[:-1])) if len(ibi) > 2 else np.nan

ranked = sorted(val_e, key=lambda e: -vol(e["beat_times"]))
picks = ranked[:3] + ranked[-2:]
out = []
for e in picks:
    a = torch.from_numpy(np.ascontiguousarray(acts[e["stem"]].astype(np.float32))).to(device)
    T = a.shape[0]; edges = np.linspace(0, T, 11).astype(int)
    with torch.no_grad():
        w_t = models["wt"].per_frame_w(a).cpu().numpy()
        lam_t = models["lamt"].per_frame_lambda(a).cpu().numpy()
    bt = e["beat_times"]
    seg_vol = []
    for i in range(10):
        t0, t1 = edges[i]/data.FPS, edges[i+1]/data.FPS
        seg = bt[(bt >= t0) & (bt < t1)]
        seg_vol.append(vol(seg) if len(seg) > 3 else None)
    rec = {"stem": e["stem"], "song_vol": vol(bt),
           "seg_vol": seg_vol,
           "seg_w": [float(w_t[edges[i]:edges[i+1]].mean()) for i in range(10)],
           "seg_lam": [float(lam_t[edges[i]:edges[i+1]].mean()) for i in range(10)]}
    out.append(rec)
    print(rec["stem"], f"song_vol={rec['song_vol']:.4f}")
    print("  seg_vol", [f"{v:.3f}" if v else "  -  " for v in seg_vol])
    print("  seg_w  ", [f"{v:.3f}" for v in rec["seg_w"]])
    print("  seg_lam", [f"{v:.1f}" for v in rec["seg_lam"]])
json.dump(out, open("results_traces.json","w"), indent=1)
