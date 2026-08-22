"""R0/R1/R2 comparison on val fold 0, same frozen BT fold_0 activations (demix cache).
All rungs consume identical [T,2] sigmoid activations; annotations grade only.
BARE decode for all (num_tempi=None, threshold=0, correct=False, obs_lambda=6 chassis)
plus BT-shipped deploy (threshold=0.2) rows. Metrics: F, CMLt(idx1), AMLt(idx3)."""
import sys, json, time
import numpy as np, torch
import mir_eval.beat as meb
from training import frontend, harness, data
from rungs.r1_2016_dbn import DBN2016
from rungs.r0_madmom_dbn import MadmomDBN
from rungs.r2_em_dbn import R2GenerativeLambda

device = "cuda:0"
ck = torch.load("runs/r2_seed0/r2_em_best.pt", map_location="cpu")
lam = ck["rung"]["transition_lambda"]; obs = ck["rung"]["observation_lambda"]
print("R2 learned:", ck["rung"], "best:", ck["best"])

chassis = DBN2016(fps=data.FPS, device=device, dtype=torch.float32, observation_lambda=6,
                  num_tempi=None, threshold=0.0, correct=False)
train_e, val_e, _ = data.load_songs(chassis.annotated_state_path)
model = frontend.load_frozen_model("checkpoints/bt_fold0_repacked.pt", device)
acts = frontend.activations_for(model, val_e, device)

def mk(name, **kw):
    base = dict(fps=data.FPS, device=device, dtype=torch.float32, bounding="none",
                observation_lambda=6, num_tempi=None)
    base.update(kw); return DBN2016(**base)

rungs = {
 "R1_bare": mk("r1", transition_lambda=100, threshold=0.0, correct=False),
 "R1_deploy": mk("r1", transition_lambda=100, threshold=0.2, correct=False),
 "R2_bare": mk("r2", transition_lambda=lam, observation_lambda=obs, threshold=0.0, correct=False),
 "R2_deploy": mk("r2", transition_lambda=lam, observation_lambda=obs, threshold=0.2, correct=False),
}
r0 = MadmomDBN(fps=data.FPS, bounding="none", observation_lambda=6,
               num_tempi=None, threshold=0.0, correct=False)

def score(ref, est):
    ref, est = meb.trim_beats(ref), meb.trim_beats(est)
    f = meb.f_measure(ref, est) if len(est) and len(ref) else 0.0
    if len(est) and len(ref) > 1:
        _, cmlt, _, amlt = meb.continuity(ref, est)
    else: cmlt = amlt = 0.0
    return f, cmlt, amlt

acc = {n: [] for n in ["R0_bare"] + list(rungs)}
t0 = time.time()
for e in val_e:
    a = acts[e["stem"]]
    for n in acc:
        ev = r0.predict(a) if n == "R0_bare" else rungs[n].predict(a)
        bf, bc, ba = score(e["beat_times"], ev["beats"])
        df, dc, da = score(e["downbeat_times"], ev["downbeats"])
        acc[n].append((bf, bc, ba, df, dc, da))
print(f"decode {time.time()-t0:.0f}s over {len(val_e)} songs")
names = ["beatF","CMLt","AMLt","downbeatF","dbCMLt","dbAMLt"]
out = {}
for n, v in acc.items():
    m = np.mean(np.array(v), axis=0)
    out[n] = dict(zip(names, map(float, m)))
    print(f"{n:10s} " + " ".join(f"{x:.4f}" for x in m))
json.dump({"n_val": len(val_e), "r2_state": ck["rung"], "table": out},
          open("results_r2_val.json","w"), indent=1)
