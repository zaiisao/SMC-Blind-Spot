"""R2 + mixture kernel: joint (w, lambda) EM on the same 300 crops/fold-0 protocol as run_r2.
Question (archive 2026-07-20): does lambda_main recover toward ~100 once the dither component
absorbs grid flips, and does deployed F hold vs plain R2's 0.9132?"""
import json, time
import numpy as np, torch
import mir_eval.beat as meb
from training import frontend, data
from rungs.r1_2016_dbn import DBN2016
from mixture import MixtureLambda

device = "cuda:0"; SEED = 0
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
chassis = DBN2016(fps=data.FPS, device=device, dtype=torch.float32, observation_lambda=6,
                  num_tempi=None, threshold=0.0, correct=False)
train_e, val_e, _ = data.load_songs(chassis.annotated_state_path)
model = frontend.load_frozen_model("checkpoints/bt_fold0_repacked.pt", device)
train_acts = frontend.activations_for(model, train_e[:300], device)
val_acts = frontend.activations_for(model, val_e, device)
crops = []
for e in train_e[:300]:
    a = train_acts[e["stem"]]
    if a.shape[0] > 701:
        s = int(rng.integers(0, a.shape[0] - 700)); a = a[s:s+700]
    crops.append(torch.from_numpy(np.ascontiguousarray(a)).float().to(device))

m = MixtureLambda(fps=data.FPS, device=device, observation_lambda=6)
t0 = time.time()
hist = []
for it in range(8):
    w, lam = m.em_step_mixture(crops)
    hist.append((w, lam))
    print(f"EM iter {it}: w={w:.3f} lambda={lam:.2f}", flush=True)
print(f"EM wall {time.time()-t0:.0f}s")

def score(ref, est):
    ref, est = meb.trim_beats(ref), meb.trim_beats(est)
    f = meb.f_measure(ref, est) if len(est) and len(ref) else 0.0
    if len(est) and len(ref) > 1:
        _, cmlt, _, amlt = meb.continuity(ref, est)
    else: cmlt = amlt = 0.0
    return f, cmlt, amlt

out = {"w": w, "lambda": lam, "history": hist, "seed": SEED}
for mode in (False, True):
    acc = []
    for e in val_e:
        ev = m.decode(val_acts[e["stem"]], deploy=mode)
        acc.append(score(e["beat_times"], ev["beats"]) + score(e["downbeat_times"], ev["downbeats"]))
    mean = np.mean(np.array(acc), axis=0)
    key = "deploy" if mode else "bare"
    out[key] = dict(zip(["beatF","CMLt","AMLt","downbeatF","dbCMLt","dbAMLt"], map(float, mean)))
    print(key, out[key], flush=True)
json.dump(out, open("results_r2mix_val.json","w"), indent=1)
torch.save({"w": w, "lambda": lam}, "runs/r2mix_seed0.pt")
