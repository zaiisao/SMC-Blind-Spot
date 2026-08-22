"""R3 revival, UNSUPERVISED (marginal likelihood), mixture kernel included.
Two arms (era lesson, r3_generative_mixture_v2.py docstring):
  wt   : per-frame DITHER WEIGHT w_t = sigmoid(logit(w0)+net(x)_t), lambda pinned (era v2, on-program)
  lamt : per-frame lambda_t = lambda0*exp(net(x)_t), w pinned (era v1 -- leaked; negative control)
Net: r3_conditioned_dbn conv stack, zero-init -> starts at the global mixture optimum.
Adam on exact meter-marginal NLL, frozen BT fold_0 frontend, fold-0 protocol. Annotations unused."""
import sys, json, time
import numpy as np, torch
import mir_eval.beat as meb
from training import frontend, data
from rungs.r1_2016_dbn import DBN2016
from rungs.r3_conditioned_dbn import R3ConditionedFactors
from rungs.bar_pointer.readout import state_path_to_events

ARM = sys.argv[1]           # wt | lamt
mix = torch.load("runs/r2mix_seed0.pt")
W0, LAM0 = float(mix["w"]), float(mix["lambda"])
device = "cuda:0"; STEPS, BATCH = 200, 16
torch.manual_seed(0); rng = np.random.default_rng(0)


from r3_model import R3Mixture

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

r3 = R3Mixture(arm=ARM, w0=W0, fps=data.FPS, device=device, observation_lambda=6, lambda_base=LAM0)
opt = torch.optim.Adam(r3.net.parameters(), lr=1e-3)
t0 = time.time()
for step in range(STEPS):
    idx = rng.choice(len(crops), BATCH, replace=False)
    loss = sum(-r3.marginal_ll(crops[i]) / crops[i].shape[0] for i in idx) / BATCH
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(r3.net.parameters(), 1.0)
    opt.step()
    if step % 25 == 0 or step == STEPS-1:
        print(f"step {step:3d} nll/frame {loss.item():.4f}", flush=True)
print(f"train wall {time.time()-t0:.0f}s", flush=True)

def score(ref, est):
    ref, est = meb.trim_beats(ref), meb.trim_beats(est)
    f = meb.f_measure(ref, est) if len(est) and len(ref) else 0.0
    if len(est) and len(ref) > 1:
        _, c, _, a = meb.continuity(ref, est)
    else: c = a = 0.0
    return f, c, a

out = {"arm": ARM, "w0": W0, "lambda0": LAM0}
for mode, key in ((False,"bare"), (True,"deploy")):
    acc = [score(e["beat_times"], (ev:=r3.decode(val_acts[e["stem"]], deploy=mode))["beats"])
           + score(e["downbeat_times"], ev["downbeats"]) for e in val_e]
    out[key] = dict(zip(["beatF","CMLt","AMLt","downbeatF","dbCMLt","dbAMLt"],
                        map(float, np.mean(np.array(acc), axis=0))))
    print(key, out[key], flush=True)
json.dump(out, open(f"results_r3_{ARM}_val.json","w"), indent=1)
torch.save({"net": r3.net.state_dict(), "arm": ARM, "w0": W0, "lambda0": LAM0},
           f"runs/r3_{ARM}_seed0.pt")
