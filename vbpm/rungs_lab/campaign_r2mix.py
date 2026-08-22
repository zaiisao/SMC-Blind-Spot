"""8-fold x 3-seed R2+mixture campaign. Fold f: train crops from folds != f, frozen OFFICIAL
BT fold_f frontend (fold-honest), eval deployed+bare on fold f. Sequential on one GPU."""
import json, sys, time
import numpy as np, torch
import mir_eval.beat as meb
from training import frontend, data
from rungs.r1_2016_dbn import DBN2016
from mixture import MixtureLambda
from smc_data import load_smc

device = "cuda:0"
smc_by_fold = {}
for e in load_smc():
    smc_by_fold.setdefault(e["fold"], []).append(e)
def score(ref, est):
    ref, est = meb.trim_beats(ref), meb.trim_beats(est)
    f = meb.f_measure(ref, est) if len(est) and len(ref) else 0.0
    if len(est) and len(ref) > 1:
        _, c, _, a = meb.continuity(ref, est)
    else: c = a = 0.0
    return f, c, a

chassis = DBN2016(fps=data.FPS, device=device, dtype=torch.float32, observation_lambda=6,
                  num_tempi=None, threshold=0.0, correct=False)
# load_songs splits by EVAL_FOLD; do our own split over all entries
data.EVAL_FOLD = -1                      # everything into train list
all_e, _, _ = data.load_songs(chassis.annotated_state_path)
out_path = "results_campaign_r2mix.json"
import os
results = json.load(open(out_path)) if os.path.exists(out_path) else []
done = {(r["fold"], r["seed"]) for r in results}
for fold in range(8):
    train_e = [e for e in all_e if e["fold"] != fold]
    val_e = [e for e in all_e if e["fold"] == fold]
    model = frontend.load_frozen_model(f"checkpoints/bt_fold{fold}_repacked.pt", device)
    val_acts = frontend.activations_for(model, val_e, device)
    for seed in range(3):
        if (fold, seed) in done:
            continue
        t0 = time.time()
        rng = np.random.default_rng(seed); torch.manual_seed(seed)
        pick = rng.permutation(len(train_e))[:300]
        entries = [train_e[i] for i in pick]
        acts = frontend.activations_for(model, entries, device)
        crops = []
        for e in entries:
            a = acts[e["stem"]]
            if a.shape[0] > 701:
                s = int(rng.integers(0, a.shape[0] - 700)); a = a[s:s+700]
            crops.append(torch.from_numpy(np.ascontiguousarray(a)).float().to(device))
        m = MixtureLambda(fps=data.FPS, device=device, observation_lambda=6)
        for it in range(8):
            w, lam = m.em_step_mixture(crops)
        row = {"fold": fold, "seed": seed, "w": w, "lambda": lam, "n_val": len(val_e)}
        for mode, key in ((False,"bare"), (True,"deploy")):
            acc = [score(e["beat_times"], (ev:=m.decode(val_acts[e["stem"]], deploy=mode))["beats"])
                   + score(e["downbeat_times"], ev["downbeats"]) for e in val_e]
            row[key] = dict(zip(["beatF","CMLt","AMLt","downbeatF","dbCMLt","dbAMLt"],
                                map(float, np.mean(np.array(acc), axis=0))))
        # SMC (beat-only), same fold-f frontend
        smc_e = smc_by_fold.get(fold, [])
        if smc_e:
            smc_acts = frontend.activations_for(model, smc_e, device)
            acc = [score(e["beat_times"], m.decode(smc_acts[e["stem"]], deploy=True)["beats"])
                   for e in smc_e]
            row["smc_deploy"] = dict(zip(["beatF","CMLt","AMLt"],
                                         map(float, np.mean(np.array(acc), axis=0))))
        row["wall_s"] = round(time.time()-t0)
        results.append(row)
        json.dump(results, open(out_path, "w"), indent=1)
        print(f"fold{fold} seed{seed}: w={w:.3f} lam={lam:.1f} "
              f"deployF={row['deploy']['beatF']:.4f} ({row['wall_s']}s)", flush=True)
print("CAMPAIGN DONE")
