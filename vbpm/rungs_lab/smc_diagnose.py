"""SMC divergence diagnosis.
1. BPM range audit: how many SMC songs fall outside the 55-215 grid?
2. Engine parity ON SMC: R0 (madmom shipped) vs R1 (MATCHED shipped config) event-identity.
3. Knob decomposition on the R1 engine: obs_lambda 16 vs 6, threshold, correct, num_tempi.
4. Wide grid: min_bpm 30 (covers SMC's slow tail) vs 55, shipped config.
Annotations grade only (and audit the BPM range -- label-side)."""
import json
import numpy as np, torch
import mir_eval.beat as meb
from training import frontend, data
from rungs.r1_2016_dbn import DBN2016
from rungs.r0_madmom_dbn import MadmomDBN
from smc_data import load_smc

device = "cuda:0"
entries = load_smc()
bpms = []
for e in entries:
    ibi = np.diff(e["beat_times"])
    if len(ibi): bpms.append(60.0/np.median(ibi))
bpms = np.array(bpms)
print(f"SMC BPM: min {bpms.min():.1f} p5 {np.percentile(bpms,5):.1f} median {np.median(bpms):.1f} "
      f"p95 {np.percentile(bpms,95):.1f} max {bpms.max():.1f}")
print(f"below 55 BPM: {(bpms<55).sum()}/{len(bpms)}  above 215: {(bpms>215).sum()}")

def score(ref, est):
    ref, est = meb.trim_beats(ref), meb.trim_beats(est)
    f = meb.f_measure(ref, est) if len(est) and len(ref) else 0.0
    if len(est) and len(ref) > 1:
        _, c, _, a = meb.continuity(ref, est)
    else: c = a = 0.0
    return f, c, a

r0 = MadmomDBN(fps=data.FPS, bounding="none")   # shipped: obs16, num_tempi 60, thr .05, correct
configs = {
 "R1_matched_shipped": dict(observation_lambda=16, num_tempi=60, threshold=0.05, correct=True),
 "R1_obs16_bare": dict(observation_lambda=16, num_tempi=None, threshold=0.0, correct=False),
 "R1_obs6_shipped": dict(observation_lambda=6, num_tempi=60, threshold=0.05, correct=True),
 "R1_obs6_thr02": dict(observation_lambda=6, num_tempi=None, threshold=0.2, correct=False),
 "R1_wide30_shipped": dict(observation_lambda=16, num_tempi=60, threshold=0.05, correct=True,
                           min_bpm=30.0),
 "R1_wide30_notempi": dict(observation_lambda=16, num_tempi=None, threshold=0.05, correct=True,
                           min_bpm=30.0),
}
models = {n: DBN2016(fps=data.FPS, device=device, dtype=torch.float32, bounding="none", **kw)
          for n, kw in configs.items()}

by_fold = {}
for e in entries: by_fold.setdefault(e["fold"], []).append(e)
acc = {n: [] for n in ["R0_shipped"] + list(models)}
identical = 0; total = 0
for fold, es in sorted(by_fold.items()):
    model = frontend.load_frozen_model(f"checkpoints/bt_fold{fold}_repacked.pt", device)
    acts = frontend.activations_for(model, es, device)
    for e in es:
        a = acts[e["stem"]]; bt = e["beat_times"]
        ev0 = r0.predict(a)
        acc["R0_shipped"].append(score(bt, ev0["beats"]))
        for n, m in models.items():
            ev = m.predict(a)
            acc[n].append(score(bt, ev["beats"]))
            if n == "R1_matched_shipped":
                total += 1
                if (len(ev["beats"]) == len(ev0["beats"])
                        and np.allclose(ev["beats"], ev0["beats"], atol=1e-6)):
                    identical += 1
print(f"\nSMC certificate (matched shipped config): {identical}/{total} event-identical to R0")
out = {"certificate": [identical, total], "bpm_below55": int((bpms<55).sum())}
for n, v in acc.items():
    m = np.mean(np.array(v), axis=0)
    out[n] = dict(zip(["beatF","CMLt","AMLt"], map(float, m)))
    print(f"{n:22s} F {m[0]:.4f} CMLt {m[1]:.4f} AMLt {m[2]:.4f}")
json.dump(out, open("results_smc_diagnosis.json","w"), indent=1)
