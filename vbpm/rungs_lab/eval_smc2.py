"""CORRECTED SMC ladder (post-diagnosis): obs_lambda=6 chassis for every learned/hand-set row,
deploy = threshold 0.05 crop + peak snap (the shipped heuristics that SMC needs; 0.2 crops
away quiet songs). Wide-grid (min_bpm=30) arms cover SMC's sub-55-BPM tail (45/217 songs).
R0_shipped (madmom obs16 shipped) = certified context. Beat-only. Same blind-spot analysis."""
import json
import numpy as np, torch
import mir_eval.beat as meb
from scipy.stats import spearmanr
from training import frontend, data
from rungs.r1_2016_dbn import DBN2016
from rungs.r0_madmom_dbn import MadmomDBN
from mixture import MixtureLambda
from r3_model import R3Mixture
from smc_data import load_smc

device = "cuda:0"
mix = torch.load("runs/r2mix_seed0.pt"); W0, LAM0 = float(mix["w"]), float(mix["lambda"])
DEC = dict(num_tempi=None, threshold=0.05, correct=True)

def r1(min_bpm=55.0):
    return DBN2016(fps=data.FPS, device=device, dtype=torch.float32, bounding="none",
                   observation_lambda=6, transition_lambda=100.0, min_bpm=min_bpm, **DEC)
def r2m(min_bpm=55.0):
    m = MixtureLambda(fps=data.FPS, device=device, observation_lambda=6, min_bpm=min_bpm)
    m.mixture_weight, m.transition_lambda = W0, LAM0
    return m
def r3(arm, min_bpm=55.0):
    ck = torch.load(f"runs/r3_{arm}_seed0.pt")
    m = R3Mixture(arm=arm, w0=W0, fps=data.FPS, device=device, observation_lambda=6,
                  lambda_base=LAM0, min_bpm=min_bpm)
    m.net.load_state_dict(ck["net"]); return m

MODELS = {"R0_shipped": MadmomDBN(fps=data.FPS, bounding="none"),
          "R1": r1(), "R1_wide30": r1(30.0),
          "R2mix": r2m(), "R2mix_wide30": r2m(30.0),
          "R3_wt": r3("wt"), "R3_lamt": r3("lamt"), "R3_lamt_wide30": r3("lamt", 30.0)}

def score(ref, est):
    ref, est = meb.trim_beats(ref), meb.trim_beats(est)
    f = meb.f_measure(ref, est) if len(est) and len(ref) else 0.0
    if len(est) and len(ref) > 1:
        _, c, _, a = meb.continuity(ref, est)
    else: c = a = 0.0
    return f, c, a

entries = load_smc()
by_fold = {}
for e in entries: by_fold.setdefault(e["fold"], []).append(e)
rows = []
for fold, es in sorted(by_fold.items()):
    model = frontend.load_frozen_model(f"checkpoints/bt_fold{fold}_repacked.pt", device)
    acts = frontend.activations_for(model, es, device)
    for e in es:
        a = acts[e["stem"]]; bt = e["beat_times"]
        ibi = np.diff(bt)
        bpm = 60.0/np.median(ibi) if len(ibi) else np.nan
        if len(ibi) >= 8:
            local = 60.0/np.array([ibi[i:i+8].mean() for i in range(len(ibi)-7)])
            mtc = float(local.max()-local.min())
        else: mtc = np.nan
        row = {"stem": e["stem"], "bpm": float(bpm), "max_tempo_change": mtc,
               "vol": float(np.std(np.diff(ibi)/ibi[:-1])) if len(ibi) > 2 else np.nan}
        for n, m in MODELS.items():
            if n == "R0_shipped": ev = m.predict(a)
            elif n.startswith("R1"): ev = m.predict(a)
            else: ev = m.decode(a, deploy=True, deploy_threshold=0.05, snap=True)
            row[n] = score(bt, ev["beats"])
        rows.append(row)
print(f"{len(rows)} songs")
NAMES = list(MODELS)
def table(sub, label):
    print(f"[{label}] n={len(sub)}")
    out = {"n": len(sub)}
    for n in NAMES:
        v = np.mean(np.array([r[n] for r in sub]), axis=0)
        out[n] = dict(zip(["beatF","CMLt","AMLt"], map(float, v)))
        print(f"  {n:16s} F {v[0]:.4f} CMLt {v[1]:.4f} AMLt {v[2]:.4f}")
    return out
bpms = np.array([r["bpm"] for r in rows]); lo, hi = np.percentile(bpms,[15,85])
mtc = np.array([r["max_tempo_change"] for r in rows]); cut = np.nanpercentile(mtc, 66.7)
res = {"overall": table(rows,"overall"),
       "bpm_extreme": table([r for r in rows if r["bpm"]<=lo or r["bpm"]>=hi],"bpm extreme"),
       "bpm_below55": table([r for r in rows if r["bpm"]<55],"bpm<55 (grid blind spot)"),
       "bpm_mid": table([r for r in rows if lo<r["bpm"]<hi],"bpm mid"),
       "tc_high": table([r for r in rows if r["max_tempo_change"]>=cut],"tempo change high"),
       "tc_low": table([r for r in rows if r["max_tempo_change"]<cut],"tempo change low")}
corr = {}
bpm_dist = np.abs(bpms-np.median(bpms))
for dn, m2 in (("R2mix","R2mix"),("R3_wt","R3_wt"),("R3_lamt","R3_lamt"),
               ("R1_wide30","R1_wide30")):
    d = np.array([r[m2][1]-r["R1"][1] for r in rows])
    for xn, xv in (("mtc",mtc),("vol",np.array([r["vol"] for r in rows])),("bpm_dist",bpm_dist)):
        ok = ~np.isnan(xv)
        s = spearmanr(xv[ok], d[ok])
        corr[f"dCMLt({dn}-R1)_vs_{xn}"] = [float(s.statistic), float(s.pvalue)]
        print(f"spearman dCMLt({dn}-R1) vs {xn}: rho={s.statistic:.3f} p={s.pvalue:.3g}")
json.dump({"strata": res, "correlations": corr, "per_song": rows},
          open("results_smc2.json","w"), indent=1)
