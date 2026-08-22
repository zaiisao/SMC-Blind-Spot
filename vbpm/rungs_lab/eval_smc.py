"""SMC blind-spot analysis: R1 / R2+mixture / R3(wt,lamt) + R0-shipped context, DEPLOYED decode,
identical BT activations. Fold-honest: song fold f -> official BT fold_f frontend (assumption:
Beat This and Beat Transformer share the official 8-fold split -- the same 8-folds.split file).
Beat-only metrics: F, CMLt (continuity idx 1), AMLt (idx 3). Annotations grade only + define
the blind-spot strata (label-side)."""
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
r1 = DBN2016(fps=data.FPS, device=device, dtype=torch.float32, bounding="none",
             observation_lambda=6, transition_lambda=100.0, num_tempi=None,
             threshold=0.2, correct=False)
r0 = MadmomDBN(fps=data.FPS, bounding="none")           # madmom shipped decode: context column
r2m = MixtureLambda(fps=data.FPS, device=device, observation_lambda=6)
r2m.mixture_weight, r2m.transition_lambda = W0, LAM0
r3s = {}
for arm in ("wt", "lamt"):
    ck = torch.load(f"runs/r3_{arm}_seed0.pt")
    m = R3Mixture(arm=arm, w0=W0, fps=data.FPS, device=device,
                  observation_lambda=6, lambda_base=LAM0)
    m.net.load_state_dict(ck["net"]); r3s[arm] = m

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
        a = acts[e["stem"]]
        bt = e["beat_times"]
        ibi = np.diff(bt)
        bpm = 60.0 / np.median(ibi) if len(ibi) else np.nan
        # largest within-song tempo change: local BPM over 8-beat windows
        if len(ibi) >= 8:
            local = 60.0 / np.array([ibi[i:i+8].mean() for i in range(len(ibi)-7)])
            max_change = float(np.max(local) - np.min(local))
        else:
            max_change = np.nan
        volat = float(np.std(np.diff(ibi)/ibi[:-1])) if len(ibi) > 2 else np.nan
        row = {"stem": e["stem"], "fold": fold, "bpm": float(bpm),
               "max_tempo_change": max_change, "vol": volat}
        row["R0_shipped"] = score(bt, r0.predict(a)["beats"])
        row["R1"] = score(bt, r1.predict(a)["beats"])
        row["R2mix"] = score(bt, r2m.decode(a, deploy=True)["beats"])
        row["R3_wt"] = score(bt, r3s["wt"].decode(a, deploy=True)["beats"])
        row["R3_lamt"] = score(bt, r3s["lamt"].decode(a, deploy=True)["beats"])
        rows.append(row)
print(f"{len(rows)} SMC songs scored")

MODELS = ["R0_shipped", "R1", "R2mix", "R3_wt", "R3_lamt"]
def table(sub, label):
    print(f"[{label}] n={len(sub)}")
    out = {"n": len(sub)}
    for m in MODELS:
        v = np.mean(np.array([r[m] for r in sub]), axis=0)
        out[m] = dict(zip(["beatF","CMLt","AMLt"], map(float, v)))
        print(f"  {m:10s} F {v[0]:.4f} CMLt {v[1]:.4f} AMLt {v[2]:.4f}")
    return out

bpms = np.array([r["bpm"] for r in rows])
lo, hi = np.percentile(bpms, [15, 85])
mtc = np.array([r["max_tempo_change"] for r in rows])
mtc_hi = np.nanpercentile(mtc, 66.7)
res = {"overall": table(rows, "overall"),
       "bpm_extreme(<=p15 or >=p85)": table([r for r in rows if r["bpm"] <= lo or r["bpm"] >= hi],
                                             "bpm extreme"),
       "bpm_mid": table([r for r in rows if lo < r["bpm"] < hi], "bpm mid"),
       "tempo_change_high(top third)": table([r for r in rows if r["max_tempo_change"] >= mtc_hi],
                                             "tempo change high"),
       "tempo_change_low": table([r for r in rows if r["max_tempo_change"] < mtc_hi],
                                 "tempo change low")}
# correlations: per-song delta(R3-R1) CMLt vs blind-spot measures
d_cmlt_wt = np.array([r["R3_wt"][1] - r["R1"][1] for r in rows])
d_cmlt_lamt = np.array([r["R3_lamt"][1] - r["R1"][1] for r in rows])
d_cmlt_r2 = np.array([r["R2mix"][1] - r["R1"][1] for r in rows])
bpm_dist = np.abs(bpms - np.median(bpms))
corr = {}
for name, dv in (("R3wt-R1", d_cmlt_wt), ("R3lamt-R1", d_cmlt_lamt), ("R2mix-R1", d_cmlt_r2)):
    for xname, xv in (("max_tempo_change", mtc), ("vol", np.array([r["vol"] for r in rows])),
                      ("bpm_dist_from_median", bpm_dist)):
        ok = ~np.isnan(xv)
        s = spearmanr(xv[ok], dv[ok])
        corr[f"{name}_vs_{xname}"] = [float(s.statistic), float(s.pvalue)]
        print(f"spearman dCMLt({name}) vs {xname}: rho={s.statistic:.3f} p={s.pvalue:.3g}")
json.dump({"strata": res, "correlations": corr, "per_song": rows,
           "bpm_cuts": [float(lo), float(hi)], "mtc_cut": float(mtc_hi)},
          open("results_smc.json","w"), indent=1)
