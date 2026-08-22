"""THE SMC LADDER inside the Analyze-SMC protocol (certified: peak-pick 0.6270 exact,
oracle-lambda 0.6525 vs their 0.653, engine parity 60/60 wide).
Base: Beat This cached activations (paper-faithful), fps 50, wide grid [30,215],
observation_lambda 16, squeeze bounding, shipped heuristics (num_tempi None,
threshold 0.05 crop + peak snap) uniformly on every structured row.
Rows: peak-pick 0.5 / fixed-100 / R2-mixture / R3 wt / R3 lamt / ML-selected per-song
lambda (grid 1..500, marginal likelihood, LABEL-FREE) / ORACLE per-song lambda (ceiling).
Annotations: grading + strata only; ML selection never sees them."""
import sys, json
import numpy as np, torch
import mir_eval
from scipy.stats import spearmanr
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts")
from _harness import track_ids, load_gt, bthis_beat_prob, bthis_downbeat_prob, peakpick, ev, FPS
from rungs.r1_2016_dbn import DBN2016
from mixture import MixtureLambda
from r3_model import R3Mixture

device = "cuda"
mixck = torch.load("runs/r2mix_seed0.pt"); W0, LAM0 = float(mixck["w"]), float(mixck["lambda"])
WIDE = dict(min_bpm=30.0, max_bpm=215.0, observation_lambda=16)
LAMBDAS = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]

r1 = DBN2016(fps=FPS, device=device, dtype=torch.float32, bounding="squeeze",
             transition_lambda=100.0, num_tempi=None, threshold=0.05, correct=True, **WIDE)
def squeeze_chassis(m):
    for c in m._chassis_cache.values() if hasattr(m, "_chassis_cache") else [m.chassis]:
        c.bounding = "squeeze"
    m.chassis.bounding = "squeeze"
    return m
r2m = squeeze_chassis(MixtureLambda(fps=FPS, device=device, **WIDE))
r2m.mixture_weight, r2m.transition_lambda = W0, LAM0
r3s = {}
for arm in ("wt", "lamt"):
    ck = torch.load(f"runs/r3_{arm}_seed0.pt")
    m = squeeze_chassis(R3Mixture(arm=arm, w0=W0, fps=FPS, device=device,
                                  lambda_base=LAM0, **WIDE))
    m.net.load_state_dict(ck["net"]); r3s[arm] = m
mlsel = squeeze_chassis(MixtureLambda(fps=FPS, device=device, **WIDE))
mlsel.mixture_weight = W0

rows = []
for t in track_ids():
    ref = load_gt(t)
    bp, dp = bthis_beat_prob(t), bthis_downbeat_prob(t)
    acts = np.vstack((bp, dp)).T                      # raw probs; squeeze happens inside
    ibi = np.diff(ref)
    bpm = 60.0/np.median(ibi) if len(ibi) else np.nan
    if len(ibi) >= 8:
        local = 60.0/np.array([ibi[i:i+8].mean() for i in range(len(ibi)-7)])
        mtc = float(local.max()-local.min())
    else: mtc = np.nan
    row = {"stem": t, "bpm": float(bpm), "max_tempo_change": mtc,
           "vol": float(np.std(np.diff(ibi)/ibi[:-1])) if len(ibi) > 2 else np.nan}
    row["peak"] = ev(ref, peakpick(bp, 0.5, FPS))
    row["fixed100"] = ev(ref, r1.predict(acts)["beats"])
    dec = dict(deploy=True, deploy_threshold=0.05, snap=True)
    row["R2mix"] = ev(ref, r2m.decode(acts, **dec)["beats"])
    row["R3_wt"] = ev(ref, r3s["wt"].decode(acts, **dec)["beats"])
    row["R3_lamt"] = ev(ref, r3s["lamt"].decode(acts, **dec)["beats"])
    # per-song lambda: forward LL on squeezed acts, then decode at selected lambda
    with torch.no_grad():
        sq = acts*(1-1e-5)+1e-5/2
        at = torch.from_numpy(np.ascontiguousarray(sq.astype(np.float32))).to(device)
        lls, evs = [], []
        for L in LAMBDAS:
            k = mlsel.log_mixture_kernel(W0, float(L))
            lls.append(float(mlsel.marginal_log_likelihood(at, k)))
            mlsel.transition_lambda = float(L)
            evs.append(ev(ref, mlsel.decode(acts, **dec)["beats"]))
    i_ml = int(np.argmax(lls))
    i_or = int(np.argmax([e["F"] for e in evs]))
    row["ML"] = evs[i_ml]; row["ORACLE"] = evs[i_or]
    row["lam_ml"] = LAMBDAS[i_ml]; row["lam_or"] = LAMBDAS[i_or]
    rows.append(row)
    if len(rows) % 50 == 0:
        print(len(rows), flush=True); json.dump(rows, open("results_smc3.json","w"))
json.dump(rows, open("results_smc3.json","w"))

NAMES = ["peak","fixed100","R2mix","R3_wt","R3_lamt","ML","ORACLE"]
def table(sub, label):
    print(f"[{label}] n={len(sub)}")
    for n in NAMES:
        f = np.mean([r[n]["F"] for r in sub]); c = np.mean([r[n]["CMLt"] for r in sub])
        a = np.mean([r[n]["AMLt"] for r in sub])
        print(f"  {n:9s} F {f:.4f} CMLt {c:.4f} AMLt {a:.4f}")
bpms = np.array([r["bpm"] for r in rows])
table(rows, "overall")
table([r for r in rows if r["bpm"] < 55], "bpm<55 blind spot")
table([r for r in rows if r["bpm"] >= 55], "bpm>=55")
mtc = np.array([r["max_tempo_change"] for r in rows]); cut = np.nanpercentile(mtc, 66.7)
table([r for r in rows if r["max_tempo_change"] >= cut], "tempo-change high")
table([r for r in rows if r["max_tempo_change"] < cut], "tempo-change low")
bpm_dist = np.abs(bpms - np.median(bpms))
for dn in ("R2mix","R3_wt","R3_lamt","ML"):
    d = np.array([r[dn]["CMLt"] - r["fixed100"]["CMLt"] for r in rows])
    for xn, xv in (("mtc", mtc), ("bpm_dist", bpm_dist)):
        ok = ~np.isnan(xv); s = spearmanr(xv[ok], d[ok])
        print(f"spearman dCMLt({dn}-fixed100) vs {xn}: rho={s.statistic:.3f} p={s.pvalue:.3g}")
lm = np.log([r["lam_ml"] for r in rows]); lo = np.log([r["lam_or"] for r in rows])
s = spearmanr(lm, lo)
print(f"spearman(log lam_ML, log lam_ORACLE) rho={s.statistic:.3f} p={s.pvalue:.3g}")
print("lam_ML quartiles:", np.percentile([r["lam_ml"] for r in rows], [10,25,50,75,90]))
print("lam_OR quartiles:", np.percentile([r["lam_or"] for r in rows], [10,25,50,75,90]))
