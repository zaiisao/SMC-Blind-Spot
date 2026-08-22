"""Stratified R1 / R2mix / R3(wt) / R3(lamt) comparison, fold-honest.
Sets: val fold 0 (BT fold_0 demix activations, fps 43.07) and gtzan (Beat This final0
activations, fps 50 -- cross-frontend TRANSFER of the learned scalars/net; caveat labeled).
Stratification (grades only): per-song tempo volatility v = std(diff(IBI)/IBI[:-1]) from
annotations; terciles within each eval set. Also: Spearman correlation between the net's
mean modulation (w_t or lambda_t) and v; per-segment traces for 2 high-volatility songs.
Metrics: beat F, CMLt (continuity idx 1), AMLt (idx 3), downbeat F. BARE decode (model comparison).
"""
import json, sys
import numpy as np, torch
import mir_eval.beat as meb
from scipy.stats import spearmanr
from training import frontend, data
from rungs.r1_2016_dbn import DBN2016
from mixture import MixtureLambda
from r3_model import R3Mixture

device = "cuda:0"
mix = torch.load("runs/r2mix_seed0.pt"); W0, LAM0 = float(mix["w"]), float(mix["lambda"])

def build_models(fps):
    r1 = DBN2016(fps=fps, device=device, dtype=torch.float32, bounding="none",
                 observation_lambda=6, transition_lambda=100.0,
                 num_tempi=None, threshold=0.0, correct=False)
    r2m = MixtureLambda(fps=fps, device=device, observation_lambda=6)
    r2m.mixture_weight, r2m.transition_lambda = W0, LAM0
    r3s = {}
    for arm in ("wt", "lamt"):
        ck = torch.load(f"runs/r3_{arm}_seed0.pt")
        r3 = R3Mixture(arm=arm, w0=W0, fps=fps, device=device,
                       observation_lambda=6, lambda_base=LAM0)
        r3.net.load_state_dict(ck["net"])
        r3s[f"R3_{arm}"] = r3
    return {"R1": r1, "R2mix": r2m, **r3s}

def volatility(beat_times):
    ibi = np.diff(beat_times)
    if len(ibi) < 3: return np.nan
    return float(np.std(np.diff(ibi) / ibi[:-1]))

def score(ref, est):
    ref, est = meb.trim_beats(ref), meb.trim_beats(est)
    f = meb.f_measure(ref, est) if len(est) and len(ref) else 0.0
    if len(est) and len(ref) > 1:
        _, c, _, a = meb.continuity(ref, est)
    else: c = a = 0.0
    return f, c, a

def evaluate_set(name, songs, get_acts, fps):
    models = build_models(fps)
    per_song = []
    for s in songs:
        acts = get_acts(s)
        bt, dbt = s["beat_times"], s["downbeat_times"]
        v = volatility(bt)
        row = {"stem": s["stem"], "vol": v}
        for mn, m in models.items():
            if mn == "R1":
                ev = m.predict(acts)
            else:
                ev = m.decode(acts, deploy=False)
            bf, bc, ba = score(bt, ev["beats"]); df, _, _ = score(dbt, ev["downbeats"])
            row[mn] = (bf, bc, ba, df)
        # net modulation summary (R3 wt & lamt)
        at = torch.from_numpy(np.ascontiguousarray(acts.astype(np.float32))).to(device)
        with torch.no_grad():
            row["mean_w"] = float(models["R3_wt"].per_frame_w(at).mean())
            row["mean_lam"] = float(models["R3_lamt"].per_frame_lambda(at).mean())
        per_song.append(row)
    vols = np.array([r["vol"] for r in per_song])
    ok = ~np.isnan(vols)
    terc = np.nanpercentile(vols[ok], [33.3, 66.7])
    strata = {"steady": lambda v: v <= terc[0],
              "mid": lambda v: terc[0] < v <= terc[1],
              "volatile": lambda v: v > terc[1]}
    out = {"set": name, "fps": fps, "terciles": terc.tolist(), "n": len(per_song), "table": {}}
    for sn, pred in [("overall", lambda v: True)] + list(strata.items()):
        rows = [r for r in per_song if not np.isnan(r["vol"]) and pred(r["vol"])]
        out["table"][sn] = {"n": len(rows)}
        for mn in ("R1", "R2mix", "R3_wt", "R3_lamt"):
            m = np.mean(np.array([r[mn] for r in rows]), axis=0)
            out["table"][sn][mn] = dict(zip(["beatF","CMLt","AMLt","downbeatF"], map(float, m)))
    rw = spearmanr(vols[ok], np.array([r["mean_w"] for r in per_song])[ok])
    rl = spearmanr(vols[ok], np.array([r["mean_lam"] for r in per_song])[ok])
    out["spearman_vol_vs_mean_w"] = [float(rw.statistic), float(rw.pvalue)]
    out["spearman_vol_vs_mean_lam"] = [float(rl.statistic), float(rl.pvalue)]
    out["per_song"] = per_song
    return out

results = {}

# --- val fold 0 (BT fold_0, native cache fps) ---
chassis = DBN2016(fps=data.FPS, device=device, dtype=torch.float32, observation_lambda=6,
                  num_tempi=None, threshold=0.0, correct=False)
_, val_e, _ = data.load_songs(chassis.annotated_state_path)
model = frontend.load_frozen_model("checkpoints/bt_fold0_repacked.pt", device)
val_acts = frontend.activations_for(model, val_e, device)
results["val_fold0"] = evaluate_set("val_fold0", val_e, lambda s: val_acts[s["stem"]], data.FPS)

# --- gtzan (Beat This final0, fps 50; transfer) ---
from data.songs import iter_songs
from tracker import build_frontend
import soundfile
fe = build_frontend("frontends.beat_this", output="activations", checkpoint="final0", device="cuda")
gt = []
for song in iter_songs(datasets=["gtzan"]):
    b, d = song.beats()
    gt.append({"stem": song.stem, "beat_times": b, "downbeat_times": d, "path": song.audio_path})
def gt_acts(s):
    sig, sr = soundfile.read(s["path"], dtype="float32")
    if sig.ndim > 1: sig = sig.mean(axis=1)
    a = fe.get_features(sig, sr)
    return 1.0/(1.0+np.exp(-np.asarray(a, dtype=np.float64)))
results["gtzan"] = evaluate_set("gtzan", gt, gt_acts, fe.fps)

json.dump(results, open("results_stratified.json","w"), indent=1)

for setname, res in results.items():
    print(f"\n== {setname} (n={res['n']}, terciles v={res['terciles']}) ==")
    for sn, tab in res["table"].items():
        print(f" [{sn}] n={tab['n']}")
        for mn in ("R1","R2mix","R3_wt","R3_lamt"):
            t = tab[mn]
            print(f"   {mn:8s} beatF {t['beatF']:.4f} CMLt {t['CMLt']:.4f} "
                  f"AMLt {t['AMLt']:.4f} downbeatF {t['downbeatF']:.4f}")
    print(" spearman(vol, mean_w)  ", res["spearman_vol_vs_mean_w"])
    print(" spearman(vol, mean_lam)", res["spearman_vol_vs_mean_lam"])
