"""Per-song lambda by MARGINAL LIKELIHOOD (label-free) vs ORACLE (F-optimal; grading ceiling).
Mixture chassis (w fixed at learned 0.390), lambda_main on a log grid. For each song:
  lambda*_ML = argmax_lambda log p(activations | lambda)      [exact meter-marginal forward]
  lambda*_OR = argmax_lambda beat F(decode(lambda))           [ORACLE -- annotations, ceiling only]
Rows: fixed 100 / fixed 104.4(learned) / peak-pick (no structure) / ML-selected / ORACLE.
Usage: lambda_select.py smc | lambda_select.py gtzan"""
import sys, json
import numpy as np, torch
import mir_eval.beat as meb
from scipy.signal import find_peaks
from training import frontend, data
from mixture import MixtureLambda

DS = sys.argv[1]
device = "cuda:0"
W0 = 0.38999998569488525
GRID = sorted(set(list(np.logspace(np.log10(20), np.log10(500), 14)) + [100.0, 104.43177795410156]))

def score(ref, est):
    ref, est = meb.trim_beats(ref), meb.trim_beats(est)
    f = meb.f_measure(ref, est) if len(est) and len(ref) else 0.0
    if len(est) and len(ref) > 1:
        _, c, _, a = meb.continuity(ref, est)
    else: c = a = 0.0
    return f, c, a

if DS == "smc":
    from smc_data import load_smc
    entries = load_smc()
    by_fold = {}
    for e in entries: by_fold.setdefault(e["fold"], []).append(e)
    fps = data.FPS
    def iter_songs_acts():
        for fold, es in sorted(by_fold.items()):
            model = frontend.load_frozen_model(f"checkpoints/bt_fold{fold}_repacked.pt", device)
            acts = frontend.activations_for(model, es, device)
            for e in es:
                yield e["stem"], e["beat_times"], acts[e["stem"]]
else:
    from data.songs import iter_songs
    from tracker import build_frontend
    import soundfile
    fe = build_frontend("frontends.beat_this", output="activations", checkpoint="final0",
                        device="cuda")
    fps = fe.fps
    def iter_songs_acts():
        for song in iter_songs(datasets=["gtzan"]):
            bt, _ = song.beats()
            sig, sr = soundfile.read(song.audio_path, dtype="float32")
            if sig.ndim > 1: sig = sig.mean(axis=1)
            a = fe.get_features(sig, sr)
            yield song.stem, bt, 1.0/(1.0+np.exp(-np.asarray(a, dtype=np.float64)))

m = MixtureLambda(fps=fps, device=device, observation_lambda=6)
m.mixture_weight = W0

def peak_pick(acts):
    beat = np.asarray(acts)[:, 0]
    dist = max(1, int(round(60.0/215.0 * fps)))
    peaks, _ = find_peaks(beat, height=0.2, distance=dist)
    return peaks / fps

rows = []
for stem, bt, a in iter_songs_acts():
    at = torch.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=np.float32))).to(device)
    with torch.no_grad():
        lls, fs, events = [], [], []
        for lam in GRID:
            k = m.log_mixture_kernel(W0, lam)
            lls.append(float(m.marginal_log_likelihood(at, k)))
            m.transition_lambda = lam
            ev = m.decode(a, deploy=True)
            events.append(ev)
            fs.append(score(bt, ev["beats"])[0])
    i_ml, i_or = int(np.argmax(lls)), int(np.argmax(fs))
    row = {"stem": stem, "lam_ml": GRID[i_ml], "lam_or": GRID[i_or],
           "ML": score(bt, events[i_ml]["beats"]), "ORACLE": score(bt, events[i_or]["beats"]),
           "fixed100": score(bt, events[GRID.index(100.0)]["beats"]),
           "peak": score(bt, peak_pick(a))}
    rows.append(row)
    if len(rows) % 50 == 0:
        print(len(rows), "songs", flush=True)
        json.dump(rows, open(f"results_lamselect_{DS}.json","w"))

json.dump(rows, open(f"results_lamselect_{DS}.json","w"))
from scipy.stats import spearmanr
print(f"== {DS} (n={len(rows)}) mixture chassis w={W0:.3f}, deployed decode ==")
for k in ("fixed100","peak","ML","ORACLE"):
    v = np.mean(np.array([r[k] for r in rows]), axis=0)
    print(f"  {k:8s} F {v[0]:.4f} CMLt {v[1]:.4f} AMLt {v[2]:.4f}")
lm = np.log([r["lam_ml"] for r in rows]); lo = np.log([r["lam_or"] for r in rows])
s = spearmanr(lm, lo)
print(f"  spearman(log lam_ML, log lam_ORACLE) rho={s.statistic:.3f} p={s.pvalue:.3g}")
qs = np.percentile([r['lam_ml'] for r in rows], [10,25,50,75,90])
print(f"  lam_ML deciles 10/25/50/75/90: {np.round(qs,1)}")
