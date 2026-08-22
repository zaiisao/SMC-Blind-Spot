"""Re-earn the R0==R1 certificate: 6 songs x 5 datasets, shipped AND bare, event-identical.

Activations: live Beat This final0 (squeeze bounding, 50 fps), same array fed to both rungs,
so this certifies the inference engine, not the frontend. Annotations are NOT consumed.
"""
import numpy as np, torch, json, time
from data.songs import iter_songs
from tracker import build_frontend
from rungs.r0_madmom_dbn import MadmomDBN
from rungs.r1_2016_dbn import DBN2016

rng = np.random.default_rng(0)
DATASETS = ["ballroom", "beatles", "gtzan", "hainsworth", "hjdb"]
songs = []
for d in DATASETS:
    pool = list(iter_songs(datasets=[d]))
    songs += [pool[i] for i in rng.choice(len(pool), 6, replace=False)]

frontend = build_frontend("frontends.beat_this", output="activations",
                          checkpoint="final0", device="cuda")
fps = frontend.fps
rungs = {
    "shipped": (MadmomDBN(fps=fps, bounding=frontend.BOUNDING),
                DBN2016(fps=fps, bounding=frontend.BOUNDING, device="cuda")),
    "bare": (MadmomDBN(fps=fps, bounding=frontend.BOUNDING,
                       num_tempi=None, threshold=0.0, correct=False),
             DBN2016(fps=fps, bounding=frontend.BOUNDING, device="cuda",
                     num_tempi=None, threshold=0.0, correct=False)),
}
import soundfile
results = {k: [] for k in rungs}
for s in songs:
    sig, sr = soundfile.read(s.audio_path, dtype="float32")
    if sig.ndim > 1: sig = sig.mean(axis=1)
    act = frontend.get_features(sig, sr)
    act = 1.0/(1.0+np.exp(-np.asarray(act, dtype=np.float64)))  # logits -> probs
    for mode, (r0, r1) in rungs.items():
        e0, e1 = r0.predict(act), r1.predict(act)
        same = (len(e0["beats"]) == len(e1["beats"]) and np.allclose(e0["beats"], e1["beats"], atol=1e-6)
                and len(e0["downbeats"]) == len(e1["downbeats"])
                and np.allclose(e0["downbeats"], e1["downbeats"], atol=1e-6))
        results[mode].append(same)
        if not same:
            print(f"DIVERGE [{mode}] {s.dataset}/{s.stem}: "
                  f"R0 {len(e0['beats'])}b/{len(e0['downbeats'])}d vs R1 {len(e1['beats'])}b/{len(e1['downbeats'])}d")
for mode, r in results.items():
    print(f"{mode}: {sum(r)}/{len(r)} event-identical")
