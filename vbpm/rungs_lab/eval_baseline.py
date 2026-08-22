"""Fold-honest R0/R1 baseline table. For song s, activations come from checkpoint fold{s.fold};
gtzan (fold=None, held out of every training set) uses final0 (fold-honest by construction).
Metrics: mir_eval beat/downbeat F, CMLt (continuity idx 1), AMLt (idx 3) -- TRUE names.
Consumes: audio -> Beat This activations (label-free decode); annotations for GRADING only.
Both rungs run SHIPPED decode (madmom defaults: num_tempi=60, threshold=0.05, correct=True).
"""
import sys, json, time
import numpy as np, soundfile
import mir_eval.beat as meb
from data.songs import iter_songs
from tracker import build_frontend
from rungs.r0_madmom_dbn import MadmomDBN
from rungs.r1_2016_dbn import DBN2016

DATASETS = ["gtzan", "ballroom", "beatles", "hainsworth", "hjdb"]

def score(ref, est):
    ref, est = meb.trim_beats(ref), meb.trim_beats(est)
    f = meb.f_measure(ref, est) if len(est) and len(ref) else 0.0
    if len(est) and len(ref) > 1:
        _, cmlt, _, amlt = meb.continuity(ref, est)
    else:
        cmlt = amlt = 0.0
    return f, cmlt, amlt

def main():
    out_path = sys.argv[1]
    rows = []
    # group songs by required checkpoint
    by_ckpt = {}
    for d in DATASETS:
        for s in iter_songs(datasets=[d]):
            ck = "final0" if s.fold is None else f"fold{s.fold}"
            by_ckpt.setdefault(ck, []).append(s)
    r0 = r1 = None
    for ck, songs in sorted(by_ckpt.items()):
        frontend = build_frontend("frontends.beat_this", output="activations",
                                  checkpoint=ck, device="cuda")
        if r0 is None:
            r0 = MadmomDBN(fps=frontend.fps, bounding=frontend.BOUNDING)
            r1 = DBN2016(fps=frontend.fps, bounding=frontend.BOUNDING, device="cuda")
        for s in songs:
            t0 = time.time()
            sig, sr = soundfile.read(s.audio_path, dtype="float32")
            if sig.ndim > 1: sig = sig.mean(axis=1)
            act = frontend.get_features(sig, sr)
            act = 1.0/(1.0+np.exp(-np.asarray(act, dtype=np.float64)))
            bt, dbt = s.beats()
            row = {"dataset": s.dataset, "stem": s.stem, "ckpt": ck}
            for name, rung in (("R0", r0), ("R1", r1)):
                ev = rung.predict(act)
                bf, bc, ba = score(bt, ev["beats"])
                df, dc, da = score(dbt, ev["downbeats"])
                row[name] = dict(beatF=bf, CMLt=bc, AMLt=ba, downbeatF=df,
                                 dbCMLt=dc, dbAMLt=da)
            row["sec"] = round(time.time()-t0, 2)
            rows.append(row)
            if len(rows) % 50 == 0:
                json.dump(rows, open(out_path, "w"))
                print(f"{len(rows)} songs done ({s.dataset})", flush=True)
    json.dump(rows, open(out_path, "w"))
    print("DONE", len(rows))

main()
