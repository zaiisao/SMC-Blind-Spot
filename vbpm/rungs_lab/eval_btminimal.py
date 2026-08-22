"""Fold-honest Beat This WITHOUT DBN: the official minimal postprocessor (their own
peak-pick) on the same activations the R0/R1 table used. Same scoring, TRUE metric names."""
import sys, json, time
from pathlib import Path

import numpy as np, soundfile, torch
import mir_eval.beat as meb

sys.path.insert(0, str(Path(__file__).resolve().parent / "external" / "beat_this"))
from beat_this.model.postprocessor import Postprocessor

from data.songs import iter_songs
from tracker import build_frontend

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
    by_ckpt = {}
    for d in DATASETS:
        for s in iter_songs(datasets=[d]):
            ck = "final0" if s.fold is None else f"fold{s.fold}"
            by_ckpt.setdefault(ck, []).append(s)
    postp = None
    for ck, songs in sorted(by_ckpt.items()):
        frontend = build_frontend("frontends.beat_this", output="activations",
                                  checkpoint=ck, device="cuda:1")
        if postp is None:
            postp = Postprocessor(type="minimal", fps=int(frontend.fps))
        for s in songs:
            sig, sr = soundfile.read(s.audio_path, dtype="float32")
            if sig.ndim > 1:
                sig = sig.mean(axis=1)
            act = np.asarray(frontend.get_features(sig, sr), dtype=np.float32)
            beat_t = torch.from_numpy(act[:, 0])
            db_t = torch.from_numpy(act[:, 1])
            times_b, times_db = postp(beat_t, db_t)
            bt, dbt = s.beats()
            bf, bc, ba = score(bt, np.asarray(times_b, dtype=np.float64))
            df, dc, da = score(dbt, np.asarray(times_db, dtype=np.float64))
            rows.append({"dataset": s.dataset, "stem": s.stem, "ckpt": ck,
                         "BTmin": dict(beatF=bf, CMLt=bc, AMLt=ba, downbeatF=df,
                                       dbCMLt=dc, dbAMLt=da)})
            if len(rows) % 200 == 0:
                json.dump(rows, open(out_path, "w"))
                print(f"{len(rows)} songs done ({s.dataset})", flush=True)
    json.dump(rows, open(out_path, "w"))
    print("DONE", len(rows), flush=True)

main()
