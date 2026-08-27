"""Score a checkpoint on ALL held-out fold-0 tracks, not just SMC's 27.

WHY. The corpus uses ONE 8-fold split across every dataset, so a fold-0 model
holds out fold 0 of gtzan, simac, ballroom, groove_midi, smc, asap, hainsworth,
beatles, guitarset, rwc and tapcorrect -- 329 tracks, all with precomputed
spectrograms. The whole 2026-08-26 session scored 27 of them.

Measured per-track paired difference sd between two arms is 0.0856, so the
minimum detectable effect at 80% power (2.80*sd/sqrt(n)) is:

    SMC fold 0 only          n=27     0.0461     <- what was used
    ALL datasets, fold 0    n=329     0.0132     <- this script, no new training
    SMC all 8 folds         n=217     0.0163     <- would cost 8 trainings

So this is better resolution than the full 8-fold SMC protocol, for free, on
checkpoints that already exist.

Reads <spect_root>/<stem>/track.npy directly -- no audio decode, no resampling.
Reports overall F and a per-dataset breakdown, because a tempo-bias fix should be
positive on SMC/asap and roughly neutral on steady-tempo sets like ballroom, and
an aggregate would hide that.
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts/sa3")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM/rungs_lab")

DATA = "/disk1/jaehoon/bt_wide/data"
SPECT = f"{DATA}/audio/spectrograms/wide"
ANN = f"{DATA}/annotations/wide/annotations/beats"
SPLIT = f"{DATA}/annotations/wide/8-folds.split"
FPS = 50


def source(stem):
    return stem.split("__")[0] if "__" in stem else stem.split("_")[0]


def fold_stems(fold):
    out = []
    for line in open(SPLIT):
        p = line.split()
        if len(p) == 2 and int(p[1]) == fold:
            out.append(p[0])
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path, or a released shortname like fold0")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out", default=None, help="write per-track F to this json")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"

    from smc_eval import load_net
    from beat_this.inference import split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from training.evaluation import beat_f_measure

    net = load_net(a.ckpt, dev)
    net.eval()
    post = Postprocessor(type="minimal", fps=FPS)

    stems = fold_stems(a.fold)
    if a.limit:
        stems = stems[: a.limit]
    rows = {}
    skipped = 0
    with torch.inference_mode():
        for stem in stems:
            sp = f"{SPECT}/{stem}/track.npy"
            bp = f"{ANN}/{stem}.beats"
            if not (os.path.exists(sp) and os.path.exists(bp)):
                skipped += 1
                continue
            truth = np.loadtxt(bp, ndmin=2)[:, 0]
            if len(truth) < 4:
                skipped += 1
                continue
            S = torch.from_numpy(np.load(sp)).float().to(dev)
            if S.ndim == 3:
                S = S[0]
            pr = split_predict_aggregate(S, 1500, 6, "keep_first", net)
            bt, _ = post(pr["beat"].float(), pr["downbeat"].float())
            est = np.asarray(bt[0] if isinstance(bt, (list, tuple)) else bt).ravel()
            lo, hi = truth.min(), truth.max()
            n_in = int(((est >= lo) & (est <= hi)).sum())
            rows[stem] = dict(F=float(beat_f_measure(truth, est)),
                              r=n_in / max(len(truth), 1),
                              bpm=float(60.0 / np.median(np.diff(truth))) if len(truth) > 2 else 0.0,
                              src=source(stem))

    F = np.array([v["F"] for v in rows.values()])
    name = os.path.basename(str(a.ckpt))[:46]
    print(f"\n  {name}")
    print(f"  fold {a.fold}: {len(rows)} tracks scored, {skipped} skipped")
    print(f"  OVERALL F = {F.mean():.4f}   (se {F.std(ddof=1)/np.sqrt(len(F)):.4f})\n")
    print(f"  {'dataset':<14}{'n':>4}{'F':>9}{'mean r':>9}{'median bpm':>12}")
    for s in sorted({v["src"] for v in rows.values()}):
        g = [v for v in rows.values() if v["src"] == s]
        f = np.array([x["F"] for x in g])
        print(f"  {s:<14}{len(g):4d}{f.mean():9.4f}"
              f"{np.mean([x['r'] for x in g]):9.3f}"
              f"{np.median([x['bpm'] for x in g]):12.1f}")
    smc = [v["F"] for v in rows.values() if v["src"] == "smc"]
    if smc:
        print(f"\n  SMC only: n={len(smc)}  F={np.mean(smc):.4f}   "
              f"(this is the number the session was using)")
    if a.out:
        json.dump(rows, open(a.out, "w"))
        print(f"  per-track -> {a.out}")


if __name__ == "__main__":
    main()
