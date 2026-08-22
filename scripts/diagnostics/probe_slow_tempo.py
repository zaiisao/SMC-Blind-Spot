"""Out-of-the-box capability test: does MERT's geometry encode SLOW tempo in a way that
EXTRAPOLATES, as web-scale pretraining should confer?

THE CLAIM UNDER TEST (C1). "MERT was trained on ~160k hours including ample unusually slow music,
so its representation should carry slow-tempo rhythm structure that Beat This (trained on ~4k mostly
percussive/pop tracks) lacks."

DESIGN. Identical tiny probes -- log2-tempo REGRESSION from temporally pooled frozen features --
trained on the 300 corpus tracks (ballroom/beatles/hainsworth/hjdb, few slow examples) and tested on
all 217 SMC tracks, stratified by true tempo. Regression in log-tempo, not classification over bins:
a softmax cannot predict bins absent from training by construction, while a regressor can
extrapolate if the FEATURE GEOMETRY orders tempo smoothly below the training range. That ordering is
exactly what pretraining exposure to slow music should produce, and what supervised training on
fast corpora has no reason to produce.

Feature sets under the SAME probe, same pooling, same seed:
    mertL6    MERT-v1-95M layer 6, cached          (the claim's subject)
    btemb     Beat This penultimate embedding      (the supervised comparison)
Pooling: mean + std over time (2x d). Frame-level tempo is contextual in both models, so pooled
statistics preserve "this passage is slow" if the representation encodes it at all.

READ-OUT: mean |log2 error| and octave-accuracy (+-5% after octave folding removed -- we report BOTH
raw +-5% and octave-folded +-5%, because the interesting failure is predicting 2x the true tempo),
stratified by SMC tempo: <55 BPM (45 tracks, mostly OUTSIDE the corpus training range) vs >=55.
If MERT extrapolates below the training range better than btemb, C1 has legs. If both collapse
equally on <55, pretraining coverage did not turn into accessible slow-tempo structure, and the
synergy premise loses its foundation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from _harness import track_ids, load_gt

MERT_SMC = Path("/disk1/jaehoon/vbpm_mert_layers/smc")
MERT_COR = Path("/disk1/jaehoon/vbpm_mert_layers/corpus")
BT_SMC = Path("/disk1/jaehoon/smc_bt_embeddings")
BT_COR = Path("/disk1/jaehoon/bt_emb")
DEV = "cuda:3"


def pool(x):
    return np.concatenate([x.mean(0), x.std(0)])


def corpus_items():
    out = []
    for p in sorted(MERT_COR.glob("*.pt")):
        d = torch.load(p, map_location="cpu")
        bt_p = None
        stem = d.get("stem", p.stem)
        ds = stem.split("_")[0]
        cand = BT_COR / ds / f"{stem}.npz"
        if cand.exists():
            bt_p = cand
        tgt = np.asarray(d["beat_targets"], np.float32)
        fr = np.nonzero(tgt > 0)[0]
        if len(fr) < 8:
            continue
        # MERT cache is 86.13 fps
        bpm = 60.0 * 86.1328125 / float(np.median(np.diff(fr)))
        out.append((stem, np.asarray(d["layers"][6], np.float32), bt_p, bpm))
    return out


def main():
    torch.manual_seed(0)
    cor = corpus_items()
    print(f"corpus items: {len(cor)}; with BT embedding: {sum(1 for c in cor if c[2])}")
    cb = np.array([c[3] for c in cor])
    print(f"corpus tempo range: {cb.min():.0f}-{cb.max():.0f} BPM; below 55: {(cb < 55).sum()} tracks")

    feats = {"mertL6": {}, "btemb": {}}
    y = {}
    for stem, me, btp, bpm in cor:
        if btp is None:
            continue
        feats["mertL6"][stem] = pool(me)
        feats["btemb"][stem] = pool(np.asarray(np.load(btp, allow_pickle=True)["emb"], np.float32))
        y[stem] = np.log2(bpm)

    smc = {}
    for tid in track_ids():
        pm, pe = MERT_SMC / f"{tid}.pt", BT_SMC / f"{tid}.npz"
        if not (pm.exists() and pe.exists()):
            continue
        gt = np.asarray(load_gt(tid), float)
        if len(gt) < 8:
            continue
        d = torch.load(pm, map_location="cpu")
        smc[tid] = {
            "mertL6": pool(np.asarray(d["layers"][6], np.float32)),
            "btemb": pool(np.asarray(np.load(pe, allow_pickle=True)["emb"], np.float32)),
            "bpm": 60.0 / float(np.median(np.diff(gt))),
        }
    print(f"SMC items: {len(smc)}")

    print(f"\n{'features':<9}{'subset':<12}{'n':>4}{'mean|log2 err|':>16}{'+-5% raw':>10}"
          f"{'+-5% oct-folded':>16}{'pred<55 when true<55':>22}")
    results = {}
    for fs in ("mertL6", "btemb"):
        ids = sorted(set(feats[fs]) & set(y))
        X = torch.tensor(np.stack([feats[fs][t] for t in ids]), dtype=torch.float32, device=DEV)
        mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-6)
        X = (X - mu) / sd
        Y = torch.tensor([y[t] for t in ids], dtype=torch.float32, device=DEV)
        net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.GELU(), nn.Dropout(0.2),
                            nn.Linear(128, 1)).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        for ep in range(400):
            i = torch.randperm(len(ids), device=DEV)[:64]
            loss = nn.functional.huber_loss(net(X[i])[:, 0], Y[i])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            tids = sorted(smc)
            Xs = torch.tensor(np.stack([smc[t][fs] for t in tids]), dtype=torch.float32,
                              device=DEV)
            pred = net((Xs - mu) / sd)[:, 0].cpu().numpy()
        true = np.array([np.log2(smc[t]["bpm"]) for t in tids])
        err = pred - true
        res = {}
        for nm, mask in (("slow <55", true < np.log2(55)), (">=55", true >= np.log2(55)),
                         ("ALL", np.ones_like(true, bool))):
            e = err[mask]
            raw5 = np.abs(e) < 0.0704
            fold5 = np.abs(e - np.round(e)) < 0.0704
            below = (pred[mask] < np.log2(55)).mean() if nm == "slow <55" else float("nan")
            print(f"{fs:<9}{nm:<12}{mask.sum():>4}{np.abs(e).mean():>16.3f}"
                  f"{100*raw5.mean():>9.1f}%{100*fold5.mean():>15.1f}%"
                  + (f"{100*below:>21.1f}%" if nm == "slow <55" else ""))
            res[nm] = {"mae_log2": float(np.abs(e).mean()), "raw5": float(raw5.mean()),
                       "fold5": float(fold5.mean())}
        results[fs] = res
    json.dump(results, open("/tmp/overnight/slow_tempo_probe.json", "w"), indent=1)
    print("\n'pred<55 when true<55' is the extrapolation question itself: does the probe even "
          "PREDICT a below-training-range tempo when the truth is below it.")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
