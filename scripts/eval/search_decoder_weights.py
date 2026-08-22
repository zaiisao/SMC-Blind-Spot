"""Direct search over the four decoder weights, scored by the DECODED METRIC. No surrogate objective.

WHY THIS INSTEAD OF THE CRF. The CRF's conditional-ML objective (§4.14) is degenerate by
construction: its loss is the gap between the constrained maximum and the free maximum, so anything
that makes those two decodes AGREE lowers the loss -- and the cheapest route to agreement is
weakening the structure that distinguishes them. Measured: corpus wrong-decode count 54 -> 244 of
250 while the loss fell, with the degenerate direction reachable through BOTH lambda (100 -> 18.36)
and w_trans (em:trans ratio 1.00 -> 1.64) -- freezing one simply moved the optimiser to the other.
A unit-norm projection stops the COLLAPSE but not the ROTATION.

The entire justification for a learned objective was joint optimisation of many parameters. There
are FOUR. So search them directly and score by the quantity we actually care about -- decoded CMLt
and F -- which cannot be gamed by flattening, because flattening makes the decodes worse and the
score falls. This is the same procedure Krebs used for lambda in 2015 (grid search on a decoding
metric), extended from one parameter to four.

SCORE. Weights are selected on CORPUS (ballroom/hainsworth/beatles/gtzan/asap, Beat This activations
from /disk1/jaehoon/bt_emb which carry the beat logits alongside the embeddings) and the winner is
then evaluated ONCE on SMC, which is never used for selection. Selection uses CMLt: the tempo
oracle buys +0.2221 CMLt but only +0.0615 F (§4.6), so F understates any tempo-side intervention
about fourfold.

BASELINE. w = [1, 0, 0, 1] with lambda=100 is exactly the shipped beat-only DBN -- verified
bit-exact in scripts/diagnostics/gate_crf_plumbing.py (identical Viterbi paths on 40/40 tracks, dF 0.00000). So
the search strictly contains the current system, and "no improvement" is a meaningful answer:
if a direct search over all four weights cannot beat [1,0,0,1] on corpus, no learned objective over
these features was going to either.

EXPECTATION. The decoder envelope on Beat This activations is F 0.6908 with per-track ORACLE
hyperparameters, against 0.6270 deployed (§4.12) -- a 0.064 F box. A single global weight vector
cannot exceed a per-track oracle, so the realistic headroom here is a fraction of that.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events
from train_contrastive_dbn_fixed import log_tempo_transition
from train_crf_decoder import acf_logratio
from train_tempo_observation import load_net

EMB = Path("/disk1/jaehoon/bt_emb")
SMC_EMB = Path("/disk1/jaehoon/smc_bt_embeddings")
STORE = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
HEAD = None  # set from --head-pt
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ballroom,hainsworth,beatles,gtzan,asap")
    ap.add_argument("--per-ds", type=int, default=40)
    ap.add_argument("--head-disjoint", type=int, default=1,
                    help="restrict the selection set to corpus fold 0, which the tempo head held "
                         "out during training -- required for w_tempo to be searchable")
    ap.add_argument("--head-pt", default="/disk1/jaehoon/tempo_obs_head.pt")
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--select-on", default="CMLt", choices=METRICS)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="/tmp/overnight/search_weights.json")
    args = ap.parse_args()
    dev = torch.device(args.device)

    m = make_beatonly_dbn(30.0, 16, 100, args.device)
    space = m.state_spaces[0]
    iv = space.interval_frames.astype(np.int64)
    s2c = m.state_to_classes[0].to(dev)
    idx = {int(v): i for i, v in enumerate(iv)}
    tmap = torch.from_numpy(np.array([idx[int(v)] for v in space.state_interval_frames])).to(dev)
    iv_t = torch.from_numpy(iv).to(dev)
    logT = log_tempo_transition(iv_t, torch.tensor(args.lam, dtype=torch.float64, device=dev),
                                "madmom").detach()
    head_pt = args.head_pt
    net = load_net(head_pt, dev)
    marg = np.load(head_pt + ".marginal.npy")
    logmarg = torch.tensor(np.log(np.clip(marg, 1e-12, None)), dtype=torch.float64, device=dev)

    def feats(act, emb):
        got = emissions_beatonly(m, act)
        if got is None:
            return None
        lcd, dec, ff = got
        T = lcd.shape[0]
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(emb, np.float32)).permute(1, 0).unsqueeze(0).to(dev)
            lq = torch.log_softmax(net(x).float(), 1)[0].T.double()
        if lq.shape[0] < ff + T:
            lq = torch.cat([lq, lq[-1:].repeat(ff + T - lq.shape[0], 1)], 0)
        return {"em": lcd[:, s2c], "tempo": (lq[ff:ff + T] - logmarg[None, :])[:, tmap],
                "acf": acf_logratio(act[ff:ff + T], iv, args.tau, dev)[tmap],
                "dec": dec, "ff": ff}

    def decode(F, w, ref):
        em = w[0] * F["em"] + w[1] * F["tempo"] + w[2] * F["acf"][None, :]
        path, _ = m.dynamic_programs[0].viterbi(
            m.log_initial_distributions[0], w[3] * logT, em,
            state_to_class=None, return_log_score=True)
        return ev(ref, state_path_to_events(path.cpu().numpy(), space, FPS,
                                            snap_to_activations=F["dec"],
                                            first_frame=F["ff"])["beats"])

    # ---------------- corpus selection set ----------------
    corpus = []
    for ds in args.datasets.split(","):
        d = EMB / ds
        if not d.exists():
            continue
        # HEAD-DISJOINT SELECTION SET. The tempo head trains on fold != 0 and holds out fold 0
        # (724 corpus tracks) as validation, so fold-0 tracks are genuinely out-of-sample FOR THE
        # HEAD and w_tempo can be searched without a leak. The previous version took
        # sorted(glob)[:per_ds], which mixes folds and therefore included head-training tracks --
        # that is what produced the +0.1137 corpus / -0.0990 SMC inversion (a 0.21 CMLt swing).
        picked = 0
        for f in sorted(d.glob("*.npz")):
            if picked >= args.per_ds:
                break
            ann = STORE / ds / "annotations" / "beats" / f"{f.stem}.beats"
            if not ann.exists():
                continue
            z = np.load(f, allow_pickle=True)
            if args.head_disjoint and int(z["fold"]) != 0:
                continue
            picked += 1
            act = 1.0 / (1.0 + np.exp(-np.asarray(z["beat"], np.float64)))
            F = feats(act, z["emb"])
            if F is None:
                continue
            sf = int(z["start_frame"])
            bt = np.array([float(l.split()[0]) for l in open(ann) if l.strip()]) * FPS - sf
            keep = (bt >= F["ff"]) & (bt < F["ff"] + F["em"].shape[0])
            if keep.sum() < 8:
                continue
            corpus.append((F, (bt[keep]) / FPS))
    print(f"corpus selection set: {len(corpus)} tracks", flush=True)

    # Coarse by design. 400 vectors x 150 tracks = 60,000 Viterbi decodes over 2,921 states was
    # 30-45 min; this is 36 vectors and answers the same question. Resolution on tempo/acf is
    # deliberately minimal: four independent lines of evidence (SPEC 4.6, 4.7, 4.11, 4.14) already
    # put their useful weight at ~0, so the question is whether they are non-zero AT ALL.
    # LEAK-FREE GRID. w_tempo is PINNED AT 0. q_theta was fitted on these very corpus tracks, so on
    # corpus its tempo predictions are IN-SAMPLE and effectively oracle-quality. Selecting w_tempo
    # there gave +0.1137 CMLt on corpus and -0.0990 (p=4.9e-05) on SMC -- a 0.21 swing, i.e. the
    # selected weights actively HURT out of sample. The emission, ACF and transition features are
    # all closed-form (no training, no leak), so searching over those three is valid on corpus.
    # Re-introducing w_tempo requires a selection set held out of the HEAD's training set, which no
    # corpus track currently is.
    GRID = {
        "em":    [0.6, 0.8, 1.0, 1.25, 1.6],
        "tempo": [0.0, 0.1, 0.2, 0.4],
        "acf":   [0.0, 0.05, 0.1, 0.2],
        "trans": [0.6, 0.8, 1.0, 1.25, 1.6],
    }
    combos = list(itertools.product(GRID["em"], GRID["tempo"], GRID["acf"], GRID["trans"]))
    print(f"grid: {len(combos)} weight vectors x {len(corpus)} tracks", flush=True)

    rows = []
    for i, w in enumerate(combos):
        sc = [decode(F, w, ref) for F, ref in corpus]
        mean = {k: float(np.mean([s[k] for s in sc])) for k in METRICS}
        rows.append({"w": list(w), **mean})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(combos)}", flush=True)
    base = [r for r in rows if r["w"] == [1.0, 0.0, 0.0, 1.0]][0]
    rows.sort(key=lambda r: -r[args.select_on])
    print(f"\nCORPUS baseline w=[1,0,0,1]: " + "  ".join(f"{k} {base[k]:.4f}" for k in METRICS))
    print(f"top 5 by {args.select_on}:")
    for r in rows[:5]:
        print(f"  w={r['w']}  " + "  ".join(f"{k} {r[k]:.4f}" for k in METRICS))
    best = rows[0]["w"]
    gain = rows[0][args.select_on] - base[args.select_on]
    print(f"\ncorpus gain of best over baseline: {gain:+.4f} {args.select_on}")

    # ---------------- one shot on SMC ----------------
    print("\nevaluating baseline and corpus-selected weights on SMC (never used for selection)",
          flush=True)
    res = {"null": {}, "sel": {}}
    for t in track_ids():
        f = SMC_EMB / f"{t}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        F = feats(np.asarray(bthis_beat_prob(t), np.float64), z["emb"])
        if F is None:
            continue
        gt = load_gt(t)
        res["null"][t] = decode(F, [1.0, 0.0, 0.0, 1.0], gt)
        res["sel"][t] = decode(F, best, gt)
    ids = sorted(set(res["null"]) & set(res["sel"]))
    A = lambda r: {k: float(np.mean([r[t][k] for t in ids])) for k in METRICS}
    print(f"\nSMC n={len(ids)}")
    print(f"  null  w=[1,0,0,1]   " + "  ".join(f"{k} {A(res['null'])[k]:.4f}" for k in METRICS))
    print(f"  sel   w={best}   " + "  ".join(f"{k} {A(res['sel'])[k]:.4f}" for k in METRICS))
    print("  paired sel - null:")
    for k in METRICS:
        d = np.array([res["sel"][t][k] - res["null"][t][k] for t in ids])
        p = wilcoxon(d).pvalue if np.any(d) else 1.0
        print(f"    {k} {d.mean():+.4f} (p={p:.3g})")
    json.dump({"best_w": best, "corpus_gain": gain, "corpus_rows": rows[:40],
               "smc": {n: A(res[n]) for n in res}}, open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
