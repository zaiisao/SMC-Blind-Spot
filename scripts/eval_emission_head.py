"""Decode SMC with a learned emission head in place of Boeck-2016's generative densities.

The head replaces ONLY the [T,3] class-density table that `DBN2016._log_class_densities` produces:

    log p(y_t | class) = log p(class | h_t) - log p(class)          (hybrid HMM-DNN scaled likelihood)

The state space, the state->class map, the tempo transition and the inference are all UNCHANGED,
so any difference is attributable to the emission alone.

NOTE ON TEMPO NEUTRALITY. Because the state->class assignment stays geometric (fractional, 1/obs),
n_beat = T/obs at every tempo, so the E(i) identity that killed the fixed-ms REGION does not apply
here: the head changes the densities, not the frame counts. Gate T1 verifies this empirically
rather than assuming it, by measuring the head's mean log-density by tempo band.

BASELINE. Compared against the TUNED hand-set configuration (obs_lambda swept: 8 is F-optimal, 16
is AMLt-optimal on raw activations), not madmom's default alone.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))

from scipy.stats import wilcoxon
from _harness import FPS, track_ids, load_gt, gt_bpm, bthis_beat_prob, bthis_downbeat_prob, ev
from train_emission_head import Head, stack_ctx
from rungs.r1_2016_dbn import DBN2016
from rungs.deployment import threshold_crop
from rungs.bar_pointer.readout import state_path_to_events

SMC_EMB = Path("/disk1/jaehoon/smc_bt_embeddings")


def head_densities(head, prior, emb, ctx, dev):
    x = stack_ctx(emb.astype(np.float32), ctx)
    with torch.no_grad():
        lg = head(torch.from_numpy(x).to(dev))
        logp = F.log_softmax(lg.float(), dim=1)
    return (logp - torch.log(torch.tensor(prior, dtype=torch.float64, device=dev)).float()).double()


def decode_with(m, lcd, dec, ff):
    best = None
    for space, dp, li, s2c in zip(m.state_spaces, m.dynamic_programs,
                                  m.log_initial_distributions, m.state_to_classes):
        path, sc = dp.viterbi(li, m.log_tempo_transition, lcd, state_to_class=s2c,
                              return_log_score=True)
        if best is None or sc > best[0]:
            best = (sc, space, path)
    _, space, path = best
    return state_path_to_events(path.cpu().numpy(), space, m.fps,
                                snap_to_activations=dec, first_frame=ff)["beats"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--obs", type=int, default=16)
    ap.add_argument("--lam", type=int, default=100)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="head")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ck = torch.load(args.head, map_location=dev, weights_only=False)
    head = Head(512, ck["hidden"], ck["ctx"]).to(dev); head.load_state_dict(ck["state"]); head.eval()
    prior = np.asarray(ck["prior"])

    m = DBN2016(fps=FPS, min_bpm=30., max_bpm=215., device=args.device,
                observation_lambda=args.obs, transition_lambda=args.lam)
    ids = track_ids()
    if args.limit:
        ids = ids[:args.limit]
    ids = [t for t in ids if (SMC_EMB / f"{t}.npz").exists()]
    refs = {t: load_gt(t) for t in ids}
    bpm = {t: gt_bpm(refs[t]) for t in ids}

    per_base, per_head, neutral = {}, {}, []
    for t in ids:
        a = np.stack([bthis_beat_prob(t), bthis_downbeat_prob(t)], -1)
        b, d, fl = m._bound(a[:, 0], a[:, 1]); dec = m._decorrelate(b, d, fl)
        dec, ff = threshold_crop(dec, m.threshold)
        if not len(dec):
            continue
        aa = a[ff:ff + len(dec)]
        base_lcd = m._log_class_densities(aa[:, 0], aa[:, 1])
        per_base[t] = ev(refs[t], decode_with(m, base_lcd, dec, ff))

        emb = np.load(SMC_EMB / f"{t}.npz")["emb"]
        hd = head_densities(head, prior, emb, ck["ctx"], dev)[ff:ff + len(dec)]
        per_head[t] = ev(refs[t], decode_with(m, hd, dec, ff))
        neutral.append((bpm[t], float(hd[:, 1].mean()), float(hd[:, 0].mean())))

    A = lambda p: {k: float(np.mean([p[t][k] for t in ids if t in p]))
                   for k in ("F", "CMLt", "AMLt", "CMLc", "AMLc")}
    ok = [t for t in ids if t in per_base and t in per_head]
    print(f"n={len(ok)}  head={Path(args.head).name}  obs={args.obs} lam={args.lam}")
    print(f"  baseline (2016 densities) {A(per_base)}")
    print(f"  learned head              {A(per_head)}")

    print("\n  GATE T1 tempo neutrality of the head's densities (mean log-density by band):")
    arr = np.array(neutral)
    for lo, hi in ((0, 55), (55, 70), (70, 90), (90, 999)):
        s = arr[(arr[:, 0] >= lo) & (arr[:, 0] < hi)]
        if len(s):
            print(f"    {lo:3}-{hi:3} BPM n={len(s):3}  BEAT {s[:,1].mean():+7.3f}  "
                  f"NO_BEAT {s[:,2].mean():+7.3f}")
    sp = arr[:, 1]
    print(f"    spread of mean BEAT log-density across bands: "
          f"{max(np.mean(arr[(arr[:,0]>=lo)&(arr[:,0]<hi)][:,1]) for lo,hi in ((0,55),(55,70),(70,90),(90,999)) if ((arr[:,0]>=lo)&(arr[:,0]<hi)).any()) - min(np.mean(arr[(arr[:,0]>=lo)&(arr[:,0]<hi)][:,1]) for lo,hi in ((0,55),(55,70),(70,90),(90,999)) if ((arr[:,0]>=lo)&(arr[:,0]<hi)).any()):.3f} nats/frame")

    print("\n  head minus baseline (paired):")
    out = {}
    for k in ("F", "CMLt", "AMLt", "CMLc", "AMLc"):
        dd = np.array([per_head[t][k] - per_base[t][k] for t in ok]); nz = dd[dd != 0]
        p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
        bt = np.array([np.mean(np.random.choice(dd, len(dd), True)) for _ in range(10000)])
        out[k] = {"delta": float(dd.mean()), "p": p,
                  "ci": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]}
        print(f"    {k:5} {dd.mean():+.4f} CI[{out[k]['ci'][0]:+.4f},{out[k]['ci'][1]:+.4f}] "
              f"p={p:.3g} w/l {int((dd>0).sum())}/{int((dd<0).sum())}")
    Path(f"/tmp/overnight/head_eval_{args.tag}.json").write_text(json.dumps(
        {"baseline": A(per_base), "head": A(per_head), "tests": out, "n": len(ok)}, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
