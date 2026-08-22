"""Inject BEAT TRANSFORMER's tempo head into the beat-only DBN, through the verified path.

WHY. Our purpose-built head (16.7M frames, phase-vocoder stretch augmentation, octave-band penalty)
is beaten on SMC by an off-the-shelf tempo head that ships with Beat Transformer:

                        correct   octave-err   median log2 bias
    Beat Transformer     56.7%      20.7%          -0.015
    ours (baseline)      52.1%      30.0%          +0.000
    ours (oct_w=20)      48.8%      23.0%          +0.013

More correct AND fewer octave errors. Since the audit proved the injection path reproduces the
oracle ceiling exactly (synthetic oracle through this same code gives CMLt 0.7906 / F 0.6756, matching
exp_oracle_tempo_beatonly.py to 16 significant digits), a better tempo distribution should convert
directly into a better decode -- with no further work on our own head.

GRID MAPPING. Beat Transformer emits 300 BPM logits; the DBN grid is 60 integer beat-INTERVALS
(14..100 frames at 50 fps = 214..30 BPM). Each grid interval is mapped to its nearest BPM bin and the
distribution renormalised over the DBN grid. This resamples a distribution onto a coarser, non-uniform
support, so it is an approximation -- the alternative (interpolating in log-BPM) is reported as a
second arm so the choice is measured rather than assumed.

DENOMINATOR. The scaled likelihood needs the estimator's own marginal, not the label prior (using the
label prior once injected 5.47 nats/tempo). Beat Transformer's marginal over SMC is computed here and
used; a uniform denominator is run as a control, since the audit measured that choice to be worth at
most 0.0062 CMLt for our head.

REFERENCE, same decoder, n=217: tempo-blind null CMLt 0.5685 / F 0.6141; our best learned head
+0.0153 (p=0.36); oracle +0.2221 (p=5.5e-21).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob, btrans_tempo
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events

METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
BETAS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"


def main():
    dev = torch.device(DEV)
    m = make_beatonly_dbn(30.0, 16, 100, DEV)
    space = m.state_spaces[0]
    iv = space.interval_frames.astype(np.int64)
    idx = {int(v): i for i, v in enumerate(iv)}
    tmap = torch.from_numpy(np.array([idx[int(v)] for v in space.state_interval_frames])).to(dev)
    bf = torch.tensor(space.state_interval_frames, dtype=torch.float64, device=dev)
    s2c = m.state_to_classes[0].to(dev)
    grid_bpm = 60.0 * FPS / iv.astype(np.float64)          # DBN grid expressed in BPM

    # ---- gather Beat Transformer tempo distributions, mapped onto the DBN grid ----
    q = {}
    for t in track_ids():
        try:
            _, lg = btrans_tempo(t)
        except Exception:
            continue
        lg = np.asarray(lg, dtype=np.float64)
        bpm_axis = np.arange(len(lg), dtype=np.float64)     # 300 bins, 1 BPM each
        j = np.clip(np.rint(grid_bpm).astype(int), 0, len(lg) - 1)
        sel = lg[j]                                         # nearest-BPM-bin gather
        q[t] = torch.tensor(sel - np.log(np.exp(sel - sel.max()).sum()) - sel.max(),
                            dtype=torch.float64, device=dev)
    print(f"Beat Transformer tempo available for {len(q)} tracks; DBN grid "
          f"{grid_bpm.min():.0f}-{grid_bpm.max():.0f} BPM", flush=True)

    # estimator's own marginal over the evaluated set (the correct scaled-likelihood denominator)
    marg = torch.log_softmax(torch.stack([v for v in q.values()]).logsumexp(0)
                             - np.log(len(q)), dim=0)
    uni = torch.full_like(marg, -np.log(len(iv)))

    res = {}
    for den_name, den in (("marginal", marg), ("uniform", uni)):
        for b in BETAS:
            res[(den_name, b)] = {}
    for t in track_ids():
        if t not in q:
            continue
        got = emissions_beatonly(m, np.asarray(bthis_beat_prob(t), np.float64))
        if got is None:
            continue
        lcd, dec, ff = got
        T = lcd.shape[0]
        gt = load_gt(t)
        for den_name, den in (("marginal", marg), ("uniform", uni)):
            ratio = (q[t] - den)[tmap]                       # [num_states]
            for b in BETAS:
                if b == 0.0:
                    em, stc = lcd, s2c
                else:
                    em = lcd[:, s2c] + b * (ratio[None, :] / bf.unsqueeze(0))
                    stc = None
                path, _ = m.dynamic_programs[0].viterbi(
                    m.log_initial_distributions[0], m.log_tempo_transition, em,
                    state_to_class=stc, return_log_score=True)
                res[(den_name, b)][t] = ev(gt, state_path_to_events(
                    path.cpu().numpy(), space, FPS, snap_to_activations=dec,
                    first_frame=ff)["beats"])

    ids = sorted(set.intersection(*[set(v) for v in res.values()]))
    A = lambda r: {k: float(np.mean([r[t][k] for k_ in [0] for k in [k_] ])) for k in METRICS}
    A = lambda r: {k: float(np.mean([r[t][k] for t in ids])) for k in METRICS}
    base = res[("marginal", 0.0)]
    print(f"\nn={len(ids)}")
    print(f"{'denom':<10}{'beta':>6}" + "".join(f"{k:>9}" for k in METRICS) + f"{'dCMLt':>9}{'p':>10}")
    for den_name in ("marginal", "uniform"):
        for b in BETAS:
            r = res[(den_name, b)]
            d = np.array([r[t]["CMLt"] - base[t]["CMLt"] for t in ids])
            p = wilcoxon(d).pvalue if np.any(d) else 1.0
            print(f"{den_name:<10}{b:>6.1f}" + "".join(f"{A(r)[k]:>9.4f}" for k in METRICS)
                  + f"{d.mean():>+9.4f}{p:>10.2g}")
    print("\nreference: null CMLt 0.5685 | our best head +0.0153 (p=0.36) | oracle +0.2221")
    json.dump({"n": len(ids), "summary": {f"{k[0]}_b{k[1]}": A(v) for k, v in res.items()}},
              open("/tmp/overnight/btrans_tempo_dbn.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
