"""How much SMC score does a tempo estimator buy, as a function of its OCTAVE error rate?

REPLACES the retracted sigma-noise study (SPECIFICATION.md §8 R-1). That study degraded the oracle
with i.i.d. per-frame lognormal noise and concluded 94-100% of the gain survives at "30.9% accuracy".
It was an artefact: per-frame noise is averaged out by the per-beat normalisation (30.9% accurate per
FRAME is ~97% accurate per BEAT), so it measured smoothing, not degradation. The noisy arms even
scored ABOVE the perfect oracle.

Real estimator errors are not i.i.d. jitter. They are per-track, temporally correlated, and
overwhelmingly METRICAL: our learned head's residual errors are 40 tracks at exactly 1/2x, 14 at 2x,
7 at 3x, 4 at 1/3x. A track tracked one octave off scores CMLt ~0 by definition, which no
symmetric-jitter model can express.

THIS EXPERIMENT. Start from the oracle tempo (which buys ΔCMLt +0.2221, n=217) and corrupt a
fraction p of TRACKS with a whole-track octave flip (x2 or /2, chosen at random). Sweep p. This
yields the achievable-gain curve as a function of octave-error rate, which is the quantity that
actually gates a learned tempo head:

    p=0.00  perfect tempo            -> the oracle ceiling
    p=0.25  ~ our current head       -> 54/217 octave-wrong
    p=0.50  half the tracks wrong

The number to read off is the p at which the curve crosses zero: above that, a tempo channel is
worse than none, and any head must beat it before it can ship. Every other arm holds the tempo
CONSTANT per track (median IBI), isolating octave correctness as the only variable -- the
within-level refinement question is answered separately (+0.1234 CMLt on the 174 non-ambiguous
tracks).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events

METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
SHARP = 0.15
BETA = float(os.environ.get("BETA", "5.0"))
PS = [0.0, 0.10, 0.25, 0.40, 0.55, 0.75, 1.0]
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"


def main():
    dev = torch.device(DEV)
    m = make_beatonly_dbn(30.0, 16, 100, DEV)
    space = m.state_spaces[0]
    iv = space.interval_frames.astype(np.int64)
    idx = {int(v): i for i, v in enumerate(iv)}
    tmap = torch.from_numpy(np.array([idx[int(v)] for v in space.state_interval_frames])).to(dev)
    bf = torch.tensor(space.state_interval_frames, dtype=torch.float64, device=dev)
    s2c = m.state_to_classes[0].to(dev)
    g = iv.astype(np.float64)
    print(f"beta={BETA}  octave-error fractions {PS}", flush=True)

    # decide once per track which tracks are corrupted at each p, nested so the curve is monotone
    rng = np.random.default_rng(0)
    ids_all = track_ids()
    u = {t: rng.random() for t in ids_all}
    flip = {t: (2.0 if rng.random() < 0.5 else 0.5) for t in ids_all}

    per = {p: {} for p in PS}
    per["null"] = {}
    for t in ids_all:
        got = emissions_beatonly(m, bthis_beat_prob(t))
        if got is None:
            continue
        lcd, dec, ff = got
        T = lcd.shape[0]
        gt = np.asarray(load_gt(t), float)
        if len(gt) < 4:
            continue
        ibi = float(np.median(np.diff(gt)) * FPS)          # constant per-track oracle tempo

        def decode(em, stc):
            path, _ = m.dynamic_programs[0].viterbi(
                m.log_initial_distributions[0], m.log_tempo_transition, em,
                state_to_class=stc, return_log_score=True)
            return ev(gt, state_path_to_events(path.cpu().numpy(), space, FPS,
                                               snap_to_activations=dec, first_frame=ff)["beats"])

        per["null"][t] = decode(lcd, s2c)
        for p in PS:
            v = ibi * (flip[t] if u[t] < p else 1.0)
            d = (np.log(g)[None, :] - np.log(v)) / SHARP
            lp = -0.5 * d ** 2
            lp = lp - torch.logsumexp(torch.tensor(lp, device=dev), dim=1, keepdim=True).cpu().numpy()
            ratio = torch.tensor(lp, dtype=torch.float64, device=dev) - np.log(1.0 / len(g))
            ratio = ratio.repeat(T, 1) if ratio.shape[0] == 1 else ratio[:T]
            em = lcd[:, s2c] + BETA * (ratio[:, tmap] / bf.unsqueeze(0))
            per[p][t] = decode(em, None)

    ids = sorted(set.intersection(*[set(per[k]) for k in per]))
    A = lambda r: {k: float(np.mean([r[t][k] for t in ids])) for k in METRICS}
    base = per["null"]
    print(f"\nn={len(ids)}")
    print(f"{'octave err':>11}" + "".join(f"{k:>9}" for k in METRICS) + f"{'dCMLt':>9}{'p':>10}")
    print(f"{'(no tempo)':>11}" + "".join(f"{A(base)[k]:>9.4f}" for k in METRICS))
    out = {}
    for p in PS:
        a = A(per[p])
        d = np.array([per[p][t]["CMLt"] - base[t]["CMLt"] for t in ids])
        pv = wilcoxon(d).pvalue if np.any(d) else 1.0
        frac = float(np.mean([u[t] < p for t in ids]))
        out[p] = {"metrics": a, "dCMLt": float(d.mean()), "p": float(pv), "frac_flipped": frac}
        print(f"{frac:>11.1%}" + "".join(f"{a[k]:>9.4f}" for k in METRICS)
              + f"{d.mean():>+9.4f}{pv:>10.2g}")
    xs = [out[p]["frac_flipped"] for p in PS]
    ys = [out[p]["dCMLt"] for p in PS]
    cross = None
    for i in range(len(xs) - 1):
        if ys[i] > 0 >= ys[i + 1]:
            cross = xs[i] + (xs[i + 1] - xs[i]) * ys[i] / (ys[i] - ys[i + 1])
            break
    print(f"\nbreak-even octave-error rate: {cross:.1%}" if cross else "\nno zero crossing in range")
    print(f"our learned head sits at 54/217 = 24.9% octave-wrong")
    json.dump({"n": len(ids), "beta": BETA, "curve": {str(k): v for k, v in out.items()},
               "null": A(base)}, open("/tmp/overnight/octave_degradation.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
