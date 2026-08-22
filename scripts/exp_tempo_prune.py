"""HARD per-track tempo pruning: restrict the DBN's state space to a band around an estimated tempo.

WHY THIS IS NOT THE EXPERIMENT WE ALREADY RAN. Per-track tempo has been injected before as a SOFT
emission term (diag_tempo_head_bracket.py, eval_tempo_head_dbn.py): add beta*log q(tempo) to each
state's score. That failed for two understood reasons -- the tempo is largely redundant with what the
beat activation already encodes combinatorially, and Viterbi's argmax turns a score shift into a step
function, so the four heads we built (48.8%-57.6% track accuracy) all gave the same +0.010-0.017 CMLt
with no monotone relation to accuracy.

A HARD prune is a different operation: it deletes wrong-metrical-level paths before Viterbi sees
them, rather than making them slightly more expensive. It is not subject to the redundancy argument.

WHY IT MIGHT STILL FAIL, stated up front so the result is not a surprise: our best tempo estimator is
57.6% accurate per track, so a narrow band excludes the TRUE tempo on ~40% of tracks and the decoder
then cannot recover at all -- the soft version at least degrades gracefully. The band ratio r is
therefore swept from generous to tight, and the ORACLE arm bounds what any estimator could buy.

IMPLEMENTATION. Rather than rebuilding a state space per track (expensive), the wide [30, 215] DBN is
kept and states whose tempo lies outside the band get a large negative additive mask -- identical in
effect for Viterbi, and the same mechanism as the MMI numerator mask. A finite penalty, not -inf, so
that a track whose band excludes every reachable state still decodes something rather than producing
an all--inf row.

TEMPO SOURCES:
  oracle    60 / median(diff(annotations))          -- the ceiling; needs ground truth
  twopass   decode with the FULL range, take the median IBI of the output, re-decode in the band
            -- fully self-contained, no external estimator, no labels
  head      our trained tempo head (57.6% track accuracy), the realistic external estimator
Bands are octave-agnostic: [t/r, t*r]. A track whose true tempo is an octave from the estimate is
lost by construction, which is exactly the risk being measured.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob, btrans_tempo
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events

METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
PENALTY = -1e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", default="1.15,1.3,1.5,2.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/tmp/overnight/tempo_prune.json")
    args = ap.parse_args()
    m = make_beatonly_dbn(30.0, 16, 100, args.device)
    space, dp0 = m.state_spaces[0], m.dynamic_programs[0]
    s2c = m.state_to_classes[0]
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition
    sif = torch.from_numpy(np.asarray(space.state_interval_frames, np.float64)).to(s2c.device)
    state_bpm = 60.0 * FPS / sif                                   # tempo of every state

    def decode(lcd, dec, ff, band=None):
        em = lcd[:, s2c]                                            # [T, num_states]
        if band is not None:
            lo, hi = band
            bad = (state_bpm < lo) | (state_bpm > hi)
            if bool(bad.all()):
                return None
            em = em + torch.where(bad, torch.full_like(state_bpm, PENALTY),
                                  torch.zeros_like(state_bpm)).unsqueeze(0)
        path, _ = dp0.viterbi(logpi, logA, em, state_to_class=None, return_log_score=True)
        return state_path_to_events(path.cpu().numpy(), space, FPS,
                                    snap_to_activations=dec, first_frame=ff)["beats"]

    RS = [float(x) for x in args.ratios.split(",")]
    SRCS = ("oracle", "twopass", "btrans", "consensus")
    arms = {"global": {}}
    for src in SRCS:
        for r in RS:
            arms[f"{src}_r{r}"] = {}
    acc = {s: [] for s in SRCS}
    n_gate = 0

    for tid in track_ids():
        got = emissions_beatonly(m, np.asarray(bthis_beat_prob(tid), np.float64))
        if got is None:
            continue
        lcd, dec, ff = got
        gt = np.asarray(load_gt(tid), float)
        if len(gt) < 4:
            continue
        base = decode(lcd, dec, ff)
        if base is None:
            continue
        arms["global"][tid] = ev(gt, base)

        t_or = 60.0 / float(np.median(np.diff(gt)))
        t_tp = (60.0 / float(np.median(np.diff(base))) if len(base) >= 4 else t_or)
        try:
            t_bt = float(btrans_tempo(tid)[0])
        except Exception:
            t_bt = t_tp
        # CONSENSUS GATE: prune only where two INDEPENDENT estimators agree within 10%. The
        # twopass/btrans arms cancel almost exactly (gains on tracks they get right, catastrophic
        # losses on octave errors), so the question is whether the damaged tracks can be identified
        # WITHOUT labels. Agreement between a decoder-derived estimate and a learned one is the
        # cheapest such signal. Where they disagree, fall back to the unpruned global range.
        agree = abs(np.log2(max(t_bt, 1e-6) / max(t_tp, 1e-6))) < 0.1375       # +-10%
        n_gate += int(agree)
        for s, t in (("oracle", t_or), ("twopass", t_tp), ("btrans", t_bt)):
            acc[s].append(float(abs(np.log2(max(t, 1e-6) / t_or)) < 0.0704))   # +-5%
        acc["consensus"].append(float(agree and abs(np.log2(max(t_bt, 1e-6) / t_or)) < 0.0704))
        for src, t_est in (("oracle", t_or), ("twopass", t_tp), ("btrans", t_bt),
                           ("consensus", t_bt)):
            for r in RS:
                band = (t_est / r, t_est * r) if (src != "consensus" or agree) else None
                e = decode(lcd, dec, ff, band)
                if e is not None:
                    arms[f"{src}_r{r}"][tid] = ev(gt, e)

    ids = sorted(set.intersection(*[set(v) for v in arms.values() if v]))
    A = lambda a: {k: float(np.mean([arms[a][t][k] for t in ids])) for k in METRICS}
    print(f"n={len(ids)}   beat-only DBN, states outside the band masked")
    for s_ in SRCS:
        print(f"  {s_:<10} tempo within +-5% of truth on {100*np.mean(acc[s_]):.1f}% of tracks")
    print(f"  consensus gate fired (estimators agree within 10%) on {n_gate}/{len(acc['twopass'])} tracks")
    print(f"\n{'arm':<16}" + "".join(f"{k:>9}" for k in METRICS) + f"{'dF':>10}{'p':>9}")
    for a in arms:
        if not arms[a]:
            continue
        d = np.array([arms[a][t]["F"] - arms["global"][t]["F"] for t in ids])
        p = wilcoxon(d).pvalue if np.any(d) else 1.0
        print(f"{a:<16}" + "".join(f"{A(a)[k]:>9.4f}" for k in METRICS)
              + f"{d.mean():>+10.4f}{p:>9.2g}")
    json.dump({"n": len(ids), "summary": {a: A(a) for a in arms if arms[a]},
               "twopass_acc": float(np.mean(acc["twopass"])), "per_track": arms},
              open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
