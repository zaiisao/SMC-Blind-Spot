"""Fuse Beat This and Beat Transformer BEAT ACTIVATIONS before decoding.

WHY THIS AND NOT SOMETHING WE ALREADY DID. Per-track ORACLE selection between the two frontends
scores 0.6978 F against our best no-oracle stack's 0.6389 -- so ~0.059 F of headroom demonstrably
exists in combining them. Two previous attempts did NOT test this:
  * Beat Transformer's DOWNBEAT channel fed into the joint DBN: neutral (+0.0025 CMLt, p=0.64).
    That swapped the downbeat evidence, not the beat evidence.
  * Path-score ARBITRATION -- decode both, keep the higher-scoring Viterbi path -- failed, and for
    a structural reason: path scores are not comparable across different tempo hypotheses, since a
    slow-tempo path accumulates a different number of emission terms than a fast one.
Combining the beat activations THEMSELVES, then decoding once, is untried.

THE GRID PROBLEM, which is the main thing that could silently ruin this. Beat This runs at 50 fps
(2001 frames per 40 s clip); Beat Transformer runs at 44100/1024 = 43.066 fps (1727 frames). Any
fusion needs BT resampled onto the 50 fps grid, and a resampling error of even one frame is 20 ms --
a third of the F-measure tolerance. resample_to() in _harness.py is linear interpolation on the true
time axes, so alignment is exact up to interpolation; the ALIGNMENT CHECK arm below verifies this
empirically by cross-correlating the two activations and reporting the lag that maximises it. A
nonzero systematic lag would mean every fusion number here is measuring a misalignment, not a
disagreement between models.

COMBINERS. Probability-space unless noted. w sweeps the mixture so we can see whether any weight
beats both singles, rather than testing one arbitrary blend:
    bthis / btrans      the two singles, for reference
    mean_w              w*bthis + (1-w)*btrans
    geo_w               exp(w*log bthis + (1-w)*log btrans)  -- log-space, penalises disagreement
    max / min           elementwise
Everything is decoded with the SAME beat-only DBN we ship, so any difference is the activation.

NO ORACLE ANYWHERE: w is a single global constant, not chosen per track. The oracle number is quoted
only as the ceiling this is aiming at.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import (FPS, BT_FPS, track_ids, load_gt, ev, bthis_beat_prob,
                      btrans_beat_prob, resample_to)
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events

METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
EPS = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="/tmp/overnight/fusion.json")
    args = ap.parse_args()
    m = make_beatonly_dbn(30.0, 16, 100, args.device)
    space, dp0 = m.state_spaces[0], m.dynamic_programs[0]
    s2c = m.state_to_classes[0]
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition

    def decode(prob):
        got = emissions_beatonly(m, np.clip(prob, EPS, 1 - EPS))
        if got is None:
            return None
        lcd, dec, ff = got
        path, _ = dp0.viterbi(logpi, logA, lcd, state_to_class=s2c, return_log_score=True)
        return state_path_to_events(path.cpu().numpy(), space, FPS,
                                    snap_to_activations=dec, first_frame=ff)["beats"]

    WS = (0.3, 0.4, 0.5, 0.6, 0.7)
    ARMS = ["bthis", "btrans"] + [f"mean_{w}" for w in WS] + [f"geo_{w}" for w in WS] \
        + ["max", "min"]
    res = {a: {} for a in ARMS}
    lags = []
    for tid in track_ids():
        a = np.asarray(bthis_beat_prob(tid), np.float64)
        try:
            b0 = np.asarray(btrans_beat_prob(tid), np.float64)
        except Exception:
            continue
        b = resample_to(b0, BT_FPS, FPS, len(a))
        # ALIGNMENT CHECK: lag (in frames) maximising cross-correlation of the two activations.
        # Should be ~0; a systematic nonzero lag means we are fusing misaligned signals.
        x, y = a - a.mean(), b - b.mean()
        cc = np.correlate(x, y, mode="full")
        lags.append(int(np.argmax(cc) - (len(x) - 1)))
        gt = np.asarray(load_gt(tid), float)
        la, lb = np.log(np.clip(a, EPS, 1)), np.log(np.clip(b, EPS, 1))
        combos = {"bthis": a, "btrans": b, "max": np.maximum(a, b), "min": np.minimum(a, b)}
        for w in WS:
            combos[f"mean_{w}"] = w * a + (1 - w) * b
            combos[f"geo_{w}"] = np.exp(w * la + (1 - w) * lb)
        for k, p in combos.items():
            e = decode(p)
            if e is not None:
                res[k][tid] = ev(gt, e)

    lags = np.array(lags)
    print(f"ALIGNMENT: cross-correlation lag median {np.median(lags):+.0f} frames "
          f"({1000*np.median(lags)/FPS:+.0f} ms), mean {lags.mean():+.2f}, "
          f"|lag|<=1 on {100*(np.abs(lags) <= 1).mean():.0f}% of tracks")
    ids = sorted(set.intersection(*[set(res[a]) for a in ARMS if res[a]]))
    A = lambda a: {k: float(np.mean([res[a][t][k] for t in ids])) for k in METRICS}
    print(f"\nn={len(ids)}   beat-only DBN, no oracle (w is a single global constant)")
    print(f"{'arm':<12}" + "".join(f"{k:>9}" for k in METRICS) + f"{'dF vs bthis':>13}{'p':>9}")
    for a in ARMS:
        if not res[a]:
            continue
        d = np.array([res[a][t]["F"] - res["bthis"][t]["F"] for t in ids])
        p = wilcoxon(d).pvalue if np.any(d) else 1.0
        print(f"{a:<12}" + "".join(f"{A(a)[k]:>9.4f}" for k in METRICS)
              + f"{d.mean():>+13.4f}{p:>9.2g}")
    # the ceiling this is aiming at, recomputed here rather than quoted from memory
    orc = np.mean([max(res["bthis"][t]["F"], res["btrans"][t]["F"]) for t in ids])
    print(f"\nper-track ORACLE selection between the two singles: F {orc:.4f}  "
          f"(+{orc - A('bthis')['F']:.4f} over Beat This alone) -- the ceiling, needs ground truth")
    json.dump({"n": len(ids), "summary": {a: A(a) for a in ARMS if res[a]},
               "oracle_F": float(orc), "per_track": res}, open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
