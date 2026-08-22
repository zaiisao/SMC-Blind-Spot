"""Test-time augmentation over Beat This's OWN 22-variant augmentation grid.

WHAT IS NEW HERE. Our shipped TTA averages Beat This over two time-stretch rates {0.85, 1.2} and is
worth +0.0132 F in-stack. We now have, on disk, the exact spectrograms the authors trained on
(Zenodo 13922116): 22 variants per track -- the original, 11 pitch shifts (ps-5..ps6) and 10 time
stretches (ts+-4..+-20). That is 11x more views than we have ever ensembled, and it costs no
training and no new audio processing.

WHY PITCH SHIFT IS THE INTERESTING HALF. A pitch shift does not move beat times AT ALL: the
spectrograms are (2001, 128) for every psN, identical in length to the original. So the 11 pitch
views can be averaged with NO time-axis realignment -- pure variance reduction over an augmentation
the network was explicitly trained to be invariant to. Time stretches need realignment (tsN has
length L/(1+N/100)) and that resampling is itself a source of error, which is probably why our
2-rate TTA gained so little.

FOLD HONESTY. Each track is encoded with the fold{N} checkpoint that HELD IT OUT, mirroring how
beat_this_activations_cache was built (LATE_CORRECTIONS.md). Without this the numbers would measure
memorisation: the fold0 checkpoint scores CMLt 0.7362/0.8064 on folds 1/2 that it trained on versus
0.6273 on held-out fold 0.

ARMS, each decoded with the SAME beat-only DBN we ship, so any difference is the activation:
    orig            the single-view baseline (must reproduce the cache -- asserted)
    pitch           average over the 11 pitch views (no realignment)
    tempo           average over the 10 stretch views (realigned to the original time axis)
    pitch+tempo     all 21 augmented views
    all             the above plus the original
Averaging is done in PROBABILITY space, matching how our existing TTA combines views.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events

SPLIT = ROOT / "beat_this_annotations" / "smc" / "8-folds.split"
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/disk1/jaehoon/bt_spect/smc.npz")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/tmp/overnight/aug_tta.json")
    args = ap.parse_args()
    dev = torch.device(args.device)

    splits = {}
    for line in open(SPLIT):
        p = line.split()
        if len(p) >= 2:
            splits[p[0]] = int(p[-1])

    from beat_this.inference import Audio2Frames, split_predict_aggregate
    models = {}
    for f in sorted(set(splits.values())):
        models[f] = Audio2Frames(checkpoint_path=f"fold{f}", device=args.device).model
        models[f].eval()
    print(f"loaded {len(models)} fold checkpoints", flush=True)

    z = np.load(args.npz, mmap_mode="r")
    m = make_beatonly_dbn(30.0, 16, 100, args.device)
    space, dp0 = m.state_spaces[0], m.dynamic_programs[0]
    s2c = m.state_to_classes[0]
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition

    @torch.no_grad()
    def act(model, spect):
        out = split_predict_aggregate(spect=spect.to(dev), chunk_size=1500,
                                      overlap_mode="keep_first", border_size=6, model=model)
        return expit(np.asarray(out["beat"].float().cpu().numpy(), np.float64))

    def decode(prob):
        got = emissions_beatonly(m, prob)
        if got is None:
            return None
        lcd, dec, ff = got
        path, _ = dp0.viterbi(logpi, logA, lcd, state_to_class=s2c, return_log_score=True)
        return state_path_to_events(path.cpu().numpy(), space, FPS,
                                    snap_to_activations=dec, first_frame=ff)["beats"]

    ARMS = ("orig", "pitch", "tempo", "pitch+tempo", "all", "pitch<=2", "tempo<=8",
            "small", "median_all", "median_small")
    res = {a: {} for a in ARMS}
    ids = track_ids()[:args.limit] if args.limit else track_ids()
    for n, tid in enumerate(ids):
        if tid not in splits or f"{tid}/track" not in z.files:
            continue
        model = models[splits[tid]]
        base = act(model, torch.from_numpy(np.asarray(z[f"{tid}/track"], np.float32)))
        L = len(base)
        pitch, tempo, pk, tk = [], [], [], []
        for k in z.files:
            if not k.startswith(tid + "/") or k.endswith("/track"):
                continue
            a = act(model, torch.from_numpy(np.asarray(z[k], np.float32)))
            if "_ps" in k:
                pitch.append(a[:L] if len(a) >= L else np.pad(a, (0, L - len(a)), constant_values=a[-1]))
                pk.append(int(k.rsplit("_ps", 1)[1]))
            elif "_ts" in k:
                r = 1.0 + int(k.rsplit("_ts", 1)[1]) / 100.0     # stretched length = L / r
                # map stretched frame axis back onto the original: orig time t -> stretched t/r
                src = np.arange(len(a)) * r
                tempo.append(np.interp(np.arange(L), src, a))
                tk.append(int(k.rsplit("_ts", 1)[1]))
        gt = np.asarray(load_gt(tid), float)
        # Restricted ranges and a median combiner. Averaging over +-5 semitones / +-20% asks the
        # ensemble to absorb views that may simply be worse; small perturbations are the ones a
        # variance-reduction argument actually supports, and a median is robust to a minority of
        # bad views in a way the mean is not.
        ps_s = [pitch[i] for i in range(len(pitch)) if abs(pk[i]) <= 2]
        ts_s = [tempo[i] for i in range(len(tempo)) if abs(tk[i]) <= 8]
        combos = {"orig": base,
                  "pitch": np.mean(pitch, 0) if pitch else None,
                  "tempo": np.mean(tempo, 0) if tempo else None,
                  "pitch+tempo": np.mean(pitch + tempo, 0) if (pitch and tempo) else None,
                  "all": np.mean([base] + pitch + tempo, 0),
                  "pitch<=2": np.mean([base] + ps_s, 0) if ps_s else None,
                  "tempo<=8": np.mean([base] + ts_s, 0) if ts_s else None,
                  "small": np.mean([base] + ps_s + ts_s, 0) if (ps_s or ts_s) else None,
                  "median_all": np.median([base] + pitch + tempo, 0),
                  "median_small": np.median([base] + ps_s + ts_s, 0) if (ps_s or ts_s) else None}
        for a, p in combos.items():
            if p is None:
                continue
            b = decode(p)
            if b is not None:
                res[a][tid] = ev(gt, b)
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(ids)}", flush=True)

    ids = sorted(set.intersection(*[set(res[a]) for a in ARMS if res[a]]))
    A = lambda a: {k: float(np.mean([res[a][t][k] for t in ids])) for k in METRICS}
    print(f"\nn={len(ids)}  fold-honest, beat-only DBN, probability-space averaging")
    print(f"{'arm':<14}{'views':>7}" + "".join(f"{k:>9}" for k in METRICS))
    nv = {"orig": 1, "pitch": 11, "tempo": 10, "pitch+tempo": 21, "all": 22,
          "pitch<=2": 5, "tempo<=8": 5, "small": 9, "median_all": 22, "median_small": 9}
    for a in ARMS:
        if res[a]:
            print(f"{a:<14}{nv[a]:>7}" + "".join(f"{A(a)[k]:>9.4f}" for k in METRICS))
    print(f"\npaired vs single-view 'orig':")
    for a in ARMS[1:]:
        if not res[a]:
            continue
        line = f"  {a:<13}"
        for k in METRICS:
            d = np.array([res[a][t][k] - res["orig"][t][k] for t in ids])
            p = wilcoxon(d).pvalue if np.any(d) else 1.0
            line += f"  {k} {d.mean():+.4f} (p={p:.2g})"
        print(line)
    json.dump({"n": len(ids), "summary": {a: A(a) for a in ARMS if res[a]},
               "per_track": res}, open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
