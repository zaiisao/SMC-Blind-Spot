"""Does generated audio actually place beats where the conditioning said?

The single measurement everything downstream depends on, so it is built to
resist the two ways it could flatter us.

NON-CIRCULARITY. No beat tracker is involved anywhere. Adherence is measured by
``sa3_beat_adherence`` from signal evidence only -- running a tracker over the
audio would certify our training labels with the model whose weakness created
the need for them, and it is least reliable on exactly the soft material we are
synthesising.

HELD-OUT GRIDS. ``--grids real`` conditions on the human annotations of tracks
in the EVALUATION fold, which no generator in this experiment has seen. That
matters because the synthetic rubato grids come from ``rate_curve``, the same
function the warped training corpus is built with -- passing on those would show
compliance with one curve family, not rubato in general.

Controls are redrawn per generation rather than fixed: a single scrambled grid
reused across samples can align with dense onsets by chance (observed at +0.31
on one run, against a calibrated corpus value of -0.002).
"""

import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "sa3"))
CACHE = Path("/disk1/jaehoon/sa3_beat_cache")


def controls(beats, rng, seconds):
    """True grid plus three nulls that isolate different failure modes."""
    ibi = np.median(np.diff(beats))
    return {
        "true": beats,
        # phase error: same tempo, wrong phase. Signed measure must go NEGATIVE.
        "half_beat": beats + ibi / 2,
        # no metrical structure at all -> chance.
        "scrambled": np.sort(rng.uniform(beats[0], beats[-1], len(beats))),
        # regular but wrong rate. If this beats `true`, the model is producing a
        # steady pulse and ignoring the requested timing.
        "wrong_tempo": np.arange(beats[0], beats[-1], ibi * 1.37),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--grids", default="all",
                    choices=["all", "metronomic", "rubato", "real"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--cfg-scale", type=float, default=6.0)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--prompt", default="solo piano, quiet, expressive, intimate recording")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from sa3_beat.build import load_beat_model
    from sa3_beat.generate import (generate, isochronous, load_adapter, rubato,
                                   _patch_conditioning_inputs)
    from sa3_beat.data import load_beats
    from sa3_beat_adherence import beat_adherence

    dev = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model, cfg = load_beat_model(device=dev)
    load_adapter(model, args.adapter, dev)
    _patch_conditioning_inputs(model)

    grids = []
    if args.grids in ("all", "metronomic"):
        grids.append(("metronomic_100bpm", *isochronous(100, args.seconds)))
    if args.grids in ("all", "rubato"):
        grids.append(("rubato_d0.3", *rubato(100, args.seconds, depth=0.3)))
    if args.grids in ("all", "real"):
        # human annotations from the held-out fold -- unseen, and not from our
        # own curve family.
        held = [l.strip()[:-4] for l in open(CACHE / f"holdout_fold{args.fold}.txt")
                if l.strip().startswith(("smc__", "asap__"))]
        for uid in held[:3]:
            b, db = load_beats(CACHE / f"{uid}.beats")
            b = b[(b >= 0) & (b < args.seconds)]
            if len(b) >= 8:
                db = None if db is None else db[(db >= 0) & (db < args.seconds)]
                grids.append((f"real:{uid[:28]}", b, db))

    results = {}
    for tag, bts, dbs in grids:
        sc = {k: [] for k in ("true", "half_beat", "scrambled", "wrong_tempo")}
        for i in range(args.n):
            wav, _ = generate(model, cfg, args.prompt, bts, dbs, args.seconds,
                              args.steps, args.cfg_scale, seed=1000 + i, device=dev)
            y = wav.mean(axis=0); y = y / (np.abs(y).max() + 1e-9)
            rng = np.random.RandomState(7000 + i)     # redrawn per generation
            for k, g in controls(np.asarray(bts), rng, args.seconds).items():
                g = np.asarray(g); g = g[(g >= 0) & (g < args.seconds)]
                if len(g) >= 4:
                    sc[k].append(beat_adherence(y, g)["phase_align"])
        m = {k: float(np.mean(v)) for k, v in sc.items()}
        se = float(np.std(sc["true"], ddof=1) / np.sqrt(len(sc["true"])))
        results[tag] = {"mean": m, "se_true": se, "n": args.n}
        print(f"{tag:34s} true={m['true']:+.4f}+/-{se:.4f}  half={m['half_beat']:+.4f}  "
              f"scram={m['scrambled']:+.4f}  wrong={m['wrong_tempo']:+.4f}", flush=True)

    print("\nreference: real SMC audio on its own annotations = +0.2301; chance = 0.000")
    if args.out:
        json.dump({"adapter": args.adapter, "args": vars(args), "results": results},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
