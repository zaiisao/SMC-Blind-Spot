"""Smooth-target frozen-lambda R2, PER DATASET: learn the transition tolerance separately for
each CV dataset (frozen vanilla frontend, jitter-free target). Reveals each genre's characteristic
tempo-change tolerance and whether the learned lambda beats/ties madmom's hand-set 100 per dataset.

Note: frozen frontend was trained on fold-0-train (spans all datasets); lambda is a property of the
tempo ANNOTATIONS not the frontend, and the learned-vs-100 F comparison is same-activations so the
relative result per dataset is clean. Reports on each dataset's full song set."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import final_eval
import mir_eval
from train_bt import FPS, BT_SHIPPED_DECODE, load_songs, sample_crop
from train_r2_frozen_smooth import smooth_beat_frames, jitter_rate
from rungs.r1_2016_dbn import DBN2016
from crf_baseline import CRFLearnedFactors

DEVICE = "cuda:0"
final_eval.DEVICE = DEVICE
OUT = Path(__file__).resolve().parent
EPOCHS = 10


def learn_lambda(entries, acts, rng):
    r2 = CRFLearnedFactors(fps=FPS, device=DEVICE,
                          observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    opt = torch.optim.Adam(r2.parameters(), lr=0.05)
    for _ in range(EPOCHS):
        rng.shuffle(entries)
        for e in entries:
            s, en = sample_crop(e, rng)
            f0, f1 = e["beat_frames"][s], e["beat_frames"][en]
            a = acts[e["stem"]]
            if f0 < 0 or f1 > a.shape[0] or f1 <= f0:
                continue
            built = r2.annotated_state_path(e["beat_frames"][s:en + 1] - f0,
                                            e["beat_in_bar"][s:en + 1], e["beats_per_bar"])
            if built is None:
                continue
            path, mi = built
            loss = r2.crf_nll(torch.from_numpy(a[f0:f1]).float().to(DEVICE), path, mi) / (f1 - f0)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward(); opt.step()
    return r2.transition_lambda


def score(entries, acts, lam):
    rung = DBN2016(fps=FPS, device=DEVICE, dtype=torch.float32, bounding="none",
                   transition_lambda=lam, **BT_SHIPPED_DECODE)
    fs = []
    for e in entries:
        ev = rung.predict(acts[e["stem"]])
        est = mir_eval.beat.trim_beats(ev["beats"])
        fs.append(mir_eval.beat.f_measure(mir_eval.beat.trim_beats(e["beat_times"]), est)
                  if len(est) else 0.0)
    return float(np.mean(fs))


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    r2_probe = CRFLearnedFactors(fps=FPS, device=DEVICE,
                                observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    train_entries, val_entries, _ = load_songs(r2_probe)
    all_entries = train_entries + val_entries
    for e in all_entries:
        sm = smooth_beat_frames(e["beat_times"])
        keep = len(e["beat_frames"])
        if len(sm) >= keep:
            e["beat_frames"] = sm[len(sm) - keep:]

    by_ds = defaultdict(list)
    for e in all_entries:
        by_ds[e["dataset"]].append(e)

    model = final_eval.load_model(OUT / "vanilla_best_prelim.pt")
    print(f"{'dataset':12s} {'n':>4} {'jump%':>6} {'learned lam':>12} {'F@lam':>8} {'F@100':>8} {'delta':>7}",
          flush=True)
    print("  " + "-" * 62, flush=True)
    for ds in sorted(by_ds):
        entries = by_ds[ds]
        acts = final_eval.activations_for(model, entries)
        lam = learn_lambda(list(entries), acts, rng)
        f_lam = score(entries, acts, lam)
        f_100 = score(entries, acts, 100.0)
        print(f"{ds:12s} {len(entries):>4} {jitter_rate(entries)*100:>5.1f}% {lam:>12.2f} "
              f"{f_lam:>8.4f} {f_100:>8.4f} {f_lam - f_100:>+7.4f}", flush=True)


if __name__ == "__main__":
    main()
