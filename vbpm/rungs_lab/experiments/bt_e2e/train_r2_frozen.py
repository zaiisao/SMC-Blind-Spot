"""PURE ladder R2: frozen frontend (yesterday's best vanilla checkpoint), learn ONLY the
transition_lambda by CRF through the exact forward. Decouples the two effects that the e2e arm
bundles: does the LEARNED FACTOR alone beat madmom's hand-set 100 at deployment decode?

Frontend frozen -> activations precomputed once per song; epochs cost seconds."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "beat_transformer" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mir_eval                                          # noqa: E402
from rungs.r1_2016_dbn import DBN2016                    # noqa: E402
from crf_baseline import CRFLearnedFactors        # noqa: E402
import final_eval                                        # noqa: E402
from train_bt import FPS, BT_SHIPPED_DECODE, load_songs, sample_crop  # noqa: E402

DEVICE = "cuda:3"
final_eval.DEVICE = DEVICE                               # its helpers pin to a module-level device
load_model, activations_for = final_eval.load_model, final_eval.activations_for
OUT = Path(__file__).resolve().parent
EPOCHS = 8


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    r2 = CRFLearnedFactors(fps=FPS, device=DEVICE,
                          observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    train_entries, val_entries, _ = load_songs(r2)
    print(f"train {len(train_entries)} | val {len(val_entries)}", flush=True)

    model = load_model(OUT / "vanilla_best_prelim.pt").to(DEVICE)
    print("precomputing frozen activations...", flush=True)
    train_acts = activations_for(model, train_entries)
    val_acts = activations_for(model, val_entries)

    optimizer = torch.optim.Adam(r2.parameters(), lr=0.05)
    for epoch in range(EPOCHS):
        rng.shuffle(train_entries)
        total, n = 0.0, 0
        for entry in train_entries:
            start, end = sample_crop(entry, rng)
            f0, f1 = entry["beat_frames"][start], entry["beat_frames"][end]
            acts = train_acts[entry["stem"]]
            if f1 > acts.shape[0]:
                continue
            built = r2.annotated_state_path(entry["beat_frames"][start:end + 1] - f0,
                                            entry["beat_in_bar"][start:end + 1],
                                            entry["beats_per_bar"])
            if built is None:
                continue
            path, meter_index = built
            probs = torch.from_numpy(acts[f0:f1]).float().to(DEVICE)   # frozen: no grad to acts
            loss = r2.crf_nll(probs, path, meter_index) / (f1 - f0)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)
            n += 1
        print(f"epoch {epoch} | crf {total / max(n, 1):.4f} | lambda {r2.transition_lambda:.2f}",
              flush=True)

    lam = r2.transition_lambda
    print(f"\nlearned lambda (frozen frontend, obs=6): {lam:.2f}", flush=True)
    for lam_name, lam_value in (("hand-set 100", 100.0), (f"learned {lam:.1f}", lam)):
        rung = DBN2016(fps=FPS, device=DEVICE, dtype=torch.float32, bounding="none",
                       transition_lambda=lam_value, **BT_SHIPPED_DECODE)
        beat_fs, downbeat_fs = [], []
        for e in val_entries:
            events = rung.predict(val_acts[e["stem"]])
            est_b = mir_eval.beat.trim_beats(events["beats"])
            beat_fs.append(mir_eval.beat.f_measure(
                mir_eval.beat.trim_beats(e["beat_times"]), est_b) if len(est_b) else 0.0)
            est_d = mir_eval.beat.trim_beats(events["downbeats"])
            downbeat_fs.append(mir_eval.beat.f_measure(
                mir_eval.beat.trim_beats(e["downbeat_times"]), est_d) if len(est_d) else 0.0)
        print(f"frozen acts + lambda {lam_name:14s}: beatF {np.mean(beat_fs):.4f}  "
              f"dbF {np.mean(downbeat_fs):.4f}", flush=True)


if __name__ == "__main__":
    main()
