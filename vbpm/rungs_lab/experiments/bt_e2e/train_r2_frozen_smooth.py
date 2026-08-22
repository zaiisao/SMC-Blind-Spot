"""Frozen-frontend R2 with a JITTER-FREE target path (the user's tolerance insight).

Per-beat rounding of annotation times onto the frame grid manufactures fake tempo jumps (57% of
boundaries jump +-1..2 frames; measured). Deployment already forgives sub-region timing
(correct=True snaps to peaks; F gives 70 ms), so the training target should too: choose each
beat's frame WITHIN +-tol of its rounded position such that the resulting interval sequence is
the SMOOTHEST representable one (min total |interval change|), then train lambda on that path.

Prediction under the jitter hypothesis: learned lambda rises substantially from ~10; where it
lands measures the data's MUSICAL tempo variability."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import final_eval
from train_bt import FPS, BT_SHIPPED_DECODE, load_songs, sample_crop
from rungs.r1_2016_dbn import DBN2016
from crf_baseline import CRFLearnedFactors
import mir_eval

DEVICE = "cuda:0"
final_eval.DEVICE = DEVICE
OUT = Path(__file__).resolve().parent
EPOCHS = 8
TOLERANCE = 1                # frames; ~23 ms at 43 fps, well inside F-measure's 70 ms


def smooth_beat_frames(beat_times, tol=TOLERANCE):
    """Smoothest-in-tolerance quantization: integer frames f_i with |f_i - round(t_i*FPS)| <= tol
    minimizing sum |interval_i - interval_{i-1}| (DP over per-beat offsets)."""
    base = np.round(beat_times * FPS).astype(np.int64)
    offsets = list(range(-tol, tol + 1))
    states = {(d, None): (0.0, None) for d in offsets}   # (offset, prev interval) -> (cost, back)
    back_tables = []
    for i in range(1, len(base)):
        gap = int(base[i] - base[i - 1])
        new_states = {}
        for (d_prev, k_prev), (cost, _) in states.items():
            for d_next in offsets:
                interval = gap + d_next - d_prev
                if interval < 2:
                    continue
                step = cost + (abs(interval - k_prev) if k_prev is not None else 0.0)
                key = (d_next, interval)
                if key not in new_states or step < new_states[key][0]:
                    new_states[key] = (step, (d_prev, k_prev))
        if not new_states:                                # pathological; give up on smoothing
            return base
        states = new_states
        back_tables.append({k: v[1] for k, v in states.items()})
    best_key = min(states, key=lambda k: states[k][0])
    chosen = [best_key]
    for table in reversed(back_tables):
        chosen.append(table[chosen[-1]])
    chosen.reverse()
    return base + np.array([c[0] for c in chosen], dtype=np.int64)


def jitter_rate(entries):
    jumps = np.concatenate([np.abs(np.diff(np.diff(e["beat_frames"]))) for e in entries])
    return float((jumps > 0).mean())


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    r2 = CRFLearnedFactors(fps=FPS, device=DEVICE,
                          observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    train_entries, val_entries, _ = load_songs(r2)

    before = jitter_rate(train_entries)
    for e in train_entries + val_entries:
        smoothed = smooth_beat_frames(e["beat_times"])
        keep = len(e["beat_frames"])                 # load_songs cut to start at first downbeat
        if len(smoothed) >= keep:
            e["beat_frames"] = smoothed[len(smoothed) - keep:]
    print(f"boundary-jump rate: before {before:.3f} -> after {jitter_rate(train_entries):.3f}",
          flush=True)

    model = final_eval.load_model(OUT / "vanilla_best_prelim.pt")
    print("precomputing frozen activations...", flush=True)
    train_acts = final_eval.activations_for(model, train_entries)
    val_acts = final_eval.activations_for(model, val_entries)

    optimizer = torch.optim.Adam(r2.parameters(), lr=0.05)
    for epoch in range(EPOCHS):
        rng.shuffle(train_entries)
        total, n = 0.0, 0
        for entry in train_entries:
            start, end = sample_crop(entry, rng)
            f0, f1 = entry["beat_frames"][start], entry["beat_frames"][end]
            acts = train_acts[entry["stem"]]
            if f0 < 0 or f1 > acts.shape[0] or f1 <= f0:
                continue
            built = r2.annotated_state_path(entry["beat_frames"][start:end + 1] - f0,
                                            entry["beat_in_bar"][start:end + 1],
                                            entry["beats_per_bar"])
            if built is None:
                continue
            path, meter_index = built
            probs = torch.from_numpy(acts[f0:f1]).float().to(DEVICE)
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
    print(f"\nlearned lambda (frozen frontend, SMOOTH target, obs=6): {lam:.2f}", flush=True)
    for lam_name, lam_value in (("hand-set 100", 100.0), (f"learned {lam:.1f}", lam)):
        rung = DBN2016(fps=FPS, device=DEVICE, dtype=torch.float32, bounding="none",
                       transition_lambda=lam_value, **BT_SHIPPED_DECODE)
        beat_fs = []
        for e in val_entries:
            events = rung.predict(val_acts[e["stem"]])
            est = mir_eval.beat.trim_beats(events["beats"])
            beat_fs.append(mir_eval.beat.f_measure(
                mir_eval.beat.trim_beats(e["beat_times"]), est) if len(est) else 0.0)
        print(f"frozen acts + lambda {lam_name:14s}: beatF {np.mean(beat_fs):.4f}", flush=True)


if __name__ == "__main__":
    main()
