"""Preliminary one-to-one experiment: vanilla Beat Transformer vs Beat Transformer + R2, e2e.

Both arms train the SAME model (official Demixed_DilatedTransformerModel, from scratch), on the
SAME songs (our 4 CV datasets, fold-0 split, restricted to songs whose annotated meter is {3,4}
and representable in the tempo grid -- so neither arm sees data the other cannot use), with the
SAME beat-aligned crops, optimizer recipe (theirs: RAdam+Lookahead, lr 1e-3, clip .5, batch 1,
ReduceLROnPlateau) and epochs. The ONLY difference is the loss/decode pair:

  vanilla : their BCE loss (targets widened twice by maximum_filter1d(size=3)*0.5, summed over
            the 2 channels; tempo-head loss dropped from BOTH arms) ->
            decode with madmom-as-BT-ships-it (obs_lambda=6, num_tempi=None, threshold=0.2)
  r2      : CRF NLL through the exact structured forward (experiments/bt_e2e/crf_baseline.py); learns the
            frontend AND transition_lambda end-to-end ->
            decode with the SAME shipped params but the LEARNED transition_lambda

Usage: PYTHONPATH=. python experiments/bt_e2e/train_bt.py --arm vanilla|r2 --device cuda:0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "beat_transformer" / "code"))

from data.songs import iter_songs                       # noqa: E402
from rungs.r1_2016_dbn import DBN2016                   # noqa: E402
from crf_baseline import CRFLearnedFactors       # noqa: E402
from DilatedTransformer import Demixed_DilatedTransformerModel  # noqa: E402
from optimizer import Lookahead                         # noqa: E402

FPS = 44100 / 1024
DEMIX_ROOT = ROOT / "cache" / "bt_demix"
OUT_ROOT = Path(__file__).resolve().parent
MODEL_KWARGS = dict(attn_len=5, instr=5, ntoken=2, dmodel=256, nhead=8,
                    d_hid=1024, nlayers=9, norm_first=True)
BT_SHIPPED_DECODE = dict(observation_lambda=6, num_tempi=None, threshold=0.2)
MAX_CROP_FRAMES = 700                                    # ~16 s
EVAL_FOLD = 0


def load_songs(r2_probe: CRFLearnedFactors):
    """(train, val) lists of dicts with demixed mel path + frame annotations. Both arms use the
    SAME eligibility filter: demix cache exists, meter in {3,4}, whole-song path representable."""
    train, val, skipped = [], [], 0
    for song in iter_songs():
        if song.fold is None or song.audio_path is None:
            continue
        mel_path = DEMIX_ROOT / song.dataset / f"{song.stem}.npz"
        if not mel_path.exists():
            skipped += 1
            continue
        beat_times, downbeat_times = song.beats()
        if len(beat_times) < 8 or len(downbeat_times) < 2:
            skipped += 1
            continue
        beat_frames = np.round(beat_times * FPS).astype(np.int64)
        is_downbeat = np.isin(np.round(beat_times, 4), np.round(downbeat_times, 4))
        downbeat_indices = np.where(is_downbeat)[0]
        gaps = np.diff(downbeat_indices)
        if len(gaps) == 0:
            skipped += 1
            continue
        beats_per_bar = int(np.median(gaps))
        # positions counted from the nearest previous downbeat
        beat_in_bar = np.zeros(len(beat_frames), dtype=np.int64)
        position = -1
        for i in range(len(beat_frames)):
            position = 0 if is_downbeat[i] else (position + 1 if position >= 0 else -1)
            beat_in_bar[i] = position
        valid_from = downbeat_indices[0]                 # positions defined from first downbeat
        entry = dict(stem=song.stem, dataset=song.dataset, fold=song.fold, mel_path=mel_path,
                     beat_frames=beat_frames[valid_from:], beat_in_bar=beat_in_bar[valid_from:],
                     beats_per_bar=beats_per_bar,
                     beat_times=beat_times, downbeat_times=downbeat_times)
        # eligibility: the whole-song annotated path must be representable (meter + tempo grid)
        if r2_probe.annotated_state_path(entry["beat_frames"], entry["beat_in_bar"],
                                         beats_per_bar) is None:
            skipped += 1
            continue
        (val if song.fold == EVAL_FOLD else train).append(entry)
    return train, val, skipped


def sample_crop(entry, rng):
    """Beat-aligned crop: [start_frame, end_frame) spanning whole beats, <= MAX_CROP_FRAMES."""
    beat_frames = entry["beat_frames"]
    start = rng.integers(0, max(1, len(beat_frames) - 4))
    end = start + 1
    while end + 1 < len(beat_frames) and beat_frames[end + 1] - beat_frames[start] <= MAX_CROP_FRAMES:
        end += 1
    return start, end                                     # beat indices; span >= 1 beat


def widened_targets(beat_frames, is_downbeat_mask, num_frames):
    """Their target construction: one-hot at beat frames, widened twice (1 / .5 / .25 taper)."""
    target = np.zeros((num_frames, 2), dtype=np.float32)
    inside = (beat_frames >= 0) & (beat_frames < num_frames)
    target[beat_frames[inside], 0] = 1.0
    downbeat_frames = beat_frames[inside & is_downbeat_mask]
    target[downbeat_frames, 1] = 1.0
    for c in range(2):
        target[:, c] = np.maximum(target[:, c], maximum_filter1d(target[:, c], size=3) * 0.5)
        target[:, c] = np.maximum(target[:, c], maximum_filter1d(target[:, c], size=3) * 0.5)
    return target



TOLERANCE = 1


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

@torch.no_grad()
def evaluate(model, entries, device, transition_lambda, max_songs=None):
    """Full-song activations -> DBN2016 with BT's shipped decode + the given lambda -> beat F."""
    import mir_eval
    rung = DBN2016(fps=FPS, device=device, dtype=torch.float32, bounding="none",
                   transition_lambda=transition_lambda, **BT_SHIPPED_DECODE)
    model.eval()
    beat_fs, downbeat_fs = [], []
    for entry in entries[:max_songs]:
        x = np.load(entry["mel_path"])["x"]                       # [5, T, 128]
        pred, _ = model(torch.from_numpy(x).unsqueeze(0).to(device))
        probs = torch.sigmoid(pred[0, :, :2]).double().cpu().numpy()
        events = rung.predict(probs)
        ref_b = mir_eval.beat.trim_beats(entry["beat_times"])
        est_b = mir_eval.beat.trim_beats(events["beats"])
        beat_fs.append(mir_eval.beat.f_measure(ref_b, est_b) if len(est_b) else 0.0)
        ref_d = mir_eval.beat.trim_beats(entry["downbeat_times"])
        est_d = mir_eval.beat.trim_beats(events["downbeats"])
        downbeat_fs.append(mir_eval.beat.f_measure(ref_d, est_d) if len(est_d) else 0.0)
    model.train()
    return float(np.mean(beat_fs)), float(np.mean(downbeat_fs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("vanilla", "r2"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-songs", type=int, default=60)
    parser.add_argument("--init-from", default=None, help="checkpoint to resume the model from")
    parser.add_argument("--smooth", action="store_true",
                        help="r2 arm: jitter-free (smoothest-in-tolerance) target path -- the\n"
                             "fix that recovers lambda~98 (see RESULTS.md)")
    args = parser.parse_args()
    device = args.device
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # observation_lambda=6 == the deployment decode (BT shipped); a mismatch co-adapts the
    # learned factors to the wrong observation world (measured, see RESULTS.md).
    r2 = CRFLearnedFactors(fps=FPS, device=device,
                          observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    train_entries, val_entries, skipped = load_songs(r2)
    if args.arm == "r2" and args.smooth:
        for e in train_entries + val_entries:
            sm = smooth_beat_frames(e["beat_times"])
            keep = len(e["beat_frames"])
            if len(sm) >= keep:
                e["beat_frames"] = sm[len(sm) - keep:]
        print("[r2] using SMOOTH (jitter-free) target paths", flush=True)
    print(f"[{args.arm}] train {len(train_entries)} | val {len(val_entries)} | "
          f"skipped {skipped} (no cache / meter / grid)", flush=True)

    model = Demixed_DilatedTransformerModel(**MODEL_KWARGS).to(device).train()
    if args.init_from:
        state = torch.load(args.init_from, map_location="cpu")
        model.load_state_dict(state["model"])
        if args.arm == "r2" and state.get("r2") is not None:
            r2.load_state_dict(state["r2"])
        print(f"[{args.arm}] resumed from {args.init_from} (epoch {state.get('epoch')})",
              flush=True)
    params = list(model.parameters()) + (list(r2.parameters()) if args.arm == "r2" else [])
    optimizer = Lookahead(torch.optim.RAdam(params, lr=1e-3), k=5, alpha=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=2, threshold=1e-3, min_lr=1e-7)
    bce = torch.nn.BCEWithLogitsLoss()

    best = {"beat_f": -1.0}
    history = []
    for epoch in range(args.epochs):
        rng.shuffle(train_entries)
        epoch_loss, epoch_crf, epoch_bce, n_crops, n_bad, n_nan = 0.0, 0.0, 0.0, 0, 0, 0
        t0 = time.time()
        for entry in train_entries:
            start, end = sample_crop(entry, rng)
            f0, f1 = entry["beat_frames"][start], entry["beat_frames"][end]
            x = np.load(entry["mel_path"])["x"][:, f0:f1]          # [5, T, 128]
            if x.shape[1] != f1 - f0:                              # annotation past audio end
                n_bad += 1
                continue
            pred, _ = model(torch.from_numpy(x).unsqueeze(0).to(device))
            logits = pred[0, :, :2]                                # [T, 2]

            crop_beats = entry["beat_frames"][start:end + 1] - f0
            is_db = entry["beat_in_bar"][start:end + 1] == 0
            target = widened_targets(crop_beats, is_db, f1 - f0)
            bce_loss = bce(logits, torch.from_numpy(target).to(device)) * 2  # sum over channels

            if args.arm == "vanilla":
                loss = bce_loss
            else:
                # HYBRID: CRF + BCE anchor. Pure CRF saturates -- it has no calibration pressure,
                # so logits blow past sigmoid saturation (measured: |logit| ~55, 100% of frames,
                # within ONE epoch), where the CRF's own gradient dies and cannot recover. BCE's
                # gradient is MAXIMAL at saturation, so it both prevents and escapes it. This also
                # sharpens the comparison: arm A = BCE, arm B = BCE + exact-forward structure, so
                # the delta IS the DBN term (+ learned lambda).
                built = r2.annotated_state_path(entry["beat_frames"][start:end + 1] - f0,
                                                entry["beat_in_bar"][start:end + 1],
                                                entry["beats_per_bar"])
                if built is None:
                    n_bad += 1
                    continue
                path, meter_index = built
                crf_loss = r2.crf_nll(torch.sigmoid(logits), path, meter_index) / (f1 - f0)
                loss = crf_loss + bce_loss
                epoch_crf += float(crf_loss)
            epoch_bce += float(bce_loss)

            # The authors' own train.py carries a NaN guard -- this model does NaN occasionally,
            # and one unguarded NaN loss poisons the weights permanently. Guard both sides:
            if not torch.isfinite(loss):
                n_nan += 1
                continue
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 0.5)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad()
                n_nan += 1
                continue
            optimizer.step()
            epoch_loss += float(loss)
            n_crops += 1

        if any(not torch.isfinite(p).all() for p in model.parameters()):
            print(f"epoch {epoch:02d}: WEIGHTS NONFINITE -- aborting (guards failed)", flush=True)
            return
        mean_loss = epoch_loss / max(n_crops, 1)
        # Anneal on the BCE component for BOTH arms: stepping on the combined loss let the r2
        # arm's still-falling CRF term keep the plateau scheduler from EVER firing (lr stayed
        # 1e-3 for 30 epochs while vanilla fine-tuned at 2e-4 -- an unfair asymmetry, and most
        # of vanilla's final margin came from its post-anneal phase).
        scheduler.step(epoch_bce / max(n_crops, 1))
        line = (f"epoch {epoch:02d} | loss {mean_loss:.4f} | crops {n_crops} "
                f"(bad {n_bad}, nan {n_nan}) | "
                f"{time.time() - t0:.0f}s | lr {optimizer.param_groups[0]['lr']:.2e}")
        if args.arm == "r2":
            line += f" | crf {epoch_crf / max(n_crops, 1):.4f} | lambda {r2.transition_lambda:.2f}"
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            lam = r2.transition_lambda if args.arm == "r2" else 100.0
            beat_f, downbeat_f = evaluate(model, val_entries, device, lam,
                                          max_songs=args.eval_songs)
            line += f" | val beatF {beat_f:.4f} dbF {downbeat_f:.4f}"
            if beat_f > best["beat_f"]:
                best = {"beat_f": beat_f, "downbeat_f": downbeat_f, "epoch": epoch,
                        "lambda": lam}
                torch.save({"model": model.state_dict(),
                            "r2": r2.state_dict() if args.arm == "r2" else None,
                            "epoch": epoch, "beat_f": beat_f},
                           OUT_ROOT / f"{args.arm}_best.pt")
        print(line, flush=True)
        history.append(line)
        (OUT_ROOT / f"{args.arm}_history.json").write_text(json.dumps(
            {"best": best, "history": history}, indent=1))

    print(f"BEST [{args.arm}]: {best}", flush=True)


if __name__ == "__main__":
    main()
