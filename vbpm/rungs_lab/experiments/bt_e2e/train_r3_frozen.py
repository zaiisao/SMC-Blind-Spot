"""Frozen-frontend R3: learn the per-frame conditioning net on frozen vanilla activations, decode
with R3's own time-varying Viterbi. Clean test of per-frame vs global lambda -- the GLOBAL baseline
is R3 with the net at init (lambda_t == lambda_base everywhere), decoded through the SAME bare path,
so the only variable is per-frame conditioning. Smooth (jitter-free) targets throughout."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import final_eval
import mir_eval
from train_bt import FPS, BT_SHIPPED_DECODE, load_songs, sample_crop, smooth_beat_frames
from rungs.r3_conditioned_dbn import R3ConditionedFactors

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--lambda-base', type=float, default=100.0)
_ap.add_argument('--lambda-reg', type=float, default=0.0,
                 help='pin mean(log lam_t) to log(lambda_base): net only REDISTRIBUTES')
_ap.add_argument('--device', default='cuda:0')
_args, _ = _ap.parse_known_args()
DEVICE = _args.device
final_eval.DEVICE = DEVICE
OUT = Path(__file__).resolve().parent
EPOCHS = 12
LAMBDA_BASE = _args.lambda_base


def decode_F(r3, entries, acts):
    beat_fs = []
    for e in entries:
        ev = r3.decode(torch.from_numpy(acts[e["stem"]]).float().to(DEVICE))
        est = mir_eval.beat.trim_beats(ev["beats"])
        beat_fs.append(mir_eval.beat.f_measure(mir_eval.beat.trim_beats(e["beat_times"]), est)
                       if len(est) else 0.0)
    return float(np.mean(beat_fs))


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    r3 = R3ConditionedFactors(fps=FPS, device=DEVICE, lambda_base=LAMBDA_BASE,
                              observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    train, val, _ = load_songs(r3)                          # r3 has annotated_state_path (probe)
    for e in train + val:                                   # smooth targets
        sm = smooth_beat_frames(e["beat_times"]); keep = len(e["beat_frames"])
        if len(sm) >= keep:
            e["beat_frames"] = sm[len(sm) - keep:]
    print(f"train {len(train)} val {len(val)}", flush=True)

    model = final_eval.load_model(OUT / "vanilla_best_prelim.pt")
    train_acts = final_eval.activations_for(model, train)
    val_acts = final_eval.activations_for(model, val)

    # GLOBAL baseline: net at init (lambda_t == lambda_base), same bare decode
    f_global = decode_F(r3, val, val_acts)
    print(f"GLOBAL lambda={LAMBDA_BASE:.0f} (net at init), bare time-varying decode: beatF {f_global:.4f}",
          flush=True)

    opt = torch.optim.Adam(r3.net.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        rng.shuffle(train)
        tot, n = 0.0, 0
        for e in train:
            s, en = sample_crop(e, rng)
            f0, f1 = e["beat_frames"][s], e["beat_frames"][en]
            a = train_acts[e["stem"]]
            if f0 < 0 or f1 > a.shape[0] or f1 <= f0:
                continue
            built = r3.annotated_state_path(e["beat_frames"][s:en + 1] - f0,
                                            e["beat_in_bar"][s:en + 1], e["beats_per_bar"])
            if built is None:
                continue
            path, mi = built
            act_t = torch.from_numpy(a[f0:f1]).float().to(DEVICE)
            loss = r3.crf_nll(act_t, path, mi) / (f1 - f0)
            if _args.lambda_reg > 0:  # pin mean(log lam_t) to log(lambda_base); net redistributes only
                mean_log_lam = r3.log_lambda_per_frame(act_t).mean()
                loss = loss + _args.lambda_reg * (mean_log_lam - float(np.log(LAMBDA_BASE))) ** 2
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(r3.net.parameters(), 1.0)
            opt.step()
            tot += float(loss); n += 1
        # how much does lambda_t actually vary now?
        with torch.no_grad():
            lam = r3.log_lambda_per_frame(
                torch.from_numpy(val_acts[val[0]["stem"]]).float().to(DEVICE)).exp()
        print(f"epoch {epoch} | crf {tot/max(n,1):.4f} | lam_t mean {float(lam.mean()):.1f} "
              f"std {float(lam.std()):.1f} range [{float(lam.min()):.0f},{float(lam.max()):.0f}]",
              flush=True)

    f_perframe = decode_F(r3, val, val_acts)
    print(f"\nPER-FRAME (trained) bare decode:  beatF {f_perframe:.4f}", flush=True)
    print(f"GLOBAL {LAMBDA_BASE:.0f}                        beatF {f_global:.4f}", flush=True)
    print(f"delta (per-frame - global): {f_perframe - f_global:+.4f}", flush=True)
    torch.save({"net": r3.net.state_dict()}, OUT / "r3_frozen_net.pt")


if __name__ == "__main__":
    main()
