"""STAGE 1 of the MMI experiment: does training Beat This through the DBN move SMC at all?

WHAT THIS IS, AND WHAT IT IS NOT. This is an UPPER-BOUND PROBE, not a deployable result. It
fine-tunes on SMC itself, which is the most favourable possible condition and therefore the fastest
way to kill the idea: if MMI cannot move SMC when trained on SMC, it will not move it when diluted
16:1 by the full corpus. A POSITIVE here proves almost nothing about deployability -- "fine-tune on
the target corpus" is precisely the objection this paper was rejected for. Stage 2 (full corpus,
SMC improving WITHOUT SMC-specific training, GTZAN reported to show no general regression) is the
result that would actually count.

THE OBJECTIVE.
    L = L_MMI / T  +  alpha * L_BCE
L_MMI = -(log Z_num - log Z_den), both from the SAME certified forward_log_likelihood, differing by
the additive numerator mask in scripts/mmi.py (gate 2: F 0.9916 / CMLt 0.9802 on the representable
tracks; the obvious weaker mask scores 0.7661 CMLt and is NOT a valid numerator). Its gradient is
the occupancy difference gamma_den - gamma_num, free via autograd on log Z, bounded by 1 (gated).
L_BCE is Beat This's OWN ShiftTolerantBCELoss, imported rather than reimplemented -- and its
tolerance of 3 frames is the same +-60 ms the numerator mask uses, so the two terms agree on what
"close enough" means. Without the CE anchor MMI overfits; alpha=0 is available as an arm precisely
so that claim gets tested rather than assumed.

WHY THE DECODER IS FROZEN. Nothing about the DBN is learned: state space, +1 advance, lambda=100
with its -inf thresholding, observation_lambda=16, the 3-class emission. The DBN is a LOSS here,
not a model with parameters -- the CTC analogy. So the decoder we evaluate with is bit-for-bit the
decoder we ship today, and any movement is attributable to the activations alone. lambda is
deliberately NOT trained: three independent methods have already converged on it to +-0.02, and
fitting it generatively actively hurt.

SPLIT DISCIPLINE. Beat This trains on SMC, so the cached activations were built with 8-fold CV
checkpoints -- every track predicted by a model that never saw it (LATE_CORRECTIONS.md). This
mirrors that exactly: start from the fold{F} checkpoint, TEST on fold F (which it never saw),
VALIDATE on fold (F+7)%8 for early stopping, TRAIN on the remaining six folds. Fold F is never
touched except to score.

WHAT n=27 CAN AND CANNOT SHOW. One held-out fold is ~27 tracks. Our established effects are +0.057
CMLt (beat-only decoder) and +0.079 (full stack) at n=217; at n=27 the noise floor is ~2.8x wider,
so a genuine +0.02 will NOT be statistically visible here. This is a go/no-go signal. A regression
or a flat line kills it cheaply; movement earns the other seven folds.

STARTUP PARITY CHECK. Before training, the eval path must reproduce the cached baseline activations
that every number in this project is built on. If it does not, no comparison here means anything, so
it is asserted rather than hoped for.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.special import expit
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, load_gt, ev, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events
from mmi import annotation_frames, numerator_mask, mmi_loss, mmi_loss_batched

SMC_AUD = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
SPLIT = ROOT / "beat_this_annotations" / "smc" / "8-folds.split"
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")


def load_splits():
    d = {}
    for line in open(SPLIT):
        parts = line.split()
        if len(parts) >= 2:
            d[parts[0]] = int(parts[-1])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0, help="weight on Beat This's own BCE anchor")
    ap.add_argument("--mmi-w", type=float, default=1.0,
                    help="weight on the MMI term. --mmi-w 0 --alpha 1 is THE CONTROL: fine-tune "
                         "with Beat This's own loss alone, identical data / lr / schedule / crops / "
                         "seed. Without it, nothing here is attributable -- the alpha sweep varies "
                         "only the BCE weight, so even alpha=10 still contains MMI and there is no "
                         "reference for how much of the observed degradation is simply what "
                         "fine-tuning 20M parameters on 138 short clips does.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.15,
                    help="fraction of total steps spent in linear warmup, mirroring "
                         "Beat This's CosineWarmupScheduler(warmup=1000)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=1500, help="Beat This's native chunk length")
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--accum", type=int, default=16,
                    help="tracks accumulated per optimiser step. Beat This trains with "
                         "batch_size 8 x accumulate_grad_batches 8 = 64 tracks per step "
                         "(launch_scripts/train.py defaults). This loop originally stepped once "
                         "PER TRACK -- a 64x difference in gradient noise. Under Adam that is "
                         "specifically destructive: the update is ~lr * m/sqrt(v), so every "
                         "parameter moves ~lr per step in a direction set by ONE 30 s clip. 828 "
                         "such steps on a converged checkpoint is how the pure-BCE control lost "
                         "0.137 val CMLt. 16 rather than 64 because fold 0 has only 138 training "
                         "tracks and 64 would leave ~2 optimiser steps per epoch.")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="mmi_f0")
    ap.add_argument("--aug-npz", default="",
                    help="Beat This's OWN precomputed augmented spectrograms "
                         "(/disk1/jaehoon/bt_spect/smc.npz, Zenodo 13922116). 4774 keys = 217 "
                         "tracks x 22 variants: track, track_ps-5..ps6 (11 pitch shifts), "
                         "track_ts+-4..+-20 (10 time stretches). Downloaded rather than recreated "
                         "with pedalboard so these are BIT-IDENTICAL to what the fold checkpoints "
                         "actually trained on -- recreating them risks a silent divergence between "
                         "the training and evaluation paths, which is the failure mode that has "
                         "already cost this project twice.\n"
                         "WHY IT MATTERS HERE: the fold0 checkpoint has MEMORISED folds 1-6 "
                         "(CMLt 0.74-0.81 on seen folds vs 0.63 held out), so plain fine-tuning has "
                         "no headroom and can only overfit -- which is exactly what the pure-BCE "
                         "control does. 22x the data restores headroom for both arms.\n"
                         "tsN scales beat times by 1/(1+N/100) (verified: ts20 -> 1667 frames = "
                         "2001/1.2, ts-20 -> 2501 = 2001/0.8); psN leaves timing untouched.")
    ap.add_argument("--clean-only", type=int, default=1,
                    help="TRAIN only on tracks whose every beat gap is inside the tempo grid. "
                         "Gate 2b: the numerator is near-exact there (CMLt 0.9802, 100%% of "
                         "annotations anchored) but only 0.8584 on the 34 tracks that need a span "
                         "freed, where ~14%% of the 'reference-consistent' mass sits at a wrong "
                         "metrical level -- and those tracks are enriched in expressive timing and "
                         "tempo incontinuity, so the contamination would land exactly on the "
                         "material we care about. Never filters the TEST or VAL sets; only what "
                         "the model learns from.")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)

    splits = load_splits()
    test = sorted(t for t, f in splits.items() if f == args.fold)
    val = sorted(t for t, f in splits.items() if f == (args.fold + 7) % 8)
    train = sorted(t for t, f in splits.items() if f not in (args.fold, (args.fold + 7) % 8))
    if args.clean_only:
        g2 = json.load(open("/tmp/overnight/gate_mmi_num2.json"))["rows"]
        dirty = {t for t in g2 if g2[t]["n_bad_spans"] > 0}
        train = [t for t in train if t not in dirty]
    print(f"fold {args.fold}:  train {len(train)}  val {len(val)}  test {len(test)}"
          f"   (clean_only={args.clean_only}; val/test never filtered)", flush=True)

    from beat_this.inference import Audio2Frames, split_predict_aggregate
    import beat_this.model.loss as btloss
    a2f = Audio2Frames(checkpoint_path=f"fold{args.fold}", device=args.device)
    model = a2f.model
    bce = btloss.ShiftTolerantBCELoss(pos_weight=1).to(dev)

    m = make_beatonly_dbn(30.0, 16, 100, args.device)
    space, dp0 = m.state_spaces[0], m.dynamic_programs[0]
    s2c, iv = m.state_to_classes[0], space.interval_frames
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition
    edtype = logpi.dtype

    print("caching spectrograms...", flush=True)
    spects, truth = {}, {}
    for t in sorted(splits):
        y, sr = sf.read(str(SMC_AUD / f"{t.upper()}.wav"))
        spects[t] = a2f.signal2spect(y, sr).cpu()
        gt = np.asarray(load_gt(t), float)
        v = np.zeros(spects[t].shape[0], np.float32)
        fr = np.round(gt * FPS).astype(int)
        v[fr[(fr >= 0) & (fr < len(v))]] = 1.0
        truth[t] = torch.from_numpy(v)

    @torch.no_grad()
    def activations(tid):
        model.eval()
        with torch.autocast(enabled=False, device_type=dev.type):
            out = split_predict_aggregate(spect=spects[tid].to(dev), chunk_size=1500,
                                          overlap_mode="keep_first", border_size=6, model=model)
        return out["beat"].float().cpu().numpy()

    def decode(tid, logits):
        prob = expit(np.asarray(logits, np.float64))
        got = emissions_beatonly(m, prob)
        if got is None:
            return None
        lcd, dec, ff = got
        path, _ = dp0.viterbi(logpi, logA, lcd, state_to_class=s2c, return_log_score=True)
        return state_path_to_events(path.cpu().numpy(), space, FPS,
                                    snap_to_activations=dec, first_frame=ff)["beats"]

    def score(ids):
        """Also returns the beat-offset distribution.

        WHY: the numerator has a known systematic bias. The beat region is
        interval/observation_lambda frames wide (up to 7 frames = 140 ms at 30 BPM), and the mask
        pins the path INSIDE the region while the read-out fires at region ENTRY -- measured median
        -22 ms. If that bias transfers to the trained model it would move detected beats EARLY, which
        hurts F (+-70 ms window) while leaving CMLt (+-17.5% of the inter-annotation interval, ~+-175
        ms at 60 BPM) untouched. That is a specific, falsifiable alternative to "MMI trades F for
        continuity", and the two are indistinguishable from the metrics alone -- so measure it.
        """
        out, offs = {}, []
        for t in ids:
            b = decode(t, activations(t))
            if b is None:
                continue
            gt = np.asarray(load_gt(t), float)
            out[t] = ev(gt, b)
            for e in b:
                offs.append(1000.0 * (e - gt[np.argmin(np.abs(gt - e))]))
        return out, np.asarray(offs)

    # ---- startup parity: our eval path must reproduce the cached baseline exactly ----
    worst = 0.0
    for t in test[:5]:
        worst = max(worst, float(np.abs(expit(activations(t)) - bthis_beat_prob(t)).max()))
    print(f"eval-path parity vs cached activations: max |diff| = {worst:.3e}", flush=True)
    assert worst < 1e-4, "eval path does not reproduce the cached baseline; comparisons invalid"

    bv, bvo = score(val)
    bt, bto = score(test)
    base, base_off = {"val": bv, "test": bt}, {"val": bvo, "test": bto}
    A = lambda r: {k: float(np.mean([v[k] for v in r.values()])) for k in METRICS}
    off1 = lambda o: f"median {np.median(o):+.1f} ms, |o|<=70ms {100*(np.abs(o)<=70).mean():.1f}%"
    print(f"baseline  val  CMLt {A(bv)['CMLt']:.4f} F {A(bv)['F']:.4f}   offsets: {off1(bvo)}")
    print(f"          test CMLt {A(bt)['CMLt']:.4f} F {A(bt)['F']:.4f}   offsets: {off1(bto)}",
          flush=True)

    # Step-0 MMI, so the epoch averages have something to be compared against. Without this a flat
    # loss (gradient not reaching the network) is indistinguishable from a converged one.
    with torch.no_grad():
        z = []
        for tid in train[:20]:
            sp = spects[tid][:args.crop].to(dev)
            model.eval()
            pr = torch.sigmoid(model(sp.unsqueeze(0))["beat"][0]).to(edtype)
            e0 = m.log_class_densities(torch.stack([pr, torch.zeros_like(pr)], 1))
            f0 = annotation_frames(np.asarray(load_gt(tid), float), FPS, 0, e0.shape[0])
            if len(f0) < 8:
                continue
            mk0, _, _ = numerator_mask(f0, e0.shape[0], iv, e0.device, e0.dtype)
            z.append(float(mmi_loss(dp0, logpi, logA, e0, s2c, mk0)[0]) / e0.shape[0])
    print(f"step-0 MMI/frame (20 train tracks): {np.mean(z):.4f}", flush=True)

    aug, aug_keys = None, {}
    if args.aug_npz:
        aug = np.load(args.aug_npz, mmap_mode="r")
        for t in train:
            aug_keys[t] = [k.split("/", 1)[1] for k in aug.files if k.startswith(t + "/")]
        nk = sum(len(v) for v in aug_keys.values())
        print(f"augmentation: {nk} variants over {len(aug_keys)} training tracks "
              f"({nk/max(1,len(aug_keys)):.0f} per track). VAL/TEST are never augmented.",
              flush=True)

    # Match Beat This's own optimiser setup. Three things were wrong before, and the pure-BCE
    # control showed the cost: fine-tuning with their loss ALONE cost -0.13 val CMLt, which a
    # checkpoint being fine-tuned on data it already trained on should not do.
    #   (a) weight decay was applied to biases and LayerNorm gains. pl_module.configure_optimizers
    #       explicitly excludes ndim<=1 ("to exclude biases and norms").
    #   (b) NO WARMUP. They use CosineWarmupScheduler(warmup=1000). The entire previous run was
    #       828 steps -- shorter than their warmup -- at full LR from step 0 with fresh Adam
    #       moments. And because Adam normalises gradient magnitude away (update ~ lr * m/sqrt(v)),
    #       each weight moves ~lr per step REGARDLESS of gradient size: 828 steps at 1e-5 is up to
    #       8.3e-3 per weight, a 10-40% relative change on weights of scale 0.02-0.1. The model was
    #       moving hard, not "barely moving" as the gradient norms misled me into concluding --
    #       grad clipping is irrelevant under Adam.
    #   (c) no LR decay, so it never settled.
    # Copied op-for-op from beat_this.model.pl_module.CosineWarmupScheduler rather than imported:
    # that module pulls in pytorch_lightning, which is not installed here. Same formula
    # (linear warmup multiplying a cosine decay), raise_last=0 so the final branch never fires.
    class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, warmup, max_iters):
            self.warmup, self.max_num_iters = warmup, max_iters
            super().__init__(optimizer)

        def get_lr(self):
            step = self.last_epoch
            if step < self.max_num_iters:
                f = 0.5 * (1 + np.cos(np.pi * step / self.max_num_iters))
                if step <= self.warmup:
                    f *= step / self.warmup
            else:
                f = 0.0
            return [base_lr * f for base_lr in self.base_lrs]

    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": nodecay, "weight_decay": 0.0}], lr=args.lr)
    total_steps = max(1, args.epochs * ((len(train) + args.accum - 1) // args.accum))
    sched = CosineWarmupScheduler(opt, warmup=max(1, int(args.warmup_frac * total_steps)),
                                  max_iters=total_steps)
    print(f"optimiser: AdamW lr={args.lr} (decay {len(decay)} tensors / no-decay {len(nodecay)}), "
          f"cosine schedule over {total_steps} steps, warmup "
          f"{max(1, int(args.warmup_frac * total_steps))}", flush=True)
    hist, best_val, best_test, best_ep = [], -float("inf"), None, -1
    rng = np.random.default_rng(0)
    for ep in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        tot_mmi = tot_bce = n = 0
        mmi_vals = []
        opt.zero_grad(set_to_none=True)
        # TRUE BATCHING, not gradient accumulation. The DP is dispatch-bound, so running args.accum
        # tracks through one batched forward costs about what ONE track costs -- gate 4 measured
        # 8.5x at B=12 while the batched path matched the certified unbatched one at 0.000e+00 on
        # both log Z and d logZ/d emission. Per-track calls were ~94% of each epoch's wall clock.
        # All SMC clips are 40 s, so crops are rectangular and no length masking is needed.
        for b0 in range(0, len(order), args.accum):
            batch = [train[i] for i in order[b0:b0 + args.accum]]
            xs, tg, msks, frs = [], [], [], []
            for tid in batch:
                sp, times = spects[tid], np.asarray(load_gt(tid), float)
                if aug is not None:
                    key = f"{tid}/{aug_keys[tid][rng.integers(len(aug_keys[tid]))]}"
                    sp = torch.from_numpy(np.asarray(aug[key], np.float32))
                    st = key.rsplit("_ts", 1)
                    if len(st) == 2:                    # tsN: tempo +N% -> times / (1 + N/100)
                        times = times / (1.0 + int(st[1]) / 100.0)
                off = int(rng.integers(0, max(1, sp.shape[0] - args.crop + 1)))
                fr = annotation_frames(times, FPS, off, args.crop)
                if len(fr) < 8 or sp.shape[0] - off < args.crop:
                    continue
                v = torch.zeros(args.crop)
                v[torch.from_numpy(fr)] = 1.0
                xs.append(sp[off:off + args.crop])
                tg.append(v)
                frs.append(fr)
            if not xs:
                continue
            x = torch.stack(xs, 0).to(dev)                              # [B, L, mel]
            L = x.shape[1]
            logits = model(x)["beat"]                                   # [B, L]
            prob = torch.sigmoid(logits).to(edtype)
            em = torch.stack([m.log_class_densities(
                torch.stack([prob[j], torch.zeros_like(prob[j])], 1)) for j in range(len(xs))], 0)

            lbce = bce(logits, torch.stack(tg, 0).to(dev),
                       torch.ones(len(xs), L, dtype=torch.bool, device=dev))
            if args.mmi_w != 0:
                mk = torch.stack([numerator_mask(f, L, iv, em.device, em.dtype)[0] for f in frs], 0)
                lmmi_b, _, _ = mmi_loss_batched(dp0, logpi, logA, em, s2c, mk)
                lmmi = lmmi_b.mean()
                per_track = (lmmi_b / L).detach().cpu().numpy().tolist()
            else:
                lmmi = torch.zeros((), dtype=em.dtype, device=em.device)
                per_track = [0.0] * len(xs)
            loss = args.mmi_w * (lmmi / L) + args.alpha * lbce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            n += len(xs)
            tot_mmi += float(lmmi) / L * len(xs)
            tot_bce += float(lbce) * len(xs)
            mmi_vals.extend(per_track)
        v, vo = score(val)
        s, so = score(test)
        av, at = A(v), A(s)
        hist.append({"epoch": ep, "mmi": tot_mmi / max(n,1), "mmi_median": float(np.median(mmi_vals)) if mmi_vals else 0.0, "bce": tot_bce / n, "val": av, "test": at,
                     "test_median_offset_ms": float(np.median(so)),
                     "test_within70": float((np.abs(so) <= 70).mean())})
        print(f"ep{ep}  MMImed {np.median(mmi_vals) if mmi_vals else 0:7.4f}  "
              f"MMImean {tot_mmi/max(n,1):8.3f}  BCE {tot_bce/max(n,1):.4f}  "
              f"| val CMLt {av['CMLt']:.4f} F {av['F']:.4f}  "
              f"| test CMLt {at['CMLt']:.4f} F {at['F']:.4f}  "
              f"| test offset {np.median(so):+.1f} ms (base {np.median(bto):+.1f})", flush=True)
        if av["CMLt"] > best_val:
            best_val, best_test, best_ep = av["CMLt"], s, ep
            print(f"      ^ new best val ({av['CMLt']:.4f}); test snapshot kept", flush=True)

    print(f"\nfold {args.fold}, alpha={args.alpha}, n_test={len(test)}  "
          f"(a real +0.02 is NOT resolvable at this n -- go/no-go only)")
    print(f"{'metric':<8}{'baseline':>10}{'mmi':>10}{'delta':>10}{'p':>10}")
    print(f"\nSELECTED epoch {best_ep} by val CMLt {best_val:.4f}.  WARNING: val is "
          f"fold {(args.fold + 7) % 8}, which the fold{args.fold} checkpoint TRAINED ON -- "
          f"it measures memorisation decay, not generalisation. Treat any selection "
          f"made on it as invalid.", flush=True)
    final = best_test if best_test is not None else score(test)[0]
    ids = sorted(set(base["test"]) & set(final))
    for k in METRICS:
        d = np.array([final[t][k] - base["test"][t][k] for t in ids])
        p = wilcoxon(d).pvalue if np.any(d) else 1.0
        print(f"{k:<8}{np.mean([base['test'][t][k] for t in ids]):>10.4f}"
              f"{np.mean([final[t][k] for t in ids]):>10.4f}{d.mean():>+10.4f}{p:>10.3g}")
    json.dump({"fold": args.fold, "alpha": args.alpha, "mmi_w": args.mmi_w, "lr": args.lr,
               "hist": hist,
               "baseline_test": base["test"], "final_test": final},
              open(f"/tmp/overnight/{args.tag}.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
