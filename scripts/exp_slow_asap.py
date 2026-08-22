"""THE DATA-HYPOTHESIS GATE: does adding SLOW EXPRESSIVE training data move SMC?

THE HYPOTHESIS, AND WHY IT IS NOW PRECISE. Everything decoder-side is exhausted (per-track oracle
hyperparameters cap at F 0.6908 and only ~22% of that survives having to commit in advance), and
foundation-model features carry LESS accessible tempo structure than Beat This's own embedding. What
remains is training-distribution coverage, and the fold checkpoints prove capacity is not the limit:
they score ~0.71 F on SMC tracks they TRAINED on versus 0.63 held out.

tempo_curve_stats.py located the gap exactly, and it is narrower than expected:
    rubato volatility sigma_d   SMC 0.152   ASAP 0.213   rest-of-corpus 0.049
    median tempo                SMC  70.9   ASAP 105.9   rest-of-corpus  123.2
    slow (<55 BPM)              SMC 20.7%              corpus  3.2%
    slow AND volatile           SMC 12.4%              corpus  1.6%       <- 8x under-represented
So EXPRESSIVE TIMING IS NOT MISSING -- ASAP has more of it than SMC, 473 annotated tracks, already
in Beat This's training set. What is missing is the CONJUNCTION: expressive AND slow.

THE INTERVENTION. Uniformly slow ASAP into SMC's tempo range. Labels transform by one scalar
(t -> t*r), so they are EXACT -- no generative model, no adherence to verify, no circularity. If the
missing regime really is "expressive and slow", this alone should move SMC, and it is a day of work
rather than the ControlNet's weeks. If it does nothing, the data hypothesis is falsified cheaply and
the generative pipeline should not be built on it.

WHY SPECTROGRAM-DOMAIN WARPING. Beat This consumes log-mel spectrograms, so we stretch the
spectrogram time axis rather than resynthesising audio. Two consequences, both wanted: it is ~100x
cheaper, and it SMEARS TRANSIENTS in proportion to the stretch -- which is precisely the soft-onset
character of slow expressive playing (SMC's 'lack of transient sounds' tag, 33% of the F-deficit).
It is not identical to a phase-vocoder stretch, and the write-up must say so.

ARMS (all fine-tune the SAME fold checkpoint with the repaired harness; test fold never trained on):
    control    ASAP at original tempo
    slowed     ASAP warped into SMC's tempo distribution
    mixed      both
The control is the point: 'slowed' beating 'control' isolates the SLOWING, not the extra ASAP data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events
from scipy.special import expit
from scipy.stats import wilcoxon

ASAP_AUD = Path("/disk1/jaehoon/dataset_store/audio_by_stem/asap")
ASAP_ANN = Path("/disk1/jaehoon/dataset_store/beat_this_annotations/asap/annotations/beats")
SMC_SPECT = Path("/disk1/jaehoon/bt_spect/smc.npz")
SPLIT = ROOT / "beat_this_annotations" / "smc" / "8-folds.split"
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")


def warp_spect(sp, r):
    """Stretch a [T, mel] spectrogram in time by factor r (r>1 = slower/longer)."""
    T = sp.shape[0]
    n = int(round(T * r))
    src = np.arange(T)
    dst = np.linspace(0, T - 1, n)
    return np.stack([np.interp(dst, src, sp[:, d]) for d in range(sp.shape[1])], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--val-fold", type=int, default=7)
    ap.add_argument("--arm", default="slowed", choices=["control", "slowed", "mixed"])
    ap.add_argument("--n-asap", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--crop", type=int, default=1500)
    ap.add_argument("--warmup-frac", type=float, default=0.15)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="slowasap")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    splits = {}
    for line in open(SPLIT):
        p = line.split()
        if len(p) >= 2:
            splits[p[0]] = int(p[-1])
    test = sorted(t for t, f in splits.items() if f == args.fold)
    val = sorted(t for t, f in splits.items() if f == args.val_fold)

    from beat_this.inference import Audio2Frames, split_predict_aggregate
    import beat_this.model.loss as btloss
    a2f = Audio2Frames(checkpoint_path=f"fold{args.fold}", device=args.device)
    model = a2f.model
    bce = btloss.ShiftTolerantBCELoss(pos_weight=1).to(dev)

    # SMC target tempo distribution, from TRAIN folds only (fold-honest curve sampling)
    tr_smc = [t for t, f in splits.items() if f not in (args.fold, args.val_fold)]
    smc_bpm = np.array([60.0 / np.median(np.diff(np.asarray(load_gt(t), float)))
                        for t in tr_smc if len(load_gt(t)) > 3])
    print(f"SMC target tempo (train folds): median {np.median(smc_bpm):.1f} BPM, "
          f"p10-p90 {np.percentile(smc_bpm,10):.0f}-{np.percentile(smc_bpm,90):.0f}", flush=True)

    print(f"loading ASAP ({args.n_asap} tracks)...", flush=True)
    items = []
    for p in sorted(ASAP_AUD.glob("*.wav"))[:args.n_asap]:
        ann = ASAP_ANN / f"{p.stem}.beats"
        if not ann.exists():
            continue
        try:
            a = np.loadtxt(str(ann))
        except Exception:
            continue
        bt = (a if a.ndim == 1 else a[:, 0]).astype(float)
        if len(bt) < 20:
            continue
        y, srr = sf.read(str(p), always_2d=True)
        sp = a2f.signal2spect(y.mean(1), srr).cpu().numpy()
        src_bpm = 60.0 / float(np.median(np.diff(bt)))
        items.append({"spect": sp, "beats": bt, "bpm": src_bpm})
        if len(items) % 40 == 0:
            print(f"  {len(items)}", flush=True)
    print(f"ASAP loaded: {len(items)} tracks, median {np.median([i['bpm'] for i in items]):.1f} BPM",
          flush=True)

    def sample_example():
        it = items[rng.integers(len(items))]
        sp, bt = it["spect"], it["beats"]
        if args.arm == "control":
            r = 1.0
        else:
            # draw a TARGET tempo from SMC's distribution; r>1 slows the track toward it
            tgt = smc_bpm[rng.integers(len(smc_bpm))]
            r = float(np.clip(it["bpm"] / tgt, 1.0, 3.0))
            if args.arm == "mixed" and rng.random() < 0.5:
                r = 1.0
        if abs(r - 1.0) > 1e-3:
            sp = warp_spect(sp, r)
            bt = bt * r
        T = sp.shape[0]
        if T < args.crop:
            return None
        off = int(rng.integers(0, T - args.crop + 1))
        fr = np.round(bt * FPS).astype(int) - off
        fr = fr[(fr >= 0) & (fr < args.crop)]
        if len(fr) < 8:
            return None
        v = np.zeros(args.crop, np.float32)
        v[fr] = 1.0
        return sp[off:off + args.crop], v

    # ---- decoder + scoring (identical to every other experiment in this project) ----
    m = make_beatonly_dbn(30.0, 16, 100, args.device)
    space, dp0 = m.state_spaces[0], m.dynamic_programs[0]
    s2c = m.state_to_classes[0]
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition
    z = np.load(SMC_SPECT, mmap_mode="r")

    @torch.no_grad()
    def score(ids):
        model.eval()
        out = {}
        for t in ids:
            sp = torch.from_numpy(np.asarray(z[f"{t}/track"], np.float32)).to(dev)
            o = split_predict_aggregate(spect=sp, chunk_size=1500, overlap_mode="keep_first",
                                        border_size=6, model=model)
            prob = expit(np.asarray(o["beat"].float().cpu().numpy(), np.float64))
            got = emissions_beatonly(m, prob)
            if got is None:
                continue
            lcd, dec, ff = got
            path, _ = dp0.viterbi(logpi, logA, lcd, state_to_class=s2c, return_log_score=True)
            b = state_path_to_events(path.cpu().numpy(), space, FPS,
                                     snap_to_activations=dec, first_frame=ff)["beats"]
            out[t] = ev(np.asarray(load_gt(t), float), b)
        return out

    A = lambda r: {k: float(np.mean([v[k] for v in r.values()])) for k in METRICS}
    base_val, base_test = score(val), score(test)
    print(f"\nBASELINE (no fine-tune)  val CMLt {A(base_val)['CMLt']:.4f} F {A(base_val)['F']:.4f}"
          f" | test CMLt {A(base_test)['CMLt']:.4f} F {A(base_test)['F']:.4f}", flush=True)

    class Cos(torch.optim.lr_scheduler._LRScheduler):
        def __init__(s, o, w, mx):
            s.w, s.mx = w, mx
            super().__init__(o)

        def get_lr(s):
            st = s.last_epoch
            f = 0.5 * (1 + np.cos(np.pi * min(st, s.mx) / s.mx))
            if st <= s.w:
                f *= st / s.w
            return [b * f for b in s.base_lrs]

    dec_p = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    nod_p = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": dec_p, "weight_decay": 0.01},
                             {"params": nod_p, "weight_decay": 0.0}], lr=args.lr)
    steps_per_ep = max(1, len(items) // args.accum)
    sched = Cos(opt, max(1, int(args.warmup_frac * args.epochs * steps_per_ep)),
                args.epochs * steps_per_ep)

    hist, best = [], (-1, None, -1)
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = nb = 0
        for _ in range(steps_per_ep):
            xs, ys = [], []
            while len(xs) < args.accum:
                e = sample_example()
                if e is not None:
                    xs.append(e[0]); ys.append(e[1])
            x = torch.from_numpy(np.stack(xs)).to(dev)
            y = torch.from_numpy(np.stack(ys)).to(dev)
            loss = bce(model(x)["beat"], y, torch.ones_like(y, dtype=torch.bool))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step(); sched.step()
            tot += float(loss); nb += 1
        v = A(score(val))
        hist.append({"epoch": ep, "bce": tot / nb, "val": v})
        print(f"ep{ep}  BCE {tot/nb:.4f}  | VAL CMLt {v['CMLt']:.4f} F {v['F']:.4f}", flush=True)
        if v["CMLt"] > best[0]:
            best = (v["CMLt"], {k: t.detach().clone() for k, t in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    print(f"\nselected epoch {best[2]} by VAL CMLt {best[0]:.4f} (test never consulted)")
    fin = score(test)
    ids = sorted(set(fin) & set(base_test))
    print(f"\nARM={args.arm}   TEST fold {args.fold}, n={len(ids)}")
    print(f"{'':<12}" + "".join(f"{k:>9}" for k in METRICS))
    print(f"{'baseline':<12}" + "".join(f"{A({t:base_test[t] for t in ids})[k]:>9.4f}" for k in METRICS))
    print(f"{'fine-tuned':<12}" + "".join(f"{A({t:fin[t] for t in ids})[k]:>9.4f}" for k in METRICS))
    for k in METRICS:
        d = np.array([fin[t][k] - base_test[t][k] for t in ids])
        print(f"  d{k:<5} {d.mean():+.4f} (p={wilcoxon(d).pvalue:.3g})")
    json.dump({"arm": args.arm, "fold": args.fold, "best_ep": best[2], "hist": hist,
               "test": fin, "base_test": base_test},
              open(f"/tmp/overnight/{args.tag}.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
