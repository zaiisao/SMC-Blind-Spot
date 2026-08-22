"""Fine-tune Beat This with a self-supervised TEMPO-EQUIVARIANCE loss alongside the beat loss.

DIAGNOSIS THIS TARGETS (measured, scripts/probe_tempo_equivariance.py, 40 SMC tracks, fold-honest,
phase-vocoder stretching). The released model satisfies f(stretch(x,s)) = f(x) - log2 s almost
exactly INSIDE its training augmentation range and collapses outside it:

        s      0.50   0.70   0.85   1.00   1.20   1.50   2.00
   implied BPM  137     98     81     69     57     46     34
   median d   +0.757 +0.029 +0.012 +0.000 -0.018 -0.411 -0.966
                             |<--- trained range --->|

At s=2.0 the defect is -0.966, one octave: slowed audio is tracked at double time. Both ends
collapse INWARD toward the ~70-100 BPM training prior. The flat region coincides with the shipped
augmentation range (0.83-1.25).

WHY EQUIVARIANCE RATHER THAN MORE AUGMENTATION. Supervised wide-range fine-tuning was null in a
pilot (dF -0.006, p=0.14 under DBN decoding, n=27). The equivariance identity is a stronger
constraint: it is dense, exact, and needs NO tempo annotation, because only the RATIO between two
views is required and we choose it. If it holds, behaviour on the well-populated 70-100 BPM region
DETERMINES behaviour at 40 BPM, converting extrapolation into interpolation.

WHAT THE PAYOFF WOULD BE. A tempo channel fed to the DBN is worth +0.0571 F / +0.2398 CMLt at
oracle accuracy, and retains 94% of the CMLt gain at only 29.6% tempo accuracy (n=217). The current
head sits at 12.2% on SMC and HURTS (-0.069 F). So the target is crossing ~30% accuracy.

THE SHORTCUT THREAT, and why stretching is audio-domain here. A ratio objective is maximally
vulnerable to solving the task from resampling artefacts instead of musical tempo. Stretching is
therefore done with a phase vocoder on the WAVEFORM, never by resampling the mel time axis (which
is what the earlier fine-tune did). This does not eliminate the shortcut -- run
scripts/control_shortcut.py, which repeats training on beat-destroyed audio and must FAIL.

FOLD HONESTY. Fine-tune fold k on corpus tracks not in fold k; evaluate only on SMC tracks in
fold k, with both peak-picking and DBN decoding, since the two disagree in sign on this pipeline.
"""
import argparse, json, sys
from pathlib import Path

import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "beat_this"))
from beat_this.inference import Audio2Frames
from beat_this.preprocessing import load_audio, LogMelSpect
from beat_this.model.loss import ShiftTolerantBCELoss
from _harness import FPS, track_ids, load_gt, ev, peakpick, make_dbn, dbn_beats
from tempo_readout import lag_grid, soft_log_period, equivariance_residual

STORE = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
AUDIO = Path("/disk1/jaehoon/dataset_store/audio_by_stem")
SMC_AUD = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")


def split_of(ds):
    p = STORE / ds / "8-folds.split"
    return {l.split("\t")[0]: int(l.split("\t")[1]) for l in open(p) if l.strip()} if p.exists() else {}


def annots(ds, tid):
    p = STORE / ds / "annotations" / "beats" / f"{tid}.beats"
    if not p.exists():
        return None, None
    t, b = [], []
    for line in open(p):
        f = line.split()
        if f:
            t.append(float(f[0])); b.append(int(float(f[1])) if len(f) > 1 else 0)
    return np.array(t), np.array(b)


def make_targets(times, downbeat, T):
    yb = np.zeros(T, np.float32); yd = np.zeros(T, np.float32)
    for tt, bn in zip(times, downbeat):
        f = int(round(tt * FPS))
        if 0 <= f < T:
            yb[f] = 1.0
            if bn == 1:
                yd[f] = 1.0
    return yb, yd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--tracks-per-ds", type=int, default=40)
    ap.add_argument("--datasets", default="ballroom,hainsworth,beatles,gtzan,asap")
    ap.add_argument("--lam-eq", type=float, default=1.0, help="weight on the equivariance loss")
    ap.add_argument("--smin", type=float, default=0.5)
    ap.add_argument("--smax", type=float, default=2.0)
    ap.add_argument("--p-identity", type=float, default=0.3,
                    help="probability a view uses s=1.0 RAW audio (no phase vocoder). With s\n                         drawn log-uniformly, s=1 has measure zero, so the model trained only on\n                         vocoder output and was then tested on real SMC audio. That domain shift\n                         cost -0.092 F while the equivariance defect improved 0.279 -> 0.062:\n                         the property trained got better, the domain tested got worse.")
    ap.add_argument("--tau", type=float, default=30.0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval", action="store_true", help="run the SMC evaluation")
    ap.add_argument("--accum", type=int, default=8,
                    help="gradient accumulation. bs=1 with a fresh random stretch each step made\n                         the loss trace pure noise (sup swung 2.5-4.2 over 25-step windows)")
    ap.add_argument("--defect-tracks", type=int, default=20,
                    help="SMC tracks for the post-training equivariance defect table (0=skip)")
    ap.add_argument("--crop-sec", type=float, default=30.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--pos-weight-beat", type=float, default=20.4)
    ap.add_argument("--pos-weight-downbeat", type=float, default=91.2)
    ap.add_argument("--shuffle-audio", action="store_true",
                    help="SHORTCUT CONTROL: destroy beat structure; equivariance must then FAIL")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="/tmp/overnight/eq")
    args = ap.parse_args()
    dev = torch.device(args.device)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    pool = []
    for ds in args.datasets.split(","):
        sp = split_of(ds)
        d = AUDIO / ds
        if not d.exists():
            continue
        stems = [p for p in sorted(d.glob("*.wav")) if sp.get(p.stem, -1) != args.fold]
        pool += [(ds, p) for p in stems[:args.tracks_per_ds]]
    print(f"fold {args.fold}: {len(pool)} training tracks, s in [{args.smin},{args.smax}], "
          f"lam_eq={args.lam_eq}, shuffle={args.shuffle_audio}", flush=True)

    a2f = Audio2Frames(checkpoint_path=f"fold{args.fold}", device=args.device)
    model = a2f.model
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    spect = LogMelSpect(device=dev)
    lags = lag_grid()
    loss_b = ShiftTolerantBCELoss(pos_weight=args.pos_weight_beat, tolerance=3).to(dev)
    loss_d = ShiftTolerantBCELoss(pos_weight=args.pos_weight_downbeat, tolerance=3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    opt.zero_grad()
    cache = {}

    def view(y, sr, s, bt, bn):
        """Phase-vocode the WAVEFORM by s (s>1 = slower), return (spect, targets, n_frames)."""
        ys = y if s == 1.0 else librosa.effects.time_stretch(y, rate=1.0 / s)
        with torch.no_grad():
            sp = spect(torch.tensor(ys, dtype=torch.float32, device=dev))
        T = sp.shape[0]
        yb, yd = make_targets(bt * s, bn, T)
        return sp, yb, yd, T

    model.train()
    run_b = run_e = 0.0
    done = 0
    for step in range(args.steps):
        ds, path = pool[rng.integers(len(pool))]
        if path not in cache:
            sig, sr = load_audio(str(path))
            y = sig.mean(1) if sig.ndim > 1 else sig
            bt, bn = annots(ds, path.stem)
            cache[path] = (np.ascontiguousarray(y.astype(np.float32)), sr, bt, bn)
            if len(cache) > 120:
                cache.pop(next(iter(cache)))
        y, sr, bt, bn = cache[path]
        if bt is None or len(bt) < 8:
            continue
        n = int(args.crop_sec * sr)
        if len(y) < n + sr:
            continue
        st = int(rng.integers(0, len(y) - n))
        yc = y[st:st + n]
        btc = bt[(bt >= st / sr) & (bt < (st + n) / sr)] - st / sr
        bnc = bn[(bt >= st / sr) & (bt < (st + n) / sr)]
        if len(btc) < 6:
            continue
        if args.shuffle_audio:
            # CONTROL: chop into 250 ms grains and permute -> spectral content preserved, beat
            # structure destroyed. A model reading artefacts can still solve equivariance here.
            g = int(0.25 * sr)
            gr = [yc[i:i + g] for i in range(0, len(yc) - g, g)]
            rng.shuffle(gr)
            yc = np.concatenate(gr)
            btc, bnc = btc[:0], bnc[:0]

        # Anchor to the real audio domain: each view independently has p_identity chance of
        # being the untouched waveform. The two views must still differ, or the equivariance
        # residual target is 0 and the pair teaches nothing.
        def draw():
            if rng.random() < args.p_identity:
                return 1.0
            return float(np.exp(rng.uniform(np.log(args.smin), np.log(args.smax))))
        s_a, s_b = draw(), draw()
        if abs(np.log2(s_b / s_a)) < 0.05:
            s_b = float(np.exp(rng.uniform(np.log(args.smin), np.log(args.smax))))
        sp_a, yb_a, yd_a, _ = view(yc, sr, float(s_a), btc, bnc)
        sp_b, _, _, _ = view(yc, sr, float(s_b), btc, bnc)

        out_a = model(sp_a.unsqueeze(0))
        out_b = model(sp_b.unsqueeze(0))
        f_a, _ = soft_log_period(torch.sigmoid(out_a["beat"].float()), lags, tau=args.tau)
        f_b, _ = soft_log_period(torch.sigmoid(out_b["beat"].float()), lags, tau=args.tau)
        eq = equivariance_residual(f_a, f_b, float(s_a), float(s_b)).pow(2).mean()

        if args.shuffle_audio:
            sup = torch.zeros((), device=dev)
        else:
            sup = (loss_b(out_a["beat"], torch.from_numpy(yb_a).to(dev).unsqueeze(0))
                   + loss_d(out_a["downbeat"], torch.from_numpy(yd_a).to(dev).unsqueeze(0)))
        loss = (sup + args.lam_eq * eq) / args.accum
        loss.backward()
        if (done + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step(); opt.zero_grad()
        run_b += float(sup); run_e += float(eq); done += 1
        if done % args.log_every == 0:
            k = args.log_every
            print(f"  step {step+1} (used {done}) sup {run_b/k:.4f}  eq {run_e/k:.4f}",
                  flush=True)
            run_b = run_e = 0.0

    torch.save(model.state_dict(), f"{args.out}/model_fold{args.fold}.pt")
    if args.shuffle_audio:
        print("CONTROL RUN: see eq loss trace above; a FALLING eq loss on beat-destroyed audio "
              "means the objective is solvable from artefacts.", flush=True)
        return

    print(f"steps requested {args.steps}, steps USED {done}", flush=True)

    # ---- PRIMARY READOUT: does the equivariance defect actually shrink? ----
    # SMC F is a poor judge of a tempo intervention (the tempo oracle buys +0.240 CMLt but only
    # +0.057 F), and on 27 tracks it is noisy. delta(s) is what the loss optimises, so measure it
    # directly, base vs trained, on the SAME held-out tracks with the SAME stretches.
    if args.defect_tracks:
        model.eval()
        smc_sp = split_of("smc")
        dids = [t for t in track_ids() if smc_sp.get(t, -1) == args.fold
                and (SMC_AUD / f"{t.upper()}.wav").exists()][:args.defect_tracks]
        SS = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
        ft_now = {k: v.detach().clone() for k, v in model.state_dict().items()}
        tab = {}
        for nm, stt in (("base", base_state), ("eq", ft_now)):
            model.load_state_dict(stt); model.eval()
            tab[nm] = {s: {} for s in SS}
            for t in dids:
                sig, sr = load_audio(str(SMC_AUD / f"{t.upper()}.wav"))
                yy = sig.mean(1) if sig.ndim > 1 else sig
                yy = np.ascontiguousarray(yy.astype(np.float32))
                for s in SS:
                    ys = yy if s == 1.0 else librosa.effects.time_stretch(yy, rate=1.0 / s)
                    with torch.no_grad():
                        bb, _ = a2f(ys, sr)
                    pk = peakpick(torch.sigmoid(bb.float()).cpu().numpy(), 0.5, FPS)
                    tab[nm][s][t] = (float(np.log2(np.median(np.diff(pk)) * FPS))
                                     if len(pk) >= 4 else float("nan"))
        print(f"\n  EQUIVARIANCE DEFECT delta(s), median over {len(dids)} SMC tracks")
        print(f"  {'s':>6} {'base':>9} {'trained':>9} {'improve':>9}")
        dsum = {}
        for s in SS:
            row = []
            for nm in ("base", "eq"):
                d = [tab[nm][s][t] - tab[nm][1.0][t] - np.log2(s) for t in dids
                     if np.isfinite(tab[nm][s].get(t, np.nan))
                     and np.isfinite(tab[nm][1.0].get(t, np.nan))]
                row.append(float(np.median(d)) if d else float("nan"))
            dsum[s] = row
            print(f"  {s:>6.2f} {row[0]:>+9.3f} {row[1]:>+9.3f} "
                  f"{abs(row[0]) - abs(row[1]):>+9.3f}")
        out_s = [s for s in SS if s != 1.0]
        mab = float(np.mean([abs(dsum[s][0]) for s in out_s]))
        mae = float(np.mean([abs(dsum[s][1]) for s in out_s]))
        print(f"  mean |delta| over s!=1:  base {mab:.3f} -> trained {mae:.3f}  "
              f"({'BETTER' if mae < mab else 'WORSE'})")
        json.dump({"fold": args.fold, "lam_eq": args.lam_eq, "defect": dsum,
                   "mean_abs_base": mab, "mean_abs_trained": mae},
                  open(f"{args.out}/defect_fold{args.fold}.json", "w"), indent=1)
        model.load_state_dict(ft_now)
    if not args.eval:
        return
    # ---- evaluate: SMC tracks of this fold, both decoders ----
    model.eval()
    smc = split_of("smc")
    ids = [t for t in track_ids() if smc.get(t, -1) == args.fold
           and (SMC_AUD / f"{t.upper()}.wav").exists()]
    ft_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    dbn = make_dbn(min_bpm=55.0, transition_lambda=100)
    res = {nm: {"peak": {}, "dbn": {}} for nm in ("base", "eq")}
    for nm, stt in (("base", base_state), ("eq", ft_state)):
        model.load_state_dict(stt); model.eval()
        for t in ids:
            sig, sr = load_audio(str(SMC_AUD / f"{t.upper()}.wav"))
            b, d = a2f(sig, sr)
            bp = torch.sigmoid(b.float()).cpu().numpy()
            dp = torch.sigmoid(d.float()).cpu().numpy()
            gt = load_gt(t)
            res[nm]["peak"][t] = ev(gt, peakpick(bp, 0.5, FPS))
            res[nm]["dbn"][t] = ev(gt, dbn_beats(bp, dp, dbn))
    A = lambda r: {m: float(np.mean([r[t][m] for t in ids])) for m in METRICS}
    print(f"\nn={len(ids)}")
    for dec in ("peak", "dbn"):
        base, eqm = A(res["base"][dec]), A(res["eq"][dec])
        dF = np.array([res["eq"][dec][t]["F"] - res["base"][dec][t]["F"] for t in ids])
        print(f"  [{dec}] base " + " ".join(f"{m} {base[m]:.4f}" for m in METRICS))
        print(f"  [{dec}] eq   " + " ".join(f"{m} {eqm[m]:.4f}" for m in METRICS)
              + f"   dF {dF.mean():+.4f} w/l {int((dF>0).sum())}/{int((dF<0).sum())}")
    json.dump({"fold": args.fold, "n": len(ids), "args": vars(args),
               "summary": {d: {n: A(res[n][d]) for n in res} for d in ("peak", "dbn")},
               "per_track": res}, open(f"{args.out}/fold{args.fold}.json", "w"), indent=1, default=str)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
