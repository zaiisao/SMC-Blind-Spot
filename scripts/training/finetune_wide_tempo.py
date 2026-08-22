"""Fine-tune Beat This with a WIDER tempo augmentation range, then test on SMC.

MOTIVATION (measured, not assumed). Beat This trains with
    augmentations["tempo"] = {"min": -20, "max": 20, "stride": 4}
i.e. stretch factors ~[0.83, 1.25]. Stretching the SAME corpus audio outside that range and
re-scoring against correspondingly scaled annotations shows a clear cliff (100 tracks, fold-honest
checkpoints, nothing retrained):

    stretch 0.85 -> F 0.9103   (in range)      stretch 1.40 -> F 0.8668  (-0.044)
    stretch 1.00 -> F 0.9104   (in range)      stretch 1.70 -> F 0.7917  (-0.119)
    stretch 1.20 -> F 0.8876   (in range)      stretch 2.00 -> F 0.7378  (-0.173)

Same music, same beats, only the rate changed. Degradation starts exactly where the augmentation
range ends and is asymmetric -- slowing hurts more than speeding, matching the fact that all three
tempo estimators we tested read slow material as too fast. That is ~0.17 F of pure tempo-robustness
loss on slow material, on the axis SMC needs (median 70.9 BPM, 21% below 55).

METHOD. Fine-tune from a released fold checkpoint on corpus spectrograms stretched over a wider
range. Stretching is done by resampling the mel TIME axis, which is the same transformation the
cliff diagnostic used, so the two are internally consistent. The confound this could introduce --
learning the resampling artefact rather than genuine slowness -- cannot help on the evaluation set,
because SMC is real slow music that is never resampled.

FOLD HONESTY. Fine-tune fold k on corpus tracks NOT in fold k, then evaluate only on the SMC tracks
that belong to fold k, exactly as the released checkpoints are used.
"""
import argparse, json, sys, zipfile
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third-party" / "beat_this"))
from beat_this.inference import load_model, split_predict_aggregate
from beat_this.model.loss import ShiftTolerantBCELoss
from _harness import FPS, track_ids, load_gt, ev, peakpick

STORE = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
SPECT = Path("/disk1/jaehoon/beat_this_spect/done")
SMC_AUD = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"


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
    """Sharp one-hot frame targets, as Beat This uses (the loss supplies shift tolerance)."""
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
    ap.add_argument("--stretches", default="0.7,0.85,1.0,1.2,1.5,1.8")
    ap.add_argument("--datasets", default="ballroom,hainsworth,beatles,gtzan,asap")
    ap.add_argument("--tracks-per-ds", type=int, default=60)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--crop", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--pos-weight-beat", type=float, default=20.4)
    ap.add_argument("--pos-weight-downbeat", type=float, default=91.2)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="/tmp/overnight/ft_wide")
    args = ap.parse_args()
    S = [float(x) for x in args.stretches.split(",")]
    dev = torch.device(args.device)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # ---- training pool: corpus tracks NOT in this fold ----
    pool = []
    handles = {}
    for ds in args.datasets.split(","):
        sp = split_of(ds)
        keys = [n[:-len("/track.npy")] for n in zipfile.ZipFile(SPECT / f"{ds}.npz").namelist()
                if n.endswith("/track.npy")]
        keys = [k for k in sorted(keys) if sp.get(k, -1) != args.fold][:args.tracks_per_ds]
        for k in keys:
            pool.append((ds, k))
    print(f"fold {args.fold}: {len(pool)} training tracks, stretches {S}", flush=True)

    model = load_model(f"fold{args.fold}", dev)
    model.train()
    # Beats are ~2% of frames. With pos_weight=1 the loss is dominated by negatives and the
    # optimum is to SUPPRESS activations: a first run showed loss falling 0.19->0.11 while F
    # collapsed 0.630->0.495. Beat This calls this weight "crucial when not using a DBN"
    # (their ablation: 92.6 -> 79.5 F without it). Measured on our corpora: beat 20.4, db 91.2.
    loss_b = ShiftTolerantBCELoss(pos_weight=args.pos_weight_beat, tolerance=3).to(dev)
    loss_d = ShiftTolerantBCELoss(pos_weight=args.pos_weight_downbeat, tolerance=3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(0)

    def batch():
        ds, tid = pool[rng.integers(len(pool))]
        if ds not in handles:
            handles[ds] = np.load(SPECT / f"{ds}.npz")
        try:
            sp = np.asarray(handles[ds][f"{tid}/track"], dtype=np.float32)
        except KeyError:
            return None
        t, bn = annots(ds, tid)
        if t is None or len(t) < 8:
            return None
        s = S[rng.integers(len(S))]
        sp2 = zoom(sp, (s, 1.0), order=1) if s != 1.0 else sp
        T = sp2.shape[0]
        if T < args.crop + 8:
            return None
        st = int(rng.integers(0, T - args.crop))
        sp2 = sp2[st:st + args.crop]
        yb, yd = make_targets(t * s - st / FPS, bn, args.crop)
        return (torch.from_numpy(sp2).to(dev),
                torch.from_numpy(yb).to(dev), torch.from_numpy(yd).to(dev))

    run = 0.0
    for step in range(args.steps):
        b = batch()
        if b is None:
            continue
        sp2, yb, yd = b
        out = model(sp2.unsqueeze(0))
        # ShiftTolerantBCELoss max-pools over time, so it needs the batch dim kept
        loss = loss_b(out["beat"], yb.unsqueeze(0)) + loss_d(out["downbeat"], yd.unsqueeze(0))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step(); run += float(loss)
        if (step + 1) % 100 == 0:
            print(f"  step {step+1}/{args.steps} loss {run/100:.4f}", flush=True); run = 0.0

    # ---- evaluate on the SMC tracks belonging to this fold ----
    model.eval()
    smc_split = split_of("smc")
    ids = [t for t in track_ids() if smc_split.get(t, -1) == args.fold]
    print(f"\nevaluating on {len(ids)} SMC tracks held out of fold {args.fold}", flush=True)
    # Use Audio2Frames rather than reimplementing the front end: signal2spect also RESAMPLES to
    # 22050 Hz, and omitting that silently produced F 0.089 instead of ~0.63 on the released model.
    from beat_this.inference import Audio2Frames
    from beat_this.preprocessing import load_audio
    a2f = Audio2Frames(checkpoint_path=f"fold{args.fold}", device=args.device)
    base_state = {k: v.clone() for k, v in a2f.model.state_dict().items()}
    ft_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rows = {"base": {}, "ft": {}}
    for t in ids:
        w = SMC_AUD / f"{t.upper()}.wav"
        if not w.exists():
            continue
        sig, sr = load_audio(str(w))
        for nm, st in (("base", base_state), ("ft", ft_state)):
            a2f.model.load_state_dict(st); a2f.model.eval()
            b, _ = a2f(sig, sr)
            est = peakpick(torch.sigmoid(b.float()).cpu().numpy(), 0.5, FPS)
            rows[nm][t] = ev(load_gt(t), est)
    ok = [t for t in ids if t in rows["base"] and t in rows["ft"]]
    A = lambda r: {m: float(np.mean([r[t][m] for t in ok])) for m in ("F", "CMLt", "AMLt", "AMLc")}
    print(f"\nn={len(ok)}")
    print(f"  released fold{args.fold}      {A(rows['base'])}")
    print(f"  wide-tempo fine-tune   {A(rows['ft'])}")
    d = np.array([rows["ft"][t]["F"] - rows["base"][t]["F"] for t in ok])
    print(f"  dF {d.mean():+.4f}  win/loss {int((d>0).sum())}/{int((d<0).sum())}")
    json.dump({"fold": args.fold, "n": len(ok), "base": A(rows["base"]), "ft": A(rows["ft"]),
               "per_track": {t: {"base": rows["base"][t], "ft": rows["ft"][t]} for t in ok}},
              open(f"{args.out}/fold{args.fold}.json", "w"), indent=1)
    torch.save(model.state_dict(), f"{args.out}/model_fold{args.fold}.pt")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
