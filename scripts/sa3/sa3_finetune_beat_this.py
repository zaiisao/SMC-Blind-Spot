"""Pilot: does fine-tuning Beat This on SA3-generated audio improve SMC?

The one experiment that tests the project's actual claim. Everything upstream --
beat control, adherence, rubato -- is instrumental to this number.

Design, mirroring the percussion_bias fine-tune recipe so results stay
comparable to that campaign's baselines:

* start from the RELEASED fold-k checkpoint, so the comparison is
  "same weights, further trained on real+synthetic" vs "same weights untouched";
* train on real fold-k training songs mixed with synthetic clips at a fixed
  ratio (--synth-frac);
* evaluate ONLY on fold-k SMC, which neither the generator nor its conditioning
  grids ever saw.

Fold discipline is the thing that killed this campaign's predecessor: an SMC
eval once used the fold0 checkpoint on all 217 songs, ~7/8 of them in that
model's training data, voiding the absolutes. Hence the explicit assertion below.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path("/home/sogang/jaehoon/VBPM")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "external" / "beat_this"))
# beat_f_measure lives under rungs_lab/, not REPO/training/
sys.path.insert(0, str(REPO / "rungs_lab"))
ANN = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
SMC_AUDIO = Path("/home/sogang/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Audio")
SR, FPS, EXCERPT = 22050, 50, 30.0
DATASETS = ("ballroom", "hainsworth", "beatles", "hjdb", "smc")


def frame_targets(times, n):
    t = torch.zeros(n)
    f = np.round(np.asarray(times) * FPS).astype(np.int64)
    t[f[(f >= 0) & (f < n)]] = 1.0
    return t


def load_wav(path):
    import soundfile as sf, soxr
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    return soxr.resample(y, sr, SR) if sr != SR else y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gen-dir", default="/disk1/jaehoon/sa3_gen_corpus_fold0")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--synth-frac", type=float, default=0.25)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = f"cuda:{args.gpu}"
    from beat_this.inference import load_model
    from beat_this.preprocessing import LogMelSpect
    from beat_this.model.loss import ShiftTolerantBCELoss
    from beat_this.model.postprocessor import Postprocessor
    from training.evaluation import beat_f_measure

    spect = LogMelSpect(device=dev)
    model = load_model(f"fold{args.fold}", device=dev)
    post = Postprocessor(type="minimal", fps=FPS)
    # pos_weight MUST match what the released model was trained with
    # ({'beat': 20, 'downbeat': 86} in its hyper_parameters). At the default 1,
    # positives are underweighted 20x, the cheapest way to cut the loss is to
    # predict fewer beats, and F collapses (-0.19 by step 1400) while the loss
    # falls -- which reads as "synthetic data hurts" when nothing was tested.
    from beat_this.inference import load_checkpoint as _lc
    _pw = _lc(f"fold{args.fold}", "cpu")["hyper_parameters"]["pos_weights"]["beat"]
    print(f"[loss] pos_weight={_pw} (from released fold{args.fold})", flush=True)
    crit = ShiftTolerantBCELoss(pos_weight=_pw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)

    fold_of = {}
    for d in DATASETS:
        for line in open(ANN / d / "8-folds.split"):
            p = line.split()
            if len(p) == 2:
                fold_of[p[0]] = int(p[1])

    real = []
    for d in DATASETS:
        for bp in sorted((ANN / d / "annotations" / "beats").glob("*.beats")):
            a = np.loadtxt(bp, ndmin=2)
            bt = a[:, 0]
            if len(bt) < 8:
                continue
            fold = fold_of.get(bp.stem)
            if fold is None:
                continue
            wav = (SMC_AUDIO / f"{bp.stem.replace('smc_', 'SMC_')}.wav" if d == "smc"
                   else Path("/disk1/jaehoon/dataset_store/audio_by_stem") / d / f"{bp.stem}.wav")
            if not wav.exists():
                continue
            real.append(dict(stem=bp.stem, dataset=d, fold=fold, beats=bt, wav=wav))

    train_real = [s for s in real if s["fold"] != args.fold]
    eval_smc = [s for s in real if s["fold"] == args.fold and s["dataset"] == "smc"]
    assert all(s["fold"] != args.fold for s in train_real), "fold leak in training set"
    assert eval_smc, "no held-out SMC songs for this fold"

    gen = sorted(Path(args.gen_dir).glob("*.wav"))
    synth = [dict(stem=w.stem, dataset="sa3gen", fold=-1,
                  beats=np.loadtxt(w.with_suffix(".beats"), ndmin=2)[:, 0], wav=w)
             for w in gen if w.with_suffix(".beats").exists()]
    print(f"[data] real train={len(train_real)}  synth={len(synth)}  "
          f"eval SMC (fold {args.fold})={len(eval_smc)}", flush=True)

    def excerpt(song):
        """Returns (audio, beats, valid_seconds).

        valid_seconds matters: synthetic clips are 20 s but EXCERPT is 30 s, so
        they get zero-padded. Unmasked, that trains the model on 10 s of silence
        per synthetic item plus a hard boundary that never occurs in real audio
        -- a systematic handicap on exactly the pool under test.
        """
        y = load_wav(song["wav"])
        n = int(EXCERPT * SR)
        if len(y) < n:
            valid = len(y) / SR
            y = np.pad(y, (0, n - len(y)))
            off = 0.0
        else:
            valid = EXCERPT
            start = rng.integers(0, len(y) - n + 1)
            y, off = y[start:start + n], start / SR
        b = song["beats"] - off
        return y, b[(b >= 0) & (b < EXCERPT)], valid

    def evaluate(tag):
        model.eval(); fs = []
        with torch.no_grad():
            for s in eval_smc:
                y = load_wav(s["wav"])
                pr = model(spect(torch.from_numpy(y).to(dev)).unsqueeze(0))
                bt, _ = post(pr["beat"][0], pr["downbeat"][0])
                fs.append(beat_f_measure(s["beats"], np.asarray(bt)))
        model.train()
        print(f"[eval {tag}] SMC fold{args.fold} n={len(fs)} F={np.mean(fs):.4f} "
              f"+/-{np.std(fs, ddof=1)/np.sqrt(len(fs)):.4f}", flush=True)
        return float(np.mean(fs)), fs

    T0 = time.time()
    base_mean, base_per = evaluate("baseline")
    print(f"[timing] baseline eval took {time.time()-T0:.1f}s "
          f"({len(eval_smc)} songs)", flush=True)
    TSTEP = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        batch_y, batch_t, batch_v = [], [], []
        for _ in range(args.batch):
            pool = synth if (synth and rng.random() < args.synth_frac) else train_real
            y, b, v = excerpt(pool[rng.integers(len(pool))])
            batch_y.append(y); batch_t.append(b); batch_v.append(v)
        # LogMelSpect.forward does spect_class(x).T -- a FULL transpose, correct
        # only for 1-D input. Batched (B,T) comes back as (frames,128,B), so the
        # model reads B as the mel dim. Spectrogram per item, then stack, which
        # is exactly what the eval path does.
        sp = torch.stack([spect(torch.from_numpy(y).to(dev)) for y in batch_y])
        pred = model(sp)
        n_fr = pred["beat"].shape[-1]
        tgt = torch.stack([frame_targets(b, n_fr) for b in batch_t]).to(dev)
        # padding mask: 1 on real frames, 0 on zero-padded tail. Beat This's own
        # loss takes this (it uses batch["padding_mask"] for exactly this case).
        mask = torch.zeros_like(tgt)
        for i, v in enumerate(batch_v):
            mask[i, :min(n_fr, int(round(v * FPS)))] = 1.0
        loss = crit(pred["beat"], tgt, mask)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        if step % 100 == 0 or step == 10:
            el = time.time() - TSTEP
            print(f"  step {step}/{args.steps} loss={loss.item():.4f} "
                  f"[{el/step:.2f}s/step, eta {(args.steps-step)*el/step/60:.1f} min]", flush=True)
        if step % args.eval_every == 0:
            m, per = evaluate(f"step{step}")
            d = np.array(per) - np.array(base_per)
            print(f"    delta vs baseline = {m-base_mean:+.4f}  "
                  f"(paired sd {d.std(ddof=1):.4f}, {int((d>0).sum())}/{len(d)} improved)", flush=True)


if __name__ == "__main__":
    main()
