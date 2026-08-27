"""Fine-tune Beat This on real + SA3-generated data using its OWN training stack.

Rationale: we are only adding data. Everything else -- pitch/tempo/mask
augmentation, positive weights with widen_target_mask=3, padding masks, cosine
LR with warmup, mixed precision, checkpointing -- already exists in
BeatDataModule/PLBeatThis and is correct there. A hand-rolled loop reproduced
four bugs in one afternoon before this script existed.

What is NOT vanilla, and why:
* weights start from the RELEASED fold-k checkpoint (equal playing field:
  "same weights, further trained" vs "same weights untouched"), loaded into
  PLBeatThis directly rather than via Trainer(ckpt_path=...), which is Lightning's
  *resume* and would restore the optimizer at epoch 100 and stop immediately;
* lr is lowered and max_epochs is small, because this is a fine-tune;
* scoring is done afterwards by sa3_finetune_beat_this.py's evaluator, which is
  already validated and reproduces the published fold-0 baseline (0.6553),
  whereas the built-in val set mixes every dataset's fold-k items together.

Fold discipline: synthetic clips are staged into folds != k (asserted at staging
time), so the eval fold reaches the tracker through no path.
"""
import argparse, sys
from pathlib import Path

BT = Path("/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
sys.path.insert(0, str(BT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/disk1/jaehoon/bt_combined/data")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=1500,
                    help="explicit optimizer-step budget. Fixing this (rather than "
                    "epochs) makes arms with different dataset sizes comparable: "
                    "identical step count AND identical CosineWarmupScheduler "
                    "horizon, the two confounds the audit flagged.")
    ap.add_argument("--real-frac", type=float, default=1.0,
                    help="deterministically subsample real training items (scaling curve)")
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    # accumulate=1 on purpose: at 8 (train.py's from-scratch setting) three
    # epochs is only ~246 OPTIMIZER steps, so a 200-step warmup would leave
    # 80% of the fine-tune still ramping the LR and nothing would move.
    ap.add_argument("--accumulate", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--train-length", type=int, default=1500)
    ap.add_argument("--out", default="/disk1/jaehoon/bt_ft_runs")
    ap.add_argument("--name", default="sa3gen")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--init-ckpt", default=None,
                    help="initialize weights from one of OUR Lightning checkpoints "
                         "(curriculum stage 2: synth-pretrained -> real training). "
                         "Weights only; fresh optimizer and schedule.")
    ap.add_argument("--from-scratch", action="store_true",
                    help="random init instead of the released checkpoint. Removes the "
                         "converged-optimum drift entirely and is the only setting that "
                         "tests whether data COMPOSITION shapes what the model learns.")
    ap.add_argument("--exclude-synth", action="store_true",
                    help="control arm: identical recipe, real data only")
    ap.add_argument("--reading-lambda", type=float, default=0.0,
                    help="weight of the contrastive metrical-reading loss "
                         "(bt_reading_loss.py): the annotated grid must outscore "
                         "offbeat/double/half readings of the same excerpt. "
                         "Targets the level/phase ambiguity that dominates SMC "
                         "failures; NOT bounded by the +0.028 decoding oracle, "
                         "since it changes the activations themselves.")
    ap.add_argument("--reading-wide", action="store_true",
                    help="widen the candidate set with 3:2, 4:3, 3x, 1/3x ratios "
                         "and third-of-beat phases. A softmax only moves mass onto "
                         "candidates it contains: with the narrow set, suppressing "
                         "`double` (29.6->11.1%%) pushed level_ok DOWN (25.9->11.1%%), "
                         "i.e. onto uncovered readings.")
    ap.add_argument("--seed", type=int, default=0,
                    help="replicate seed. Fold 0 is 27 tracks (se~0.024 on mean F), "
                         "so a single run cannot resolve the ~+0.05 this project "
                         "needs; run several and test the paired per-track delta.")
    args = ap.parse_args()

    import torch
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from beat_this.dataset import BeatDataModule
    from beat_this.model.pl_module import PLBeatThis
    from beat_this.inference import load_checkpoint

    seed_everything(args.seed, workers=True)

    augmentations = {} if args.no_augment else {
        "tempo": {"min": -20, "max": 20, "stride": 4},
        "pitch": {"min": -5, "max": 6},
        "mask": {"kind": "permute", "min_count": 1, "max_count": 6,
                 "min_len": 0.1, "max_len": 2, "min_parts": 5, "max_parts": 9},
    }

    dm = BeatDataModule(
        Path(args.data_dir), batch_size=args.batch_size,
        train_length=args.train_length, spect_fps=50,
        num_workers=args.num_workers, test_dataset="gtzan",
        length_based_oversampling_factor=0, augmentations=augmentations,
        no_val=False, fold=args.fold,
    )
    dm.setup(stage="fit")

    if args.real_frac < 1.0:
        # Subsample AFTER setup, then REBUILD train_dataset. Filtering
        # dm.train_items alone is cosmetic: setup() already constructed
        # dm.train_dataset from the unfiltered list and the dataloader reads the
        # dataset, not the list -- two "different" fractions trained on identical
        # data and produced bit-identical curves.
        import zlib
        from beat_this.dataset.dataset import BeatTrackingDataset
        before = len(dm.train_items)
        dm.train_items = [i for i in dm.train_items
                          if (zlib.crc32(i.encode()) % 10000) / 10000.0 < args.real_frac]
        dm.train_dataset = BeatTrackingDataset(
            dm.train_items, deterministic=False, augmentations=dm.augmentations,
            train_length=dm.train_length, data_folder=dm.data_dir,
            spect_fps=dm.spect_fps,
            length_based_oversampling_factor=dm.length_based_oversampling_factor)
        print(f"[data] real-frac {args.real_frac}: {before} -> {len(dm.train_items)} items "
              f"(train_dataset rebuilt: {len(dm.train_dataset)} samples)", flush=True)

    if args.exclude_synth:
        # Same trap as --real-frac above: filtering dm.train_items is cosmetic,
        # because dm.train_dataset was already built from the unfiltered list and
        # train_dataloader() reads the dataset. Re-running dm.setup("fit") makes it
        # worse -- setup() resets train_items from the split files, so it undoes the
        # filter and rebuilds the dataset with every synthetic clip. Filter, then
        # rebuild the dataset from the filtered list.
        from beat_this.dataset.dataset import BeatTrackingDataset
        before = len(dm.train_items)
        dm.train_items = [i for i in dm.train_items if not i.startswith("sa3gen/")]
        dm.train_dataset = BeatTrackingDataset(
            dm.train_items, deterministic=False, augmentations=dm.augmentations,
            train_length=dm.train_length, data_folder=dm.data_dir,
            spect_fps=dm.spect_fps,
            length_based_oversampling_factor=dm.length_based_oversampling_factor)
        print(f"[control] dropped synthetic: {before} -> {len(dm.train_items)} items "
              f"(train_dataset rebuilt: {len(dm.train_dataset)} samples)", flush=True)

    n_synth = sum(1 for i in dm.train_items if i.startswith("sa3gen/"))
    assert not any(i.startswith("sa3gen/") for i in dm.val_items), \
        "synthetic clip in validation fold"
    print(f"[data] train={len(dm.train_items)} (synthetic {n_synth}, "
          f"{100*n_synth/max(1,len(dm.train_items)):.1f}%)  val={len(dm.val_items)}",
          flush=True)

    # FIXED loss weights -- the released checkpoint's own values. Auto-deriving
    # per-arm gave {21,95} vs {19,88}, i.e. the arms optimized different
    # objectives; the audit measured the resulting calibration shift as larger
    # than the reported effect.
    pos_weights = {"beat": 20, "downbeat": 86}
    print("[data] positive weights (fixed):", pos_weights, flush=True)

    ckpt = load_checkpoint(f"fold{args.fold}", "cpu")
    hp = ckpt["hyper_parameters"]
    pl_model = PLBeatThis(
        spect_dim=128, fps=50,
        transformer_dim=hp.get("transformer_dim", 512),
        ff_mult=hp.get("ff_mult", 4), n_layers=hp.get("n_layers", 6),
        stem_dim=hp.get("stem_dim", 32),
        dropout=hp.get("dropout", {"frontend": 0.1, "transformer": 0.2}),
        lr=args.lr, weight_decay=hp.get("weight_decay", 0.01),
        pos_weights=pos_weights, head_dim=hp.get("head_dim", 32),
        loss_type=hp.get("loss_type", "shift_tolerant_weighted_bce"),
        warmup_steps=args.warmup_steps, max_epochs=args.max_epochs,
        use_dbn=False, eval_trim_beats=hp.get("eval_trim_beats", 5),
        sum_head=hp.get("sum_head", True),
        partial_transformers=hp.get("partial_transformers", True),
    )
    if args.init_ckpt:
        blob = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        miss, unexp = pl_model.load_state_dict(blob["state_dict"], strict=False)
        assert not miss and not unexp, f"init-ckpt mismatch: {miss[:3]} {unexp[:3]}"
        n = sum(1 for _ in blob["state_dict"])
        print(f"[ckpt] curriculum init from {args.init_ckpt} ({n} tensors)", flush=True)
    elif args.from_scratch:
        print("[ckpt] FROM SCRATCH: random init (released weights not loaded)", flush=True)
    else:
        missing, unexpected = pl_model.load_state_dict(ckpt["state_dict"], strict=False)
        assert not [m for m in missing if "model." in m], f"missing model weights: {missing[:5]}"
        print(f"[ckpt] loaded released fold{args.fold}: "
              f"{len(missing)} missing / {len(unexpected)} unexpected", flush=True)

    if args.reading_lambda > 0:
        from bt_reading_loss import attach
        attach(pl_model, args.reading_lambda, wide=args.reading_wide)
        print(f"[loss] contrastive reading term ON  lambda={args.reading_lambda} "
              f"wide={args.reading_wide}",
              flush=True)

    # Score held-out SMC after every epoch, with the evaluator that reproduces
    # the published fold-0 baseline. Watching the trajectory matters because
    # ΔF and its paired sd both grow with training -- the best-powered point is
    # not necessarily the last one.
    import numpy as np, torch as _t
    from pytorch_lightning.callbacks import Callback
    from beat_this.preprocessing import LogMelSpect
    from beat_this.model.postprocessor import Postprocessor
    sys.path.insert(0, "/home/sogang/jaehoon/VBPM/rungs_lab")
    from training.evaluation import beat_f_measure

    ANN = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
    SMC_AUDIO = Path("/home/sogang/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Audio")

    class SMCFoldEval(Callback):
        def __init__(self, fold, dump, min_step_interval=250):
            self.fold = fold; self.base = None; self.base_per = None; self.dump = dump
            self.min_step_interval = min_step_interval; self._last_step = -10**9

        def _score(self, pl_module):
            # the ONE evaluator: chunked split_predict_aggregate, validated at
            # 0.6510 against data/results.csv. Never inline scoring again.
            sys.path.insert(0, str(Path(__file__).parent))
            from smc_eval import score_fold
            was = pl_module.training
            pl_module.eval()
            res = score_fold(pl_module.model, self.fold, pl_module.device)
            if was:
                pl_module.train()
            self.stems = sorted(res)
            self.res = res
            self.ap = float(np.mean([res[s]["ap"] for s in self.stems]))
            self.bpm = np.array([res[s]["bpm"] for s in self.stems])
            self.lvl = 100.0 * float(np.mean([res[s]["level_ok"] for s in self.stems]))
            self.dbl = 100.0 * float(np.mean([res[s]["doubled"] for s in self.stems]))
            return np.array([res[s]["F"] for s in self.stems])

        def on_fit_start(self, trainer, pl_module):
            f = self._score(pl_module); self.base, self.base_per = f.mean(), f
            print(f"[smc-eval] baseline F={f.mean():.4f} frameAP={self.ap:.4f} "
                  f"level_ok={self.lvl:.1f}% doubled={self.dbl:.1f}%", flush=True)

        def on_train_epoch_end(self, trainer, pl_module):
            # step-gated: small --real-frac arms cycle epochs every ~60 steps and
            # would spend more time evaluating than training
            last = trainer.global_step >= trainer.max_steps
            if trainer.global_step - self._last_step < self.min_step_interval and not last:
                return
            self._last_step = trainer.global_step
            f = self._score(pl_module); d = f - self.base_per
            # Per-track dump. The mean alone cannot support a PAIRED arm-vs-arm
            # test, and the effects at stake (0.005-0.02) live inside the noise
            # of an unpaired comparison.
            import csv as _csv
            with open(self.dump % trainer.global_step, "w", newline="") as fh:
                w = _csv.writer(fh); w.writerow(["track", "F", "frame_ap", "level_ok", "doubled"])
                for st in self.stems:
                    r = self.res[st]
                    w.writerow([st, f"{r['F']:.6f}", f"{r['ap']:.6f}",
                                int(r["level_ok"]), int(r["doubled"])])
            print(f"[smc-eval] step {trainer.global_step} ep{trainer.current_epoch}: F={f.mean():.4f}  "
                  f"frameAP={self.ap:.4f}  level_ok={self.lvl:.1f}%  doubled={self.dbl:.1f}%  "
                  f"delta={f.mean()-self.base:+.4f}  paired sd={d.std(ddof=1):.4f}  "
                  f"{int((d>0).sum())}/{len(d)} improved  "
                  f"r(pre,post)={np.corrcoef(self.base_per, f)[0,1]:.4f}", flush=True)
            qq = np.quantile(self.bpm, [1/3, 2/3])
            bands = []
            for lo, hi, lbl in ((-1, qq[0], "slow"), (qq[0], qq[1], "mid"), (qq[1], 1e9, "fast")):
                m = (self.bpm > lo) & (self.bpm <= hi)
                lo_ok = 100 * np.mean([self.res[s]["level_ok"] for s, k in zip(self.stems, m) if k])
                bands.append(f"{lbl} n={m.sum()} F={f[m].mean():.4f} lvl={lo_ok:.0f}%")
            print("           by tempo: " + " | ".join(bands), flush=True)

    outdir = Path(args.out) / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        max_epochs=args.max_epochs, max_steps=args.max_steps, deterministic=True,
        accelerator="gpu", devices=[args.gpu],
        num_sanity_val_steps=0, logger=False, log_every_n_steps=10,
        precision="16-mixed", accumulate_grad_batches=args.accumulate,
        check_val_every_n_epoch=args.max_epochs,   # skip mixed-dataset val
        # save_on_train_epoch_end=True is REQUIRED here. ModelCheckpoint defaults to
        # saving at validation end, and check_val_every_n_epoch=max_epochs means
        # validation effectively never runs -- so every_n_epochs never fired and only
        # last.ckpt (written at fit end) survived. Arms killed mid-run left nothing
        # recoverable: the fold-7 pair died at step 1323 with zero checkpoints.
        callbacks=[ModelCheckpoint(dirpath=str(outdir), save_last=True, save_top_k=-1,
                                   every_n_epochs=5, save_on_train_epoch_end=True,
                                   filename="{epoch:03d}"),
                   SMCFoldEval(args.fold, str(outdir / "pertrack_step%06d.csv"))],
        enable_progress_bar=True,
    )
    trainer.fit(pl_model, dm)
    print(f"[done] checkpoints in {outdir}", flush=True)


if __name__ == "__main__":
    main()
