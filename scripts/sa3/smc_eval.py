"""The one SMC fold evaluator. Every experiment scores through this file.

Why it exists: three copy-pasted evaluators drifted apart, and all of them fed
whole 2001-frame SMC pieces into a model trained on 1500-frame excerpts. The
canonical Beat This inference path (Spect2Frames) chunks with
split_predict_aggregate(chunk_size=1500, border_size=6, overlap_mode='keep_first');
skipping it biased every reported number by ~+0.008 in the direction that
flattered fine-tuned arms. With chunking, the released fold-0 checkpoint scores
0.6510 on fold 0 -- exactly matching data/results.csv.

Reports, per track: F (mir_eval, trim=5s), and a decoder-free frame-level
average precision (targets widened +/-3 frames at fps=50, i.e. +/-60 ms) so
activation-quality changes are visible even when the peak-picker masks them.
smc_197 is flagged in output: it is a cross-fold near-duplicate of training
track smc_178 (same performance, offset ~0.29 s).
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np

BT = Path("/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
sys.path.insert(0, str(BT)); sys.path.insert(0, "/home/sogang/jaehoon/VBPM/rungs_lab")
ANN = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
SMC_AUDIO = Path("/home/sogang/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Audio")
SR, FPS = 22050, 50
FLAGGED = {"smc_197"}   # near-duplicate of smc_178 (fold 2, training)


def fold_songs(fold):
    fo = {}
    for line in open(ANN / "smc" / "8-folds.split"):
        p = line.split()
        if len(p) == 2:
            fo[p[0]] = int(p[1])
    out = []
    for bp in sorted((ANN / "smc" / "annotations" / "beats").glob("*.beats")):
        if fo.get(bp.stem) != fold:
            continue
        wav = SMC_AUDIO / f"{bp.stem.replace('smc_', 'SMC_')}.wav"
        if wav.exists():
            out.append((bp.stem, wav, np.loadtxt(bp, ndmin=2)[:, 0]))
    return out


def load_net(ckpt, device):
    """Accepts a released shortname ('fold0'), a Lightning .ckpt path, or an
    nn.Module. Always returns the bare BeatThis network."""
    import torch
    if hasattr(ckpt, "forward"):
        return ckpt
    from beat_this.inference import load_model, load_checkpoint
    if not str(ckpt).endswith(".ckpt"):
        return load_model(str(ckpt), device)
    from beat_this.model.pl_module import PLBeatThis
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    keep = ("spect_dim", "fps", "transformer_dim", "ff_mult", "n_layers", "stem_dim",
            "dropout", "lr", "weight_decay", "pos_weights", "head_dim", "loss_type",
            "warmup_steps", "max_epochs", "use_dbn", "eval_trim_beats", "sum_head",
            "partial_transformers")
    hp = {k: v for k, v in blob["hyper_parameters"].items() if k in keep}
    # The loss is irrelevant at inference -- only the weights matter. Pin it to
    # the stock loss so a checkpoint trained with a loss class that has since
    # been deleted (the refuted arms: per_item, count, density, gated_phase,
    # bidirectional_phase, tempo_relative_collar) still loads and can be
    # re-scored. Without this, deleting dead code bricks the archived arms.
    hp["loss_type"] = "shift_tolerant_weighted_bce"
    m = PLBeatThis(**hp)
    miss, unexp = m.load_state_dict(blob["state_dict"], strict=False)
    assert not miss and not unexp, f"{ckpt}: missing {miss[:3]} unexpected {unexp[:3]}"
    return m.model.to(device)


def score_fold(ckpt, fold=0, device="cuda:0", postp_type="minimal"):
    """-> dict stem -> {'F': float, 'ap': float, 'flagged': bool}"""
    import torch, soundfile as sf, soxr
    import torch.nn.functional as TF
    from sklearn.metrics import average_precision_score
    from beat_this.preprocessing import LogMelSpect
    from beat_this.inference import split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from training.evaluation import beat_f_measure

    net = load_net(ckpt, device)
    net.eval()
    spect = LogMelSpect(device=device)
    post = Postprocessor(type=postp_type, fps=FPS)
    out = {}
    with torch.inference_mode():
        for stem, wav, truth in fold_songs(fold):
            y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != SR:
                y = soxr.resample(y, sr, SR)
            S = spect(torch.from_numpy(y).to(device))            # (T,128)
            pr = split_predict_aggregate(S, 1500, 6, "keep_first", net)
            bt, _ = post(pr["beat"].float(), pr["downbeat"].float())
            n = pr["beat"].shape[-1]
            tgt = torch.zeros(n)
            f = np.round(truth * FPS).astype(int)
            tgt[f[(f >= 0) & (f < n)]] = 1.0
            wide = TF.max_pool1d(tgt.view(1, 1, -1), 7, 1, 3).view(-1).numpy()
            probs = torch.sigmoid(pr["beat"].float()).cpu().numpy()
            est = np.asarray(bt)
            # metrical-level ratio: >1 means the tracker is FASTER than the
            # annotation (subdivision/doubling). F conflates "no pulse found"
            # with "wrong level chosen"; this separates them, and level choice is
            # where SMC's headroom is (F=0.72 when right, 0.48 when wrong).
            lvl = float("nan")
            if len(est) >= 8 and len(truth) >= 8:
                lvl = float(np.median(np.diff(truth)) / np.median(np.diff(est)))
            out[stem] = dict(
                bpm=float(60.0 / np.median(np.diff(truth))),
                F=float(beat_f_measure(truth, est)),
                ap=float(average_precision_score(wide, probs)),
                level_ratio=lvl,
                level_ok=bool(np.isfinite(lvl) and abs(np.log2(lvl)) < 0.18),
                doubled=bool(np.isfinite(lvl) and abs(np.log2(lvl / 2)) < 0.18),
                flagged=stem in FLAGGED)
    return out


def summarize(res, label=""):
    F = np.array([r["F"] for r in res.values()])
    ap = np.array([r["ap"] for r in res.values()])
    ok = np.array([not r["flagged"] for r in res.values()])
    lvl = np.array([r.get("level_ok", False) for r in res.values()])
    dbl = np.array([r.get("doubled", False) for r in res.values()])
    print(f"[{label}] n={len(F)}  F={F.mean():.4f}  frameAP={ap.mean():.4f}  "
          f"level_ok={100*lvl.mean():.1f}%  doubled={100*dbl.mean():.1f}%   "
          f"(excl. flagged n={ok.sum()}: F={F[ok].mean():.4f})", flush=True)
    return float(F.mean())


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--ckpt", default="fold0")
    ap_.add_argument("--fold", type=int, default=0)
    ap_.add_argument("--postp", default="minimal", choices=["minimal", "dbn"])
    ap_.add_argument("--gpu", type=int, default=0)
    ap_.add_argument("--out", default=None)
    a = ap_.parse_args()
    res = score_fold(a.ckpt, a.fold, f"cuda:{a.gpu}", a.postp)
    summarize(res, f"{a.ckpt} fold{a.fold} {a.postp}")
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["track", "F", "frame_ap", "flagged"])
            for s, r in sorted(res.items()):
                w.writerow([s, f"{r['F']:.4f}", f"{r['ap']:.4f}", int(r["flagged"])])
        print(f"[out] {a.out}")
