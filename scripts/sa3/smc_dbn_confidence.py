"""Use the DBN's Viterbi log-likelihood as a confidence signal, and as a router.

Motivation: TTA (averaging activations over pitch-shifted copies) gains +0.0068 F
over 217 tracks but wins on only 105/217 -- it helps and harms in almost equal
measure. If we could tell in advance which, we would keep the gains. The archived
work found all TTA routing schemes lost to always-on, but those used heuristic
confidence; the DBN log-likelihood is a model-based alternative that asks whether
the activation curve supports a coherent metrical structure at all.

Two uses, and they are not equally defensible:

  ROUTE   log P(base activations) is low  ->  the model has no coherent reading,
          so try TTA. This compares one curve against a threshold and is sound.

  SELECT  pick whichever of base/TTA has the higher log P. This compares
          likelihoods of DIFFERENT observation sequences, so a systematic bias is
          possible -- averaging smooths the curve, which may raise likelihood
          regardless of whether the beats got better. Tested, but read with care.

Also caches activations so later analyses cost nothing.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def dbn_logprob(proc, beat_prob, downbeat_prob):
    """Best Viterbi log-likelihood per frame over the DBN's HMMs."""
    from madmom.features.downbeats import _process_dbn
    eps = 1e-5
    act = np.vstack((np.maximum(beat_prob - downbeat_prob, eps / 2),
                     downbeat_prob)).T
    # madmom takes log(1 - sum(act)) per frame, so a saturated frame gives -inf
    # and kills the whole path. Observed peaks reach 0.989, close enough to clip.
    act = np.clip(act, eps / 2, 1.0)
    tot = act.sum(axis=1, keepdims=True)
    over = (tot > 1.0 - 1e-3).ravel()
    if over.any():
        act[over] = act[over] / tot[over] * (1.0 - 1e-3)
    best = -np.inf
    for hmm in proc.hmms:
        try:
            _, lp = _process_dbn((hmm, act))
        except Exception:
            continue
        best = max(best, float(lp))
    return best / max(1, len(act))          # per frame, so lengths compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shifts", type=int, nargs="+", default=[-4, -2, 0, 2, 4])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--out", default="/disk1/jaehoon/dbn_confidence.csv")
    ap.add_argument("--act-cache", default="/disk1/jaehoon/smc_activations.npz")
    a = ap.parse_args()

    import torch, soundfile as sf, soxr, librosa, csv
    import mir_eval.beat as mb
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    from beat_this.inference import load_model, split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect
    from smc_eval import fold_songs, FPS

    dev = f"cuda:{a.gpu}"
    spect = LogMelSpect(device=dev)
    post = Postprocessor(type="minimal", fps=FPS)
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0,
                                        max_bpm=215.0, fps=FPS,
                                        transition_lambda=100)

    rows, cache = [], {}
    for fold in a.folds:
        net = load_model(f"fold{fold}", device=dev).eval()
        for stem, wav, truth in fold_songs(fold):
            truth = np.asarray(truth)
            if len(truth) < 8:
                continue
            y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            y22 = soxr.resample(y, sr, 22050) if sr != 22050 else y

            accb = accd = None
            for s in a.shifts:
                ys = y22 if s == 0 else librosa.effects.pitch_shift(
                    y=y22, sr=22050, n_steps=s)
                with torch.inference_mode():
                    pr = split_predict_aggregate(
                        spect(torch.from_numpy(ys).to(dev)), 1500, 6, "keep_first", net)
                b, d = pr["beat"].float(), pr["downbeat"].float()
                if s == 0:
                    b0, d0 = b, d
                if accb is None:
                    accb, accd = b.clone(), d.clone()
                else:
                    n = min(len(accb), len(b))
                    accb = accb[:n] + b[:n]; accd = accd[:n] + d[:n]
            accb /= len(a.shifts); accd /= len(a.shifts)

            rt = mb.trim_beats(truth)
            out = {}
            for tag, bb, dd in (("base", b0, d0), ("tta", accb, accd)):
                est, _ = post(bb, dd)
                est = np.asarray(est)
                out[f"F_{tag}"] = (mb.f_measure(rt, mb.trim_beats(est))
                                   if len(est) >= 4 else 0.0)
                pb = torch.sigmoid(bb).cpu().numpy()
                pd = torch.sigmoid(dd).cpu().numpy()
                out[f"lp_{tag}"] = dbn_logprob(proc, pb, pd)
                cache[f"{stem}/{tag}/beat"] = pb.astype(np.float16)
                cache[f"{stem}/{tag}/downbeat"] = pd.astype(np.float16)
            rows.append((stem, out["F_base"], out["F_tta"], out["lp_base"], out["lp_tta"]))
        print(f"[fold {fold}] done", flush=True)

    np.savez_compressed(a.act_cache, **cache)
    with open(a.out, "w") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "F_base", "F_tta", "lp_base", "lp_tta"])
        w.writerows(rows)
    print(f"[dump] {a.out}  and activations -> {a.act_cache}")

    r = np.array([[x[1], x[2], x[3], x[4]] for x in rows])
    Fb, Ft, Lb, Lt = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    from scipy import stats
    print(f"\nn={len(r)}")
    print(f"  always-base   F={Fb.mean():.4f}")
    print(f"  always-TTA    F={Ft.mean():.4f}   ({Ft.mean()-Fb.mean():+.4f})")
    print(f"  ORACLE router F={np.maximum(Fb,Ft).mean():.4f}   "
          f"({np.maximum(Fb,Ft).mean()-Fb.mean():+.4f})   <- ceiling for ANY router")
    sel = np.where(Lt > Lb, Ft, Fb)
    print(f"  SELECT by lp  F={sel.mean():.4f}   ({sel.mean()-Fb.mean():+.4f})")
    print(f"\n  corr(lp_base, F_base)          r={stats.pearsonr(Lb,Fb)[0]:+.3f} "
          f"p={stats.pearsonr(Lb,Fb)[1]:.3g}   <- is lp a valid confidence signal?")
    gain = Ft - Fb
    print(f"  corr(lp_base, TTA gain)        r={stats.pearsonr(Lb,gain)[0]:+.3f} "
          f"p={stats.pearsonr(Lb,gain)[1]:.3g}   <- can lp predict WHEN TTA helps?")
    print(f"  corr(lp_tta - lp_base, gain)   r={stats.pearsonr(Lt-Lb,gain)[0]:+.3f} "
          f"p={stats.pearsonr(Lt-Lb,gain)[1]:.3g}")
    for q in (0.25, 0.5):
        thr = np.quantile(Lb, q)
        routed = np.where(Lb <= thr, Ft, Fb)
        print(f"  ROUTE lowest {int(100*q)}% lp -> TTA:  F={routed.mean():.4f} "
              f"({routed.mean()-Fb.mean():+.4f})")


if __name__ == "__main__":
    main()
