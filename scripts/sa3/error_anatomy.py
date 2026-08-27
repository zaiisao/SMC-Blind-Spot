"""Where exactly do beat-tracker errors live? Diagnostics A, B, C.

The loss under-penalises double-tempo firing (a miss costs pos_weight=32 at one
frame; a spurious peak costs 1 across the ~7 frames its +/-3 spreading covers,
so roughly 4-5:1 in favour of firing). The proposed fix -- hard negatives at
subdivision positions -- ASSUMES the false positives actually sit at subdivisions.
This measures that instead of assuming it.

A. FALSE POSITIVE PHASE. For every predicted beat matching no true beat, its
   phase within the enclosing true inter-beat interval, in [0,1).
     peak at 0.50        -> duple subdivision; midpoint hard negatives are right
     peaks at 0.33/0.67  -> triple; thirds too
     peak near 0.60-0.66 -> swing; a fixed midpoint penalty would MISS it
     flat                -> the subdivision story is wrong; do not build it

B. MISS PATTERN. True beats with no prediction within 70 ms. If misses alternate
   (every other beat) that is half-tempo; if scattered, it is detection failure.
   A tells us what to SUPPRESS, B what to ENCOURAGE.

C. LOGIT PROFILE. Mean logit at true beats / subdivision midpoints / random
   frames, per tempo band. Confident logits at midpoints are the direct image of
   "nothing ever penalised them", and the before/after picture for the paper.

Fold-honest: each track scored with the checkpoint that held its fold out.
"""
import argparse, glob, json, os
import numpy as np

ANN = "/disk1/jaehoon/dataset_store/beat_this_annotations/smc"
AUDIO = "/disk1/jaehoon/dataset_store/audio_by_stem/smc"
BANDS = [(0, 55, "<55"), (55, 70, "55-70"), (70, 100, "70-100"), (100, 1e9, ">100")]
FPS = 50
TOL = 0.070


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--released", default=None, help="'honest' = per-fold released")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch, soxr, soundfile as sf
    from beat_this.inference import split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect

    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    spect = LogMelSpect(device=dev)
    post = Postprocessor(type="minimal", fps=FPS)
    nets = {}

    def net_for(fold):
        if a.released == "honest":
            if fold not in nets:
                from beat_this.inference import load_model
                nets[fold] = load_model(f"fold{fold}", device=dev).eval()
            return nets[fold]
        if "n" not in nets:
            ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
            from beat_this.model.pl_module import PLBeatThis
            pl = PLBeatThis(**dict(ck["hyper_parameters"]))
            pl.load_state_dict(ck["state_dict"])
            nets["n"] = pl.model.to(dev).eval()
        return nets["n"]

    folds = {}
    for line in open(f"{ANN}/8-folds.split"):
        p = line.split()
        if len(p) >= 2:
            folds[p[0]] = int(p[1])

    rows = []
    for bt in sorted(glob.glob(f"{ANN}/annotations/beats/*.beats")):
        stem = os.path.basename(bt)[:-6]
        if stem not in folds:
            continue
        wav = None
        for e in (".wav", ".flac", ".mp3"):
            p = os.path.join(AUDIO, stem + e)
            if os.path.exists(p):
                wav = p
                break
        if wav is None:
            continue
        truth = np.loadtxt(bt, ndmin=2)[:, 0]
        if len(truth) < 8:
            continue
        y, sr = sf.read(wav, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        y22 = soxr.resample(y, sr, 22050) if sr != 22050 else y
        with torch.inference_mode():
            pr = split_predict_aggregate(
                spect(torch.from_numpy(np.ascontiguousarray(y22)).to(dev)),
                1500, 6, "keep_first", net_for(folds[stem]))
            logit = pr["beat"].float().cpu().numpy().ravel()
            est, _ = post(pr["beat"].float(), pr["downbeat"].float())
        est = np.asarray(est)
        bpm = 60.0 / float(np.median(np.diff(truth)))
        # Tracks where the decoder emitted almost nothing are a REAL failure of
        # that model, not a reason to shrink its denominator. Dropping them made
        # arms comparable only over their own surviving tracks, so differences of
        # fractions were taken over different track sets. Keep them, flagged.
        degenerate = len(est) < 4

        # Restrict predictions to the ANNOTATED SPAN before counting. Annotations
        # often do not cover the whole file (mean ~1.0 s of unannotated lead-in
        # and ~0.65 s of tail here, and >2 s on 23% of tracks), so comparing a
        # whole-file prediction count against a partial reference inflates the
        # est/true ratio -- exactly the statistic the doubling numbers rest on.
        lo, hi = truth[0], truth[-1]
        est_in = est[(est >= lo - TOL) & (est <= hi + TOL)] if len(est) else est

        # ---- A: phase of unmatched predictions inside the enclosing true IBI
        # ONE-TO-ONE matching: a nearest-neighbour test lets several predictions
        # all "match" one beat, so duplicate peaks near a true beat were never
        # counted as false positives -- and those are precisely the ones that
        # would land at small phase, biasing the histogram toward the
        # subdivision bins this diagnostic is testing for.
        used = np.zeros(len(truth), dtype=bool)
        is_fp = np.zeros(len(est_in), dtype=bool)
        for i, e in enumerate(est_in):
            d = np.abs(truth - e)
            d[used] = np.inf
            j = int(np.argmin(d)) if len(d) else -1
            if j >= 0 and d[j] <= TOL:
                used[j] = True
            else:
                is_fp[i] = True

        fp_phase = []
        for e in est_in[is_fp]:
            j = np.searchsorted(truth, e) - 1
            if 0 <= j < len(truth) - 1:
                ibi = truth[j + 1] - truth[j]
                if ibi > 0:
                    fp_phase.append(float((e - truth[j]) / ibi))

        # ---- B: which true beats were missed, and do misses alternate?
        matched = used
        miss_idx = np.where(~matched)[0]
        if len(miss_idx) >= 4:
            d = np.diff(miss_idx)
            alt = float(np.mean(d == 2))      # every-other-beat = half tempo
        else:
            alt = np.nan

        # ---- C: logit at beats vs subdivision midpoints vs random
        n = len(logit)
        def at(ts):
            # round, not truncate: truncation biases the index ~10 ms early,
            # and upstream (prepare_annotations) rounds
            i = np.clip(np.rint(np.asarray(ts) * FPS).astype(int), 0, n - 1)
            return float(np.mean([logit[max(0, k - 1): k + 2].max() for k in i]))
        mids = (truth[:-1] + truth[1:]) / 2.0
        rng = np.random.RandomState(0)
        rnd = rng.uniform(truth[0], truth[-1], len(truth))

        rows.append(dict(stem=stem, bpm=bpm,
                         n_est=int(len(est_in)), n_true=len(truth),
                         n_est_whole=int(len(est)), degenerate=bool(degenerate),
                         span=float(hi - lo),
                         fp_phase=fp_phase, miss_frac=float((~matched).mean()),
                         alt_frac=alt,
                         logit_beat=at(truth), logit_mid=at(mids), logit_rand=at(rnd)))

    json.dump(rows, open(a.out, "w"))

    bpm = np.array([r["bpm"] for r in rows])
    print(f"\n  n={len(rows)} tracks\n")
    print("  === A. FALSE-POSITIVE PHASE inside the true IBI (0=on beat, 0.5=midpoint)")
    edges = np.linspace(0, 1, 11)
    for lo, hi, tag in BANDS:
        m = (bpm >= lo) & (bpm < hi)
        ph = np.concatenate([rows[i]["fp_phase"] for i in np.where(m)[0]
                             if rows[i]["fp_phase"]]) if m.sum() else np.array([])
        if len(ph) < 20:
            print(f"  {tag:>8}  (too few FPs: {len(ph)})")
            continue
        h, _ = np.histogram(ph, bins=edges)
        h = h / h.sum()
        bar = " ".join(f"{v:.2f}" for v in h)
        print(f"  {tag:>8} n_fp={len(ph):5d}  {bar}")
    print(f"  {'':>8} bins:          " +
          " ".join(f"{(edges[i]+edges[i+1])/2:.2f}" for i in range(10)))

    print("\n  === B. MISSES")
    print(f"  {'band':>8}{'miss frac':>11}{'alternating':>13}  (alt~1.0 = half-tempo)")
    for lo, hi, tag in BANDS:
        m = (bpm >= lo) & (bpm < hi)
        if not m.sum():
            continue
        mf = np.mean([rows[i]["miss_frac"] for i in np.where(m)[0]])
        al = np.nanmean([rows[i]["alt_frac"] for i in np.where(m)[0]])
        print(f"  {tag:>8}{mf:11.3f}{al:13.3f}")

    print("\n  === C. LOGITS  (beat vs subdivision midpoint vs random)")
    print(f"  {'band':>8}{'beat':>9}{'midpoint':>10}{'random':>9}{'mid-rand':>10}")
    for lo, hi, tag in BANDS:
        m = (bpm >= lo) & (bpm < hi)
        if not m.sum():
            continue
        b = np.mean([rows[i]["logit_beat"] for i in np.where(m)[0]])
        mid = np.mean([rows[i]["logit_mid"] for i in np.where(m)[0]])
        r = np.mean([rows[i]["logit_rand"] for i in np.where(m)[0]])
        print(f"  {tag:>8}{b:9.3f}{mid:10.3f}{r:9.3f}{mid-r:10.3f}")

    ne = np.array([r["n_est"] for r in rows], dtype=float)
    nt = np.array([r["n_true"] for r in rows], dtype=float)
    print(f"\n  === est/true beat-count ratio (>1.5 ~ double tempo)")
    print(f"  {'band':>8}{'median':>9}{'frac>1.5':>10}{'frac<0.7':>10}")
    for lo, hi, tag in BANDS:
        m = (bpm >= lo) & (bpm < hi)
        if not m.sum():
            continue
        rr = ne[m] / nt[m]
        print(f"  {tag:>8}{np.median(rr):9.2f}{np.mean(rr > 1.5):10.3f}"
              f"{np.mean(rr < 0.7):10.3f}")


if __name__ == "__main__":
    main()
