"""Does the SA3 VAE MOVE beat positions on SMC, or only soften attacks?

Pass 1 (--stage rt): round-trip all 217 SMC tracks through the SA3 autoencoder
(encode->decode, no diffusion), save to --rtdir at 22050 Hz mono (the rate the
adherence verifier uses).  Also records raw-waveform lag and length delta.

Pass 2 (--stage score): paired comparison of sa3_beat_adherence on REAL vs
ROUND-TRIP with the SAME true annotations, plus
  - per-track onset-envelope cross-correlation lag (constant-latency confound)
  - adherence after correcting for that lag, and after correcting for the
    corpus-median lag (a single global constant)
  - per-beat peak LOCATION shift vs peak SHARPNESS change (position vs shape)
  - stratification by envelope crest factor.
"""
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from sa3_beat_adherence import (beat_adherence, combined_score, smc_pairs,
                                onset_envelope, SR, HOP, TOL)

RTDIR = Path("/tmp/sa3_rt_smc")


# --------------------------------------------------------------------------- #
def stage_rt(args):
    import torch, librosa, soundfile as sf
    from sa3_onset_reference import load_ae
    dev = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    ae, sr = load_ae(dev)
    pairs = smc_pairs()
    out = Path(args.rtdir); out.mkdir(parents=True, exist_ok=True)
    meta = {}
    DS = 4096  # patch_size 256 * stride 16
    for i, (w, _) in enumerate(pairs):
        y, _ = librosa.load(str(w), sr=sr, mono=True)
        N = len(y)
        pad = (-N) % DS
        yp = np.pad(y, (0, pad))
        x = torch.from_numpy(np.stack([yp, yp])).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            r = ae.decode(ae.encode(x))
        rec_full = r[0].float().cpu().numpy().mean(0)
        rec = rec_full[:N]
        # raw waveform lag, full-rate, from cross-correlation of the two signals
        n = min(len(y), len(rec))
        a = y[:n] - y[:n].mean(); b = rec[:n] - rec[:n].mean()
        mx = int(0.050 * sr)
        from scipy.signal import correlate
        cc = correlate(b, a, "full", method="fft")
        c0 = len(a) - 1
        seg = cc[c0 - mx: c0 + mx + 1]
        lag_raw = (int(np.argmax(np.abs(seg))) - mx) / sr
        meta[w.stem] = {"len_in": int(N), "len_out_full": int(len(rec_full)),
                        "pad": int(pad), "wave_lag_s": float(lag_raw),
                        "wave_xcorr_peak_r": float(seg[int(np.argmax(np.abs(seg)))] /
                                                   (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))}
        y22 = librosa.resample(rec, orig_sr=sr, target_sr=SR)
        sf.write(str(out / f"{w.stem}.wav"), y22, SR)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(pairs)}]", flush=True)
    json.dump(meta, open(out / "_rt_meta.json", "w"), indent=1)
    print(f"[rt] wrote {len(meta)} files to {out}")


# --------------------------------------------------------------------------- #
def _xcorr_lag(e1, e2, max_lag_frames):
    """lag (frames) that best aligns e2 onto e1; positive = e2 is LATE."""
    a = e1 - e1.mean(); b = e2 - e2.mean()
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    from scipy.signal import correlate
    cc = correlate(b, a, "full", method="fft")
    c0 = n - 1
    seg = cc[c0 - max_lag_frames: c0 + max_lag_frames + 1]
    k = int(np.argmax(seg))
    lag = k - max_lag_frames
    r = float(seg[k] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return lag, r


def _peak_stats(env, beats, shift=0.0):
    """Per-beat: argmax location (s, relative to the beat) and sharpness
    (peak / mean of the window) inside +/-TOL."""
    k = max(1, int(round(TOL * SR / HOP)))
    locs, sharp, pk = [], [], []
    for b in beats:
        c = int(round((b + shift) * SR / HOP))
        lo, hi = c - k, c + k + 1
        if lo < 0 or hi > len(env):
            continue
        w = env[lo:hi]
        j = int(np.argmax(w))
        locs.append((j - k) * HOP / SR)
        m = float(w.mean())
        sharp.append(float(w[j] / (m + 1e-12)))
        pk.append(float(w[j]))
    return np.array(locs), np.array(sharp), np.array(pk)


def _one(job):
    wav, ann, rt, seed = job
    import librosa
    y, _ = librosa.load(str(wav), sr=SR, mono=True)
    z, _ = librosa.load(str(rt), sr=SR, mono=True)
    beats = np.loadtxt(str(ann)).astype(float).ravel()
    dur = len(y) / SR
    beats = beats[(beats >= 0) & (beats < dur)]
    if len(beats) < 6:
        return None
    ey, ez = onset_envelope(y), onset_envelope(z)
    max_lag = int(round(0.100 * SR / HOP))
    lagf, xr = _xcorr_lag(ey, ez, max_lag)
    lag_s = lagf * HOP / SR

    ly, sy, py = _peak_stats(ey, beats)
    lz, sz, pz = _peak_stats(ez, beats)
    n = min(len(ly), len(lz))
    return {
        "track": wav.stem,
        "n_beats": int(len(beats)),
        "real": beat_adherence(y, beats, seed=seed),
        "rt": beat_adherence(z, beats, seed=seed),
        "rt_corr": beat_adherence(z, beats + lag_s, seed=seed),
        "env_lag_s": float(lag_s), "env_xcorr_r": float(xr),
        "peak_loc_real_mean": float(ly[:n].mean()), "peak_loc_rt_mean": float(lz[:n].mean()),
        "peak_loc_shift_mean": float((lz[:n] - ly[:n]).mean()),
        "peak_loc_shift_absmean": float(np.abs(lz[:n] - ly[:n]).mean()),
        "peak_loc_shift_sd": float((lz[:n] - ly[:n]).std()),
        "frac_peak_moved_gt10ms": float(np.mean(np.abs(lz[:n] - ly[:n]) > 0.010)),
        "sharp_real": float(sy[:n].mean()), "sharp_rt": float(sz[:n].mean()),
        "n_peaks": int(n),
    }


def _agg(vals):
    v = np.asarray(vals, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "median": float(np.median(v)), "n": int(len(v))}


def stage_score(args):
    from scipy import stats
    pairs = smc_pairs()
    rt = Path(args.rtdir)
    jobs = [(w, a, rt / f"{w.stem}.wav", i) for i, (w, a) in enumerate(pairs)
            if (rt / f"{w.stem}.wav").exists()]
    print(f"[setup] {len(pairs)} pairs, {len(jobs)} with round-trip audio", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(_one, jobs)):
            if r:
                rows.append(r)
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(jobs)}]", flush=True)

    S = {}
    lag = np.array([r["env_lag_s"] for r in rows])
    S["env_lag"] = _agg(lag)
    S["env_lag"]["iqr"] = [float(np.quantile(lag, .25)), float(np.quantile(lag, .75))]
    S["env_lag"]["frac_within_1_frame_of_median"] = float(
        np.mean(np.abs(lag - np.median(lag)) <= HOP / SR + 1e-9))
    S["env_lag"]["hist"] = {f"{v*1000:+.1f}ms": int(c) for v, c in
                            zip(*np.unique(np.round(lag, 6), return_counts=True))}
    S["env_xcorr_r"] = _agg([r["env_xcorr_r"] for r in rows])

    gmed = float(np.median(lag))
    print(f"\n[lag] env xcorr lag: mean {lag.mean()*1000:+.2f} ms  median {gmed*1000:+.2f} ms"
          f"  sd {lag.std(ddof=1)*1000:.2f} ms  IQR {S['env_lag']['iqr'][0]*1000:+.1f}"
          f"..{S['env_lag']['iqr'][1]*1000:+.1f} ms")
    print(f"       {S['env_lag']['frac_within_1_frame_of_median']*100:.1f}% of tracks "
          f"within +/-1 frame ({HOP/SR*1000:.2f} ms) of the median")
    print(f"       hist: {S['env_lag']['hist']}")

    # global-constant-corrected pass, single median offset for the whole corpus
    if abs(gmed) > 1e-9:
        gj = [(w, a, rt / f"{w.stem}.wav", i, gmed) for i, (w, a) in enumerate(pairs)
              if (rt / f"{w.stem}.wav").exists()]
        with ProcessPoolExecutor(args.workers) as ex:
            gres = list(ex.map(_global_corr, gj))
        for r, g in zip(rows, gres):
            r["rt_gcorr"] = g
    else:
        for r in rows:
            r["rt_gcorr"] = r["rt"]

    conds = ["real", "rt", "rt_corr", "rt_gcorr"]
    print(f"\n{'condition':10s} {'phase_align':>24s} {'beat_energy_z':>22s} {'combined':>10s}")
    for c in conds:
        pa = np.array([r[c]["phase_align"] for r in rows])
        bz = np.array([r[c]["beat_energy_z"] for r in rows])
        cb = np.array([combined_score(r[c]) for r in rows])
        S[c] = {"phase_align": _agg(pa), "beat_energy_z": _agg(bz), "combined": _agg(cb)}
        print(f"{c:10s} mean {pa.mean():+.4f} sd {pa.std(ddof=1):.4f} "
              f"  mean {bz.mean():+6.2f} sd {bz.std(ddof=1):5.2f}  {cb.mean():+9.3f}")

    print(f"\n{'paired test (per-track delta vs real)':46s}{'delta':>10}{'95% CI':>22}"
          f"{'t p':>10}{'wilcoxon p':>12}{'dz':>7}")
    S["paired"] = {}
    for c in conds[1:]:
        for key in ("phase_align", "beat_energy_z"):
            a = np.array([r["real"][key] for r in rows])
            b = np.array([r[c][key] for r in rows])
            d = b - a
            t, p = stats.ttest_rel(b, a)
            try:
                _, pw = stats.wilcoxon(b, a)
            except ValueError:
                pw = float("nan")
            se = d.std(ddof=1) / np.sqrt(len(d))
            ci = (d.mean() - 1.96 * se, d.mean() + 1.96 * se)
            dz = d.mean() / d.std(ddof=1)
            S["paired"][f"{c}:{key}"] = {"delta": float(d.mean()),
                                         "ci95": [float(ci[0]), float(ci[1])],
                                         "t": float(t), "p_ttest": float(p),
                                         "p_wilcoxon": float(pw), "dz": float(dz),
                                         "sd_delta": float(d.std(ddof=1)), "n": len(d)}
            print(f"{c+' - real  ['+key+']':46s}{d.mean():>+10.4f}"
                  f"{f'[{ci[0]:+.4f},{ci[1]:+.4f}]':>22}{p:>10.2g}{pw:>12.2g}{dz:>+7.2f}")

    # position vs shape
    sh = np.array([r["peak_loc_shift_mean"] for r in rows])
    sha = np.array([r["peak_loc_shift_absmean"] for r in rows])
    sr_ = np.array([r["sharp_real"] for r in rows])
    sz_ = np.array([r["sharp_rt"] for r in rows])
    ts, ps = stats.ttest_rel(sz_, sr_)
    S["position_vs_shape"] = {
        "peak_loc_shift_mean_ms": _agg(sh * 1000),
        "peak_loc_shift_absmean_ms": _agg(sha * 1000),
        "peak_loc_shift_sd_within_track_ms": _agg(
            [r["peak_loc_shift_sd"] * 1000 for r in rows]),
        "frac_peaks_moved_gt10ms": _agg([r["frac_peak_moved_gt10ms"] for r in rows]),
        "sharpness_real": _agg(sr_), "sharpness_rt": _agg(sz_),
        "sharpness_delta": float((sz_ - sr_).mean()),
        "sharpness_rel_change": float((sz_ - sr_).mean() / sr_.mean()),
        "sharpness_p_ttest_rel": float(ps),
        "frame_resolution_ms": HOP / SR * 1000,
    }
    print(f"\n[position vs shape]  per-beat onset-envelope peak inside +/-{TOL*1000:.0f} ms")
    print(f"  peak LOCATION shift : mean {sh.mean()*1000:+.2f} ms, |shift| {sha.mean()*1000:.2f} ms,"
          f" within-track sd {np.mean([r['peak_loc_shift_sd'] for r in rows])*1000:.2f} ms")
    print(f"  frac of beats whose peak moved >10 ms: "
          f"{np.mean([r['frac_peak_moved_gt10ms'] for r in rows]):.3f}")
    print(f"  peak SHARPNESS      : real {sr_.mean():.3f} -> rt {sz_.mean():.3f} "
          f"({(sz_.mean()-sr_.mean())/sr_.mean()*100:+.2f}%, paired p={ps:.2g})")

    # strata by crest factor of the REAL envelope
    cr = np.array([r["real"]["onset_strength"] for r in rows])
    q = np.quantile(cr, [.25, .5, .75])
    edges = [-np.inf, q[0], q[1], q[2], np.inf]
    names = ["Q1 softest", "Q2", "Q3", "Q4 most impulsive"]
    print(f"\ncrest quartiles {q.round(2)}")
    print(f"{'stratum':20s}{'n':>4}{'real pa':>9}{'rt pa':>9}{'rt_corr pa':>12}"
          f"{'d(rt_corr)':>12}{'p':>10}{'|peak shift|ms':>16}")
    S["strata"] = {}
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        m = (cr >= lo) & (cr < hi)
        sub = [r for r, k in zip(rows, m) if k]
        if len(sub) < 3:
            continue
        a = np.array([r["real"]["phase_align"] for r in sub])
        b = np.array([r["rt"]["phase_align"] for r in sub])
        c = np.array([r["rt_corr"]["phase_align"] for r in sub])
        _, p = stats.ttest_rel(c, a)
        d = c - a
        se = d.std(ddof=1) / np.sqrt(len(d))
        pmv = np.mean([r["peak_loc_shift_absmean"] for r in sub]) * 1000
        S["strata"][nm] = {"n": len(sub), "crest_range": [float(cr[m].min()), float(cr[m].max())],
                           "real_phase_align": float(a.mean()), "rt_phase_align": float(b.mean()),
                           "rt_corr_phase_align": float(c.mean()),
                           "delta_rt_corr": float(d.mean()),
                           "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                           "p_ttest_rel": float(p),
                           "peak_loc_shift_absmean_ms": float(pmv)}
        print(f"{nm:20s}{len(sub):>4}{a.mean():>+9.4f}{b.mean():>+9.4f}{c.mean():>+12.4f}"
              f"{d.mean():>+12.4f}{p:>10.2g}{pmv:>16.2f}")

    meta_p = Path(args.rtdir) / "_rt_meta.json"
    if meta_p.exists():
        mt = json.load(open(meta_p))
        wl = np.array([v["wave_lag_s"] for v in mt.values()])
        wr = np.array([v["wave_xcorr_peak_r"] for v in mt.values()])
        S["waveform"] = {"lag_ms": _agg(wl * 1000), "xcorr_r": _agg(wr),
                         "len_in_eq_len_out": bool(all(v["len_in"] <= v["len_out_full"]
                                                       for v in mt.values())),
                         "pad_samples": _agg([v["pad"] for v in mt.values()])}
        print(f"\n[waveform] raw lag mean {wl.mean()*1000:+.3f} ms sd {wl.std(ddof=1)*1000:.3f} ms;"
              f" xcorr r mean {wr.mean():.3f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": {"sr": SR, "hop": HOP, "tol": TOL, "n_tracks": len(rows),
                          "global_median_lag_s": gmed},
               "summary": S, "per_track": rows}, open(out, "w"), indent=1)
    print(f"\n[out] {out}")


def _global_corr(job):
    wav, ann, rtp, seed, shift = job
    import librosa
    z, _ = librosa.load(str(rtp), sr=SR, mono=True)
    y, _ = librosa.load(str(wav), sr=SR, mono=True)
    beats = np.loadtxt(str(ann)).astype(float).ravel()
    dur = len(y) / SR
    beats = beats[(beats >= 0) & (beats < dur)]
    return beat_adherence(z, beats + shift, seed=seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["rt", "score"], required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rtdir", default=str(RTDIR))
    ap.add_argument("--out", default=str(ROOT / "cache" / "sa3_beat_position_roundtrip.json"))
    a = ap.parse_args()
    (stage_rt if a.stage == "rt" else stage_score)(a)
