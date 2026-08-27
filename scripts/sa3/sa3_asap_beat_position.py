"""Does the SA3 VAE MOVE beat positions on piano, or only soften attacks?

Corpus: ASAP performances that have audio.  Annotations come from
asap_annotations.json (performance_beats_type keys = beats+downbeats, in
performance-audio time).  Join: audio symlink target -> repo-relative .mid key.

Pipeline per track: load a fixed segment, score it against the true beat grid
and the three control grids with the CALIBRATED verifier in
sa3_beat_adherence.py; round-trip the same segment through the VAE
(encode+decode, no diffusion); re-score against the SAME grid, raw and after
correcting a per-track constant offset estimated by onset-envelope
cross-correlation.  Finally decompose per-beat into POSITION (did the envelope
peak move?) and SHAPE (did it flatten?).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
HOP_FINE = 32
from sa3_beat_adherence import beat_adherence, combined_score, control_grids, onset_envelope, SR, HOP  # noqa
from sa3_onset_reference import load_ae  # noqa

ASAP_AUDIO = Path("/disk1/jaehoon/dataset_store/audio_by_stem/asap")
ASAP_REPO = Path("/disk3/jaehoon/beat-tracking-dataset/_repos/asap-dataset")


def pairs():
    ann = json.load(open(ASAP_REPO / "asap_annotations.json"))
    out, unmatched = [], []
    for w in sorted(ASAP_AUDIO.glob("*.wav")):
        tgt = Path(w).resolve()
        try:
            rel = tgt.relative_to(ASAP_REPO)
        except ValueError:
            unmatched.append((w.name, "not-under-repo")); continue
        key = str(rel.with_suffix(".mid"))
        if key not in ann:
            unmatched.append((w.name, key)); continue
        b = sorted(float(t) for t in ann[key]["performance_beats_type"])
        db = sorted(float(t) for t in ann[key]["performance_downbeats"])
        out.append((w, key, np.array(b), np.array(db)))
    return out, unmatched


def xcorr_lag(a, b, sr, max_lag_s):
    """lag (s) that best aligns b to a: b(t) ~ a(t - lag). positive = b delayed."""
    n = min(len(a), len(b)); a = a[:n] - a[:n].mean(); b = b[:n] - b[:n].mean()
    m = int(round(max_lag_s * sr))
    f = np.fft.rfft(a, 2 * n); g = np.fft.rfft(b, 2 * n)
    cc = np.fft.irfft(g * np.conj(f), 2 * n)
    cc = np.concatenate([cc[-m:], cc[:m + 1]])
    lags = np.arange(-m, m + 1) / sr
    i = int(np.argmax(cc))
    # parabolic refinement
    if 0 < i < len(cc) - 1:
        y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    else:
        d = 0.0
    return float(lags[i] + d / sr), float(cc[i] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def peak_stats(env, times, beats, tol=0.060):
    """Per-beat envelope peak LOCATION (s, relative to beat) and SHARPNESS.

    sharpness = (peak - mean of the window) / (peak + eps): 1 = impulse,
    0 = flat.  Location moving = POSITION damage; sharpness falling with
    location fixed = SHAPE damage."""
    dt = times[1] - times[0]
    k = int(round(tol / dt))
    loc, shp, amp = [], [], []
    for bt in beats:
        c = int(round(bt / dt))
        lo, hi = max(0, c - k), min(len(env), c + k + 1)
        w = env[lo:hi]
        if len(w) < 3 or w.max() <= 0:
            continue
        j = int(np.argmax(w))
        loc.append((lo + j) * dt - bt)
        shp.append(float((w.max() - w.mean()) / (w.max() + 1e-12)))
        amp.append(float(w.max()))
    return np.array(loc), np.array(shp), np.array(amp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--shuffle", type=int, default=0, help="seed; spread the sample over the whole corpus instead of taking the alphabetically-first N (which are all Bach)")
    ap.add_argument("--offset", type=float, default=10.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default=str(ROOT / "cache" / "sa3_asap_beat_position.json"))
    args = ap.parse_args()

    import librosa
    P, unmatched = pairs()
    print(f"[join] {len(P)} audio<->annotation matches, {len(unmatched)} unmatched", flush=True)
    for u in unmatched[:5]:
        print("   unmatched:", u)
    if args.shuffle:
        np.random.default_rng(args.shuffle).shuffle(P)
    P = P[: args.n]

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    ae, MSR = load_ae(device)
    print(f"[setup] device={device} model_sr={MSR} seg=[{args.offset},{args.offset+args.seconds}]s "
          f"verifier_sr={SR} hop={HOP}", flush=True)

    rows = []
    for n, (w, key, beats, dbs) in enumerate(P):
        y, _ = librosa.load(str(w), sr=MSR, mono=True, offset=args.offset, duration=args.seconds)
        if len(y) < MSR * args.seconds * 0.9:
            continue
        y = y / (np.abs(y).max() + 1e-9)
        N = len(y)
        x = torch.from_numpy(np.stack([y, y])).unsqueeze(0).to(device)
        with torch.no_grad():
            lat = ae.encode(x)
            dec = ae.decode(lat)
        len_change = int(dec.shape[-1]) - N
        rec = dec[0, :, :N].float().cpu().numpy().mean(0)
        rec = rec / (np.abs(rec).max() + 1e-9)

        # verifier-rate versions
        yv = librosa.resample(y, orig_sr=MSR, target_sr=SR)
        rv = librosa.resample(rec, orig_sr=MSR, target_sr=SR)
        b = beats[(beats >= args.offset) & (beats < args.offset + args.seconds)] - args.offset
        if len(b) < 8:
            continue
        dur = len(yv) / SR

        # --- constant-offset estimates -------------------------------------
        ev_r = onset_envelope(yv); ev_t = onset_envelope(rv)
        # fine-hop envelopes (1.45 ms) purely for timing: the 4.99 ms scoring hop
        # cannot resolve a shift smaller than itself.
        fr = onset_envelope(yv, hop=HOP_FINE); ft = onset_envelope(rv, hop=HOP_FINE)
        lag_env, cc_env = xcorr_lag(fr, ft, SR / HOP_FINE, 0.100)
        lag_wav, cc_wav = xcorr_lag(y, rec, MSR, 0.050)

        rec_by = {}
        for tag, sig in (("real", yv), ("rt", rv)):
            r = {"true": beat_adherence(sig, b, seed=n)}
            for cn, g in control_grids(b, dur, seed=n).items():
                r[cn] = beat_adherence(sig, g, seed=n)
            rec_by[tag] = r
        # offset-corrected: shift the grid by the estimated lag (rt is delayed
        # by lag, so the beats in rt sit at b + lag)
        rec_by["rt_corr"] = {"true": beat_adherence(rv, b + lag_wav, seed=n)}
        rec_by["rt_corr_env"] = {"true": beat_adherence(rv, b + lag_env, seed=n)}
        if rec_by["real"]["true"] is None or rec_by["rt"]["true"] is None:
            continue

        # --- position vs shape ---------------------------------------------
        tf = np.arange(len(fr)) * HOP_FINE / SR
        lr, sr_, ar = peak_stats(fr, tf, b)
        lt, st, at = peak_stats(ft, np.arange(len(ft)) * HOP_FINE / SR, b)
        m = min(len(lr), len(lt))
        dshift = lt[:m] - lr[:m]
        jit = dshift - np.median(dshift)
        rows.append({
            "track": w.stem, "key": key, "n_beats": int(len(b)),
            "len_change_samples": len_change,
            "lag_env_s": lag_env, "cc_env": cc_env,
            "lag_wav_s": lag_wav, "cc_wav": cc_wav,
            "scores": {k: {c: v for c, v in d.items()} for k, d in rec_by.items()},
            "peak_loc_real_mean": float(lr.mean()), "peak_loc_rt_mean": float(lt.mean()),
            "peak_loc_shift_mean": float(np.mean(lt[:m] - lr[:m])),
            "peak_loc_shift_absmean": float(np.mean(np.abs(lt[:m] - lr[:m]))),
            "peak_loc_shift_median_abs": float(np.median(np.abs(lt[:m] - lr[:m]))),
            "peak_shift_median_ms": float(np.median(dshift) * 1000),
            "peak_jitter_sd_ms": float(jit.std() * 1000),
            "peak_jitter_iqr_ms": float((np.percentile(jit, 75) - np.percentile(jit, 25)) * 1000),
            "frac_beats_shift_gt_10ms": float(np.mean(np.abs(jit) > 0.010)),
            "frac_beats_shift_gt_25ms": float(np.mean(np.abs(jit) > 0.025)),
            "peak_loc_sd_real": float(lr.std()), "peak_loc_sd_rt": float(lt.std()),
            "sharp_real": float(sr_.mean()), "sharp_rt": float(st.mean()),
            "crest_real": rec_by["real"]["true"]["onset_strength"],
            "crest_rt": rec_by["rt"]["true"]["onset_strength"],
        })
        if (n + 1) % 10 == 0:
            print(f"  [{n+1}/{len(P)}] {w.stem}  pa real {rec_by['real']['true']['phase_align']:+.4f} "
                  f"rt {rec_by['rt']['true']['phase_align']:+.4f} lag_wav {lag_wav*1000:+.1f}ms", flush=True)

    print(f"\n[n] {len(rows)} tracks scored", flush=True)
    A = lambda p, c, k: np.array([r["scores"][p][c][k] for r in rows if r["scores"][p].get(c)])
    from scipy import stats

    out = {"config": vars(args), "n_tracks": len(rows), "per_track": rows, "summary": {}}
    S = out["summary"]

    print(f"\n{'cond':<24}{'phase_align':>22}{'beat_energy_z':>20}{'combined':>12}")
    for p in ("real", "rt", "rt_corr", "rt_corr_env"):
        for c in (("true", "half_beat", "scrambled", "wrong_tempo") if p in ("real", "rt") else ("true",)):
            pa, bz = A(p, c, "phase_align"), A(p, c, "beat_energy_z")
            cs = np.minimum(10 * pa, bz)
            S[f"{p}/{c}"] = {"n": len(pa), "phase_align_mean": float(pa.mean()),
                             "phase_align_sd": float(pa.std()),
                             "beat_energy_z_mean": float(bz.mean()),
                             "beat_energy_z_sd": float(bz.std()),
                             "combined_mean": float(cs.mean())}
            print(f"{p+'/'+c:<24}{pa.mean():+.4f} sd {pa.std():.4f}{bz.mean():>14.2f} sd {bz.std():5.2f}{cs.mean():>10.2f}")

    print("\nPAIRED real -> roundtrip (same tracks, same grid):")
    for k in ("phase_align", "beat_energy_z"):
        for p in ("rt", "rt_corr", "rt_corr_env"):
            a, bb = A("real", "true", k), A(p, "true", k)
            d = bb - a
            t, pt = stats.ttest_rel(bb, a)
            try:
                _, pw = stats.wilcoxon(bb, a)
            except Exception:
                pw = float("nan")
            dz = d.mean() / (d.std(ddof=1) + 1e-12)
            S[f"paired_{k}_{p}"] = {"delta_mean": float(d.mean()), "delta_sd": float(d.std(ddof=1)),
                                    "t": float(t), "p_ttest": float(pt), "p_wilcoxon": float(pw),
                                    "cohen_dz": float(dz), "n": len(d),
                                    "ci95": [float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                                             float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d)))]}
            print(f"  {k:<14} real->{p:<12} d={d.mean():+.4f} sd {d.std(ddof=1):.4f} "
                  f"dz={dz:+.2f} t={t:+.2f} p_t={pt:.3g} p_w={pw:.3g}")

    lw = np.array([r["lag_wav_s"] for r in rows]) * 1000
    le = np.array([r["lag_env_s"] for r in rows]) * 1000
    lc = np.array([r["len_change_samples"] for r in rows])
    S["lag_ms"] = {"wav_mean": float(lw.mean()), "wav_sd": float(lw.std(ddof=1)),
                   "wav_median": float(np.median(lw)), "wav_iqr": [float(np.percentile(lw, 25)), float(np.percentile(lw, 75))],
                   "env_mean": float(le.mean()), "env_sd": float(le.std(ddof=1)),
                   "env_median": float(np.median(le)),
                   "len_change_samples_unique": sorted(set(int(v) for v in lc))}
    print(f"\nconstant-offset: waveform lag mean {lw.mean():+.2f} ms sd {lw.std(ddof=1):.2f} "
          f"median {np.median(lw):+.2f} IQR [{np.percentile(lw,25):+.2f},{np.percentile(lw,75):+.2f}]")
    print(f"                 envelope lag mean {le.mean():+.2f} ms sd {le.std(ddof=1):.2f} median {np.median(le):+.2f}")
    print(f"                 decoded length change (samples): {sorted(set(int(v) for v in lc))}")

    ps = np.array([r["peak_loc_shift_mean"] for r in rows]) * 1000
    pa_ = np.array([r["peak_loc_shift_absmean"] for r in rows]) * 1000
    pm = np.array([r["peak_loc_shift_median_abs"] for r in rows]) * 1000
    shr = np.array([r["sharp_real"] for r in rows]); sht = np.array([r["sharp_rt"] for r in rows])
    ts, pss = stats.ttest_rel(sht, shr)
    S["position_vs_shape"] = {
        "peak_loc_shift_mean_ms": float(ps.mean()), "peak_loc_shift_sd_ms": float(ps.std(ddof=1)),
        "peak_loc_absshift_mean_ms": float(pa_.mean()), "peak_loc_absshift_median_ms": float(np.median(pm)),
        "sharpness_real": float(shr.mean()), "sharpness_rt": float(sht.mean()),
        "sharpness_delta": float((sht - shr).mean()),
        "sharpness_dz": float((sht - shr).mean() / ((sht - shr).std(ddof=1) + 1e-12)),
        "sharpness_p": float(pss),
        "crest_real": float(np.mean([r["crest_real"] for r in rows])),
        "crest_rt": float(np.mean([r["crest_rt"] for r in rows])),
    }
    pjm = np.array([r["peak_shift_median_ms"] for r in rows])
    pjs = np.array([r["peak_jitter_sd_ms"] for r in rows])
    pji = np.array([r["peak_jitter_iqr_ms"] for r in rows])
    f10 = np.array([r["frac_beats_shift_gt_10ms"] for r in rows])
    f25 = np.array([r["frac_beats_shift_gt_25ms"] for r in rows])
    S["position_fine"] = {"hop_fine_ms": HOP_FINE / SR * 1000,
        "per_track_median_shift_ms_mean": float(pjm.mean()),
        "per_track_median_shift_ms_sd": float(pjm.std(ddof=1)),
        "per_track_median_shift_ms_iqr": [float(np.percentile(pjm, 25)), float(np.percentile(pjm, 75))],
        "within_track_jitter_sd_ms_mean": float(pjs.mean()),
        "within_track_jitter_iqr_ms_mean": float(pji.mean()),
        "frac_beats_jitter_gt_10ms": float(f10.mean()),
        "frac_beats_jitter_gt_25ms": float(f25.mean()),
        "annot_to_peak_sd_ms_real": float(np.mean([r["peak_loc_sd_real"] for r in rows]) * 1000),
        "annot_to_peak_sd_ms_rt": float(np.mean([r["peak_loc_sd_rt"] for r in rows]) * 1000)}
    print(f"\nPOSITION (fine hop {HOP_FINE/SR*1000:.2f} ms):")
    print(f"  per-track MEDIAN peak shift: mean {pjm.mean():+.2f} ms sd {pjm.std(ddof=1):.2f} "
          f"IQR [{np.percentile(pjm,25):+.2f},{np.percentile(pjm,75):+.2f}]  (constant component)")
    print(f"  within-track JITTER (shift - track median): sd {pjs.mean():.2f} ms, IQR {pji.mean():.2f} ms")
    print(f"  reference scale: sd of annotation->peak offset within real audio "
          f"{np.mean([r['peak_loc_sd_real'] for r in rows])*1000:.2f} ms "
          f"(roundtrip {np.mean([r['peak_loc_sd_rt'] for r in rows])*1000:.2f} ms)")
    print(f"  beats with |jitter| > 10 ms: {f10.mean()*100:.1f}%   > 25 ms: {f25.mean()*100:.1f}%")
    print(f"\nPOSITION: per-beat envelope-peak shift rt-real: mean {ps.mean():+.2f} ms (sd {ps.std(ddof=1):.2f}), "
          f"|shift| mean {pa_.mean():.2f} ms, median |shift| {np.median(pm):.2f} ms")
    print(f"SHAPE   : peak sharpness real {shr.mean():.4f} -> rt {sht.mean():.4f} "
          f"(d {(sht-shr).mean():+.4f}, dz {(sht-shr).mean()/(sht-shr).std(ddof=1):+.2f}, p={pss:.3g})")
    print(f"          envelope crest real {np.mean([r['crest_real'] for r in rows]):.2f} -> "
          f"{np.mean([r['crest_rt'] for r in rows]):.2f}")

    # stratify by real crest factor
    cr = np.array([r["crest_real"] for r in rows])
    q = np.quantile(cr, [1/3, 2/3])
    dpa = A("rt", "true", "phase_align") - A("real", "true", "phase_align")
    print(f"\nstratified by real onset crest (tertiles {q.round(2)}):")
    print(f"{'stratum':<12}{'n':>4}{'real pa':>10}{'rt pa':>10}{'d pa':>10}{'|peakshift| ms':>16}")
    for nm, m_ in (("soft", cr < q[0]), ("mid", (cr >= q[0]) & (cr < q[1])), ("impulsive", cr >= q[1])):
        if m_.sum() == 0: continue
        S.setdefault("strata", {})[nm] = {
            "n": int(m_.sum()), "real_pa": float(A("real", "true", "phase_align")[m_].mean()),
            "rt_pa": float(A("rt", "true", "phase_align")[m_].mean()),
            "d_pa": float(dpa[m_].mean()), "abs_peak_shift_ms": float(pa_[m_].mean())}
        print(f"{nm:<12}{m_.sum():>4}{A('real','true','phase_align')[m_].mean():>+10.4f}"
              f"{A('rt','true','phase_align')[m_].mean():>+10.4f}{dpa[m_].mean():>+10.4f}{pa_[m_].mean():>16.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
