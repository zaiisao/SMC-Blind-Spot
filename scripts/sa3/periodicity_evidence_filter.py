"""Periodicity family of acoustic-evidence measures for generated slow-tempo clips.

Every measure is computed from the log-mel spectrogram plus the inherited beat
times only.  No beat model, no tracker output.

Family idea: instead of asking "is there onset energy at the labelled frames",
ask "does the signal contain PERIODIC structure at the labelled period AND at
the labelled phase, specifically at that metrical level and not at half of it".

Evaluation follows the pre-registered criterion:
    rho_partial = partial Spearman(e, damage | difficulty)   want strongly negative
    rho_diff    = Spearman(e, difficulty)                    want |.| < 0.15
SELECT/REPORT split by md5 of the stem.
"""

import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata

BT = "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"
PAIRED = "/home/sogang/jaehoon/Analyze-SMC/cache/sdedit_vs_source.csv"
FPS = 50


# ---------------------------------------------------------------- features --
def flux(spect):
    d = np.diff(spect, axis=0)
    f = np.maximum(d, 0).sum(axis=1)
    f = np.concatenate([[0.0], f])
    f = f - np.median(f)
    return f / (f.std() + 1e-9)


def comb_response(f, period, phase, tol=2.0):
    """Mean flux under a Gaussian-ish comb of the given period and phase.

    period, phase in frames (float).  Uses a triangular kernel of half-width
    `tol` frames around every tooth so that sub-frame phases are handled.
    """
    T = len(f)
    n0 = int(np.ceil((0.0 - phase) / period))
    n1 = int(np.floor((T - 1 - phase) / period))
    if n1 - n0 < 2:
        return np.nan
    centres = phase + np.arange(n0, n1 + 1) * period
    w = np.arange(-int(np.ceil(tol)), int(np.ceil(tol)) + 1)
    pos = np.round(centres[:, None]).astype(int) + w[None, :]
    off = (centres[:, None] + 0.0) - np.round(centres[:, None])
    wgt = np.clip(1.0 - np.abs(w[None, :] - off) / (tol + 1e-9), 0, None)
    ok = (pos >= 0) & (pos < T)
    pos = pos.clip(0, T - 1)
    vals = f[pos] * wgt * ok
    norm = (wgt * ok).sum(axis=1) + 1e-9
    return float((vals.sum(axis=1) / norm).mean())


def phase_profile(f, period, n_phase=24, tol=2.0):
    ph = np.linspace(0.0, period, n_phase, endpoint=False)
    return np.array([comb_response(f, period, p, tol) for p in ph]), ph


def autocorr(f):
    n = len(f)
    x = f - f.mean()
    F = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(F * np.conj(F), 2 * n)[:n]
    denom = np.arange(n, 0, -1)
    ac = ac / (denom * (x.var() + 1e-9))
    return ac


def lag_interp(ac, lag):
    i = int(np.floor(lag))
    if i < 1 or i + 1 >= len(ac):
        return np.nan
    a = lag - i
    return float((1 - a) * ac[i] + a * ac[i + 1])


def measures(spect, beats):
    """Return dict of periodicity-family scalars for one clip."""
    f = flux(spect)
    T = len(f)
    out = {}
    if len(beats) < 4 or T < 100:
        return None
    ibis = np.diff(beats)
    period = float(np.median(ibis)) * FPS          # frames per beat
    if not np.isfinite(period) or period < 8 or period > 0.4 * T:
        return None
    phase0 = float(beats[0] * FPS) % period

    # --- autocorrelation-based -------------------------------------------
    ac = autocorr(f)
    maxlag = min(len(ac) - 2, int(4 * period))
    lo, hi = max(2, int(0.35 * period)), maxlag
    band = ac[lo:hi]
    a_beat = lag_interp(ac, period)
    a_half = lag_interp(ac, period / 2.0)
    a_dbl = lag_interp(ac, 2 * period) if 2 * period + 1 < len(ac) else np.nan

    # A1: raw autocorrelation at the labelled period (no phase info)
    out["ac_at_period"] = a_beat
    # A2: labelled period vs the best competing lag in the band -> is the
    #     labelled level the dominant one at all?
    out["ac_period_vs_best"] = a_beat - float(np.nanmax(band)) if band.size else np.nan
    # A3: labelled level vs HALF that period (doubling check)
    out["ac_beat_vs_half"] = a_beat - (a_half if np.isfinite(a_half) else 0.0)

    # --- comb filter, phase aware ----------------------------------------
    prof, _ = phase_profile(f, period, n_phase=24)
    r_lab = comb_response(f, period, phase0)
    best = float(np.nanmax(prof))
    other = prof.copy()
    # competing phases at least a quarter period away from the labelled one
    ph = np.linspace(0.0, period, 24, endpoint=False)
    dist = np.minimum(np.abs(ph - phase0), period - np.abs(ph - phase0))
    other[dist < 0.25 * period] = np.nan
    med_other = float(np.nanmedian(other))
    max_other = float(np.nanmax(other))

    # C1: labelled-phase comb response minus the best competing phase
    out["comb_phase_margin"] = r_lab - max_other
    # C2: labelled-phase response vs the median over competing phases
    #     (how much the comb sticks out of the phase background)
    sd = float(np.nanstd(prof)) + 1e-9
    out["comb_phase_z"] = (r_lab - med_other) / sd

    # --- half-period comb: the doubling test with phase ------------------
    half = period / 2.0
    r_half_lab = comb_response(f, half, phase0)
    prof_h, _ = phase_profile(f, half, n_phase=16)
    r_half_best = float(np.nanmax(prof_h))
    # C3: does the labelled LEVEL beat the half level?  If the clip is really
    #     at double tempo, the half-period comb fires just as hard.
    out["comb_level_margin"] = r_lab - r_half_best
    out["comb_level_ratio"] = r_lab / (r_half_lab + 1e-9)

    # --- Fourier / tempogram concentration --------------------------------
    # DFT of flux at the beat frequency and its phase alignment.
    n = T
    k = n / period                                  # cycles over the clip
    t = np.arange(n)
    z = (f * np.exp(-2j * np.pi * k * t / n)).sum() / n
    mag = np.abs(z)
    ang = np.angle(z)
    exp_ang = -2 * np.pi * k * phase0 / n
    align = np.cos(ang - exp_ang)
    # background magnitude over neighbouring frequencies
    ks = k + np.concatenate([np.arange(-6, -1), np.arange(2, 7)]) * 0.5
    bg = np.array([np.abs((f * np.exp(-2j * np.pi * kk * t / n)).sum() / n)
                   for kk in ks if kk > 0.5])
    # F1: spectral prominence at the beat frequency
    out["dft_prominence"] = float(mag / (np.median(bg) + 1e-9))
    # F2: same, multiplied by phase agreement with the labels
    out["dft_phase_locked"] = float(mag / (np.median(bg) + 1e-9) * align)
    # F3: pure phase agreement
    out["dft_phase_align"] = float(align)

    # --- combined: phase margin gated by level margin ---------------------
    out["comb_phase_x_level"] = out["comb_phase_z"] * np.tanh(
        np.clip(out["comb_level_margin"], -5, 5))
    out["ac_x_comb"] = (a_beat if np.isfinite(a_beat) else 0.0) * out["comb_phase_z"]
    if np.isfinite(a_dbl):
        out["ac_beat_vs_double"] = a_beat - a_dbl
    else:
        out["ac_beat_vs_double"] = np.nan

    # --- second round: variants around "labelled level vs best rival" -----
    # A4: same as ac_period_vs_best but excluding lags near integer ratios of
    #     the labelled period, so metrical relatives are not counted as rivals.
    lags = np.arange(lo, hi)
    rel = np.array([2.0, 1.5, 4.0 / 3, 1.0, 0.75, 0.5])
    keep = np.ones(len(lags), bool)
    for rr in rel:
        keep &= np.abs(lags - rr * period) > 0.12 * period
    out["ac_vs_best_nonmetrical"] = (
        a_beat - float(np.nanmax(ac[lo:hi][keep])) if keep.any() else np.nan)
    # A5: rank of the labelled lag among all lags in the band (scale free)
    out["ac_period_rank"] = float((band < a_beat).mean()) if band.size else np.nan
    # A6: z-score of the labelled lag against the whole band
    out["ac_period_z"] = float((a_beat - np.nanmean(band)) /
                               (np.nanstd(band) + 1e-9)) if band.size else np.nan

    # C4: full 2-D margin -- labelled (period, phase) comb against the best
    #     comb over a grid of rival periods and their own best phases.
    rival = []
    for pf in (0.5, 0.6, 0.75, 1.25, 1.5, 2.0, 3.0):
        pp = period * pf
        if pp < 8 or pp > 0.4 * T:
            continue
        pr, _ = phase_profile(f, pp, n_phase=12)
        rival.append(np.nanmax(pr))
    out["comb_2d_margin"] = r_lab - float(np.nanmax(rival)) if rival else np.nan
    # C5: labelled comb minus best rival, divided by the phase spread, i.e.
    #     the 2-D margin made scale free.
    out["comb_2d_margin_z"] = out["comb_2d_margin"] / sd
    return out


# ------------------------------------------------------------- evaluation --
def partial_spearman(x, y, z):
    """Spearman of x,y controlling for z, via ranks + linear residualisation."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    rx, ry, rz = rankdata(x[m]), rankdata(y[m]), rankdata(z[m])

    def resid(a):
        A = np.vstack([rz, np.ones_like(rz)]).T
        beta, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ beta
    r, p = spearmanr(resid(rx), resid(ry))
    return float(r), float(p), int(m.sum())


def main():
    os.chdir(BT)
    sys.path.insert(0, BT)
    from beat_this.dataset.mmnpz import MemmappedNpzFile

    paired = {r["stem"]: r for r in csv.DictReader(open(PAIRED))}
    npz = MemmappedNpzFile(Path("data/audio/spectrograms/sdedit.npz"))
    stems = sorted({k.split("/")[0] for k in npz.files})
    annd = Path("data/annotations/sdedit/annotations/beats")

    rows = []
    for i, s in enumerate(stems):
        if s not in paired:
            continue
        ref = np.loadtxt(annd / f"{s}.beats", ndmin=2)[:, 0]
        sp = np.asarray(npz[f"{s}/track"], dtype=np.float32)
        m = measures(sp, ref)
        if m is None:
            continue
        p = paired[s]
        m["stem"] = s
        m["damage"] = float(p["F_src"]) - float(p["F_gen"])
        m["difficulty"] = 1.0 - float(p["F_src"])
        h = int(hashlib.md5(s.encode()).hexdigest(), 16)
        m["split"] = "SELECT" if h % 2 == 0 else "REPORT"
        rows.append(m)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(stems)}", flush=True)

    names = [k for k in rows[0] if k not in ("stem", "damage", "difficulty", "split")]
    arr = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in names}
    dam = np.array([r["damage"] for r in rows])
    dif = np.array([r["difficulty"] for r in rows])
    spl = np.array([r["split"] for r in rows])

    out = "/home/sogang/.tmp/claude-1003/-home-sogang-jaehoon-Analyze-SMC/" \
          "e9b382f3-cceb-471e-91d6-43bf1f39be4a/scratchpad/wf/periodicity_scores2.csv"
    with open(out, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=["stem", "split", "damage", "difficulty"] + names)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nn = {len(rows)}  SELECT={int((spl=='SELECT').sum())} "
          f"REPORT={int((spl=='REPORT').sum())}\n")
    hdr = f"{'measure':26s} {'rp_SEL':>8s} {'rd_SEL':>8s} {'rp_REP':>8s} {'rd_REP':>8s} {'pass':>5s}"
    print(hdr)
    print("-" * len(hdr))
    res = {}
    for k in names:
        row = []
        for sp_ in ("SELECT", "REPORT"):
            m = spl == sp_
            rp, _, _ = partial_spearman(arr[k][m], dam[m], dif[m])
            good = np.isfinite(arr[k][m]) & np.isfinite(dif[m])
            rd = float(spearmanr(arr[k][m][good], dif[m][good])[0])
            row += [rp, rd]
        ok = abs(row[2]) > 0.25 and abs(row[3]) < 0.15
        res[k] = row + [ok]
        print(f"{k:26s} {row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f} {row[3]:8.3f} {str(ok):>5s}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
