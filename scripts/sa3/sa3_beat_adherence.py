"""HANDOVER §5.3: non-circular beat-adherence verification.

The plan in §4/§6 is to use a conditioning beat grid as the *label* for audio the
model generates.  That is only sound if the model actually followed the grid, and
the obvious way to check -- run a beat tracker on the output and compare -- is
circular: the tracker we would use is the same family as the tracker we are
trying to train, so a high score means "our tracker agrees with our tracker", not
"the audio has beats there".  §3.3 rejected classifier guidance for the same
reason.

This module measures adherence from *signal-level evidence only*.  No learned
model, no beat tracker, no peak-picking heuristic with a tuned threshold.  Given
audio and a candidate beat grid it asks one question:

    is the audio's onset energy concentrated at phase 0 of this grid?

Two statistics, both computed from a single onset-strength envelope:

  phase_align  = Re( sum_t w(t) exp(i*phi(t)) / sum_t w(t) )      in [-1, 1]

      phi(t) is continuous beat phase, obtained by piecewise-linear interpolation
      between consecutive candidate beats (phase 0 at each beat, 2*pi just before
      the next).  w(t) is the SQUARE of the positive part of the onset envelope
      after median removal (squaring concentrates the weight on actual onsets;
      measured on 40 SMC tracks it lifts true-vs-half-beat d' from 2.17 to 2.24
      and roughly doubles the dynamic range, so it costs nothing to prefer it).  This is the real part of the first circular moment, so it is
      *signed*: energy at the beats gives +, energy exactly between the beats
      (a half-beat error) gives -, and no relationship gives ~0.  The signedness
      is the reason we take Re(C) and not |C|: |C| cannot tell a correct grid
      from a half-beat-shifted one, which is the single most important control.

  beat_energy_z = z-score of the mean envelope value in a +/-40 ms window around
      each beat time (max within the window: performers and annotators are not
      accurate to 5 ms, and sampling the envelope exactly at the annotated
      instant throws most of the evidence away -- measured, the windowed version
      raises the true-grid z from 0.6 to 4.9), against a null built by rigidly rotating the whole grid by random offsets
      (wrapped modulo the track duration).  The rotation null preserves the
      grid's IOI structure and its tempo -- it destroys only the *phase*.  So
      this reads as "above chance" by construction, and chance is 0 by
      definition of a z-score.

`combined_score()` folds the two into one calibrated number.

Deliberately NOT z-scoring phase_align against the rotation null: rotating a grid
turns Re(C) into R*cos(random angle), whose null sd is R/sqrt(2), so the z-score
saturates at sqrt(2) no matter how strong the periodicity.  phase_align is
already bounded and interpretable on its own scale, so it is reported raw.

CLI: calibrate on real SMC audio, where ground truth exists, against three
deliberately-wrong controls.

    /disk1/jaehoon/sa3env/bin/python scripts/sa3/sa3_beat_adherence.py --gpu 1

NOTE ON SCOPE: the beat adapter in sa3_beat/ is zero-init and untrained, i.e. a
mathematical no-op, so running this on generated audio right now is guaranteed to
return chance.  This is the instrument, calibrated; it is not a result about the
model.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SMC_AUDIO = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
SMC_ANN = ROOT / "SMC_MIREX" / "SMC_MIREX_Annotations"

SR = 22050
HOP = 110          # 4.99 ms -- well under any beat tolerance we care about
N_NULL = 200       # rotations per track for the chance distribution
TOL = 0.040        # +/-40 ms window around each beat; beat-tracking convention
                   # is 70 ms, and 40 ms measured slightly better here


# --------------------------------------------------------------------------- #
# the measure
# --------------------------------------------------------------------------- #

def onset_envelope(audio, sr=SR, hop=HOP):
    """Positive spectral flux envelope, median-removed and half-wave rectified.

    Median removal (not mean) so that sustained material does not contribute a
    constant floor that would dilute the phase weighting; rectification so that
    w(t) >= 0 and the circular moment stays a proper weighted average.
    """
    import librosa
    env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
    env = env - np.median(env)
    return np.maximum(env, 0.0)


def beat_phase(times, beats):
    """Continuous phase in [0, 2pi) of each frame time w.r.t. a beat grid.

    Piecewise linear between consecutive beats.  Frames outside the grid's span
    are marked invalid rather than extrapolated -- extrapolating would invent
    agreement at the edges.
    """
    beats = np.asarray(beats, dtype=float)
    idx = np.searchsorted(beats, times) - 1
    valid = (idx >= 0) & (idx < len(beats) - 1)
    phase = np.zeros_like(times)
    i = idx[valid]
    span = beats[i + 1] - beats[i]
    phase[valid] = 2 * np.pi * (times[valid] - beats[i]) / np.maximum(span, 1e-9)
    return phase, valid


def _phase_align(env, times, beats):
    phase, valid = beat_phase(times, beats)
    w = env[valid] ** 2
    if w.sum() <= 0:
        return 0.0, 0.0
    c = np.sum(w * np.exp(1j * phase[valid])) / w.sum()
    return float(c.real), float(np.abs(c))


def _beat_energy(env, times, beats, hop=HOP, sr=SR, tol=TOL):
    """Mean of the envelope max inside +/-tol of each beat."""
    k = max(1, int(round(tol * sr / hop)))
    idx = np.clip(np.round(beats * sr / hop).astype(int), 0, len(env) - 1)
    return float(np.mean([env[max(0, i - k): i + k + 1].max() for i in idx]))


def _rotate(beats, delta, duration):
    """Rigid rotation of the grid, wrapped modulo duration.

    Wrapping keeps the same number of beats and the same IOI histogram (bar one
    seam interval), so the null differs from the true grid in phase alone.
    """
    r = np.sort((np.asarray(beats, float) + delta) % duration)
    return r


def beat_adherence(audio, beats, sr=SR, hop=HOP, n_null=N_NULL, seed=0):
    """Non-circular adherence of `audio` to candidate beat grid `beats`.

    Returns dict with:
      phase_align    Re(first circular moment) of the squared envelope, [-1, 1].  + = onset energy on
                     the beat, - = on the off-beat, ~0 = unrelated.
      phase_r        |C|, the unsigned concentration (periodicity at this period,
                     phase-blind).  Reported for diagnosis only.
      beat_energy_z  z of mean envelope at beat times vs rotation null.
      onset_strength crest factor of the envelope: how impulsive the material is.
                     Used to stratify -- the measure needs onsets to see.
    """
    audio = np.asarray(audio, dtype=np.float64)
    beats = np.asarray(sorted(float(b) for b in beats), dtype=float)
    duration = len(audio) / sr
    beats = beats[(beats >= 0) & (beats < duration)]
    if len(beats) < 4:
        return None

    env = onset_envelope(audio, sr, hop)
    times = np.arange(len(env)) * hop / sr

    align, r = _phase_align(env, times, beats)
    obs = _beat_energy(env, times, beats)

    rng = np.random.default_rng(seed)
    null = np.array([
        _beat_energy(env, times, _rotate(beats, d, duration))
        for d in rng.uniform(0, duration, n_null)
    ])
    z = (obs - null.mean()) / (null.std() + 1e-12)

    pos = env[env > 0]
    crest = float(env.max() / (pos.mean() + 1e-12)) if len(pos) else 0.0

    return {
        "phase_align": align,
        "phase_r": r,
        "beat_energy_z": float(z),
        "beat_energy": obs,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "onset_strength": crest,
        "n_beats": int(len(beats)),
        "duration": float(duration),
    }


def combined_score(res):
    """One number per track, calibrated on SMC (see cache/sa3_beat_adherence).

    min() of the two statistics, because they fail on different controls and a
    conjunction is what we want: phase_align is the only one that rejects a
    half-beat shift (beat_energy_z barely moves, +3.60 -> +0.18 vs
    +0.23 -> -0.20), while beat_energy_z is much the stronger rejector of
    scrambled and wrong-tempo grids (AUC 0.95/0.94 vs 0.83/0.82).  The 10x puts
    phase_align on a comparable scale to the z.

    Calibrated operating point on 217 SMC tracks:
        threshold 1.5 -> TPR 0.562, FPR 0.005 (half-beat) / 0.009 (scrambled)
                         / 0.028 (wrong tempo)
    i.e. per-track it is a *specific* but not *sensitive* test: a track scoring
    above 1.5 is almost certainly really on the grid, but a track below 1.5 is
    not evidence against.  Use it per track only to accept; use the corpus mean
    to reject.
    """
    if res is None:
        return None
    return min(10.0 * res["phase_align"], res["beat_energy_z"])


# --------------------------------------------------------------------------- #
# controls -- the wrong grids the verifier must reject
# --------------------------------------------------------------------------- #

def control_grids(beats, duration, seed=0):
    rng = np.random.default_rng(seed + 7717)
    b = np.asarray(beats, float)
    ioi = np.diff(b)

    # half-beat: each beat moved to the midpoint of its own interval, so tempo
    # and rubato are preserved exactly and only the phase is wrong.  This is the
    # hardest control and the one a phase-blind measure cannot reject.
    half = b[:-1] + ioi / 2

    # scrambled: same beat count, same duration, no temporal structure at all.
    scram = np.sort(rng.uniform(b[0], b[-1], len(b)))

    # wrong tempo: isochronous at 1.37x the median IOI (irrational-ish ratio, so
    # it cannot accidentally lock to a metrical level of the truth), random phase.
    step = float(np.median(ioi)) * 1.37
    wrong = np.arange(rng.uniform(0, step), duration, step)

    return {"half_beat": half, "scrambled": scram, "wrong_tempo": wrong}


# --------------------------------------------------------------------------- #
# SMC calibration
# --------------------------------------------------------------------------- #

def smc_pairs(audio_dir=SMC_AUDIO, ann_dir=SMC_ANN):
    """Join audio to annotations.  Stems differ: SMC_001.wav <-> SMC_001_2_1_1_a.txt,
    so the join key is the first two underscore-separated tokens."""
    ann = {}
    for p in sorted(Path(ann_dir).glob("*.txt")):
        ann.setdefault("_".join(p.stem.split("_")[:2]), p)
    out = []
    for w in sorted(Path(audio_dir).glob("*.wav")):
        a = ann.get(w.stem)
        if a is not None:
            out.append((w, a))
    return out


def _one(job):
    wav, ann, seed = job
    import librosa
    y, _ = librosa.load(str(wav), sr=SR, mono=True)
    beats = np.loadtxt(str(ann)).astype(float).ravel()
    dur = len(y) / SR
    beats = beats[(beats >= 0) & (beats < dur)]
    if len(beats) < 6:
        return None
    res = {"track": wav.stem,
           "true": beat_adherence(y, beats, seed=seed)}
    for name, g in control_grids(beats, dur, seed=seed).items():
        res[name] = beat_adherence(y, g, seed=seed)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=1, help="unused; kept for CLI parity")
    ap.add_argument("--n", type=int, default=0, help="limit tracks (0 = all)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "cache" / "sa3_beat_adherence"))
    args = ap.parse_args()

    pairs = smc_pairs()
    if args.n:
        pairs = pairs[: args.n]
    print(f"[setup] {len(pairs)} SMC audio/annotation pairs, sr={SR} hop={HOP} "
          f"({HOP/SR*1000:.2f} ms), {N_NULL} rotations/track\n")

    jobs = [(w, a, i) for i, (w, a) in enumerate(pairs)]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(_one, jobs)):
            if r:
                rows.append(r)
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(jobs)}]")

    conds = ["true", "half_beat", "scrambled", "wrong_tempo"]
    summary = {}
    print(f"\n{'condition':12s} {'phase_align':>26s}   {'beat_energy_z':>20s}")
    for c in conds:
        pa = np.array([r[c]["phase_align"] for r in rows if r.get(c)])
        bz = np.array([r[c]["beat_energy_z"] for r in rows if r.get(c)])
        summary[c] = {
            "n": int(len(pa)),
            "phase_align_mean": float(pa.mean()), "phase_align_sd": float(pa.std()),
            "phase_align_median": float(np.median(pa)),
            "beat_energy_z_mean": float(bz.mean()), "beat_energy_z_sd": float(bz.std()),
            "beat_energy_z_median": float(np.median(bz)),
        }
        print(f"{c:12s}  mean {pa.mean():+.4f} sd {pa.std():.4f} med {np.median(pa):+.4f}"
              f"   mean {bz.mean():+6.2f} sd {bz.std():5.2f}")

    # separation: d' between true and each control, and accuracy of the single
    # best threshold on phase_align.
    t_pa = np.array([r["true"]["phase_align"] for r in rows])
    t_bz = np.array([r["true"]["beat_energy_z"] for r in rows])
    print()
    for c in conds[1:]:
        for key, tv in (("phase_align", t_pa), ("beat_energy_z", t_bz)):
            cv = np.array([r[c][key] for r in rows])
            d = (tv.mean() - cv.mean()) / np.sqrt((tv.var() + cv.var()) / 2)
            # AUC via rank statistic
            allv = np.concatenate([tv, cv])
            ranks = allv.argsort().argsort() + 1
            auc = (ranks[: len(tv)].sum() - len(tv) * (len(tv) + 1) / 2) / (len(tv) * len(cv))
            # best single threshold
            cand = np.unique(allv)
            acc = max(float(((tv >= th).sum() + (cv < th).sum()) / (len(tv) + len(cv)))
                      for th in cand)
            summary.setdefault("separation", {}).setdefault(c, {})[key] = {
                "d_prime": float(d), "auc": float(auc), "best_acc": acc}
            print(f"true vs {c:12s} [{key:14s}]  d'={d:6.2f}  AUC={auc:.4f}  acc={acc:.4f}")

    # stratify by how impulsive the material is -- the honest question is whether
    # this works on the soft-onset expressive audio the project actually targets.
    cr = np.array([r["true"]["onset_strength"] for r in rows])
    q = np.quantile(cr, [0.25, 0.5, 0.75])
    print(f"\nonset_strength (envelope crest) quartiles: {q.round(2)}")
    print(f"{'stratum':22s} {'n':>4s} {'true pa':>9s} {'half pa':>9s} "
          f"{'true z':>8s} {'AUC(t/half)':>12s} {'AUC(t/scram)':>13s}")
    edges = [-np.inf, q[0], q[1], q[2], np.inf]
    names = ["Q1 softest", "Q2", "Q3", "Q4 most impulsive"]
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        m = (cr >= lo) & (cr < hi)
        if m.sum() == 0:
            continue
        sub = [r for r, k in zip(rows, m) if k]
        tv = np.array([r["true"]["phase_align"] for r in sub])
        hv = np.array([r["half_beat"]["phase_align"] for r in sub])
        sv = np.array([r["scrambled"]["phase_align"] for r in sub])
        tz = np.array([r["true"]["beat_energy_z"] for r in sub])
        def _auc(a, b):
            allv = np.concatenate([a, b]); rk = allv.argsort().argsort() + 1
            return (rk[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        summary.setdefault("strata", {})[nm] = {
            "n": int(m.sum()), "true_phase_align": float(tv.mean()),
            "half_phase_align": float(hv.mean()),
            "true_beat_energy_z": float(tz.mean()),
            "auc_true_vs_half": float(_auc(tv, hv)),
            "auc_true_vs_scrambled": float(_auc(tv, sv)),
            "crest_range": [float(cr[m].min()), float(cr[m].max())],
        }
        print(f"{nm:22s} {m.sum():4d} {tv.mean():+9.4f} {hv.mean():+9.4f} "
              f"{tz.mean():8.2f} {_auc(tv,hv):12.4f} {_auc(tv,sv):13.4f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "smc_calibration.json", "w") as f:
        json.dump({"config": {"sr": SR, "hop": HOP, "n_null": N_NULL,
                              "n_tracks": len(rows)},
                   "summary": summary, "per_track": rows}, f, indent=2)
    print(f"\n[out] {out}/smc_calibration.json")


if __name__ == "__main__":
    main()
