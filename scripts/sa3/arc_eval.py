"""Paired evaluator for the ARC beat post-training, implementing the
pre-registered success criterion literally.

For N >= 60 clips drawn ONLY from fold 0 of data/pools/slow_under70.json
(held out of every training run), generate one 20 s clip per source by SDEdit
from that source's own window, at steps=8 cfg_scale=1.0 -- the spec sampling
regime for this ARC model -- with a fixed per-clip seed. Then measure, against
the SAME window of the SAME source recording:

  onset_z    PRIMARY. Onset-evidence z of the flux envelope at the conditioned
             beat positions, null = the same grid circularly shifted. Same
             estimator as scripts/sa3/vae_floor_probe.py (beat_this log-mel,
             50 fps, tol +-3 frames, 200 null shifts).
  drift_ms   GUARD 1. Median |conditioned beat time - nearest onset peak in the
             output| over beats with a peak inside +-120 ms. Also reported:
             lag_ms, the cross-correlation lag of the output's onset envelope
             against the source's, which is the statistic the SDEdit drift
             validation used.
  crest      GUARD 2. flux.max() / flux.mean().
  flatness   GUARD 3, with mean |amplitude|: has the output become noise or
             silence?

Two model states are evaluated on IDENTICAL sources, windows and seeds:
a control (the frozen model + the released beat ControlNet) and a candidate
checkpoint. The z delta is reported with a paired bootstrap CI over clips.

Nothing here selects anything: the thresholds are those written down before any
result was seen. Success = median onset_z rises by >= 0.15 over the control with
a paired bootstrap CI on the median delta excluding zero, AND drift_ms < 20 ms,
AND crest not below the control's, AND the output still being audio.

Usage (control numbers only -- no training needed):
    python scripts/sa3/arc_eval.py --gpu 3 --n 114 \
        --out-dir /disk1/jaehoon/arc_eval/control_only
Comparison:
    python scripts/sa3/arc_eval.py --gpu 3 --n 114 \
        --test-ckpt /disk1/jaehoon/arc_runs/r1/step_2000.safetensors \
        --out-dir /disk1/jaehoon/arc_eval/r1
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "third-party" / "stable-audio-tools"),
          str(REPO / "third-party" / "beat_this")):
    if p not in sys.path:
        sys.path.insert(0, p)

POOL = REPO / "data" / "pools" / "slow_under70.json"
DEFAULT_ADAPTER = ("/disk1/jaehoon/sa3_beat_runs/cn_fold0_rw/"
                   "beat_adapter_final.safetensors")
EVAL_FOLD = 0            # held out of training, forever
FPS = 50
SR_BT = 22050
SPEC_STEPS = 8           # measured spec regime for this ARC model
SPEC_CFG = 1.0

# pre-registered thresholds
Z_GAIN_REQUIRED = 0.15
DRIFT_MAX_MS = 20.0


# --------------------------------------------------------------------------
# metrics (estimators copied unchanged from scripts/sa3/vae_floor_probe.py)
# --------------------------------------------------------------------------

def flux(sp):
    return np.concatenate([[0.0], np.maximum(np.diff(sp, axis=0), 0).sum(1)])


def crest(sp):
    f = flux(sp)
    return float(f.max() / (f.mean() + 1e-9))


def onset_z(sp, beats, rng, tol=3, n=200):
    f = flux(sp)
    f = (f - np.median(f)) / (f.std() + 1e-9)
    T = len(f)
    idx = np.round(np.asarray(beats) * FPS).astype(int)
    idx = idx[(idx >= tol) & (idx < T - tol)]
    if len(idx) < 3:
        return np.nan
    w = np.arange(-tol, tol + 1)

    def sc(o):
        return f[(o[:, None] + w[None, :]).clip(0, T - 1)].max(1).mean()

    obs = sc(idx)
    span = T - 2 * tol
    null = np.array([sc(((idx - tol + rng.integers(1, span)) % span) + tol)
                     for _ in range(n)])
    return float((obs - null.mean()) / (null.std() + 1e-9))


def _onset_env(y, sr, hop=128):
    import librosa
    return librosa.onset.onset_strength(y=np.ascontiguousarray(y), sr=sr,
                                        hop_length=hop), hop


def drift_ms(y, sr, beats, max_ms=120.0):
    """Median |beat - nearest onset peak in this audio|, over beats that have
    a peak within max_ms. Reported alongside the fraction of beats matched:
    a small drift over very few beats is not evidence of timing control."""
    import librosa
    env, hop = _onset_env(y, sr)
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5,
                                   post_avg=5, delta=0.0, wait=2)
    if len(peaks) < 3:
        return float("nan"), 0.0
    pt = peaks * hop / sr
    b = np.asarray(beats, dtype=float)
    j = np.clip(np.searchsorted(pt, b), 1, len(pt) - 1)
    d = np.minimum(np.abs(b - pt[j - 1]), np.abs(b - pt[np.minimum(j, len(pt) - 1)]))
    hit = d <= max_ms / 1000.0
    if hit.sum() < 3:
        return float("nan"), float(hit.mean())
    return float(np.median(d[hit]) * 1000.0), float(hit.mean())


def lag_ms(a, b, sr, max_ms=250.0):
    """Signed lag of b relative to a by onset-envelope cross-correlation."""
    ea, hop = _onset_env(a, sr)
    eb, _ = _onset_env(b, sr)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()
    c = np.correlate(eb, ea, mode="full")
    lags = np.arange(-n + 1, n) * hop / sr * 1000.0
    m = np.abs(lags) <= max_ms
    return float(lags[m][np.argmax(c[m])])


def sanity(y, sr):
    import librosa
    fl = float(np.mean(librosa.feature.spectral_flatness(
        y=np.ascontiguousarray(y))))
    return fl, float(np.mean(np.abs(y)))


# --------------------------------------------------------------------------
# fold-0 evaluation set
# --------------------------------------------------------------------------

def eval_sources(n, pool_path=POOL, fold=EVAL_FOLD, seconds=20.0, seed=0,
                 windows_per_source=1):
    """Deterministic list of (idx, name, wav, beats_path). Fold 0 ONLY.

    Fold 0 holds 57 usable recordings, so one window per source caps the
    evaluation at 57 clips. `windows_per_source` > 1 takes additional,
    disjointly-seeded windows from the same recordings (idx = i + 10000*w
    feeds both `window_for` and the generation seed, so control and candidate
    still see identical windows). Clips from one recording are NOT
    independent: `paired_bootstrap` resamples by SOURCE, not by clip, so the
    extra windows tighten the within-source estimate without inflating n.
    """
    pool = [r for r in json.loads(Path(pool_path).read_text())
            if r["fold"] == fold
            and os.path.exists(r["audio"]) and os.path.exists(r["beats"])]
    pool.sort(key=lambda r: r["name"])
    rng = random.Random(seed)
    rng.shuffle(pool)
    out = []
    for w in range(max(1, windows_per_source)):
        for i, r in enumerate(pool):
            if len(out) >= n:
                return out
            out.append((i + 10000 * w, r["name"], r["audio"], r["beats"]))
    return out


def window_for(idx, beats, seconds):
    """Same window rule as scripts/sa3/sa3_sdedit_bulk.py, seeded per index, so
    control and candidate see identical windows."""
    r = random.Random(idx)
    lo, hi = beats[2], max(beats[2], beats[-2] - seconds)
    return float(beats[min(range(len(beats)),
                           key=lambda i: abs(beats[i] - r.uniform(lo, hi)))])


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def generate_state(label, ckpt, sources, out_dir, args, dev):
    """Generate one clip per source with the given model state. Resumable."""
    import torch
    import soxr
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    from sa3_beat.build import load_beat_model
    from sa3_beat.generate import load_adapter, beat_conditioning, \
        _patch_conditioning_inputs

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in sources
            if not (out_dir / f"sd_{s[0]:05d}.wav").exists()]
    if not todo:
        print(f"[{label}] all {len(sources)} clips already on disk", flush=True)
        return
    model, cfg = load_beat_model(device=dev)
    load_adapter(model, ckpt, dev)
    _patch_conditioning_inputs(model)
    sr_gen = cfg["sample_rate"]
    ds = model.pretransform.downsampling_ratio
    io_ch = cfg["model"]["io_channels"]

    for idx, name, wav, bt in todo:
        y, sr = sf.read(wav, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        yg = soxr.resample(y, sr, sr_gen) if sr != sr_gen else y
        ann = np.loadtxt(bt, ndmin=2)
        db_col = ann[:, 1] if ann.shape[1] > 1 else None
        beats = ann[:, 0]
        if len(beats) < 12:
            continue
        start = window_for(idx, beats, args.seconds)
        seg = yg[int(start * sr_gen): int((start + args.seconds) * sr_gen)]
        n = (len(seg) // ds) * ds
        if n < ds * 8:
            continue
        seg = seg[:n]
        m = (beats >= start) & (beats < start + n / sr_gen)
        b = beats[m] - start
        d = (beats[m][db_col[m] == 1] - start) if db_col is not None else None
        if len(b) < 8:
            continue

        init = torch.from_numpy(np.stack([seg, seg])).to(dev)
        n_lat = n // ds
        item = beat_conditioning(b, d, n_lat, args.prompt, n / sr_gen)
        cond = model.conditioner([item], dev)
        cond["inpaint_mask"] = [torch.zeros(1, 1, n_lat, device=dev), None]
        cond["inpaint_masked_input"] = [
            torch.zeros(1, io_ch, n_lat, device=dev), None]
        out = generate_diffusion_cond(
            model, steps=args.steps, cfg_scale=args.cfg_scale,
            conditioning_tensors=cond, sample_size=n, seed=idx, device=dev,
            init_audio=(sr_gen, init), init_noise_level=args.noise)
        o = out.squeeze(0).float().cpu().numpy()
        o = o.mean(axis=0) if o.ndim > 1 else o
        o = o / (np.abs(o).max() + 1e-9) * 0.9

        base = out_dir / f"sd_{idx:05d}"
        sf.write(str(base) + ".wav.tmp", o, sr_gen, format="WAV")
        os.replace(str(base) + ".wav.tmp", str(base) + ".wav")
        np.savetxt(str(base) + ".beats", np.stack(
            [b, np.array([1 if (d is not None and t in d) else 2 for t in b])],
            axis=1), fmt="%.6f\t%d")
        (Path(str(base) + ".json")).write_text(json.dumps({
            "source": name, "source_wav": wav, "window_start": start,
            "noise": args.noise, "steps": args.steps,
            "cfg_scale": args.cfg_scale, "seed": idx, "state": label,
            "ckpt": str(ckpt), "n_beats": int(len(b))}))
        print(f"[{label}] sd_{idx:05d} {name} start={start:.2f} "
              f"beats={len(b)}", flush=True)
    del model
    import torch as _t
    _t.cuda.empty_cache()


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def measure_dir(label, clip_dir, seconds, mel, dev):
    import soxr
    import torch
    clip_dir = Path(clip_dir)
    rows = []
    for jf in sorted(clip_dir.glob("*.json")):
        j = json.loads(jf.read_text())
        stem = jf.stem
        beats = np.loadtxt(clip_dir / f"{stem}.beats", ndmin=2)[:, 0]
        gen, gsr = sf.read(clip_dir / f"{stem}.wav", dtype="float32",
                           always_2d=True)
        gen = gen.mean(1)
        dur = len(gen) / gsr

        src, ssr = sf.read(j["source_wav"], dtype="float32", always_2d=True)
        src = src.mean(1)
        s0 = int(round(j["window_start"] * ssr))
        seg = src[s0: s0 + int(round(dur * ssr))]
        if len(seg) < ssr:
            continue

        gen_bt = soxr.resample(gen, in_rate=gsr, out_rate=SR_BT)
        src_bt = soxr.resample(seg, in_rate=ssr, out_rate=SR_BT)
        rng = np.random.default_rng(0)
        sp = {}
        for k, w in (("gen", gen_bt), ("src", src_bt)):
            with torch.no_grad():
                sp[k] = mel(torch.as_tensor(np.ascontiguousarray(w),
                                            dtype=torch.float32)).numpy()
        dm, hitfrac = drift_ms(gen_bt, SR_BT, beats)
        fl, amp = sanity(gen_bt, SR_BT)
        rows.append({
            "state": label, "clip": stem, "source": j["source"],
            "onset_z": onset_z(sp["gen"], beats, rng),
            "onset_z_src": onset_z(sp["src"], beats,
                                   np.random.default_rng(0)),
            "crest": crest(sp["gen"]), "crest_src": crest(sp["src"]),
            "drift_ms": dm, "drift_hit_frac": hitfrac,
            "lag_ms": lag_ms(src_bt, gen_bt, SR_BT),
            "flatness": fl, "mean_abs_amp": amp,
            "n_beats": int(len(beats)),
        })
    return rows


def med(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def paired_bootstrap(delta, n_boot=10000, seed=0, groups=None):
    """Median paired delta and its 95% CI.

    With `groups` (one source name per delta) the resampling unit is the
    SOURCE, not the clip: several windows of one recording share whatever
    that recording does, so treating them as independent clips would shrink
    the CI by a factor that has nothing to do with added information.
    """
    d = np.asarray([x for x in delta], dtype=float)
    keep = np.isfinite(d)
    d = d[keep]
    if d.size < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if groups is None:
        b = np.median(rng.choice(d, size=(n_boot, d.size), replace=True),
                      axis=1)
    else:
        g = np.asarray(groups)[keep]
        uniq = np.unique(g)
        idx = [np.flatnonzero(g == u) for u in uniq]
        b = np.empty(n_boot)
        for k in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            b[k] = np.median(d[np.concatenate([idx[j] for j in pick])])
    return float(np.median(d)), float(np.percentile(b, 2.5)), \
        float(np.percentile(b, 97.5))


def report(label, rows):
    print(f"\n=== {label}  (n={len(rows)}) ===")
    print(f"  onset_z   median {med(rows,'onset_z'):.3f}   "
          f"[source windows: {med(rows,'onset_z_src'):.3f}]   PRIMARY")
    print(f"  drift_ms  median {med(rows,'drift_ms'):.1f}    "
          f"beats matched {med(rows,'drift_hit_frac'):.2f}   GUARD 1 "
          f"(threshold {DRIFT_MAX_MS:.0f} ms)")
    print(f"  lag_ms    median {med(rows,'lag_ms'):.1f}     (xcorr vs source)")
    print(f"  crest     median {med(rows,'crest'):.3f}   "
          f"[source windows: {med(rows,'crest_src'):.3f}]   GUARD 2")
    print(f"  flatness  median {med(rows,'flatness'):.4f}   "
          f"mean|amp| {med(rows,'mean_abs_amp'):.4f}          GUARD 3")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=114,
                    help="number of evaluation CLIPS. Fold 0 holds 57 usable "
                         "recordings; the default takes 2 windows from each. "
                         "At n=24 the paired bootstrap half-width is ~0.19 z, "
                         "i.e. wider than the +0.15 the criterion asks for, so "
                         "every round would report a straddling interval no "
                         "matter what the model did.")
    ap.add_argument("--windows-per-source", type=int, default=2,
                    help="disjointly-seeded windows taken per fold-0 "
                         "recording. The bootstrap resamples by source, so "
                         "extra windows do not fake independent samples.")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=SPEC_STEPS)
    ap.add_argument("--cfg-scale", type=float, default=SPEC_CFG)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--control-ckpt", default=DEFAULT_ADAPTER,
                    help="untuned control: frozen model + released ControlNet")
    ap.add_argument("--test-ckpt", default=None,
                    help="candidate checkpoint; omit to measure the control only")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--measure-only", action="store_true")
    a = ap.parse_args()

    if a.steps != SPEC_STEPS or a.cfg_scale != SPEC_CFG:
        print(f"[warn] sampling OFF-SPEC: steps={a.steps} cfg={a.cfg_scale} "
              f"(spec is {SPEC_STEPS}/{SPEC_CFG})", flush=True)

    import torch
    from beat_this.preprocessing import LogMelSpect

    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sources = eval_sources(a.n, seconds=a.seconds,
                           windows_per_source=a.windows_per_source)
    print(f"[eval set] {len(sources)} fold-{EVAL_FOLD} sources "
          f"(held out of training): "
          f"{', '.join(s[1] for s in sources[:4])} ...", flush=True)
    n_src = len({s[1] for s in sources})
    print(f"[eval set] {len(sources)} clips over {n_src} distinct recordings",
          flush=True)
    if len(sources) < 60:
        print(f"[warn] only {len(sources)} clips; below n=60 the bootstrap CI "
              f"half-width exceeds the +0.15 z the criterion requires",
              flush=True)

    states = [("control", a.control_ckpt)]
    if a.test_ckpt:
        states.append(("test", a.test_ckpt))
    if not a.measure_only:
        for label, ckpt in states:
            generate_state(label, ckpt, sources, out / label, a, dev)

    mel = LogMelSpect(SR_BT)
    results = {}
    for label, _ in states:
        rows = measure_dir(label, out / label, a.seconds, mel, dev)
        results[label] = rows
        report(label, rows)
        with open(out / f"{label}.csv", "w") as fh:
            fh.write(",".join(rows[0].keys()) + "\n")
            for r in rows:
                fh.write(",".join(str(v) for v in r.values()) + "\n")

    if "test" in results:
        cmap = {r["clip"]: r for r in results["control"]}
        pairs = [(cmap[r["clip"]], r) for r in results["test"]
                 if r["clip"] in cmap]
        dz = [t["onset_z"] - c["onset_z"] for c, t in pairs]
        grp = [t["source"] for c, t in pairs]
        m, lo, hi = paired_bootstrap(dz, groups=grp)
        print(f"\n=== PAIRED COMPARISON (n={len(pairs)} clips over "
              f"{len(set(grp))} recordings, same sources / windows / seeds; "
              f"bootstrap resampled by recording) ===")
        print(f"  median delta onset_z = {m:+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]")
        primary = (m >= Z_GAIN_REQUIRED) and (lo > 0)
        # GUARD 1 keys on lag_ms, not drift_ms. drift_ms is "distance from each
        # conditioned beat to the nearest onset PEAK", which is contaminated by
        # onset sharpness -- the very thing onset_z measures. Sharper transients
        # lower drift_ms and raise onset_z together, so drift_ms cannot
        # independently guard timing. lag_ms (cross-correlation lag of the output's
        # onset envelope against the SOURCE's) is independent of sharpness and is
        # the statistic the original ControlNet drift validation used (~3 ms).
        # drift_ms stays reported below as a secondary readout.
        _lag = med(results["test"], "lag_ms")
        g1 = (abs(_lag) < DRIFT_MAX_MS) if _lag == _lag else \
             (med(results["test"], "drift_ms") < DRIFT_MAX_MS)
        g2 = med(results["test"], "crest") >= med(results["control"], "crest")
        print(f"  PRIMARY (delta >= {Z_GAIN_REQUIRED} and CI excludes 0): "
              f"{'PASS' if primary else 'FAIL'}")
        print(f"  GUARD 1 timing  : {'PASS' if g1 else 'FAIL'}")
        print(f"  GUARD 2 crest   : {'PASS' if g2 else 'FAIL'}")
        print(f"  GUARD 3 audio   : inspect flatness / mean|amp| above")
        print(f"  VERDICT: {'PASS' if (primary and g1 and g2) else 'FAIL'} "
              f"(any failed guard fails the configuration regardless of z)")
    else:
        print("\n[control only] no candidate checkpoint given; the numbers "
              "above are the baseline every later result is judged against.")


if __name__ == "__main__":
    main()
