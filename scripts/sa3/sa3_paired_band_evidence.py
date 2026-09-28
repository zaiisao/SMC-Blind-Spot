"""Paired band-rank onset evidence for generated clips (audio only, no beat model).

For each generated clip we compare the ON-beat vs OFF-beat onset contrast at the
inherited beat times against the SAME contrast measured on the same musical
window of the source audio it was generated from. Damage shows up as a drop in
that contrast relative to the source; music that is merely difficult has a low
contrast in BOTH, so the paired difference stays near zero. Four mel bands are
scored separately and rank-averaged.

CPU only.
"""
import csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

BT = "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"
PROV = Path("/disk1/jaehoon/sdedit_slow")
FPS = 50
BANDS = {"full": (0, 128), "bass": (0, 32), "mid": (32, 80), "hi": (80, 128)}
# (band, on-tolerance, off-tolerance, off-beat offsets as fractions of the IBI,
#  combiner over the off-beat peaks, clip aggregate) -- chosen on SELECT only
COMPONENTS = [
    ("full", 3, 5, (0.25,),      "max",  "mean"),
    ("hi",   3, 5, (0.25,),      "max",  "tmean"),
    ("bass", 3, 5, (0.25, 0.5),  "mean", "mean"),
    ("mid",  3, 5, (0.25,),      "max",  "mean"),
]

def flux(sp, lo, hi):
    d = np.diff(sp[:, lo:hi], axis=0)
    return np.concatenate([[0.0], np.maximum(d, 0).sum(axis=1)])

def peaks(f, idx, tol):
    w = np.arange(-tol, tol + 1)
    return f[np.clip(idx[:, None] + w[None, :], 0, len(f) - 1)].max(axis=1)

def contrast(f, idx, half, ton, toff, fracs, comb):
    on = peaks(f, idx, ton)
    offs = []
    for fr in fracs:
        d = max(2, int(round(half * 2 * fr)))
        offs += [peaks(f, idx - d, toff), peaks(f, idx + d, toff)]
    O = np.stack(offs)
    o = O.max(axis=0) if comb == "max" else O.mean(axis=0)
    return np.log((on + 1e-6) / (o + 1e-6))

def agg(v, how):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3: return np.nan
    if how == "mean": return float(v.mean())
    s = np.sort(v); k = max(1, len(v) // 10)      # tmean: 10% trimmed
    return float(s[k:len(v) - k].mean())

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])

def partial(e, y, c):
    e, y, c = (np.asarray(x, float) for x in (e, y, c))
    m = np.isfinite(e) & np.isfinite(y) & np.isfinite(c)
    R = [rankdata(x[m]) for x in (e, y, c)]
    R = [(r - r.mean()) / (r.std() + 1e-12) for r in R]
    n = len(R[0])
    re = R[0] - np.dot(R[0], R[2]) / n * R[2]
    ry = R[1] - np.dot(R[1], R[2]) / n * R[2]
    return float(np.corrcoef(re, ry)[0, 1])

def main():
    os.chdir(BT); sys.path.insert(0, BT)
    from beat_this.dataset.mmnpz import MemmappedNpzFile
    truth = {r["stem"]: r for r in csv.DictReader(
        open("/home/sogang/jaehoon/Analyze-SMC/cache/sdedit_vs_source.csv"))}
    npz = MemmappedNpzFile(Path("data/audio/spectrograms/sdedit.npz"))
    annd = Path("data/annotations/sdedit/annotations/beats")
    src = {ds: MemmappedNpzFile(Path(f"data/audio/spectrograms/{ds}.npz"))
           for ds in ("smc", "asap", "hainsworth", "simac", "rwc")
           if Path(f"data/audio/spectrograms/{ds}.npz").exists()}
    cache = {}
    def sflux(c, st):
        if (c, st) not in cache:
            try:
                sp = np.asarray(src[c][f"{st}/track"], dtype=np.float32)
                cache[(c, st)] = {b: flux(sp, *r) for b, r in BANDS.items()}
            except Exception:
                cache[(c, st)] = None
        return cache[(c, st)]

    comp = {i: {} for i in range(len(COMPONENTS))}
    meta = {}
    for s in sorted(set(truth) & {k.split("/")[0] for k in npz.files}):
        ref = np.loadtxt(annd / f"{s}.beats", ndmin=2)[:, 0]
        sp = np.asarray(npz[f"{s}/track"], dtype=np.float32)
        fg = {b: flux(sp, *r) for b, r in BANDS.items()}
        T = len(fg["full"])
        ibi = float(np.median(np.diff(ref))) if len(ref) > 1 else 1.0
        half = max(8, int(round(ibi * FPS * 0.5)))
        idx = np.round(ref * FPS).astype(int)
        keep = (idx >= 8) & (idx < T - 8)
        idx, rk = idx[keep], ref[keep]
        t = truth[s]
        meta[s] = (float(t["F_src"]) - float(t["F_gen"]), 1.0 - float(t["F_src"]),
                   "SELECT" if int(hashlib.md5(s.encode()).hexdigest(), 16) % 2 == 0 else "REPORT")
        prov = json.loads((PROV / f"{s}.json").read_text())
        c, st = prov["source"].split("__", 1)
        sf = sflux(c, st)
        if sf is None or len(idx) < 3:
            continue
        si = np.round((rk + prov["window_start"]) * FPS).astype(int)
        ok = (si >= 8) & (si < len(sf["full"]) - 8)
        if ok.sum() < 3:
            continue
        gi, si = idx[ok], si[ok]
        for i, (band, ton, toff, fr, cb, how) in enumerate(COMPONENTS):
            v = (contrast(fg[band], gi, half, ton, toff, fr, cb)
                 - contrast(sf[band], si, half, ton, toff, fr, cb))
            comp[i][s] = agg(v, how)

    # rank-average the four band components (normalised ranks, higher = more evidence)
    stems = [s for s in meta if all(s in comp[i] and np.isfinite(comp[i][s]) for i in comp)]
    ranks = []
    for i in comp:
        r = rankdata([comp[i][s] for s in stems]) / len(stems)
        ranks.append(r)
    e = dict(zip(stems, np.mean(np.stack(ranks), axis=0)))

    out = "/home/sogang/jaehoon/Analyze-SMC/cache/sdedit_paired_band_evidence.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "evidence", "c_full", "c_hi", "c_bass", "c_mid", "damage", "difficulty", "split"])
        for s in stems:
            w.writerow([s, f"{e[s]:.6f}"] + [f"{comp[i][s]:.6f}" for i in range(4)]
                       + [f"{meta[s][0]:.6f}", f"{meta[s][1]:.6f}", meta[s][2]])
    for sp in ("SELECT", "REPORT"):
        ss = [s for s in stems if meta[s][2] == sp]
        rp = partial([e[s] for s in ss], [meta[s][0] for s in ss], [meta[s][1] for s in ss])
        rd = spearman([e[s] for s in ss], [meta[s][1] for s in ss])
        print(f"{sp}: n={len(ss)}  rho_partial(damage|difficulty)={rp:+.3f}  rho(difficulty)={rd:+.3f}")
    print("wrote", out)

main()
