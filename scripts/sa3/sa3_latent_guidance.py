"""Steer SA3 sampling toward SMC's region of latent space, analytically.

Motivation: the corpus was targeted with six English prompt strings, which is an
indirect and lossy handle -- the resulting clips are the least diverse set we
measured (intra-set 7.46 vs SMC's 9.76) and cover only 42.6% of SMC's manifold.
The target region is better defined by the 217 encoded SMC latents we already have.

Why guidance rather than sampling z ~ N(mu_SMC, Sigma_SMC): that draws from an
ellipsoid whose mass is mostly OFF the decodable manifold -- cache/median_z/median.wav
is the artifact, spectrally plausible but crest 11.2 against real SMC's 25.9.
Guidance instead perturbs a sampler whose prior IS the diffusion model, so the
denoiser re-projects onto the manifold at every step; the SMC term only tilts
which realistic sample is reached.

Why the target is a DRAW, not the mean: guiding toward a fitted density's mode
reproduces exactly the median-blur failure. Each generation draws s* from the
empirical distribution -- one real SMC song's statistics -- so the target set has
190 modes instead of one.

Statistic S(z), per channel, over time: mean, std, and mean |z[t+1]-z[t]| (latent
flux). The flux term is load-bearing: pooled mean/std are BLIND to temporal
structure, and the measured gap to SMC is largest in exactly that dimension
(temporal std of spectral centroid d=1.15, onset rate d=0.77). Guiding on
marginals alone would optimise the one thing that is not broken.

The seam: sample_discrete_euler builds `denoised` only for the callback and
discards it (the update is x = x + dt*v), so mutating `denoised` is a no-op --
verified. `x` is the live tensor and z0 = x - t*v depends on x with coefficient
1, so steering z0 by delta is exactly adding delta to x in place. No fork.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/sogang/jaehoon/Analyze-SMC")
sys.path.insert(0, str(ROOT))
CACHE = Path("/disk1/jaehoon/sa3_beat_cache")
ANN = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")


def stats_t(z):
    """z: [B,C,T] torch -> [B, 3C]. Differentiable, cheap, no model involved."""
    import torch
    m = z.mean(-1)
    s = z.std(-1)
    f = (z[..., 1:] - z[..., :-1]).abs().mean(-1)
    return torch.cat([m, s, f], dim=-1)


def build_cloud_bank(eval_fold, device, max_songs=190):
    """Per-song frame clouds for SMC songs outside the eval fold."""
    import torch
    fold = {}
    for line in open(ANN / "smc" / "8-folds.split"):
        p = line.split()
        if len(p) == 2:
            fold[p[0]] = int(p[1])
    out, names = [], []
    for f in sorted(CACHE.glob("smc__*.npy")):
        stem = f.stem
        if "__w" in stem or "__tempo" in stem:
            continue
        song = stem.split("__", 1)[1]
        if fold.get(song, -1) == eval_fold:
            continue
        z = torch.from_numpy(np.load(f)).float().to(device)
        out.append(cloud_pair(z))
        names.append(song)
        if len(out) >= max_songs:
            break
    return out, names


def build_bank(eval_fold, device):
    """Per-song statistics for every SMC song OUTSIDE the eval fold."""
    import torch
    fold = {}
    for line in open(ANN / "smc" / "8-folds.split"):
        p = line.split()
        if len(p) == 2:
            fold[p[0]] = int(p[1])
    rows, names = [], []
    for f in sorted(CACHE.glob("smc__*.npy")):
        stem = f.stem
        if "__w" in stem or "__tempo" in stem:      # skip augmented variants
            continue
        song = stem.split("__", 1)[1]
        if fold.get(song, -1) == eval_fold:          # fold discipline
            continue
        z = torch.from_numpy(np.load(f)).float().unsqueeze(0)
        rows.append(stats_t(z)[0])
        names.append(song)
    S = torch.stack(rows).to(device)
    w = 1.0 / (S.var(0) + 1e-6)                      # z-score units per statistic
    return S, w, names



def audio_feats(y, sr, n_mels=24):
    """48-dim: per-band log-mel mean and TEMPORAL std.

    Same construction the audit used to show the prompt-based corpus was the
    least diverse set measured (intra-set 7.46 vs SMC's 9.76) and covered only
    42.6% of SMC's manifold. Temporal std is included deliberately: band means
    alone cannot distinguish impulsive from stationary material, which is the
    largest measured gap to SMC.
    """
    import librosa
    y = np.asarray(y, dtype=float)
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    L = np.log1p(1000.0 * M)
    return np.concatenate([L.mean(1), L.std(1)])


def diversity_and_coverage(G, R, k=5):
    """G, R: [n, d] generated and reference features, standardized on R.

    Returns mean intra-set pairwise distance for each, plus kNN precision
    (fraction of generated inside R's support) and recall (fraction of R
    covered), size-matched so the numbers are comparable across lambda.
    """
    from scipy.spatial.distance import cdist
    n = min(len(G), len(R))
    G, R = G[:n], R[:n]
    def intra(X):
        D = cdist(X, X); iu = np.triu_indices(len(X), 1); return float(D[iu].mean())
    DRR = cdist(R, R); np.fill_diagonal(DRR, np.inf)
    r_rad = np.sort(DRR, 1)[:, k - 1]              # each real point's kNN radius
    DGR = cdist(G, R)
    precision = float((DGR <= r_rad[None, :]).any(1).mean())   # gen inside real support
    recall = float((DGR <= r_rad[None, :]).any(0).mean())      # real covered by gen
    return dict(intra_gen=intra(G), intra_ref=intra(R),
                precision=precision, recall=recall)



def sliced_w2(A, B, n_proj=64, gen=None):
    """Sliced Wasserstein-2 between two clouds A[n,d], B[m,d]. Differentiable in A.

    Order-invariant by construction: A and B are BAGS of latent frames, not
    trajectories. That is the whole point -- matching the trajectory would drag
    the target song's own beat placement into the output (the failure that makes
    SDEdit unusable here) and could reproduce the song outright. Matching only
    the frame distribution transfers timbral/dynamic character while leaving the
    generator free to place events wherever the conditioned grid demands.
    """
    import torch
    d = A.shape[1]
    P = torch.randn(d, n_proj, device=A.device, generator=gen)
    P = P / P.norm(dim=0, keepdim=True)
    a = (A @ P).sort(dim=0).values                    # [n, n_proj]
    b = (B @ P).sort(dim=0).values                    # [m, n_proj]
    n = min(a.shape[0], b.shape[0])
    # resample both to a common quantile grid
    ia = torch.linspace(0, a.shape[0] - 1, n, device=A.device).round().long()
    ib = torch.linspace(0, b.shape[0] - 1, n, device=A.device).round().long()
    return ((a[ia] - b[ib]) ** 2).mean()


def cloud_pair(z):
    """z [C,T] -> (frames [T,C], frame-deltas [T-1,C]). Both order-invariant bags."""
    zt = z.transpose(0, 1)
    return zt, zt[1:] - zt[:-1]


def make_callback(target, weights, lam, log=None):
    import torch

    def cb(d):
        if lam == 0:
            return
        # Sampling runs under no_grad/inference_mode, so the callback must both
        # re-enable grad AND rebuild the tensor -- an inference tensor cannot join
        # an autograd graph even after .clone().
        with torch.enable_grad():
            z0 = torch.empty_like(d["denoised"]).copy_(d["denoised"]).requires_grad_(True)
            if isinstance(target, tuple):          # cloud target: (frames, deltas)
                Fz, Dz = cloud_pair(z0[0])
                loss = sliced_w2(Fz, target[0]) + sliced_w2(Dz, target[1])
            else:                                  # legacy pooled-statistic target
                loss = (weights * (stats_t(z0)[0] - target) ** 2).sum()
            g, = torch.autograd.grad(loss, z0)
        # Scale the step to a FRACTION OF x's OWN RMS. Normalising g to unit RMS
        # alone made every step displace the latent by more than its own magnitude
        # (x has RMS ~0.72), so the first sweep ran entirely in the over-guidance
        # regime and the objective diverged 3-4 orders of magnitude. lam is now
        # "fraction of ||x||_rms per step".
        g = g / (g.pow(2).mean().sqrt() + 1e-12)
        x_rms = d["x"].pow(2).mean().sqrt()
        sigma = d["t"].view(-1, 1, 1)
        d["x"].add_(-lam * sigma * x_rms * g)
        if log is not None:
            log.append(float(loss))
    return cb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="/disk1/jaehoon/sa3_beat_runs/cn_fold0_rw/beat_adapter_final.safetensors")
    ap.add_argument("--lams", type=float, nargs="+", default=[0, 1, 2, 4, 8, 16])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--cfg-scale", type=float, default=6.0)
    ap.add_argument("--eval-fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--objective", choices=["stat", "cloud"], default="cloud",
                    help="cloud = sliced-Wasserstein between frame bags (order-invariant, "
                         "cannot transfer beat placement or reproduce the song); "
                         "stat = legacy pooled mean/std/flux")
    ap.add_argument("--out", default="cache/latent_guidance_sweep.json")
    a = ap.parse_args()

    import torch, soundfile as sf
    sys.path.insert(0, str(ROOT / "scripts" / "sa3"))
    from sa3_beat.build import load_beat_model
    from sa3_beat.generate import load_adapter, _patch_conditioning_inputs, beat_conditioning
    from sa3_beat.data import load_beats
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    from sa3_beat_adherence import beat_adherence

    dev = f"cuda:{a.gpu}"
    model, cfg = load_beat_model(device=dev)
    load_adapter(model, a.adapter, dev)
    _patch_conditioning_inputs(model)
    sr, ds = cfg["sample_rate"], 4096
    sample_size = (int(a.seconds * sr) // ds) * ds
    n_lat = sample_size // ds

    if a.objective == "cloud":
        cbank, names = build_cloud_bank(a.eval_fold, dev)
        bank, w = None, None
        print(f"[bank] cloud objective: {len(names)} SMC songs outside fold {a.eval_fold}; "
              f"frames/song ~{cbank[0][0].shape[0]}", flush=True)
    else:
        bank, w, names = build_bank(a.eval_fold, dev)
        cbank = None
        print(f"[bank] stat objective: {len(names)} songs; stat dim {bank.shape[1]}", flush=True)

    # real SMC reference for the realism readout
    fold_of = {}
    for line in open(ANN / "smc" / "8-folds.split"):
        pp = line.split()
        if len(pp) == 2:
            fold_of[pp[0]] = int(pp[1])
    real_crest, real_feats = [], []
    for p in sorted(Path(ROOT / "SMC_MIREX" / "SMC_MIREX_Audio").glob("*.wav")):
        song = p.stem.lower()
        if fold_of.get(song, -1) == a.eval_fold:      # keep the eval fold out of the reference too
            continue
        y, s = sf.read(str(p), dtype="float32", always_2d=True)
        y = y.mean(1)[: int(a.seconds * s)]
        real_crest.append(crest(y, s)); real_feats.append(audio_feats(y, s))
        if len(real_feats) >= 60:
            break
    real_feats = np.stack(real_feats)
    mu, sd = real_feats.mean(0), real_feats.std(0) + 1e-9
    Rz = (real_feats - mu) / sd
    print(f"[ref] real SMC (fold != {a.eval_fold}): n={len(Rz)} crest={np.mean(real_crest):.2f} "
          f"intra-set spread={float(__import__('scipy.spatial.distance', fromlist=['cdist']).cdist(Rz, Rz)[np.triu_indices(len(Rz),1)].mean()):.2f}", flush=True)

    grids = []
    for f in sorted(CACHE.glob("smc__*.beats"))[:200]:
        if "__w" in f.stem or "__tempo" in f.stem:
            continue
        b, _ = load_beats(f)
        b = b[(b >= 10) & (b < 10 + a.seconds)] - 10.0
        if len(b) >= 8:
            grids.append(b)
        if len(grids) >= a.n:
            break

    rng = np.random.RandomState(0)
    out = {}
    for lam in a.lams:
        rows = []
        for k in range(a.n):
            tgt = (cbank[rng.randint(len(cbank))] if cbank is not None
                   else bank[rng.randint(len(bank))])      # a DRAW, not the mean
            g = grids[k % len(grids)]
            item = beat_conditioning(g, None, n_lat, "solo piano, quiet, expressive", a.seconds)
            cond = model.conditioner([item], dev)
            io = cfg["model"]["io_channels"]
            cond["inpaint_mask"] = [torch.zeros(1, 1, n_lat, device=dev), None]
            cond["inpaint_masked_input"] = [torch.zeros(1, io, n_lat, device=dev), None]
            losses = []
            audio = generate_diffusion_cond(
                model, steps=a.steps, cfg_scale=a.cfg_scale, conditioning_tensors=cond,
                sample_size=sample_size, seed=1000 + k, device=dev,
                callback=make_callback(tgt, w, lam, losses))
            y = audio.squeeze(0).float().cpu().numpy()
            y = y.mean(0) if y.ndim > 1 else y
            ad = beat_adherence(y, g, sr=sr)
            rows.append(dict(feat=audio_feats(y, sr).tolist(), crest=crest(y, sr),
                             adherence=float(ad["phase_align"]) if ad else 0.0,
                             half=float(beat_adherence(y, g + np.median(np.diff(g)) / 2, sr=sr)["phase_align"]),
                             loss0=losses[0] if losses else None,
                             lossN=losses[-1] if losses else None))
        out[str(lam)] = rows
        c = np.array([r["crest"] for r in rows]); ad = np.array([r["adherence"] for r in rows])
        hf = np.array([r["half"] for r in rows])
        Gz = (np.array([r["feat"] for r in rows]) - mu) / sd
        dc = diversity_and_coverage(Gz, Rz)
        print(f"          diversity={dc['intra_gen']:.2f} (SMC {dc['intra_ref']:.2f})  "
              f"precision={dc['precision']:.3f}  recall={dc['recall']:.3f}", flush=True)
        out[str(lam) + "_divcov"] = dc
        l0 = [r["loss0"] for r in rows if r["loss0"]]; lN = [r["lossN"] for r in rows if r["lossN"]]
        print(f"lam={lam:6.3f}  crest={c.mean():6.2f}  adherence={ad.mean():+.4f}+/-{ad.std(ddof=1)/np.sqrt(len(ad)):.4f}  "
              f"half={hf.mean():+.4f}  stat-loss {np.mean(l0) if l0 else float('nan'):.0f}->"
              f"{np.mean(lN) if lN else float('nan'):.0f}"
              f"  ({'converging' if l0 and lN and np.mean(lN) < np.mean(l0) else 'DIVERGING' if l0 else ''})", flush=True)
    json.dump({"real_smc_crest": float(np.mean(real_crest)), "sweep": out}, open(a.out, "w"))
    print(f"\n[out] {a.out}")


def crest(y, sr):
    import librosa
    env = librosa.onset.onset_strength(y=np.asarray(y, dtype=float), sr=sr)
    env = np.maximum(env - np.median(env), 0)
    pos = env[env > 0]
    return float(env.max() / (pos.mean() + 1e-12)) if len(pos) else 0.0


if __name__ == "__main__":
    main()
