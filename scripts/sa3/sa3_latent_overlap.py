"""Do generated clips occupy the same region of SA3's latent space as SMC?

Everything measured so far is timing. This asks the other question: is the audio
itself SMC-like? SA3's latent space is the relevant one -- it is where the
generator works, and it is what the DiT models.

A distance is meaningless alone, so three references are computed alongside:

  SMC vs SMC        two random halves -- the FLOOR, i.e. how far a corpus sits
                    from itself at this sample size
  SMC vs ballroom   a genuinely different corpus -- the SCALE
  SMC vs asap       a related but distinct corpus -- an intermediate

Distance is Frechet on time-pooled latents (per-channel mean and std -> 512-dim),
the same construction as Frechet Audio Distance but in SA3's own space rather
than VGGish's.
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
from scipy import linalg

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path("/disk1/jaehoon/sa3_beat_cache")
SA3_DIR = Path("/disk1/jaehoon/sa3_small_music")
VARIANT = re.compile(r"__(w|tempo)\d+$")


def descriptors(latents):
    """(N, 512): per-channel mean and std over time."""
    return np.stack([np.concatenate([z.mean(axis=-1), z.std(axis=-1)]) for z in latents])


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False) + 1e-6 * np.eye(a.shape[1])
    s2 = np.cov(b, rowvar=False) + 1e-6 * np.eye(b.shape[1])
    covmean = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def load_corpus(prefix, n, rng):
    files = [p for p in sorted(CACHE.glob(f"{prefix}__*.npy")) if not VARIANT.search(p.stem)]
    if len(files) > n:
        files = [files[i] for i in rng.choice(len(files), n, replace=False)]
    return [np.load(p)[:, :215] for p in files]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", default="/disk1/jaehoon/sa3_gen_corpus_fold0")
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()

    import librosa, torch
    from safetensors.torch import load_file
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict
    import json as _json

    rng = np.random.RandomState(0)
    cfg = _json.loads((SA3_DIR / "model_config.json").read_text())
    model = create_model_from_config(cfg)
    copy_state_dict(model, load_file(str(SA3_DIR / "model.safetensors")))
    dev = f"cuda:{args.gpu}"
    pt = model.pretransform.to(dev).eval()
    sr, ds = cfg["sample_rate"], model.pretransform.downsampling_ratio

    wavs = sorted(Path(args.gen_dir).glob("*.wav"))
    wavs = [wavs[i] for i in rng.choice(len(wavs), min(args.n, len(wavs)), replace=False)]
    gen = []
    for w in wavs:
        y, _ = librosa.load(str(w), sr=sr, mono=False, duration=20.0)
        if y.ndim == 1:
            y = np.stack([y, y])
        n = (y.shape[1] // ds) * ds
        with torch.no_grad():
            z = pt.encode(torch.tensor(y[:, :n], dtype=torch.float32, device=dev)[None])
        gen.append(z.squeeze(0).float().cpu().numpy()[:, :215])
    print(f"[latent] generated n={len(gen)}", flush=True)

    corpora = {k: load_corpus(k, args.n, rng) for k in ("smc", "ballroom", "asap")}
    for k, v in corpora.items():
        print(f"[latent] {k} n={len(v)}")

    D = {k: descriptors(v) for k, v in corpora.items()}
    D["generated"] = descriptors(gen)

    smc = D["smc"]
    half = rng.permutation(len(smc))
    a, b = smc[half[: len(smc) // 2]], smc[half[len(smc) // 2:]]

    print("\nFrechet distance in SA3 latent space (lower = more overlap)")
    print(f"  SMC vs SMC (halves)   {frechet(a, b):10.2f}   <- floor")
    print(f"  SMC vs generated      {frechet(smc, D['generated']):10.2f}")
    print(f"  SMC vs asap           {frechet(smc, D['asap']):10.2f}")
    print(f"  SMC vs ballroom       {frechet(smc, D['ballroom']):10.2f}   <- different corpus")
    print(f"  asap vs ballroom      {frechet(D['asap'], D['ballroom']):10.2f}")


if __name__ == "__main__":
    main()
