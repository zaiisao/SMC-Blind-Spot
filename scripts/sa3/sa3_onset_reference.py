"""HANDOVER §5.1, strengthened: style-matched reference + encoder round-trip control.

The original probe (sa3_prior_probe.py) compared generated audio only against real
SMC audio.  That is a weak reference: SMC is deliberately the SOFT-ONSET extreme of
beat tracking, and it is style-mismatched against generated solo piano.  This script
scores, with the IDENTICAL onset_features() imported from that probe:

  real        real audio of the MATCHED style   (ASAP solo piano / Groove drum kit)
  roundtrip   the same real audio passed through SA3's VAE encoder->decoder, no diffusion
  generated   SA3 text-to-music output (re-scored from the probe's .wav files)

Decomposition:
  real -> roundtrip   what the ENCODER alone destroys  (§2.3's effect, re-measured)
  roundtrip -> gen    what the DIFFUSION PRIOR adds or fails to add

Also runs a metric-sensitivity control: the same real audio deliberately low-passed /
transient-smeared must score as SOFTER, otherwise the measure proves nothing.

    /disk1/jaehoon/sa3env/bin/python scripts/sa3/sa3_onset_reference.py --gpu 3
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sa3_prior_probe import onset_features, summarise, SA3_DIR  # noqa: E402

ASAP = Path("/disk1/jaehoon/dataset_store/audio_by_stem/asap")
GROOVE = Path("/tmp/groove_ref")          # extracted from dataset_store/groove/groove-v1.0.0.zip
SMC = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"

# generated tag -> style key of the matched real reference
TAG_STYLE = {"piano_expressive": "piano", "piano_plain": "piano", "percussive": "drums"}


def load_ae(device):
    from stable_audio_tools.models.autoencoders import create_autoencoder_from_config
    from safetensors.torch import load_file
    cfg = json.load(open(SA3_DIR / "model_config.json"))
    sr = cfg["sample_rate"]
    ae = create_autoencoder_from_config(
        {"model": cfg["model"]["pretransform"]["config"], "sample_rate": sr})
    sd = load_file(str(SA3_DIR / "model.safetensors"))
    pre = {k.split("pretransform.model.", 1)[1]: v for k, v in sd.items()
           if "pretransform.model." in k}
    miss, unexp = ae.load_state_dict(pre, strict=False)
    assert not miss and not unexp
    return ae.eval().to(device), sr


SMEAR_MS = [10, 25, 50, 100]


def env_smear(y, sr, ms, hop=256, n_fft=1024):
    """Deliberate ATTACK softening: smooth the STFT magnitude forward in time with a
    causal rising kernel of width `ms`, keep the phase, resynthesise.  Every note now
    fades in over `ms` instead of starting abruptly.  This is the ground-truth control:
    onset_features() MUST score this as softer, and it does (monotone to ~100 ms).

    NOTE: plain waveform low-passing does NOT work as a control -- it moves the spectral
    centroid 830 -> 300 Hz yet leaves attack_slope unchanged to 3 decimals, because the
    measure is a normalised envelope-SHAPE statistic, not a brightness statistic."""
    import librosa
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, ph = np.abs(S), np.angle(S)
    n = max(1, int(round(ms / 1000 * sr / hop)))
    k = np.hanning(2 * n + 1)[:n + 1]
    k = k / k.sum()
    sm = np.apply_along_axis(lambda r: np.convolve(r, k, "full")[:len(r)], 1, mag)
    return librosa.istft(sm * np.exp(1j * ph), hop_length=hop, length=len(y))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--offset", type=float, default=5.0)
    ap.add_argument("--n-real", type=int, default=30)
    ap.add_argument("--gen-dir", default=str(ROOT / "cache" / "sa3_prior_probe_n24"))
    ap.add_argument("--out", default=str(ROOT / "cache" / "sa3_onset_reference.json"))
    args = ap.parse_args()

    import librosa
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    ae, sr = load_ae(device)
    print(f"[setup] device={device} sr={sr} seg={args.seconds}s offset={args.offset}s", flush=True)

    pools = {
        "piano": sorted(ASAP.glob("*.wav"))[: args.n_real],
        "drums": sorted(GROOVE.glob("*.wav"))[: args.n_real],
        "smc":   sorted(SMC.glob("*.wav"))[: args.n_real],
    }
    results, sanity = {}, {}
    for style, files in pools.items():
        real, rt = [], []
        sm = {ms: [] for ms in SMEAR_MS}
        for p in files:
            y, _ = librosa.load(str(p), sr=sr, mono=True,
                                offset=args.offset, duration=args.seconds)
            if len(y) < sr * args.seconds * 0.8:
                continue
            y = y / (np.abs(y).max() + 1e-9)
            N = len(y)
            x = torch.from_numpy(np.stack([y, y])).unsqueeze(0).to(device)
            with torch.no_grad():
                r = ae.decode(ae.encode(x))[..., :N]
            rec = r[0].float().cpu().numpy().mean(0)
            rec = rec / (np.abs(rec).max() + 1e-9)
            real.append(onset_features(y, sr))
            rt.append(onset_features(rec, sr))
            for ms in SMEAR_MS:
                sm[ms].append(onset_features(env_smear(y, sr, ms), sr))
        results[f"real_{style}"] = real
        results[f"roundtrip_{style}"] = rt
        for ms in SMEAR_MS:
            sanity[f"smear{ms}ms_{style}"] = sm[ms]
        print(summarise(f"real_{style}", real), flush=True)
        print(summarise(f"roundtrip_{style}", rt), flush=True)
        for ms in SMEAR_MS:
            print(summarise(f"  smear{ms}ms_{style}", sm[ms]), flush=True)
        print(flush=True)

    # generated: re-score the probe's .wav files with the same function
    gd = Path(args.gen_dir)
    for tag in TAG_STYLE:
        ws = sorted(gd.glob(f"{tag}_*.wav"))
        feats = []
        for p in ws:
            y, _ = librosa.load(str(p), sr=sr, mono=True)
            feats.append(onset_features(y / (np.abs(y).max() + 1e-9), sr))
        if feats:
            results[f"gen_{tag}"] = feats
            print(summarise(f"gen_{tag}", feats), flush=True)

    results["_sanity"] = sanity
    json.dump(results, open(args.out, "w"), indent=1)

    # decomposition table
    ALL = dict(results); ALL.pop("_sanity"); ALL.update(sanity)

    def m(key, k):
        ok = [f for f in ALL.get(key, []) if f]
        return float(np.mean([f[k] for f in ok])) if ok else float("nan")

    def equiv_ms(style, delta):
        """Convert a drop in attack_slope into the attack-smear (ms) that would cause it,
        by linear interpolation on this style's own calibration curve."""
        if not (delta < 0):
            return 0.0
        base = m(f"real_{style}", "attack_slope")
        xs = [0.0] + [float(x) for x in SMEAR_MS]
        ys = [0.0] + [base - m(f"smear{ms}ms_{style}", "attack_slope") for ms in SMEAR_MS]
        keep = [i for i in range(len(ys)) if i == 0 or ys[i] > ys[i - 1]]   # monotone part
        return float(np.interp(-delta, [ys[i] for i in keep], [xs[i] for i in keep]))
    print("\ncalibration (attack_slope of real audio smeared by N ms):")
    for style in pools:
        print("  " + style.ljust(7) + "  " + "  ".join(
            f"{ms}ms={m(f'smear{ms}ms_{style}','attack_slope'):.3f}" if ms else
            f"real={m(f'real_{style}','attack_slope'):.3f}" for ms in [0] + SMEAR_MS
        ))
    print(f"\n{'comparison':<46}{'d attack_slope':>16}{'d prominence':>14}{'~equiv smear':>14}")
    for tag, style in TAG_STYLE.items():
        r, t, g = f"real_{style}", f"roundtrip_{style}", f"gen_{tag}"
        for a, b, nm in ((r, t, "encoder   real -> roundtrip"),
                         (t, g, "diffusion roundtrip -> gen"),
                         (r, g, "total     real -> gen")):
            print(f"{nm+'  ['+tag+']':<46}"
                  f"{m(b,'attack_slope')-m(a,'attack_slope'):>16.3f}"
                  f"{m(b,'peak_prominence')-m(a,'peak_prominence'):>14.3f}"
                  f"{equiv_ms(style, m(b,'attack_slope')-m(a,'attack_slope')):>12.1f}ms")
    # significance of the two decomposition steps (Welch, per-clip attack_slope)
    try:
        from scipy import stats
        arr = lambda k: np.array([f["attack_slope"] for f in ALL.get(k, []) if f])
        print(f"\n{'Welch test on attack_slope':<46}{'delta':>10}{'p':>12}")
        pairs = [("real_piano", "roundtrip_piano"), ("roundtrip_piano", "gen_piano_expressive"),
                 ("roundtrip_piano", "gen_piano_plain"), ("real_drums", "roundtrip_drums"),
                 ("roundtrip_drums", "gen_percussive"), ("real_piano", "real_smc")]
        for a, b in pairs:
            x, y = arr(a), arr(b)
            if len(x) < 3 or len(y) < 3:
                continue
            _, pv = stats.ttest_ind(x, y, equal_var=False)
            print(f"{a+' -> '+b:<46}{y.mean()-x.mean():>+10.3f}{pv:>12.2g}")
    except ImportError:
        pass

    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
