"""DECISIVE PROBE: is SA3's failure to reach SMC's soft-onset region a STEERING
problem or a SUPPORT problem?

(a) STEERING  -> lowering cfg_scale (1.0 = raw unguided prior) should WIDEN the
    generated attack_slope distribution and extend its LOWER tail into SMC's range.
(b) SUPPORT   -> lowering cfg makes output worse/noisier without extending the soft
    tail; the distribution stays narrow, or widens symmetrically / upward.

Reuses onset_features() from sa3_prior_probe.py unchanged.  Also sweeps text
conditioning (target-aimed prompts, and negative prompting against percussive /
sharp attacks), and measures crude quality proxies so that "wider because it is
noise" can be told apart from "wider because it is soft music".

    CUDA_VISIBLE_DEVICES=3 /disk1/jaehoon/sa3env/bin/python \
        scripts/sa3/sa3_cfg_prompt_sweep.py
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sa3_prior_probe import onset_features, SA3_DIR  # noqa: E402

SMC = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"

P_EXPR = "solo piano, quiet, expressive rubato, classical, intimate recording"
P_TARGET = ("very slow solo piano, extremely quiet, soft legato touch, sparse sustained "
            "notes, unaccented, distant room recording with long reverb, no percussion, "
            "no drums, free tempo rubato")
P_AMBIENT = ("slow ambient drone, soft swelling strings, no percussion, no attacks, "
             "distant reverberant hall, very quiet, sustained")
NEG_SHARP = ("percussive, drums, sharp attack, staccato, plucked, transient, loud, "
             "close-miked, bright, rhythmic")


def quality_features(y, sr):
    """Crude objective degradation signals. Noise/incoherence -> high spectral
    flatness, high ZCR, low harmonic energy fraction, flat (unstructured) loudness."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    flat = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, hop_length=512)))
    H, P = librosa.decompose.hpss(S)
    he = float((H ** 2).sum()); pe = float((P ** 2).sum())
    harm = he / (he + pe + 1e-12)
    rms = librosa.feature.rms(S=S)[0]
    return {
        "spectral_flatness": flat,
        "zcr": zcr,
        "harmonic_frac": harm,
        "rms_cv": float(np.std(rms) / (np.mean(rms) + 1e-12)),
        "centroid": float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))),
    }


def stats(vals):
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "sem": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "cache" / "sa3_cfg_prompt_sweep"))
    args = ap.parse_args()

    import soundfile as sf, librosa
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict
    from safetensors.torch import load_file

    device = "cuda:0" if torch.cuda.is_available() else "cpu"   # CUDA_VISIBLE_DEVICES=3
    out = Path(args.out); (out / "wav").mkdir(parents=True, exist_ok=True)

    # ---- real SMC reference: all 217 files, scored identically -----------------
    ref_path = out / "smc_reference.json"
    if ref_path.exists():
        smc_slopes = json.load(open(ref_path))["attack_slope"]
    else:
        smc_slopes = []
        for p in sorted(SMC.glob("*.wav")):
            y, _ = librosa.load(str(p), sr=44100, mono=True, duration=args.seconds)
            f = onset_features(y, 44100)
            if f:
                smc_slopes.append(f["attack_slope"])
        json.dump({"attack_slope": smc_slopes}, open(ref_path, "w"))
    smc = np.asarray(smc_slopes)
    print(f"[smc] n={smc.size} mean={smc.mean():.3f} sd={smc.std(ddof=1):.3f} "
          f"min={smc.min():.3f} max={smc.max():.3f}", flush=True)

    cfg_json = json.load(open(SA3_DIR / "model_config.json"))
    model = create_model_from_config(cfg_json)
    copy_state_dict(model, load_file(str(SA3_DIR / "model.safetensors")))
    model = model.to(device).eval().requires_grad_(False)
    sr = cfg_json["sample_rate"]
    ds = model.pretransform.downsampling_ratio
    sample_size = (int(args.seconds * sr) // ds) * ds
    n_latent = sample_size // ds
    io_ch = cfg_json["model"]["io_channels"]

    def cond_for(prompt, B):
        c = model.conditioner([{"prompt": prompt, "seconds_total": args.seconds}] * B, device)
        c["inpaint_mask"] = [torch.zeros(B, 1, n_latent, device=device), None]
        c["inpaint_masked_input"] = [torch.zeros(B, io_ch, n_latent, device=device), None]
        return c

    # ---- the settings grid -----------------------------------------------------
    RUNS = []
    for c in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
        RUNS.append((f"cfg{c:g}_expr", P_EXPR, None, c))
    for tag, pr in [("target", P_TARGET), ("ambient", P_AMBIENT)]:
        for c in [1.0, 3.0, 6.0]:
            RUNS.append((f"cfg{c:g}_{tag}", pr, None, c))
    # negative prompting only has an effect when cfg > 1
    RUNS.append(("cfg6_expr_neg", P_EXPR, NEG_SHARP, 6.0))
    RUNS.append(("cfg6_target_neg", P_TARGET, NEG_SHARP, 6.0))
    RUNS.append(("cfg12_target_neg", P_TARGET, NEG_SHARP, 12.0))

    results = {"_config": vars(args), "_smc": stats(smc), "_prompts":
               {"expr": P_EXPR, "target": P_TARGET, "ambient": P_AMBIENT, "neg": NEG_SHARP},
               "runs": {}}
    res_path = out / "results.json"

    for tag, prompt, neg, cfg_scale in RUNS:
        t0 = time.time()
        feats, quals, n_unusable = [], [], 0
        B = args.batch
        pos = cond_for(prompt, B)
        negt = cond_for(neg, B) if neg else None
        for bi in range(0, args.n, B):
            audio = generate_diffusion_cond(
                model, steps=args.steps, cfg_scale=cfg_scale,
                conditioning_tensors=pos,
                negative_conditioning_tensors=negt,
                batch_size=B, sample_size=sample_size,
                seed=2000 + bi, device=device)
            for j in range(B):
                w = audio[j].to(torch.float32).cpu().numpy()
                w = w / (np.abs(w).max() + 1e-9)
                sf.write(str(out / "wav" / f"{tag}_{bi+j:02d}.wav"), w.T, sr)
                m = w.mean(axis=0)
                f = onset_features(m, sr)
                if f is None:
                    n_unusable += 1
                else:
                    feats.append(f)
                quals.append(quality_features(m, sr))
        sl = [f["attack_slope"] for f in feats]
        st = stats(sl)
        q = {k: stats([d[k] for d in quals]) for k in quals[0]}
        rec = {"prompt": prompt, "negative": neg, "cfg_scale": cfg_scale,
               "n_unusable": n_unusable, "attack_slope": st,
               "other": {k: stats([f[k] for f in feats]) for k in
                         ("peak_prominence", "ioi_cv", "onset_rate_hz", "n_onsets")},
               "quality": q, "raw_attack_slope": sl,
               "secs": round(time.time() - t0, 1)}
        if st:
            rec["frac_smc_inside_gen_range"] = float(np.mean((smc >= st["min"]) & (smc <= st["max"])))
            rec["frac_smc_softer_than_softest_gen"] = float(np.mean(smc < st["min"]))
        results["runs"][tag] = rec
        json.dump(results, open(res_path, "w"), indent=1)
        print(f"{tag:18s} cfg={cfg_scale:<5g} n={st['n'] if st else 0:2d} unusable={n_unusable} "
              f"slope mean={st['mean']:.3f} sd={st['sd']:.3f} sem={st['sem']:.3f} "
              f"[{st['min']:.3f},{st['max']:.3f}]  SMC inside={rec['frac_smc_inside_gen_range']*100:.1f}% "
              f"softer={rec['frac_smc_softer_than_softest_gen']*100:.1f}%  "
              f"flat={q['spectral_flatness']['mean']:.4f} harm={q['harmonic_frac']['mean']:.3f} "
              f"({rec['secs']}s)", flush=True)

    print(f"\n[out] {res_path}\n[wav] {out/'wav'}")


if __name__ == "__main__":
    main()
