"""GATE 1b: is SA3's timing loss an ENCODER artifact or a DECODER capacity limit?

WHY THE DISTINCTION DECIDES THE PROJECT. GENERATION NEVER USES THE ENCODER. A ControlNet (or a CVAE
posterior tilt, or classifier guidance) samples z from the diffusion prior and decodes it. So the
round-trip result -- median onset shift 11.6 ms, but per-track p90 81.3 ms -- conflates two causes
with opposite consequences:

  ENCODER-SIDE  the decoder CAN render precise transients; the encoder discards timing when
                compressing real audio. Generation skips that path entirely, so the tail is
                IRRELEVANT to a conditioning plan, and a beat-conditioned posterior is the right
                fix for the round-trip case.
  DECODER-SIDE  the decoder cannot place transients more finely at a 10.77 Hz latent. Then ~80 ms
                of timing error on a tenth of events is a FLOOR under every method built on this
                autoencoder, on a task whose tolerance is +-70 ms. That is a stop sign for the whole
                architecture family -- worth an hour now rather than weeks of ControlNet training.

THE PROBE. Freeze the decoder. Discard the encoder's z and instead OPTIMISE z directly by gradient
descent to reconstruct the original audio (multi-resolution log-mel L1, which is differentiable;
onset detection is not). Then measure onset displacement of the optimised reconstruction with the
same matched-onset metric as gate 1. This is GAN-inversion used as a capacity probe: it asks whether
ANY latent renders the timing, not whether the encoder happens to find it.

  optimised z MUCH better than encoder z  -> encoder-side, plan unaffected
  optimised z no better                   -> decoder capacity limit, hard floor

CONTROLS THAT MAKE THE ANSWER INTERPRETABLE:
  * z is initialised FROM the encoder, so optimisation can only improve on it; any gain is a lower
    bound on the decoder's spare capacity.
  * the loss is reported at start and end -- if it barely moves, the optimisation failed and the
    result says nothing about the decoder (the failure mode that would masquerade as case (b)).
  * a random-init arm on a few tracks checks we are not merely polishing the encoder's answer.
Short excerpts (--secs) keep this to minutes; the question is about precision, not duration.
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
SMC_AUD = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
SA3 = Path("/disk1/jaehoon/sa3_small_music")


def onsets(y, sr):
    return librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)


def nn_shift(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0)
    return np.array([1000.0 * (b[np.argmin(np.abs(b - t))] - t) for t in a])


def mel_loss(a, b, sr, device, flux_w=10.0):
    """Multi-resolution log-magnitude L1 PLUS an onset-envelope term.

    WHY THE FLUX TERM IS NOT OPTIONAL. Magnitude-spectrogram distance is phase-blind: the first
    version of this probe minimised it alone, and the loss fell (1.88 -> 1.31) while onset p90 got
    WORSE (48.8 -> 106.8 ms). The optimiser happily smears transients to improve average spectral
    match. Since the question is whether the decoder CAN place transients, the objective has to
    reward exactly what we measure -- so we add positive spectral flux summed over frequency, which
    is the onset-strength function librosa.onset_detect thresholds. Weighted 10x because the flux
    residual is far smaller in magnitude than the spectral one.
    """
    tot = 0.0
    for n_fft, hop in ((2048, 256), (1024, 128), (512, 64)):
        win = torch.hann_window(n_fft, device=device)
        A = torch.stft(a, n_fft, hop, window=win, return_complex=True).abs()
        B = torch.stft(b, n_fft, hop, window=win, return_complex=True).abs()
        la, lb = torch.log1p(1000 * A), torch.log1p(1000 * B)
        tot = tot + (la - lb).abs().mean()
        fa = torch.relu(la[..., 1:] - la[..., :-1]).sum(-2)     # onset envelope, per frame
        fb = torch.relu(lb[..., 1:] - lb[..., :-1]).sum(-2)
        tot = tot + flux_w * (fa - fb).abs().mean() / max(fa.abs().mean().item(), 1e-6)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tracks", type=int, default=12)
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--random-init", type=int, default=3, help="tracks to also run from noise")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/tmp/overnight/sa3_latent_opt.json")
    args = ap.parse_args()
    dev = args.device

    from stable_audio_tools.models.autoencoders import create_autoencoder_from_config
    from safetensors.torch import load_file
    cfg = json.load(open(SA3 / "model_config.json"))
    sr = cfg["sample_rate"]
    ae = create_autoencoder_from_config(
        {"model": cfg["model"]["pretransform"]["config"], "sample_rate": sr})
    sd = load_file(str(SA3 / "model.safetensors"))
    pre = {k.split("pretransform.model.", 1)[1]: v for k, v in sd.items()
           if "pretransform.model." in k}
    assert not any(ae.load_state_dict(pre, strict=False))
    ae = ae.eval().to(dev)
    for p in ae.parameters():
        p.requires_grad_(False)
    # AudioAutoencoder.decode() routes its NESTED pretransform (a wavelet stage) through a
    # no_grad branch unless enable_grad is set, which silently severs the graph from z and makes
    # loss.backward() raise "does not require grad". The decoder weights stay frozen above; this
    # only permits gradient to FLOW BACK to the latent we are optimising.
    if getattr(ae, "pretransform", None) is not None:
        ae.pretransform.enable_grad = True
        print(f"nested pretransform: {type(ae.pretransform).__name__}, enable_grad -> True")
    print(f"decoder frozen; latent {sr/4096:.2f} Hz; optimising {args.steps} steps", flush=True)

    # prefer the tracks where the round trip did WORST -- that is where the question lives
    rt = json.load(open("/tmp/overnight/sa3_roundtrip.json"))["rows"]
    order = sorted(rt, key=lambda t: -rt[t]["onset_p90_abs_ms"])[:args.n_tracks]
    N = int(args.secs * sr)

    rows = {}
    for i, tid in enumerate(order):
        y, fsr = sf.read(str(SMC_AUD / f"{tid.upper()}.wav"), always_2d=True)
        y = y.T.astype(np.float32)
        if fsr != sr:
            y = np.stack([librosa.resample(c, orig_sr=fsr, target_sr=sr) for c in y])
        if y.shape[0] == 1:
            y = np.repeat(y, 2, 0)
        y = y[:, :N]
        x = torch.from_numpy(y).unsqueeze(0).to(dev)
        mo = y.mean(0)
        on_o = onsets(mo, sr)

        with torch.no_grad():
            z0 = ae.encode(x)
            rec_enc = ae.decode(z0)[..., :N]
        d_enc = nn_shift(on_o, onsets(rec_enc[0].float().cpu().numpy().mean(0), sr))

        res = {"encoder": {"med": float(np.median(np.abs(d_enc))) if len(d_enc) else np.nan,
                           "p90": float(np.percentile(np.abs(d_enc), 90)) if len(d_enc) else np.nan}}
        for tag, init in (("opt_from_enc", z0.clone()),
                          ("opt_from_noise", torch.randn_like(z0) * z0.std())):
            if tag == "opt_from_noise" and i >= args.random_init:
                continue
            z = init.clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=args.lr)
            xm = x.mean(1)
            l0 = None
            for s in range(args.steps):
                r = ae.decode(z)[..., :N]
                loss = mel_loss(r.mean(1), xm, sr, dev)
                if s == 0:
                    l0 = float(loss)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                r = ae.decode(z)[..., :N]
            d = nn_shift(on_o, onsets(r[0].float().cpu().numpy().mean(0), sr))
            res[tag] = {"med": float(np.median(np.abs(d))) if len(d) else np.nan,
                        "p90": float(np.percentile(np.abs(d), 90)) if len(d) else np.nan,
                        "loss0": l0, "loss1": float(loss)}
        rows[tid] = res
        e, o = res["encoder"], res["opt_from_enc"]
        print(f"  {tid}  encoder med {e['med']:.1f}/p90 {e['p90']:.1f} ms  ->  "
              f"optimised med {o['med']:.1f}/p90 {o['p90']:.1f} ms   "
              f"(mel loss {o['loss0']:.3f} -> {o['loss1']:.3f})", flush=True)

    A = lambda tag, k: float(np.nanmean([r[tag][k] for r in rows.values() if tag in r]))
    print(f"\nn={len(rows)} hardest tracks, {args.secs:g}s each")
    print(f"{'arm':<16}{'onset med ms':>14}{'onset p90 ms':>14}{'mel loss end':>14}")
    for tag in ("encoder", "opt_from_enc", "opt_from_noise"):
        if any(tag in r for r in rows.values()):
            le = A(tag, "loss1") if tag != "encoder" else float("nan")
            print(f"{tag:<16}{A(tag,'med'):>14.1f}{A(tag,'p90'):>14.1f}"
                  + (f"{le:>14.3f}" if tag != "encoder" else f"{'-':>14}"))
    print("\nread: optimised p90 MUCH lower than encoder p90 -> ENCODER-side, generation unaffected.")
    print("      optimised p90 ~= encoder p90 (and mel loss did drop) -> DECODER capacity floor.")
    print("      mel loss barely moving -> optimisation failed; result says nothing.")
    json.dump(rows, open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
