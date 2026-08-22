"""GATE 1: does Stable Audio 3's SAME autoencoder preserve BEAT TIMING through encode->decode?

WHY THIS GATES EVERYTHING. Any beat-conditioned generation plan (ControlNet on SA3, or a CVAE
posterior tilt q(z|x,b) prop p(z|x)p(b|x,z)) puts the decoder DOWNSTREAM of the conditioning. If a
round trip already displaces onsets by more than the label tolerance, no conditioning scheme can
recover it, and latent audio diffusion at this compression is unusable for label-exact data
synthesis -- regardless of how good the control signal is.

THE COMPRESSION IS SEVERE AND THAT IS THE WHOLE QUESTION. downsampling_ratio=4096 at 44.1 kHz means
one 256-dim latent per 90.9 ms. A beat annotation is meaningful to +-70 ms (F-measure) and the
project's decoder work resolves to ~20 ms. Whether transients survive 4096x temporal compression is
an empirical property of the learned autoencoder, not something the ratio alone decides -- the
decoder reconstructs a full 44.1 kHz waveform and can in principle place a transient anywhere inside
a latent frame.

WHAT IS MEASURED, on the same 217 SMC tracks the whole project uses:
  1. ONSET DISPLACEMENT. librosa onsets before vs after, matched nearest-neighbour. This is the
     direct, tracker-free measurement: it asks whether the physical transients moved.
  2. ANNOTATION-ANCHORED displacement. For each ground-truth beat, the nearest onset before vs
     after the round trip. Restricting to annotated beats avoids being dominated by spurious
     onsets in reverberant/quiet passages.
  3. Spectral sanity: SNR and log-mel distance, to distinguish "timing preserved but timbre
     degraded" from "reconstruction failed outright".
Reported per-track and stratified by tempo (<55 BPM vs rest), because the slow/soft-onset material
is exactly where we would need synthesis to work and exactly where transients are weakest.

PADDING. The AE pads input to a multiple of 4096 samples, so the output is longer than the input
(measured: 176400 -> 180224). The output is trimmed to the input length before any comparison;
without that every displacement would carry a constant offset.

Audio is written nowhere and nothing is retained -- this is a measurement, not a preprocessing step.
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
GT_DIR = ROOT / "beat_this_annotations" / "smc" / "annotations" / "beats"


def load_gt(tid):
    p = GT_DIR / f"{tid}.beats"
    if not p.exists():
        p = GT_DIR / f"{tid.upper()}.beats"
    d = np.loadtxt(str(p))
    return d if d.ndim == 1 else d[:, 1]


def onsets(y, sr):
    return librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)


def nn_shift(a, b):
    """For each element of a, signed distance to nearest element of b (ms)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0)
    return np.array([1000.0 * (b[np.argmin(np.abs(b - t))] - t) for t in a])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/tmp/overnight/sa3_roundtrip.json")
    args = ap.parse_args()

    import json as _json
    from stable_audio_tools.models.autoencoders import create_autoencoder_from_config
    cfg = _json.load(open(SA3 / "model_config.json"))
    sr = cfg["sample_rate"]
    ae = create_autoencoder_from_config(
        {"model": cfg["model"]["pretransform"]["config"], "sample_rate": sr})
    from safetensors.torch import load_file
    sd = load_file(str(SA3 / "model.safetensors"))
    pre = {k.split("pretransform.model.", 1)[1]: v for k, v in sd.items()
           if "pretransform.model." in k}
    miss, unexp = ae.load_state_dict(pre, strict=False)
    assert not miss and not unexp, f"weight mismatch: {len(miss)} missing, {len(unexp)} unexpected"
    ae = ae.eval().to(args.device)
    print(f"SAME autoencoder loaded ({sum(p.numel() for p in ae.parameters())/1e6:.1f}M), "
          f"latent {sr/4096:.2f} Hz", flush=True)

    tids = sorted(p.stem.lower() for p in SMC_AUD.glob("*.wav"))
    if args.limit:
        tids = tids[:args.limit]
    rows = {}
    for n, tid in enumerate(tids):
        w = SMC_AUD / f"{tid.upper()}.wav"
        y, fsr = sf.read(str(w), always_2d=True)
        y = y.T.astype(np.float32)
        if fsr != sr:
            y = np.stack([librosa.resample(c, orig_sr=fsr, target_sr=sr) for c in y])
        if y.shape[0] == 1:
            y = np.repeat(y, 2, 0)
        N = y.shape[-1]
        x = torch.from_numpy(y).unsqueeze(0).to(args.device)
        with torch.no_grad():
            z = ae.encode(x)
            r = ae.decode(z)[..., :N]                      # TRIM the AE's 4096-multiple padding
        rec = r[0].float().cpu().numpy()
        mo, mr = y.mean(0), rec.mean(0)

        on_o, on_r = onsets(mo, sr), onsets(mr, sr)
        d_on = nn_shift(on_o, on_r)
        gt = np.asarray(load_gt(tid), float)
        # annotation-anchored: nearest onset to each true beat, before vs after
        b_o, b_r = nn_shift(gt, on_o), nn_shift(gt, on_r)
        d_beat = b_r - b_o

        noise = mo[:len(mr)] - mr[:len(mo)]
        snr = 10 * np.log10(max(np.mean(mo ** 2), 1e-12) / max(np.mean(noise ** 2), 1e-12))
        M = lambda a: librosa.power_to_db(librosa.feature.melspectrogram(y=a, sr=sr, n_mels=64))
        mel = float(np.mean(np.abs(M(mo)[:, :min(len(mo), len(mr)) // 512]
                                   - M(mr)[:, :min(len(mo), len(mr)) // 512])))
        rows[tid] = {
            "bpm": float(60.0 / np.median(np.diff(gt))) if len(gt) > 2 else float("nan"),
            "n_onset_orig": int(len(on_o)), "n_onset_rec": int(len(on_r)),
            "onset_med_abs_ms": float(np.median(np.abs(d_on))) if len(d_on) else float("nan"),
            "onset_p90_abs_ms": float(np.percentile(np.abs(d_on), 90)) if len(d_on) else float("nan"),
            "beat_anchored_med_abs_ms": float(np.median(np.abs(d_beat))) if len(d_beat) else float("nan"),
            "beat_anchored_within20": float(np.mean(np.abs(d_beat) <= 20)) if len(d_beat) else float("nan"),
            "snr_db": float(snr), "mel_l1_db": mel,
        }
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(tids)}", flush=True)

    ids = sorted(rows)
    A = lambda k, S: float(np.nanmean([rows[t][k] for t in S]))
    slow = [t for t in ids if rows[t]["bpm"] < 55]
    rest = [t for t in ids if rows[t]["bpm"] >= 55]
    print(f"\nn={len(ids)}  SAME autoencoder round trip (encode -> decode), SMC audio")
    print(f"{'subset':<14}{'n':>4}{'onset|d| med':>14}{'onset|d| p90':>14}"
          f"{'beat-anch med':>15}{'within 20ms':>13}{'SNR dB':>9}{'mel L1 dB':>11}")
    for nm, S in (("ALL", ids), ("slow <55", slow), (">=55", rest)):
        if not S:
            continue
        print(f"{nm:<14}{len(S):>4}{A('onset_med_abs_ms', S):>14.1f}{A('onset_p90_abs_ms', S):>14.1f}"
              f"{A('beat_anchored_med_abs_ms', S):>15.1f}{100*A('beat_anchored_within20', S):>12.1f}%"
              f"{A('snr_db', S):>9.1f}{A('mel_l1_db', S):>11.2f}")
    print("\nread: 'beat-anchored med' is the shift of the nearest onset to each TRUE beat.")
    print("      <~10 ms -> timing survives, conditioning plans are viable")
    print("      >~35 ms -> half the F tolerance is gone before any generation error")
    json.dump({"n": len(ids), "rows": rows}, open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
