"""Step 2: can the encoder's beat seam steer timing at all? Overfit ONE track.

The smallest experiment that can fail informatively.  If a beat-conditioned
encoder cannot improve beat preservation on a single track it is allowed to
memorise, it will not do so on 700, and we learn that in minutes.

What is trained: only the 0.199M-parameter beat projection.  The encoder
backbone, the bottleneck and the decoder are all frozen.  Freezing the decoder
is not incidental -- SA3's bottleneck is a frozen softnorm with no KL term, so
nothing else keeps latents on the manifold the DiT was trained on.  With the
decoder fixed, the only way to lower the loss is to emit latents this decoder
already understands.

The decoder is also never shown the beat signal (HANDOVER §3.2).  If it could
see the beats, the encoder would be free to leave timing out of ``z`` and let
the decoder reconstruct it from the side channel.  Keeping it blind is what
forces ``z`` to carry the timing -- which is the point, since ``z`` is all the
DiT ever produces.

READ THE RESULT NARROWLY.  A single-track overfit cannot distinguish steering
from stamping: the cheapest way to lower an onset loss on one track is to
memorise "emit a transient at frame 37, 58, 79" and paint the beats in.  The
adherence verifier would reward that and it would mean nothing.  So this tests
capacity and plumbing only.  Two guards are reported: adherence ABOVE the
real-audio ceiling is a red flag rather than a win, and the decoded audio is
written out so beat-stamping can be heard.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SA3_DIR = Path("/disk1/jaehoon/sa3_small_music")
ASAP_AUDIO = Path("/disk1/jaehoon/dataset_store/audio_by_stem/asap")

sys.path.insert(0, str(ROOT))


def asap_track(name):
    """Audio path and its sibling ``*_annotations.txt`` in the ASAP repo."""
    wav = ASAP_AUDIO / f"{name}.wav"
    real = wav.resolve()
    ann = real.parent / f"{real.stem}_annotations.txt"
    if not ann.exists():
        raise FileNotFoundError(ann)
    return real, ann


def stft_mag(x, n_fft, hop):
    w = torch.hann_window(n_fft, device=x.device)
    s = torch.stft(x.reshape(-1, x.shape[-1]), n_fft=n_fft, hop_length=hop,
                   window=w, return_complex=True)
    return s.abs().clamp_min(1e-7)


def onset_envelope(mag):
    """Differentiable spectral flux: half-wave rectified magnitude difference."""
    d = mag[..., 1:] - mag[..., :-1]
    return d.clamp_min(0).sum(dim=-2)


def losses(recon, target):
    """Multi-resolution log-magnitude reconstruction + an explicit onset term.

    The onset term is not optional dressing.  Attack energy is a tiny fraction
    of total energy, so a plain spectral loss barely notices a smeared
    transient (HANDOVER §2.4) -- without this there is no gradient pressure
    toward the thing being fixed.
    """
    rec, ons = 0.0, 0.0
    for n_fft, hop in ((2048, 512), (1024, 256), (512, 128)):
        m_r, m_t = stft_mag(recon, n_fft, hop), stft_mag(target, n_fft, hop)
        rec = rec + (m_r.log() - m_t.log()).abs().mean()
        e_r, e_t = onset_envelope(m_r), onset_envelope(m_t)
        scale = e_t.abs().mean().clamp_min(1e-6)
        ons = ons + ((e_r - e_t).abs().mean() / scale)
    return rec / 3, ons / 3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", default="Bach_Fugue_bwv_846_Shi05M")
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--onset-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "cache" / "sa3_encoder_overfit"))
    args = ap.parse_args()

    import librosa
    import soundfile as sf
    from safetensors.torch import load_file
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict

    from sa3_beat.conditioners import BeatConditioner
    from sa3_beat.data import load_beats
    from sa3_beat.encoder import BeatConditionedEncoder, latent_frames
    sys.path.insert(0, str(ROOT / "scripts" / "sa3"))
    from sa3_beat_adherence import beat_adherence

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    config = json.loads((SA3_DIR / "model_config.json").read_text())
    model = create_model_from_config(config)
    copy_state_dict(model, load_file(str(SA3_DIR / "model.safetensors")))
    ae = model.pretransform.model.to(device).eval().requires_grad_(False)
    # The inner "patched" pretransform is a parameterless reshape, but it ships
    # with enable_grad=False, which runs the final decode under no_grad and
    # silently severs the graph. Nothing is trained by flipping this.
    ae.pretransform.enable_grad = True
    sr = config["sample_rate"]
    ds = model.pretransform.downsampling_ratio

    wav_path, ann_path = asap_track(args.track)
    beats, downbeats = load_beats(ann_path)
    n_samples = (int(args.seconds * sr) // ds) * ds
    audio, _ = librosa.load(str(wav_path), sr=sr, mono=False, duration=args.seconds)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    audio = torch.tensor(audio[:, :n_samples], dtype=torch.float32, device=device)[None]

    wrapped = BeatConditionedEncoder(ae.encoder, include_downbeat=True).to(device)
    wrapped.freeze_backbone()
    block = wrapped.encoder.layers[0]

    with torch.no_grad():
        patched = ae.pretransform.encode(audio)
    n_frames = latent_frames(patched.shape[-1], block)
    cond = BeatConditioner(include_downbeat=True)
    feat = cond.encode_one(beats, downbeats, n_frames)
    beat = torch.tensor(feat, dtype=torch.float32, device=device)[None]

    in_span = beats[(beats >= 0) & (beats <= args.seconds)]
    print(f"[setup] {args.track}  {args.seconds}s  latent_frames={n_frames}")
    if len(in_span) < 16:
        print(f"[warn] only {len(in_span)} beats in window -- adherence "
              f"statistics are unreliable below ~16; use a longer --seconds")
    print(f"[setup] beats in window: {len(in_span)}  downbeats: "
          f"{0 if downbeats is None else int(((downbeats>=0)&(downbeats<=args.seconds)).sum())}")

    def run(beat_signal):
        z = wrapped(patched, beat=beat_signal)
        z, _ = ae.bottleneck.encode(z, return_info=True)
        # The block pads its input up to a multiple of chunk_size, so the
        # decode is a whole latent frame longer than the input; crop back.
        return ae.decode(z)[..., :n_samples], z

    def score(wav):
        y = wav[0].detach().float().cpu().numpy().mean(axis=0)
        r = beat_adherence(y, in_span)
        return r["phase_align"], r["beat_energy_z"]

    real_np = audio[0].cpu().numpy().mean(axis=0)
    real_r = beat_adherence(real_np, in_span)
    with torch.no_grad():
        base_wav, base_z = run(None)
    base_pa, base_bz = score(base_wav)
    z_ref_mean, z_ref_std = base_z.mean().item(), base_z.std().item()
    print(f"[ceiling ] real          phase_align={real_r['phase_align']:+.4f} "
          f"beat_energy_z={real_r['beat_energy_z']:+.3f}")
    print(f"[baseline] unconditioned phase_align={base_pa:+.4f} "
          f"beat_energy_z={base_bz:+.3f}")

    opt = torch.optim.AdamW(wrapped.beat_parameters(), lr=args.lr)
    hist = []
    for step in range(1, args.steps + 1):
        recon, z = run(beat)
        rec_l, ons_l = losses(recon, audio)
        loss = rec_l + args.onset_weight * ons_l
        opt.zero_grad(); loss.backward()
        # The seam adds to `new_tokens`, whose learned std is only ~1.4e-2, so
        # an unclipped step easily swamps the queries it is meant to nudge.
        torch.nn.utils.clip_grad_norm_(list(wrapped.beat_parameters()), args.clip)
        opt.step()

        if step % args.eval_every == 0 or step == 1:
            with torch.no_grad():
                ev_wav, ev_z = run(beat)
            pa, bz = score(ev_wav)
            # Frozen softnorm stats mean encoder drift is silent in the loss.
            drift = abs(ev_z.mean().item() - z_ref_mean), abs(ev_z.std().item() - z_ref_std)
            hist.append(dict(step=step, rec=rec_l.item(), onset=ons_l.item(),
                             phase_align=pa, beat_energy_z=bz,
                             z_mean=ev_z.mean().item(), z_std=ev_z.std().item()))
            tok = wrapped.beat_proj(beat)
            print(f"  step {step:4d}  |tok|={tok.abs().max().item():.2e} "
                  f"rec={rec_l.item():.4f} onset={ons_l.item():.4f} "
                  f"| phase_align={pa:+.4f} bz={bz:+.3f} "
                  f"| drift mean={drift[0]:.4f} std={drift[1]:.4f}")

    with torch.no_grad():
        fin_wav, _ = run(beat)
    fin_pa, fin_bz = score(fin_wav)

    for tag, w in (("real", audio), ("uncond", base_wav), ("cond", fin_wav)):
        sf.write(str(out / f"{args.track}_{tag}.wav"),
                 w[0].detach().float().cpu().numpy().T, sr)

    ceiling = real_r["phase_align"]
    print(f"\n  real (ceiling)  phase_align={ceiling:+.4f}")
    print(f"  unconditioned   phase_align={base_pa:+.4f}")
    print(f"  conditioned     phase_align={fin_pa:+.4f}   "
          f"delta_vs_uncond={fin_pa-base_pa:+.4f}")
    if fin_pa > ceiling:
        print("  ** ABOVE the real-audio ceiling -- suspect beat stamping, "
              "not timing recovery. Listen to the wav. **")

    json.dump(dict(track=args.track, args=vars(args),
                   real=real_r, baseline=dict(phase_align=base_pa, beat_energy_z=base_bz),
                   final=dict(phase_align=fin_pa, beat_energy_z=fin_bz),
                   history=hist), open(out / f"{args.track}.json", "w"), indent=1)
    print(f"[out] {out}/{args.track}.json")


if __name__ == "__main__":
    main()
