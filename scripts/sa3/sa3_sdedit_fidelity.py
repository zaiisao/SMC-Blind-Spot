"""Does SDEdit preserve beat TIMING on real annotated audio? Go/no-go for augmentation.

The proposal is to augment Beat This training by SDEditing real annotated clips
with their true grid as ControlNet conditioning: new timbre, same timing, labels
inherited from human annotation rather than asserted. That only works if the
audio does not move. F's tolerance is 70 ms, so a systematic drift of even 30-50 ms
would silently corrupt every augmented label.

Three measurements per clip, at each noise level:

  lag_ms      cross-correlation lag between source and output. This is the
              direct test: it must be near zero.
  salience    onset energy at the TRUE beats vs random times. Must not collapse;
              on the one style-transfer case measured it actually rose (2.63 -> 3.42).
  tracker     F/CMLt of the released tracker against the true grid, before vs
              after. A drop is not automatically bad -- harder is the goal -- but
              it must be interpretable alongside lag and salience.
"""
import argparse, os, sys, glob
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

ANN = "/disk1/jaehoon/dataset_store/beat_this_annotations"


def lag_ms(a, b, sr, max_ms=250):
    """Signed lag of b relative to a, by cross-correlation of onset envelopes."""
    import librosa
    ea = librosa.onset.onset_strength(y=a, sr=sr, hop_length=128)
    eb = librosa.onset.onset_strength(y=b, sr=sr, hop_length=128)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()
    c = np.correlate(eb, ea, mode="full")
    lags = np.arange(-n + 1, n) * 128 / sr * 1000
    m = np.abs(lags) <= max_ms
    return float(lags[m][np.argmax(c[m])])


def salience(y, sr, beats):
    import librosa
    hop = 128
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env = np.maximum(env - np.median(env), 0.0)
    te = np.arange(len(env)) * hop / sr
    ibi = float(np.median(np.diff(beats)))
    k = max(1, int(round(0.10 * ibi / (te[1] - te[0]))))
    def E(t):
        i = np.clip(np.searchsorted(te, t), 0, len(env) - 1)
        return float(np.mean([env[max(0, j - k): j + k + 1].max() for j in i]))
    rng = np.random.RandomState(0)
    return E(beats) / (E(rng.uniform(beats[0], beats[-1], len(beats))) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["smc", "asap"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0],
                    help="beat-pathway guidance values. g>1 extrapolates away "
                         "from the METRONOMIC TWIN of the clip's true grid, "
                         "amplifying exactly the rubato component -- measured "
                         "+0.131 CMLt (p=0.014) on from-scratch rubato "
                         "generation. Question here: does it extend the "
                         "no-drift noise regime for SDEdit?")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--adapter", default="/disk1/jaehoon/sa3_beat_runs/cn_fold0_rw/"
                                         "beat_adapter_final.safetensors")
    ap.add_argument("--no-adapter", action="store_true",
                    help="skip loading trained adapter weights. The ControlNet is "
                         "zero-init, so this is vanilla SA3 SDEdit -- the control for "
                         "'does the beat conditioning contribute anything at all here?'")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--cfg-scale", type=float, default=6.0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dump-dir", default=None)
    a = ap.parse_args()

    import torch, soxr
    import mir_eval.beat as mb
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    from sa3_beat.build import load_beat_model
    from sa3_beat.generate import load_adapter, beat_conditioning, _patch_conditioning_inputs
    from beat_this.inference import load_model, split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect

    dev = f"cuda:{a.gpu}"
    model, cfg = load_beat_model(device=dev)
    if a.no_adapter:
        print("[control] NO adapter loaded -- zero-init ControlNet, vanilla SA3", flush=True)
    else:
        load_adapter(model, a.adapter, dev)
    _patch_conditioning_inputs(model)
    sr_gen = cfg["sample_rate"]
    ds = model.pretransform.downsampling_ratio
    io_ch = cfg["model"]["io_channels"]
    from sa3_beat.build import BEAT_COND_ID
    cn = getattr(model.model.model, "beat_controlnet", None)
    if any(g != 1.0 for g in a.guidance):
        assert cn is not None, "guidance requested but no beat_controlnet found"

    def metronomic_twin(beats):
        b = np.asarray(beats, dtype=np.float64)
        if len(b) < 3:
            return b
        ibi = float(np.median(np.diff(b)))
        return b[0] + np.arange(len(b)) * ibi

    net = load_model("final0", device=dev).eval()
    spect = LogMelSpect(device=dev)
    post = Postprocessor(type="minimal", fps=50)

    def track(y22):
        with torch.inference_mode():
            pr = split_predict_aggregate(spect(torch.from_numpy(y22).to(dev)),
                                         1500, 6, "keep_first", net)
            est, _ = post(pr["beat"].float(), pr["downbeat"].float())
        return np.asarray(est)

    # Real annotated clips. The cache carries stems like `smc__smc_001` with a
    # matching .beats; audio lives in the flat per-corpus store.
    AUDIO = "/disk1/jaehoon/dataset_store/audio_by_stem"
    items = []
    for dsname in a.datasets:
        for bt in sorted(glob.glob(f"/disk1/jaehoon/sa3_beat_cache/{dsname}__*.beats")):
            stem = os.path.basename(bt)[:-6]
            name = stem.split("__", 1)[1]
            for ext in (".wav", ".flac", ".mp3", ".ogg"):
                w = os.path.join(AUDIO, dsname, name + ext)
                if os.path.exists(w):
                    items.append((stem, w, bt))
                    break
    rng = np.random.RandomState(0)
    items = [items[i] for i in rng.choice(len(items), min(a.n, len(items)), replace=False)]

    if a.dump_dir:
        os.makedirs(a.dump_dir, exist_ok=True)

    rows = {(nl, g): [] for nl in a.noise for g in a.guidance}
    base_rows = []
    for stem, wav, bt in items:
        try:
            y, sr = sf.read(wav, dtype="float32", always_2d=True)
        except Exception:
            continue
        y = y.mean(axis=1)
        yg = soxr.resample(y, sr, sr_gen) if sr != sr_gen else y
        beats = np.loadtxt(bt, ndmin=2)[:, 0]
        if len(beats) < 12:
            continue
        # a window that starts on a beat and holds `seconds`
        start = float(beats[len(beats) // 4])
        seg = yg[int(start * sr_gen): int((start + a.seconds) * sr_gen)]
        n = (len(seg) // ds) * ds
        if n < ds * 8:
            continue
        seg = seg[:n]
        b = beats[(beats >= start) & (beats < start + n / sr_gen)] - start
        if len(b) < 8:
            continue

        y22 = soxr.resample(seg, sr_gen, 22050)
        est0 = track(y22)
        rt = mb.trim_beats(b)
        if len(est0) >= 4:
            e0 = mb.trim_beats(est0)
            base_rows.append((mb.f_measure(rt, e0), mb.continuity(rt, e0)[1],
                              salience(seg, sr_gen, b)))

        init = torch.from_numpy(np.stack([seg, seg])).to(dev)
        n_lat = n // ds
        item = beat_conditioning(b, None, n_lat, a.prompt, n / sr_gen)
        cond = model.conditioner([item], dev)
        cond["inpaint_mask"] = [torch.zeros(1, 1, n_lat, device=dev), None]
        cond["inpaint_masked_input"] = [torch.zeros(1, io_ch, n_lat, device=dev), None]
        # twin conditioning for guidance: same layout fix as the sweep -- the
        # conditioner emits (B, beat_dim, T); cond_in wants (B, T, beat_dim)
        tw = None
        if any(g != 1.0 for g in a.guidance):
            twin_item = beat_conditioning(metronomic_twin(b), None, n_lat,
                                          a.prompt, n / sr_gen)
            tw = model.conditioner([twin_item], dev)[BEAT_COND_ID][0]
            if tw.shape[1] < tw.shape[2]:
                tw = tw.transpose(1, 2)

        for nl in a.noise:
            for g in a.guidance:
                if cn is not None:
                    cn.guidance = float(g)
                    cn.twin_beat = None if g == 1.0 else tw
                out = generate_diffusion_cond(
                    model, steps=a.steps, cfg_scale=a.cfg_scale,
                    conditioning_tensors=cond, sample_size=n, seed=0, device=dev,
                    init_audio=(sr_gen, init), init_noise_level=nl)
                o = out.squeeze(0).float().cpu().numpy()
                o = o.mean(axis=0) if o.ndim > 1 else o
                L = lag_ms(seg, o[:len(seg)], sr_gen)
                S = salience(o, sr_gen, b)
                est = track(soxr.resample(o, sr_gen, 22050))
                if len(est) >= 4:
                    e = mb.trim_beats(est)
                    F, C = mb.f_measure(rt, e), mb.continuity(rt, e)[1]
                else:
                    F = C = 0.0
                rows[(nl, g)].append((L, S, F, C))
                if a.dump_dir:
                    sf.write(os.path.join(a.dump_dir,
                                          f"{stem}__n{nl:.1f}_g{g:.1f}.wav"),
                             o, sr_gen, format="WAV")
        if cn is not None:
            cn.guidance, cn.twin_beat = 1.0, None
        print(f"  {stem[:34]:<34} done", flush=True)

    b = np.array(base_rows)
    print(f"\nORIGINAL real audio  n={len(b)}   F={b[:,0].mean():.4f}  "
          f"CMLt={b[:,1].mean():.4f}  salience={np.median(b[:,2]):.3f}")
    print(f"\n{'noise':>6}{'guid':>6}{'median lag':>12}{'|lag|>20ms':>12}"
          f"{'salience':>10}{'F':>8}{'CMLt':>8}")
    for nl in a.noise:
        for g in a.guidance:
            r = np.array(rows[(nl, g)])
            if not len(r):
                continue
            print(f"{nl:6.2f}{g:6.1f}{np.median(r[:,0]):10.1f}ms"
                  f"{100*np.mean(np.abs(r[:,0])>20):11.0f}%"
                  f"{np.median(r[:,1]):10.3f}{r[:,2].mean():8.4f}{r[:,3].mean():8.4f}")
    import json
    json.dump({f"{nl}_{g}": rows[(nl, g)] for nl in a.noise for g in a.guidance},
              open("/home/sogang/jaehoon/Analyze-SMC/cache/sdedit_guidance_fidelity.json", "w"))


if __name__ == "__main__":
    main()
