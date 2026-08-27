"""Beat-pathway guidance sweep: does it rescue rubato adherence?

Prof. Jung's tutorial (§4.3) derives classifier-free guidance on the beat pathway:
extrapolate away from a null-beat reference so the guidance direction isolates the
beat condition's contribution. Our ControlNet has the analogous mechanism already
written -- `controlnet.py:107`, extrapolating away from a METRONOMIC twin of the
requested grid rather than a null:

    residual = res(twin) + guidance * (res(true) - res(twin))

...and it has never run. `twin_beat` has no setter anywhere in the codebase, so
`if self.twin_beat is not None and self.guidance != 1.0` has never fired: every
clip this project ever generated used guidance=1.0.

That matters because the measured failure is specifically rubato. Same prompts,
same tempo, only the grid's timing structure differing:
    hainsworth grids (steady)  CMLt 0.529      asap grids (rubato)  CMLt 0.273
    rwc grids                  CMLt 0.504      smc grids            CMLt 0.176
The model collapses irregular grids toward a steady pulse. Guidance amplifies
exactly the true-vs-metronomic difference, so it is the one knob aimed at that
collapse -- and both endpoints are in-distribution, so it is not asking the model
for conditioning it never saw.

Reports CMLt/F of the released tracker against the REQUESTED grid, per guidance
value, on rubato source grids. If adherence at slow/rubato settings improves, the
generator direction reopens; if it is flat, the collapse is in the prior, not in
how strongly the condition is applied.
"""
import argparse, glob, json, os, sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = "/disk1/jaehoon/sa3_beat_cache"


def metronomic_twin(beats):
    """Constant-tempo grid with the same start and the same median IBI."""
    b = np.asarray(beats, dtype=np.float64)
    if len(b) < 3:
        return b
    ibi = float(np.median(np.diff(b)))
    return b[0] + np.arange(len(b)) * ibi


def ibi_cv(beats):
    d = np.diff(np.asarray(beats, dtype=np.float64))
    return float(np.std(d) / (np.mean(d) + 1e-12)) if len(d) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/disk1/jaehoon/sa3_beat_runs/cn_fold0_rw/"
                                         "beat_adapter_final.safetensors")
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--datasets", nargs="+", default=["smc", "asap"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--cfg-scale", type=float, default=6.0)
    ap.add_argument("--prompt", default="solo piano, expressive, clear note attacks")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", default="/disk1/jaehoon/sa3_guidance")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    import torch, soxr
    import mir_eval.beat as mb
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    from sa3_beat.build import load_beat_model, BEAT_COND_ID
    from sa3_beat.generate import load_adapter, beat_conditioning, _patch_conditioning_inputs
    from beat_this.inference import load_model, split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect

    dev = f"cuda:{a.gpu}"
    model, cfg = load_beat_model(device=dev)
    load_adapter(model, a.adapter, dev)
    _patch_conditioning_inputs(model)
    sr_gen = cfg["sample_rate"]
    ds = model.pretransform.downsampling_ratio
    io_ch = cfg["model"]["io_channels"]
    cn = getattr(model.model.model, "beat_controlnet", None)
    assert cn is not None, "no beat_controlnet on the DiT"

    net = load_model("final0", device=dev).eval()
    spect = LogMelSpect(device=dev)
    post = Postprocessor(type="minimal", fps=50)

    # rubato source grids, most irregular first
    grids = []
    for dsn in a.datasets:
        for bt in sorted(glob.glob(f"{CACHE}/{dsn}__*.beats")):
            t = np.loadtxt(bt, ndmin=2)
            if len(t) < 20:
                continue
            b = t[:, 0]
            w = b[(b >= b[2]) & (b < b[2] + a.seconds)] - b[2]
            if len(w) < 10:
                continue
            cv = ibi_cv(w)
            # Sorting by -cv picks PATHOLOGY, not rubato: the extreme tail is
            # grids with long pauses (measured cv 1.18, i.e. std > mean). SMC's
            # actual rubato drift is ~0.09, so band-limit to real expressive
            # timing and take the most rubato-heavy grids inside that band.
            if 0.04 <= cv <= 0.35:
                grids.append((os.path.basename(bt)[:-6], w, cv))
    grids.sort(key=lambda x: -x[2])
    grids = grids[: a.n]
    print(f"[grids] {len(grids)} rubato grids, IBI cv "
          f"{np.median([g[2] for g in grids]):.3f} median", flush=True)

    rows = {g: [] for g in a.guidance}
    for stem, beats, cv in grids:
        n = (int(a.seconds * sr_gen) // ds) * ds
        n_lat = n // ds
        twin = metronomic_twin(beats)
        item = beat_conditioning(beats, None, n_lat, a.prompt, n / sr_gen)
        twin_item = beat_conditioning(twin, None, n_lat, a.prompt, n / sr_gen)
        cond = model.conditioner([item], dev)
        twin_cond = model.conditioner([twin_item], dev)
        cond["inpaint_mask"] = [torch.zeros(1, 1, n_lat, device=dev), None]
        cond["inpaint_masked_input"] = [torch.zeros(1, io_ch, n_lat, device=dev), None]

        for g in a.guidance:
            cn.guidance = float(g)
            # the conditioner emits (B, beat_dim, T); the control branch's cond_in
            # is a Linear over beat_dim, so it needs (B, T, beat_dim) -- the same
            # layout modular_local_cond carries on the live path.
            tw = twin_cond[BEAT_COND_ID][0]
            if tw.shape[1] < tw.shape[2]:
                tw = tw.transpose(1, 2)
            cn.twin_beat = None if g == 1.0 else tw
            out = generate_diffusion_cond(
                model, steps=a.steps, cfg_scale=a.cfg_scale, conditioning_tensors=cond,
                sample_size=n, seed=0, device=dev)
            o = out.squeeze(0).float().cpu().numpy()
            o = o.mean(axis=0) if o.ndim > 1 else o
            o = o / (np.abs(o).max() + 1e-9) * 0.9
            y22 = soxr.resample(o, sr_gen, 22050)
            with torch.inference_mode():
                pr = split_predict_aggregate(
                    spect(torch.from_numpy(np.ascontiguousarray(y22)).to(dev)),
                    1500, 6, "keep_first", net)
                est, _ = post(pr["beat"].float(), pr["downbeat"].float())
            est = np.asarray(est)
            rt = mb.trim_beats(beats)
            if len(est) >= 4:
                et = mb.trim_beats(est)
                c = mb.continuity(rt, et)
                rows[g].append((mb.f_measure(rt, et), c[1], c[3], ibi_cv(est)))
            else:
                rows[g].append((0.0, 0.0, 0.0, 0.0))
            sf.write(os.path.join(a.out_dir, f"{stem}__g{g:.1f}.wav"), o, sr_gen,
                     format="WAV")
        cn.guidance, cn.twin_beat = 1.0, None
        print(f"  {stem[:38]:<38} done", flush=True)

    print(f"\nn={len(grids)} rubato grids   requested-grid IBI cv "
          f"{np.median([g[2] for g in grids]):.3f}\n")
    print(f"{'guidance':>9}{'F':>9}{'CMLt':>9}{'AMLt':>9}{'est IBI cv':>12}")
    for g in a.guidance:
        r = np.array(rows[g])
        print(f"{g:9.1f}{r[:,0].mean():9.4f}{r[:,1].mean():9.4f}{r[:,2].mean():9.4f}"
              f"{np.median(r[:,3]):12.3f}")
    base = np.array(rows[a.guidance[0]])
    from scipy import stats
    for g in a.guidance[1:]:
        r = np.array(rows[g])
        d = r[:, 1] - base[:, 1]
        print(f"  CMLt vs g={a.guidance[0]}: g={g}: {d.mean():+.4f} "
              f"p={stats.ttest_rel(r[:,1], base[:,1]).pvalue:.3f} "
              f"better {int((d>0).sum())}/{len(d)}")
    json.dump({str(g): rows[g] for g in a.guidance},
              open(os.path.join(a.out_dir, "sweep.json"), "w"))


if __name__ == "__main__":
    main()
