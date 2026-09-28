"""Paired check: does SDEdit hallucinate subdivisions?

For every generated clip, score the model on (a) the generated audio and
(b) the SAME time window of the original source audio, against the SAME beat
annotations. Anything that differs is attributable to the generation, not to
tempo, not to the annotation, not to which corpus the track came from.

Confound to keep in mind: the source tracks live in training folds 1-7, so the
checkpoint has seen them and the original side is flattered. That biases the
comparison TOWARD finding the generated audio worse, so a null result here is
strong and a positive one is an upper bound.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import mir_eval
import numpy as np
import torch

BT = "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"
SD = "/disk1/jaehoon/sdedit_slow"   # default; override with --clip-dir
FPS = 50


def f1(est, ref):
    if len(est) < 2 or len(ref) < 2:
        return 0.0
    e = mir_eval.beat.trim_beats(est)
    r = mir_eval.beat.trim_beats(ref)
    if len(e) < 1 or len(r) < 1:
        return 0.0
    return mir_eval.beat.f_measure(r, e, f_measure_threshold=0.07)


THRESH = 0.70


def classify(est, ref):
    """Same taxonomy as classify_errors.py -- keep the two in step, the labels
    are compared across scripts."""
    dbl = np.sort(np.concatenate([ref, (ref[:-1] + ref[1:]) / 2]))
    sc = {
        "correct": f1(est, ref),
        "doubled": f1(est, dbl),
        "halved": f1(est, ref[::2]),
        "phase": f1(est, (ref[:-1] + ref[1:]) / 2),
    }
    if sc["correct"] >= THRESH:
        return "correct", sc
    best = max(sc, key=sc.get)
    return (best if sc[best] >= THRESH else "other"), sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clip-dir", default=SD,
                    help="corpus of generated clips to score against their sources")
    ap.add_argument("--dataset", default="sdedit",
                    help="staged dataset name; selects the spectrogram npz and the "
                         "annotation dir, which must correspond to --clip-dir")
    ap.add_argument("--data-dir", default="data",
                    help="staging root under beat_this (the r3 corpus lives in data_r3)")
    ap.add_argument("--from-wav", action="store_true",
                    help="read the generated audio and its beats straight from "
                         "--clip-dir instead of a staged npz, so an ablation "
                         "corpus can be scored without being staged first. "
                         "The source side is unchanged; use the same mode for "
                         "every corpus being compared.")
    a = ap.parse_args()

    os.chdir(BT)
    sys.path.insert(0, BT)
    from beat_this.dataset.mmnpz import MemmappedNpzFile
    from beat_this.model.pl_module import PLBeatThis
    from beat_this.preprocessing import LogMelSpect, load_audio

    dev = f"cuda:{a.gpu}"
    m = PLBeatThis(
        spect_dim=128, fps=50, transformer_dim=512, ff_mult=4, n_layers=6,
        stem_dim=32, dropout={"frontend": 0.1, "transformer": 0.2}, lr=8e-4,
        weight_decay=0.01, pos_weights={"beat": 20, "downbeat": 86},
        head_dim=32, loss_type="shift_tolerant_weighted_bce",
        warmup_steps=1000, max_epochs=100,
    )
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu",
                                 weights_only=False)["state_dict"], strict=False)
    m = m.eval().to(dev)
    mel = LogMelSpect(22050).to(dev)

    if a.from_wav:
        npz, annd = None, Path(a.clip_dir)
    else:
        root = Path(a.data_dir)
        npz = MemmappedNpzFile(root / "audio/spectrograms" / f"{a.dataset}.npz")
        annd = root / "annotations" / a.dataset / "annotations/beats"
        if not annd.is_dir():
            sys.exit(f"no annotations at {annd} -- check --data-dir/--dataset")

    # group clips by source wav so each original is decoded once
    by_src = defaultdict(list)
    for jf in sorted(Path(a.clip_dir).glob("*.json")):
        j = json.load(open(jf))
        by_src[j["source_wav"]].append((jf.stem, j))
    srcs = list(by_src.items())
    if a.limit:
        srcs = srcs[: a.limit]
    print(f"  {sum(len(v) for _,v in srcs)} clips from {len(srcs)} source files", flush=True)

    def run(spect):
        # fp32, no autocast: under fp16 autocast this was not reproducible per
        # clip (~50% of F values changed on rerun), which made the two committed
        # CSVs two different runs and left F/CMLt and the beat ratio quotable
        # only from separate files.
        x = torch.as_tensor(np.asarray(spect, dtype=np.float32))[None].to(dev)
        pad = torch.ones(1, x.shape[1], dtype=torch.bool, device=dev)
        with torch.no_grad():
            p = m.model(x)
        p = {k: v.float() for k, v in p.items()}
        pb, _ = m.postprocessor(p["beat"], p["downbeat"], pad)
        return np.asarray(pb if not isinstance(pb, tuple) else pb[0])

    fh = open(a.out, "w")
    fh.write("stem,source,bpm,F_gen,F_src,Fd_gen,Fd_src,r_gen,r_src,n_ref,"
             "C_gen,C_src,A_gen,A_src,cls_gen,cls_src,"
             "Fh_gen,Fh_src,Fp_gen,Fp_src\n")
    done = 0
    for wav, clips in srcs:
        try:
            wave, sr = load_audio(wav)
        except Exception as e:
            print(f"    skip {wav}: {e}", flush=True)
            continue
        if wave.ndim > 1:
            wave = wave.mean(1)
        import soxr
        if sr != 22050:
            wave = soxr.resample(wave, in_rate=sr, out_rate=22050)
        for stem, j in clips:
            ref = np.loadtxt(annd / f"{stem}.beats", ndmin=2)[:, 0]
            if len(ref) < 3:
                continue
            if npz is None:
                gw, gsr = load_audio(str(Path(a.clip_dir) / f"{stem}.wav"))
                if gw.ndim > 1:
                    gw = gw.mean(1)
                import soxr as _soxr
                if gsr != 22050:
                    gw = _soxr.resample(gw, in_rate=gsr, out_rate=22050)
                with torch.no_grad():
                    gen = mel(torch.as_tensor(
                        gw, dtype=torch.float32, device=dev)).cpu().numpy()
            else:
                gen = np.asarray(npz[f"{stem}/track"], dtype=np.float32)
            dur = gen.shape[0] / FPS
            s0 = int(round(j["window_start"] * 22050))
            s1 = s0 + int(round(dur * 22050))
            seg = wave[s0:s1]
            if len(seg) < 22050:
                continue
            with torch.no_grad():
                ssp = mel(torch.as_tensor(seg, dtype=torch.float32, device=dev))
            e_g, e_s = run(gen), run(ssp.cpu().numpy())
            dbl = np.sort(np.concatenate([ref, (ref[:-1] + ref[1:]) / 2]))

            def cont(est):
                """CMLt/AMLt; returns zeros when mir_eval cannot score the pair."""
                try:
                    c = mir_eval.beat.continuity(
                        mir_eval.beat.trim_beats(ref), mir_eval.beat.trim_beats(est))
                    return c[1], c[3]
                except Exception:
                    return 0.0, 0.0

            cg, ag = cont(e_g)
            cs, as_ = cont(e_s)

            def ratio(est):
                """Predicted beats per annotated beat, counted INSIDE the
                annotated span only. len(est)/len(ref) on untrimmed estimates
                reads ~1.51 even on real held-out audio, because predictions
                before the first and after the last annotation inflate it on a
                20 s window; that made the raw ratio useless as a damage measure."""
                lo, hi = ref[0], ref[-1]
                inside = np.asarray(est)[(np.asarray(est) >= lo) & (np.asarray(est) <= hi)]
                return len(inside) / max(len(ref), 1)

            kg, scg = classify(e_g, ref)
            ks, scs = classify(e_s, ref)
            fh.write(f"{stem},{j['source']},{60.0/np.median(np.diff(ref)):.3f},"
                     f"{f1(e_g,ref):.6f},{f1(e_s,ref):.6f},"
                     f"{f1(e_g,dbl):.6f},{f1(e_s,dbl):.6f},"
                     f"{ratio(e_g):.4f},{ratio(e_s):.4f},{len(ref)},"
                     f"{cg:.6f},{cs:.6f},{ag:.6f},{as_:.6f},{kg},{ks},"
                     f"{scg['halved']:.6f},{scs['halved']:.6f},"
                     f"{scg['phase']:.6f},{scs['phase']:.6f}\n")
            done += 1
        fh.flush()
        if done % 100 < len(clips):
            print(f"    {done} clips", flush=True)
    fh.close()
    print(f"  wrote {a.out}  ({done} paired clips)")


if __name__ == "__main__":
    main()
