"""Score a checkpoint on GTZAN's SLOW tracks -- a properly-powered slow readout.

SMC fold 0 has only 9 songs under 55 bpm, so its slow band carries noise of about
+/-0.10; vanilla's own slow trace moved 0.406 -> 0.353 -> 0.424 between consecutive
checkpoints of the SAME run. Any slow-targeted intervention screened on 9 songs is
unmeasurable.

GTZAN is test-only for Beat This and appears in no arm's training set, so its slow
tracks are a clean, larger sample: 16 under 55 bpm, 61 in 55-70, 119 in 70-90.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np

BT = Path("/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
sys.path.insert(0, str(BT)); sys.path.insert(0, "/home/sogang/jaehoon/VBPM/rungs_lab")
sys.path.insert(0, str(Path(__file__).parent))
ANN = Path("/disk1/jaehoon/dataset_store/beat_this_annotations/gtzan/annotations/beats")
AUD = Path("/disk1/jaehoon/dataset_store/audio_by_stem/gtzan")
SR, FPS = 22050, 50


def slow_songs(max_bpm):
    out = []
    for bp in sorted(ANN.glob("*.beats")):
        w = AUD / f"{bp.stem}.wav"
        if not w.exists():
            continue
        try:
            a = np.loadtxt(bp, ndmin=2)
        except Exception:
            continue
        if a.size == 0 or len(a) < 8:
            continue
        t = a[:, 0]
        if 60.0 / np.median(np.diff(t)) < max_bpm:
            out.append((bp.stem, w, t))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-bpm", type=float, default=70.0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch, soundfile as sf, soxr
    from smc_eval import load_net
    from beat_this.preprocessing import LogMelSpect
    from beat_this.inference import split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from training.evaluation import beat_f_measure

    dev = f"cuda:{a.gpu}"
    net = load_net(a.ckpt, dev); net.eval()
    spect = LogMelSpect(device=dev); post = Postprocessor(type="minimal", fps=FPS)
    songs = slow_songs(a.max_bpm)
    rows = []
    with torch.inference_mode():
        for stem, wav, truth in songs:
            y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != SR:
                y = soxr.resample(y, sr, SR)
            S = spect(torch.from_numpy(y).to(dev))
            pr = split_predict_aggregate(S, 1500, 6, "keep_first", net)
            bt, _ = post(pr["beat"].float(), pr["downbeat"].float())
            est = np.asarray(bt)
            lvl = (float(np.median(np.diff(truth)) / np.median(np.diff(est)))
                   if len(est) >= 8 else float("nan"))
            rows.append(dict(track=stem, bpm=60.0 / np.median(np.diff(truth)),
                             F=float(beat_f_measure(truth, est)),
                             level_ok=int(np.isfinite(lvl) and abs(np.log2(lvl)) < 0.18),
                             doubled=int(np.isfinite(lvl) and abs(np.log2(lvl / 2)) < 0.18)))
    F = np.array([r["F"] for r in rows])
    print(f"[gtzan<{a.max_bpm:.0f}bpm] {Path(a.ckpt).parent.name:12s} n={len(F)}  "
          f"F={F.mean():.4f} +/-{F.std(ddof=1)/np.sqrt(len(F)):.4f}  "
          f"level_ok={100*np.mean([r['level_ok'] for r in rows]):.1f}%  "
          f"doubled={100*np.mean([r['doubled'] for r in rows]):.1f}%", flush=True)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
