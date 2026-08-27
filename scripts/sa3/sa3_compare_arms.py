"""Per-track comparison of the fine-tuned arms against baseline on held-out SMC.

The headline delta can be flat while a real mechanism hides in a stratum, so this
scores every track under every model and breaks the result down by SMC's own
difficulty tags. Pre-registered prediction: if beat-conditioned synthetic audio
works, its gains land on the expressive-timing / slow-tempo / lack-of-transient
tracks, not scattered uniformly.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np

BT = Path("/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
sys.path.insert(0, str(BT)); sys.path.insert(0, "/home/sogang/jaehoon/VBPM/rungs_lab")
ANN = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
SMC_AUDIO = Path("/home/sogang/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Audio")


def songs(fold):
    fo = {}
    for line in open(ANN / "smc" / "8-folds.split"):
        p = line.split()
        if len(p) == 2:
            fo[p[0]] = int(p[1])
    out = []
    for bp in sorted((ANN / "smc" / "annotations" / "beats").glob("*.beats")):
        if fo.get(bp.stem) != fold:
            continue
        w = SMC_AUDIO / f"{bp.stem.replace('smc_', 'SMC_')}.wav"
        if w.exists():
            out.append((bp.stem, w, np.loadtxt(bp, ndmin=2)[:, 0]))
    return out


def score(model, songs, dev):
    import torch, soundfile as sf, soxr
    from beat_this.preprocessing import LogMelSpect
    from beat_this.model.postprocessor import Postprocessor
    from training.evaluation import beat_f_measure
    spect = LogMelSpect(device=dev); post = Postprocessor(type="minimal", fps=50)
    model.eval(); out = {}
    with torch.no_grad():
        for stem, wav, truth in songs:
            y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != 22050:
                y = soxr.resample(y, sr, 22050)
            pr = model(spect(torch.from_numpy(y).to(dev)).unsqueeze(0))
            bt, _ = post(pr["beat"][0].float(), pr["downbeat"][0].float())
            out[stem] = beat_f_measure(truth, np.asarray(bt))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--synth", default="/disk1/jaehoon/bt_ft_runs/synth/epoch=02.ckpt")
    ap.add_argument("--control", default="/disk1/jaehoon/bt_ft_runs/control/epoch=02.ckpt")
    ap.add_argument("--out", default="cache/arm_comparison.csv")
    a = ap.parse_args()

    import torch
    from beat_this.inference import load_model
    from beat_this.model.pl_module import PLBeatThis
    dev = f"cuda:{a.gpu}"
    S = songs(a.fold)
    print(f"[eval] {len(S)} held-out fold-{a.fold} SMC songs", flush=True)

    res = {"baseline": score(load_model(f"fold{a.fold}", device=dev), S, dev)}
    for name, ck in (("synth", a.synth), ("control", a.control)):
        # weights_only=False: these are our own checkpoints from this session, and
        # Lightning stores a PosixPath in hyper_parameters which the torch>=2.6
        # safe unpickler rejects.
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        hp = {k: v for k, v in blob["hyper_parameters"].items()
              if k in ("spect_dim", "fps", "transformer_dim", "ff_mult", "n_layers",
                       "stem_dim", "dropout", "lr", "weight_decay", "pos_weights",
                       "head_dim", "loss_type", "warmup_steps", "max_epochs",
                       "use_dbn", "eval_trim_beats", "sum_head", "partial_transformers")}
        m = PLBeatThis(**hp)
        miss, unexp = m.load_state_dict(blob["state_dict"], strict=False)
        assert not miss and not unexp, f"{name}: {len(miss)} missing / {len(unexp)} unexpected"
        m = m.to(dev)
        res[name] = score(m.model, S, dev)
        del m, blob; torch.cuda.empty_cache()

    stems = [s for s, _, _ in S]
    b = np.array([res["baseline"][s] for s in stems])
    sy = np.array([res["synth"][s] for s in stems])
    co = np.array([res["control"][s] for s in stems])

    def rep(name, x, y):
        d = x - y
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se else 0.0
        print(f"  {name:26s} delta={d.mean():+.4f}  sd={d.std(ddof=1):.4f}  "
              f"se={se:.4f}  t={t:+.2f}  {int((d>0).sum())}/{len(d)} improved")
    print(f"\nmeans: baseline {b.mean():.4f}  synth {sy.mean():.4f}  control {co.mean():.4f}")
    rep("synth  - baseline", sy, b)
    rep("control- baseline", co, b)
    rep("synth  - control ", sy, co)

    # tag strata -- the pre-registered question
    tags = {}
    for row in csv.DictReader(open("data/results.csv")):
        if row["fold"] != str(a.fold):
            continue
        tags[row["track_id"]] = {k[4:] for k in row if k.startswith("tag:") and row[k] == "1"}
    print("\nsynth - control by SMC difficulty tag:")
    allt = sorted({t for v in tags.values() for t in v})
    for t in allt:
        m = np.array([t in tags.get(s, set()) for s in stems])
        if m.sum() < 4:
            continue
        d = (sy - co)[m]
        print(f"  {t:34s} n={m.sum():2d}  synth-control={d.mean():+.4f}  "
              f"baseline F={b[m].mean():.4f}")
    key = np.array([bool(tags.get(s, set()) & {"expressive timing", "slow tempo",
                                               "lack of transient sounds"}) for s in stems])
    print(f"\n  TARGET regime (expressive/slow/no-transients) n={key.sum():2d}  "
          f"synth-control={(sy-co)[key].mean():+.4f}")
    print(f"  rest                                          n={(~key).sum():2d}  "
          f"synth-control={(sy-co)[~key].mean():+.4f}")

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["track", "baseline", "synth", "control"])
        for s in stems:
            w.writerow([s, res["baseline"][s], res["synth"][s], res["control"][s]])
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
