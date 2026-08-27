"""Encode the beat-annotated corpora into SA3 latents for ControlNet training.

Produces, in one flat directory keyed by a corpus-qualified stem:

    <id>.npy     latents, [256, frames], float32
    <id>.json    sidecar the library requires: path / seconds_total / padding_mask
    <id>.beats   the Beat This annotation, copied so the metadata hook resolves it

plus ``fold_<k>.txt`` filelists, so a per-fold generator trains on everything
EXCEPT fold k without rebuilding the cache.

Scope decisions, both deliberate:

* GTZAN is excluded. It has no ``8-folds.split`` because Beat This holds it out
  as the test set (``if dataset == self.test_set_name: continue``), so putting it
  in the generator would contaminate the held-out evaluation.
* Only corpora with real WAVEFORM audio are included. The Beat This spectrogram
  packs (simac, harmonix, ...) cannot be used here: SA3's VAE encodes waveform,
  and inverting 128-bin mel would destroy phase and smear exactly the transients
  this project measures. Those corpora belong to the tracker's training, not the
  generator's.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SA3_DIR = Path("/disk1/jaehoon/sa3_small_music")
AUDIO = Path("/disk1/jaehoon/dataset_store/audio_by_stem")
ANNOT = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")

# GTZAN deliberately absent -- see module docstring.
CORPORA = ["asap", "ballroom", "beatles", "hainsworth", "hjdb", "rwc", "smc"]


def load_folds(corpus):
    """stem -> fold index, from the canonical 8-folds.split."""
    path = ANNOT / corpus / "8-folds.split"
    folds = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        stem, part = line.split("\t")
        folds[stem] = int(part)
    return folds


def collect(corpora):
    """(corpus, stem, audio_path, annotation_path, fold) for every usable track."""
    items, skipped = [], []
    for c in corpora:
        folds = load_folds(c)
        beats_dir = ANNOT / c / "annotations" / "beats"
        for wav in sorted((AUDIO / c).iterdir()):
            stem = wav.stem
            ann = beats_dir / f"{stem}.beats"
            if not ann.exists():
                skipped.append((c, stem, "no annotation")); continue
            if stem not in folds:
                skipped.append((c, stem, "not in 8-folds.split")); continue
            items.append((c, stem, wav, ann, folds[stem]))
    return items, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/disk1/jaehoon/sa3_beat_cache")
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--max-seconds", type=float, default=120.0,
                    help="cap per track; crops during training are 20 s anyway")
    ap.add_argument("--corpora", nargs="*", default=CORPORA)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items, skipped = collect(args.corpora)
    if args.limit:
        items = items[: args.limit]
    by_corpus = {}
    for c, *_ in items:
        by_corpus[c] = by_corpus.get(c, 0) + 1
    print(f"[plan] {len(items)} tracks: " +
          "  ".join(f"{c}={n}" for c, n in sorted(by_corpus.items())))
    if skipped:
        print(f"[plan] skipped {len(skipped)}, e.g. {skipped[:3]}")
    if args.dry_run:
        return

    import librosa
    from safetensors.torch import load_file
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    config = json.loads((SA3_DIR / "model_config.json").read_text())
    model = create_model_from_config(config)
    copy_state_dict(model, load_file(str(SA3_DIR / "model.safetensors")))
    pt = model.pretransform.to(device).eval()
    sr, ds = config["sample_rate"], model.pretransform.downsampling_ratio
    print(f"[setup] sr={sr} downsampling={ds} latent_rate={sr/ds:.3f} Hz device={device}")

    folds = {k: [] for k in range(8)}
    done = failed = 0
    for i, (corpus, stem, wav, ann, fold) in enumerate(items, 1):
        uid = f"{corpus}__{stem}"
        folds[fold].append(uid)
        npy = out / f"{uid}.npy"
        if npy.exists() and (out / f"{uid}.json").exists():
            done += 1
            continue
        try:
            y, _ = librosa.load(str(wav), sr=sr, mono=False,
                                duration=args.max_seconds)
            if y.ndim == 1:
                y = np.stack([y, y])
            real = y.shape[1]
            n = max(ds, (real // ds) * ds)
            if real < n:
                y = np.pad(y, ((0, 0), (0, n - real)))
            x = torch.tensor(y[:, :n], dtype=torch.float32, device=device)[None]
            with torch.no_grad():
                z = pt.encode(x).squeeze(0).float().cpu().numpy()
            np.save(npy, z)
            nf = z.shape[-1]
            rf = min(nf, int(math.ceil(min(real, n) / ds)))
            json.dump({"path": f"{uid}.wav",
                       "seconds_total": round(min(real, n) / sr, 3),
                       "padding_mask": [1] * rf + [0] * (nf - rf),
                       "corpus": corpus, "fold": fold},
                      open(out / f"{uid}.json", "w"))
            shutil.copy(ann, out / f"{uid}.beats")
            done += 1
        except Exception as e:                      # noqa: BLE001
            failed += 1
            print(f"[fail] {uid}: {type(e).__name__}: {e}")
        if i % 100 == 0:
            print(f"[{i}/{len(items)}] done={done} failed={failed}", flush=True)

    # Per-fold filelists: fold_k.txt lists everything EXCEPT fold k, i.e. the
    # training pool for the generator that will be evaluated on fold k.
    for k in range(8):
        train = [u for kk, us in folds.items() if kk != k for u in us]
        (out / f"train_excl_fold{k}.txt").write_text(
            "\n".join(f"{u}.npy" for u in sorted(train)) + "\n")
        (out / f"holdout_fold{k}.txt").write_text(
            "\n".join(f"{u}.npy" for u in sorted(folds[k])) + "\n")
    print(f"\n[done] {done} encoded, {failed} failed -> {out}")
    print("[folds] " + "  ".join(f"{k}:{len(v)}" for k, v in sorted(folds.items())))


if __name__ == "__main__":
    main()
