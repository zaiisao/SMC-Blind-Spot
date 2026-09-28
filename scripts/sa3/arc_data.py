"""Data path for conditional-beat ARC post-training.

Yields ``(latents, metadata)`` batches in exactly the shape
``ARCTrainingWrapper.training_step`` consumes with ``pre_encoded=True``:

    reals, metadata = batch
    padding_masks = torch.stack([md["padding_mask"][0] for md in metadata])
    ... conditioning = model.conditioner(metadata, device)

Three item kinds, all latent-native so that a batch can mix them freely:

    "real"  -- real annotated audio with its OWN grid            -> POSITIVE
    "gen"   -- our generated audio with its inherited grid       -> NEGATIVE (i)
    (ii) real audio with a WRONG grid is not an item kind; it is produced at
         training time by :func:`roll_beat_metadata`, which is what the ARC
         contrastive discriminator branch needs.

Fold 0 of ``data/pools/slow_under70.json`` is excluded here, unconditionally,
for the real side. There is no flag to include it.

Latent convention: ``.npy`` files hold ``pretransform.encode(x)`` output, i.e.
ALREADY multiplied by ``pretransform.scale`` -- the same convention as
``scripts/sa3/sa3_build_latent_cache.py`` and the same one ARC's
``pre_encoded`` branch undoes by dividing by ``scale``. Do not save raw
unscaled latents into these directories.

Self-test:  python scripts/sa3/arc_data.py --self-test
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_SAT = REPO / "third-party" / "stable-audio-tools"
if str(_SAT) not in sys.path:
    sys.path.insert(0, str(_SAT))

from sa3_beat.conditioners import LATENT_RATE  # noqa: E402
from sa3_beat.data import load_beats  # noqa: E402

POOL = REPO / "data" / "pools" / "slow_under70.json"
REAL_CACHE = Path("/disk1/jaehoon/sa3_beat_cache")
SA3_DIR = Path("/disk1/jaehoon/sa3_small_music")

# 215 latent frames = 19.969 s, the length every generated clip already has.
CROP = 215
HELD_OUT_FOLD = 0
DEFAULT_PROMPT = "solo piano"


# --------------------------------------------------------------------------
# item construction
# --------------------------------------------------------------------------

def real_items(pool_path=POOL, cache=REAL_CACHE, exclude_fold=HELD_OUT_FOLD,
               require_bpm=None):
    """Cached real slow tracks, fold `exclude_fold` removed."""
    pool = json.loads(Path(pool_path).read_text())
    items, missing = [], []
    for t in pool:
        if t["fold"] == exclude_fold:
            continue
        if require_bpm is not None and t["bpm"] > require_bpm:
            continue
        npy = Path(cache) / f"{t['name']}.npy"
        beats = Path(cache) / f"{t['name']}.beats"
        if not (npy.exists() and beats.exists()):
            missing.append(t["name"])
            continue
        items.append({
            "kind": "real", "name": t["name"], "latent": npy, "beats": beats,
            "meta": Path(cache) / f"{t['name']}.json",
            "fold": t["fold"], "corpus": t["corpus"], "bpm": t["bpm"],
        })
    return items, missing


def gen_items(latent_dir, clip_dir=None):
    """Generated clips whose latents have been cached by `encode_clip_dir`."""
    latent_dir = Path(latent_dir)
    clip_dir = Path(clip_dir) if clip_dir else latent_dir
    items = []
    for npy in sorted(latent_dir.glob("*.npy")):
        beats = clip_dir / f"{npy.stem}.beats"
        if not beats.exists():
            continue
        items.append({
            "kind": "gen", "name": f"{latent_dir.name}/{npy.stem}",
            "latent": npy, "beats": beats,
            "meta": latent_dir / f"{npy.stem}.json",
            "fold": None, "corpus": "generated", "bpm": None,
        })
    return items


def _held_out_sources(pool_path=POOL, fold=HELD_OUT_FOLD):
    return {t["name"] for t in json.loads(Path(pool_path).read_text())
            if t["fold"] == fold}


def filter_gen_by_source(items, clip_dir, pool_path=POOL, fold=HELD_OUT_FOLD):
    """Drop generated clips whose SOURCE recording is in the held-out fold.

    A generated clip inherits its source's content; training on one made from a
    fold-0 recording would leak the evaluation set through the back door.
    """
    held = _held_out_sources(pool_path, fold)
    clip_dir = Path(clip_dir)
    kept, dropped = [], []
    for it in items:
        sidecar = clip_dir / f"{Path(it['latent']).stem}.json"
        src = None
        if sidecar.exists():
            src = json.loads(sidecar.read_text()).get("source")
        (dropped if src in held else kept).append(it)
    return kept, dropped


# --------------------------------------------------------------------------
# latent caching for generated clips
# --------------------------------------------------------------------------

def encode_clip_dir(clip_dir, out_dir, gpu=3, limit=0, sr_expect=None):
    """Encode ``clip_dir/*.wav`` to latents with the SA3 pretransform.

    Uses ``pretransform.encode`` so the stored latents carry the same scale as
    ``sa3_build_latent_cache.py``'s.
    """
    import soundfile as sf
    from safetensors.torch import load_file
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict

    clip_dir, out_dir = Path(clip_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(clip_dir.glob("*.wav"))
    if limit:
        wavs = wavs[:limit]
    todo = [w for w in wavs if not (out_dir / f"{w.stem}.npy").exists()]
    if not todo:
        return 0

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    config = json.loads((SA3_DIR / "model_config.json").read_text())
    model = create_model_from_config(config)
    copy_state_dict(model, load_file(str(SA3_DIR / "model.safetensors")))
    pt = model.pretransform.to(device).eval()
    sr, ds = config["sample_rate"], model.pretransform.downsampling_ratio

    n = 0
    for w in todo:
        y, file_sr = sf.read(str(w), dtype="float32", always_2d=True)
        if file_sr != sr:
            raise ValueError(f"{w} is {file_sr} Hz, model wants {sr}")
        y = y.T
        if y.shape[0] == 1:
            y = np.concatenate([y, y], axis=0)
        real = y.shape[1]
        n_samp = max(ds, (real // ds) * ds)
        if real < n_samp:
            y = np.pad(y, ((0, 0), (0, n_samp - real)))
        x = torch.tensor(y[:, :n_samp], device=device)[None]
        with torch.no_grad():
            z = pt.encode(x).squeeze(0).float().cpu().numpy()
        np.save(out_dir / f"{w.stem}.npy", z)
        nf = z.shape[-1]
        rf = min(nf, int(math.ceil(min(real, n_samp) / ds)))
        (out_dir / f"{w.stem}.json").write_text(json.dumps({
            "path": w.name, "seconds_total": round(min(real, n_samp) / sr, 3),
            "padding_mask": [1] * rf + [0] * (nf - rf), "corpus": "generated"}))
        n += 1
    return n


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

class BeatARCDataset(torch.utils.data.Dataset):
    """(latent crop, metadata) pairs with beat-grid conditioning attached.

    Every returned metadata dict carries what the model's conditioner needs
    (``prompt``, ``seconds_total``, ``beat``) plus what ARC's training_step
    needs (``padding_mask`` as a one-element list).
    """

    def __init__(self, items, crop=CROP, random_crop=True,
                 prompt=DEFAULT_PROMPT, min_beats=2, latent_rate=LATENT_RATE,
                 seed=0, max_resample=8):
        if not items:
            raise ValueError("BeatARCDataset was given no items")
        self.items = list(items)
        self.crop = int(crop)
        self.random_crop = random_crop
        self.prompt = prompt
        self.min_beats = int(min_beats)
        self.latent_rate = float(latent_rate)
        self.max_resample = int(max_resample)
        self._rng = random.Random(seed)
        self._beats_cache = {}

    def __len__(self):
        return len(self.items)

    def _beats(self, path):
        key = str(path)
        if key not in self._beats_cache:
            self._beats_cache[key] = load_beats(path)
        return self._beats_cache[key]

    def _load(self, idx):
        it = self.items[idx]
        z = np.load(it["latent"])                    # [C, N], scaled latents
        n = z.shape[1]
        beats, downbeats = self._beats(it["beats"])
        if beats.size < 2:
            return None

        if n >= self.crop:
            hi = n - self.crop
            start = self._rng.randint(0, hi) if (self.random_crop and hi > 0) else 0
            z = z[:, start:start + self.crop]
            valid = self.crop
        else:
            start = 0
            z = np.pad(z, ((0, 0), (0, self.crop - n)))
            valid = n

        t0 = start / self.latent_rate
        t1 = (start + valid) / self.latent_rate
        inside = beats[(beats >= t0) & (beats < t1)]
        if inside.size < self.min_beats:
            return None

        pad = torch.zeros(self.crop, dtype=torch.long)
        pad[:valid] = 1
        info = {
            "prompt": self.prompt,
            "seconds_total": float(valid / self.latent_rate),
            "seconds_start": float(t0),
            "padding_mask": [pad],
            "beat": {
                "beats": np.asarray(beats, dtype=np.float64),
                "downbeats": (None if downbeats is None
                              else np.asarray(downbeats, dtype=np.float64)),
                "num_frames": self.crop,
                "frame_offset": int(start),
            },
            "kind": it["kind"],
            "name": it["name"],
            "latent_crop_start": int(start),
            "n_beats_in_crop": int(inside.size),
        }
        return torch.from_numpy(np.ascontiguousarray(z)).float(), info

    def __getitem__(self, idx):
        for _ in range(self.max_resample):
            got = self._load(idx)
            if got is not None:
                return got
            idx = self._rng.randrange(len(self.items))
        raise RuntimeError(
            "could not find an item with a usable beat grid after "
            f"{self.max_resample} tries -- the grids and latents disagree")


def collate(samples):
    """(B, C, crop) float32 latents + list of B metadata dicts."""
    latents = torch.stack([s[0] for s in samples], dim=0)
    return latents, [s[1] for s in samples]


def make_sampler(items, weights, seed=0, num_samples=None):
    """WeightedRandomSampler over item kinds/corpora, e.g. {'gen': 0.5}.

    `weights` maps a kind ("real"/"gen") or a corpus name to a relative weight;
    an unlisted item gets 1.0. Corpus wins over kind when both are listed.
    """
    w = []
    for it in items:
        w.append(float(weights.get(it["corpus"], weights.get(it["kind"], 1.0))))
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(
        torch.tensor(w, dtype=torch.double),
        num_samples=num_samples or len(items), replacement=True, generator=g)


# --------------------------------------------------------------------------
# negative type (ii): real audio, wrong grid
# --------------------------------------------------------------------------

def _grid_distance(a, b, seconds=CROP / LATENT_RATE, n=256):
    """Mean |sin phase difference| between two grids over the window.

    0 = the two grids are the same pulse in the same phase; ~0.6 = unrelated.
    """
    from sa3_beat.conditioners import _continuous_phase
    t = np.linspace(0, seconds, n)
    pa = _continuous_phase(np.asarray(a["beats"]) - a.get("t0", 0.0), t)
    pb = _continuous_phase(np.asarray(b["beats"]) - b.get("t0", 0.0), t)
    return float(np.abs(np.sin(2 * np.pi * pa) - np.sin(2 * np.pi * pb)).mean())


def roll_beat_metadata(metadata, min_distance=0.15, roll_prompt=False):
    """Negative type (ii): each item keeps its audio and gets a NEIGHBOUR's grid.

    Returns a NEW list of new dicts -- it does not mutate the input, unlike
    ARC's built-in contrastive block, which rolls in place and only rolls
    "prompt". Feed the result to ``discriminator.conditioner(...)``.

    The rolled grid keeps its donor's absolute beat times but is re-anchored to
    the RECEIVER's crop offset, so the wrong grid covers the same window as the
    audio it is paired with.

    `min_distance` rejects a roll that happens to land on essentially the same
    pulse (two 60 bpm tracks in phase) and searches further round the batch;
    an item with no distinguishable partner is returned UNROLLED and reported,
    so the caller can drop it rather than train on a mislabelled negative.
    """
    b = len(metadata)
    out, unrolled = [], []
    for i in range(b):
        own = metadata[i]["beat"]
        own_t0 = own.get("frame_offset", 0) / LATENT_RATE
        chosen = None
        for step in range(1, b):
            cand = metadata[(i + step) % b]["beat"]
            cand_t0 = cand.get("frame_offset", 0) / LATENT_RATE
            d = _grid_distance({"beats": own["beats"], "t0": own_t0},
                               {"beats": cand["beats"], "t0": cand_t0})
            if d >= min_distance:
                chosen = (cand, cand_t0)
                break
        md = dict(metadata[i])
        if chosen is None:
            unrolled.append(i)
        else:
            cand, cand_t0 = chosen
            shift = cand_t0 - own_t0
            md["beat"] = {
                "beats": np.asarray(cand["beats"], dtype=np.float64) - shift,
                "downbeats": (None if cand["downbeats"] is None else
                              np.asarray(cand["downbeats"], dtype=np.float64) - shift),
                "num_frames": own["num_frames"],
                "frame_offset": own.get("frame_offset", 0),
            }
        if roll_prompt:
            md["prompt"] = metadata[(i + 1) % b]["prompt"]
        out.append(md)
    return out, unrolled


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _self_test(args):
    items, missing = real_items()
    print(f"[real] {len(items)} cached slow tracks excluding fold "
          f"{HELD_OUT_FOLD}; {len(missing)} pool tracks have no latent yet")
    folds = sorted({it["fold"] for it in items})
    assert HELD_OUT_FOLD not in folds, "fold 0 leaked into the training items"
    print(f"[real] folds present: {folds}")

    gen = []
    if args.gen_latents:
        if args.encode_from:
            n = encode_clip_dir(args.encode_from, args.gen_latents,
                                gpu=args.gpu, limit=args.encode_limit)
            print(f"[gen] encoded {n} new clips -> {args.gen_latents}")
        gen = gen_items(args.gen_latents, clip_dir=args.encode_from)
        if args.encode_from:
            gen, dropped = filter_gen_by_source(gen, args.encode_from)
            print(f"[gen] {len(gen)} clips kept, {len(dropped)} dropped "
                  f"(source in held-out fold {HELD_OUT_FOLD})")

    all_items = items + gen
    ds = BeatARCDataset(all_items, seed=0)
    sampler = make_sampler(all_items, {"gen": 3.0}, seed=0, num_samples=64)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                     sampler=sampler, collate_fn=collate,
                                     num_workers=args.num_workers)

    seen_kinds = {}
    for bi, (lat, md) in enumerate(dl):
        if bi == 0:
            print(f"\n[batch] latents {tuple(lat.shape)} {lat.dtype} "
                  f"min={lat.min():.3f} max={lat.max():.3f} std={lat.std():.3f}")
            m0 = md[0]
            print(f"[batch] metadata keys: {sorted(m0)}")
            print(f"[batch] padding_mask: list[{len(m0['padding_mask'])}] of "
                  f"{tuple(m0['padding_mask'][0].shape)} {m0['padding_mask'][0].dtype}")
            bt = m0["beat"]
            print(f"[batch] beat: beats {bt['beats'].shape} {bt['beats'].dtype}, "
                  f"downbeats {None if bt['downbeats'] is None else bt['downbeats'].shape}, "
                  f"num_frames={bt['num_frames']} frame_offset={bt['frame_offset']}")
            stacked = torch.stack([m["padding_mask"][0] for m in md], dim=0)
            print(f"[batch] ARC padding_masks stack: {tuple(stacked.shape)} "
                  f"{stacked.dtype}")
            rolled, unrolled = roll_beat_metadata(md)
            d_self, d_roll = [], []
            for a, r in zip(md, rolled):
                own = {"beats": a["beat"]["beats"],
                       "t0": a["beat"]["frame_offset"] / LATENT_RATE}
                rol = {"beats": r["beat"]["beats"],
                       "t0": r["beat"]["frame_offset"] / LATENT_RATE}
                d_self.append(_grid_distance(own, own))
                d_roll.append(_grid_distance(own, rol))
            print(f"[roll] grid distance own-vs-own {max(d_self):.4f} (must be 0), "
                  f"own-vs-rolled median {np.median(d_roll):.3f} "
                  f"min {min(d_roll):.3f}, {len(unrolled)} items unrollable")
        for m in md:
            seen_kinds[m["kind"]] = seen_kinds.get(m["kind"], 0) + 1
        if bi + 1 >= args.batches:
            break
    print(f"\n[kinds] over {args.batches} batches: {seen_kinds}")

    if args.conditioner:
        from sa3_beat.conditioners import BeatConditioner
        c = BeatConditioner(include_downbeat=True)
        # MultiConditioner routes md["beat"] to the BeatConditioner; called
        # directly it wants that inner dict.
        t, mask = c([m["beat"] for m in md], "cpu")
        print(f"[cond] beat tensor {tuple(t.shape)} {t.dtype}, "
              f"mask {tuple(mask.shape)} {mask.dtype}, "
              f"|sin/cos| max {t.abs().max():.3f}")
        rt, _ = c([m["beat"] for m in rolled], "cpu")
        print(f"[cond] rolled tensor differs from own: "
              f"max abs diff {(t - rt).abs().max():.3f}")
    print("\n[ok] self-test complete")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--gen-latents", default=None,
                    help="directory holding (or to hold) generated-clip latents")
    ap.add_argument("--encode-from", default=None,
                    help="clip directory of sd_*.wav to encode into --gen-latents")
    ap.add_argument("--encode-limit", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--conditioner", action="store_true",
                    help="also run the BeatConditioner on a batch")
    args = ap.parse_args()
    if args.self_test:
        _self_test(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
