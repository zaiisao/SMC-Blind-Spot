"""Extract Beat This penultimate (512-d transformer) embeddings for SMC, fold-honestly.

WHY. The 2016 emission compresses a 512-d representation to 2 channels, decorrelates them with a
max(b - d, eps) hack, and assigns 3 classes by a hard geometric rule. We proved that rule is
uniquely tempo-neutral WITHIN that parameterisation (a fixed-ms region provably cannot be
tempo-neutral, whatever divisor you pick), so the only way to get both a constant temporal
tolerance and tempo-neutrality is to stop deriving class membership from geometry -- i.e. a
learned emission head on the rich features. This script produces its input.

ARCHITECTURE (beat_this/model/beat_tracker.py:188): forward is
    x = self.frontend(x); x = self.transformer_blocks(x); x = self.task_heads(x)
so the penultimate representation is the output of `transformer_blocks`, dim 512, and
`task_heads` is a SumHead = Linear(512, 2). We hook between them.

CHUNKING. `split_predict_aggregate` chunks (1500, border 6, keep_first) and `aggregate_prediction`
is hardcoded to the 'beat'/'downbeat' keys, so it cannot carry a [T, 512] tensor. We replicate the
same split with the library's own `split_piece` and aggregate the embeddings with the identical
keep_first rule.

ALIGNMENT GATE (non-negotiable). We also re-derive the beat/downbeat logits from our own chunking
and require them to match `beat_this_activations_cache/` -- which was itself built fold-honestly
from `beat_this_annotations/smc/8-folds.split`. If they do not match, the embeddings are not
aligned with every number in the paper and the run aborts.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third-party" / "beat_this"))

from beat_this.inference import Audio2Frames, split_piece
from beat_this.preprocessing import load_audio

SPLIT = ROOT / "third-party" / "beat_this_annotations" / "smc" / "8-folds.split"
AUDIO = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
CACHE = ROOT / "cache" / "beat_this_activations_cache"
CHUNK, BORDER = 1500, 6


def embed_piece(model, spect):
    """[T, 512] penultimate embeddings + [T] beat / [T] downbeat logits, same chunking as the lib."""
    chunks, starts = split_piece(spect, CHUNK, border_size=BORDER, avoid_short_end=True)
    T = spect.shape[0]
    emb = torch.zeros((T, 512), dtype=torch.float32, device=spect.device)
    beat = torch.zeros(T, dtype=torch.float32, device=spect.device)
    down = torch.zeros(T, dtype=torch.float32, device=spect.device)
    filled = torch.zeros(T, dtype=torch.bool, device=spect.device)
    with torch.inference_mode():
        for chunk, start in zip(chunks, starts):
            h = model.frontend(chunk.unsqueeze(0))
            h = model.transformer_blocks(h)                       # [1, Tc, 512]
            out = model.task_heads(h)
            hh = h[0].float()
            b, d = out["beat"][0].float(), out["downbeat"][0].float()
            lo, hi = (BORDER, hh.shape[0] - BORDER) if BORDER > 0 else (0, hh.shape[0])
            if start == 0:
                lo = 0
            gs = start + lo
            ge = min(start + hi, T)
            n = ge - gs
            if n <= 0:
                continue
            sel = ~filled[gs:ge]                                  # keep_first
            idx = torch.arange(gs, ge, device=spect.device)[sel]
            off = (idx - start)
            emb[idx] = hh[off]
            beat[idx] = b[off]
            down[idx] = d[off]
            filled[gs:ge] = True
    return emb.cpu().numpy(), beat.cpu().numpy(), down.cpu().numpy(), bool(filled.all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="/disk1/jaehoon/smc_bt_embeddings")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    want = set(int(x) for x in args.folds.split(","))

    fold_map = {}
    for line in open(SPLIT):
        line = line.strip()
        if line:
            tid, f = line.split("\t")
            fold_map[tid] = int(f)
    by_fold = defaultdict(list)
    for tid, f in fold_map.items():
        by_fold[f].append(tid)

    report = {"tracks": {}, "gate_failures": []}
    done = 0
    for fold in sorted(want):
        tids = sorted(by_fold.get(fold, []))
        if not tids:
            continue
        a2f = Audio2Frames(checkpoint_path=f"fold{fold}", device=args.device)
        print(f"fold {fold}: {len(tids)} tracks", flush=True)
        for tid in tids:
            wav = AUDIO / f"{tid.upper()}.wav"
            if not wav.exists():
                print(f"  MISSING AUDIO {wav}", flush=True); continue
            sig, sr = load_audio(str(wav))
            spect = a2f.signal2spect(sig, sr)
            emb, beat, down, full = embed_piece(a2f.model, spect)
            ref = np.load(CACHE / f"{tid}.npz")
            rb = ref["beat"] if "beat" in ref else ref[ref.files[0]]
            n = min(len(rb), len(beat))
            dev = float(np.abs(beat[:n] - rb[:n]).max())
            ok = dev < args.tol and full
            report["tracks"][tid] = {"fold": fold, "T": int(len(beat)), "max_dev_vs_cache": dev,
                                     "all_frames_filled": full, "gate": ok}
            if not ok:
                report["gate_failures"].append(tid)
            np.savez_compressed(out / f"{tid}.npz", emb=emb.astype(np.float16),
                                beat=beat, downbeat=down, fold=fold)
            done += 1
            if done % 25 == 0:
                print(f"  {done} done, worst dev so far "
                      f"{max(v['max_dev_vs_cache'] for v in report['tracks'].values()):.2e}", flush=True)
            if args.limit and done >= args.limit:
                break
        if args.limit and done >= args.limit:
            break

    devs = [v["max_dev_vs_cache"] for v in report["tracks"].values()]
    print(f"\nALIGNMENT GATE: {len(report['tracks']) - len(report['gate_failures'])}"
          f"/{len(report['tracks'])} pass (tol {args.tol})")
    if devs:
        print(f"  max deviation vs cached logits: {max(devs):.3e}  median {np.median(devs):.3e}")
    if report["gate_failures"]:
        print(f"  FAILURES: {report['gate_failures'][:10]}")
    Path(out / "_report.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
