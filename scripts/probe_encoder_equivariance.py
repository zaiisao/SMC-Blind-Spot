"""WHERE does the octave collapse happen -- in the encoder, or in the beat head on top of it?

This decides the architecture of the fix, so it is worth its own measurement.

  * If the 512-d penultimate embeddings are EQUIVARIANT under time-stretch while the beat
    activations are not, the tempo information survives the encoder and is destroyed downstream.
    A frozen-encoder TEMPO HEAD can then recover it -- cheap, safe (it cannot degrade beat
    tracking, since nothing upstream changes), and precomputable, which is what makes training on
    a large unlabelled corpus practical.
  * If the embeddings collapse too, a head is capped no matter how it is trained, and only
    fine-tuning the encoder can work.

MEASUREMENT. Same defect as scripts/probe_tempo_equivariance.py,
    delta(s) = log2(period(s) / period(1)) - log2 s,
which is 0 under perfect equivariance and +/-1 for an octave error. Two readouts from ONE forward
pass per (track, stretch), so the two levels are compared on identical audio:

  BEAT LEVEL      soft_log_period of sigmoid(beat logits)      -- reproduces the known collapse
  EMBEDDING LEVEL soft_log_period of the mean per-dimension autocorrelation of the 512-d sequence

The embedding readout requires NO training: each dimension is mean-centred and variance-normalised,
its unbiased ACF is taken, and the ACFs are averaged over dimensions before the soft arg-max. That
asks what period the representation oscillates at, without first committing to a decoder. Dimensions
are variance-normalised so a few high-variance channels cannot dominate the average.

Stretching is audio-domain (phase vocoder), never mel-axis resampling, for the reason that governs
this whole line of work: a ratio objective can otherwise be solved from resampling artefacts.
Fold-honest: each SMC track uses the checkpoint that held it out.
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "beat_this"))
from beat_this.inference import Audio2Frames
from beat_this.preprocessing import load_audio
from _harness import FPS, track_ids
from extract_bt_embeddings import embed_piece
from tempo_readout import lag_grid, soft_log_period

SMC_AUD = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
STORE = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
STRETCH = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
N_TRACKS = 30
DEV = "cuda:2"


def emb_log_period(emb, lags, tau=30.0):
    """[T, D] -> soft log2 period from the dimension-averaged unbiased ACF."""
    x = torch.as_tensor(emb, dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    x = x / (x.std(dim=0, keepdim=True) + 1e-6)          # no channel may dominate the average
    T, D = x.shape
    r = []
    for l in lags.tolist():
        r.append((x[:T - l] * x[l:]).sum(dim=0) / float(T - l) if T - l >= 8
                 else torch.full((D,), -1e9))
    r = torch.stack(r, dim=0).mean(dim=1)                # [L], averaged over dimensions
    r = r / (r.abs().max() + 1e-8)
    w = torch.softmax(tau * r, dim=0)
    return float((w * torch.log2(lags.float())).sum())


def main():
    split = {l.split("\t")[0]: int(l.split("\t")[1])
             for l in open(STORE / "smc" / "8-folds.split") if l.strip()}
    ids = [t for t in track_ids() if (SMC_AUD / f"{t.upper()}.wav").exists()][:N_TRACKS]
    by_fold = {}
    for t in ids:
        by_fold.setdefault(split.get(t, 0), []).append(t)

    lags = lag_grid()
    beat_lp = {s: {} for s in STRETCH}
    emb_lp = {s: {} for s in STRETCH}
    for fold, tracks in sorted(by_fold.items()):
        a2f = Audio2Frames(checkpoint_path=f"fold{fold}", device=DEV)
        for t in tracks:
            sig, sr = load_audio(str(SMC_AUD / f"{t.upper()}.wav"))
            y = sig.mean(1) if sig.ndim > 1 else sig
            y = np.ascontiguousarray(y.astype(np.float32))
            for s in STRETCH:
                ys = y if s == 1.0 else librosa.effects.time_stretch(y, rate=1.0 / s)
                spect = a2f.signal2spect(ys, sr)
                emb, beat, _, ok = embed_piece(a2f.model, spect)
                if not ok:
                    continue
                f_b, _ = soft_log_period(torch.sigmoid(torch.from_numpy(beat)), lags)
                beat_lp[s][t] = float(f_b[0])
                emb_lp[s][t] = emb_log_period(emb, lags)
        print(f"  fold {fold}: {len(tracks)} done", flush=True)

    def defect(tab):
        out = {}
        for s in STRETCH:
            d = [tab[s][t] - tab[1.0][t] - np.log2(s) for t in ids
                 if t in tab[s] and t in tab[1.0]]
            out[s] = float(np.median(d)) if d else float("nan")
        return out

    db, de = defect(beat_lp), defect(emb_lp)
    print(f"\nEQUIVARIANCE DEFECT delta(s), median over {len(ids)} SMC tracks")
    print(f"{'s':>6} {'BEAT level':>12} {'EMBED level':>12}")
    for s in STRETCH:
        print(f"{s:>6.2f} {db[s]:>+12.3f} {de[s]:>+12.3f}")
    out_s = [s for s in STRETCH if s != 1.0]
    mb = float(np.mean([abs(db[s]) for s in out_s]))
    me = float(np.mean([abs(de[s]) for s in out_s]))
    print(f"\n  mean |delta| (s!=1):  beat {mb:.3f}   embed {me:.3f}")
    if me < mb - 0.1:
        print("  => encoder retains tempo better than the beat head does: A TEMPO HEAD CAN HELP")
    elif me > mb + 0.1:
        print("  => embeddings are worse than the beat head: head is capped, fine-tune the encoder")
    else:
        print("  => both levels collapse alike: the loss is IN THE ENCODER, a head is capped")
    json.dump({"n": len(ids), "beat": db, "embed": de, "mean_abs_beat": mb, "mean_abs_embed": me},
              open("/tmp/overnight/encoder_equivariance.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
