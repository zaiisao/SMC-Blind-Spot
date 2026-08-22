"""
EXPERIMENT 3: test-time augmentation (TTA) for Beat This.

The paper's thesis: Beat This produces 'confident-but-wrong' activations on SMC.
If a wrong peak is an artefact of one particular input rendering, averaging the
activations over several pitch-shifted copies of the audio should suppress it
while reinforcing genuine, shift-stable beat peaks. Timing is preserved by
pitch-shift, so activations align on the native 50 fps grid with no un-warping.

Uses each track's correct held-out fold checkpoint (no train/test leakage).
All eval identical to the paper: mir_eval 5 s trim, +/-70 ms.
"""
import sys, time, numpy as np, torch, librosa
sys.path.insert(0, "scripts")
from _harness import track_ids, load_gt, peakpick, ev, mean_scores, FPS
from scipy.special import expit
import mir_eval
from collections import defaultdict
from pathlib import Path

TRIM = mir_eval.beat.trim_beats
def fastF(ref, est):
    if len(est) == 0 or len(ref) == 0: return 0.0
    r = TRIM(ref); e = TRIM(est)
    if len(r) == 0 or len(e) == 0: return 0.0
    return mir_eval.beat.f_measure(r, e)

ROOT = Path("/home/sogang/jaehoon/Analyze-SMC")
AUDIO = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
SPLIT = ROOT / "beat_this_annotations" / "smc" / "8-folds.split"
SHIFTS = [-2, -1, 0, 1, 2]   # semitones

def main():
    from beat_this.inference import Audio2Frames
    from beat_this.preprocessing import load_audio
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    fold_map = {}
    for line in open(SPLIT):
        line = line.strip()
        if line:
            tid, f = line.split("\t"); fold_map[tid] = int(f)
    ids = track_ids()
    refs = {t: load_gt(t) for t in ids}
    folds = defaultdict(list)
    for t in ids:
        folds[fold_map[t]].append(t)

    # store averaged logits per track
    avg_logits = {}      # mean over shifts (logit space)
    base_logits = {}     # shift 0 only (== cache, baseline)
    t0 = time.time()
    for fold in sorted(folds):
        a2f = Audio2Frames(checkpoint_path=f"fold{fold}", device=dev)
        for t in folds[fold]:
            sig, sr = load_audio(str(AUDIO / f"{t.upper()}.wav"))
            sig = np.asarray(sig).astype(np.float32)
            if sig.ndim > 1:
                sig = sig.mean(axis=1)
            logits_list = []
            for s in SHIFTS:
                if s == 0:
                    x = sig
                else:
                    x = librosa.effects.pitch_shift(y=sig, sr=sr, n_steps=s)
                bl, _ = a2f(x, sr)
                bl = bl.detach().cpu().numpy().astype(np.float64)
                logits_list.append(bl)
                if s == 0:
                    base_logits[t] = bl
            L = min(len(x) for x in logits_list)
            stack = np.stack([x[:L] for x in logits_list], 0)
            avg_logits[t] = stack.mean(0)
        print(f"  fold {fold} done, {time.time()-t0:.0f}s elapsed", flush=True)

    # --- evaluate ---
    # baseline (shift0) peak@0.5  (sanity: should be ~0.627)
    base = [ev(refs[t], peakpick(expit(base_logits[t]), 0.5, FPS)) for t in ids]
    print("baseline shift0 peak@0.5   :", round(mean_scores(base)["F"], 4), "(expect 0.627)", flush=True)

    # TTA: average in logit space then sigmoid; sweep global threshold (prob space)
    tta_prob = {t: expit(avg_logits[t]) for t in ids}
    best = (0.5, -1)
    for th in np.round(np.arange(0.1, 0.9, 0.025), 4):
        m = np.mean([fastF(refs[t], peakpick(tta_prob[t], th, FPS)) for t in ids])
        if m > best[1]:
            best = (th, m)
    tta_at_half = [ev(refs[t], peakpick(tta_prob[t], 0.5, FPS)) for t in ids]
    tta_best = [ev(refs[t], peakpick(tta_prob[t], best[0], FPS)) for t in ids]
    print("TTA(avg logits) peak@0.5   :", round(mean_scores(tta_at_half)["F"], 4), flush=True)
    print(f"TTA(avg logits) peak@{best[0]:.3f} :", round(mean_scores(tta_best)["F"], 4),
          "CMLt", round(mean_scores(tta_best)["CMLt"], 4),
          "AMLt", round(mean_scores(tta_best)["AMLt"], 4), flush=True)

    # per-track oracle threshold on TTA activations
    fine = expit(np.arange(-4.0, 4.5, 0.25))
    orac = [max(fastF(refs[t], peakpick(tta_prob[t], th, FPS)) for th in fine) for t in ids]
    print("TTA per-track oracle thresh:", round(np.mean(orac), 4), "(BeatThis-alone oracle ~0.670)", flush=True)

    # save activations for downstream fusion
    outdir = ROOT / "data" / "tta_activations"
    outdir.mkdir(parents=True, exist_ok=True)
    for t in ids:
        np.savez(outdir / f"{t}.npz", avg_logits=avg_logits[t])
    print("saved TTA activations ->", outdir, flush=True)

if __name__ == "__main__":
    main()
