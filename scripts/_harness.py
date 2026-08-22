"""
Shared evaluation harness for SMC blind-spot follow-up experiments.

Everything here mirrors the paper's pipeline so new experiments are
directly comparable:
  - Beat This: cached logits at 50 fps  (keys: 'beat','downbeat')
  - Beat Transformer: cached probs at 44100/1024 fps (keys: 'beat_act','downbeat_act','tempo_logits','tempo_bpm')
  - peak-picking: max-pool width 7 (+/-70 ms) local maxima above a threshold
  - DBN: madmom DBNDownBeatTrackingProcessor(beats_per_bar=[3,4])
  - eval: mir_eval.beat.evaluate  (DEFAULT 5 s trim, +/-70 ms tolerance)

Use one protocol for every row of any comparison table.
"""
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import mir_eval
from scipy.special import expit
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
GT_DIR = ROOT / "beat_this_annotations" / "smc" / "annotations" / "beats"
BTHIS_DIR = ROOT / "beat_this_activations_cache"
BTRANS_DIR = ROOT / "bt_transformer_cache"

FPS = 50                 # Beat This
BT_FPS = 44100 / 1024    # Beat Transformer (~43.066)


def track_ids():
    return sorted(p.stem for p in BTHIS_DIR.glob("*.npz"))


def load_gt(tid):
    p = GT_DIR / f"{tid}.beats"
    if not p.exists():
        p = GT_DIR / f"{tid.upper()}.beats"
    d = np.loadtxt(str(p))
    return d if d.ndim == 1 else d[:, 1]


def gt_bpm(ref):
    return 60.0 / np.median(np.diff(ref)) if len(ref) >= 2 else 0.0


# ---- activation loaders (always return PROBABILITY arrays) ----
def bthis_beat_prob(tid):
    return expit(np.load(str(BTHIS_DIR / f"{tid}.npz"))["beat"].astype(np.float64))

def bthis_downbeat_prob(tid):
    return expit(np.load(str(BTHIS_DIR / f"{tid}.npz"))["downbeat"].astype(np.float64))

def btrans_beat_prob(tid):
    return np.load(str(BTRANS_DIR / f"{tid}.npz"))["beat_act"].astype(np.float64)

def btrans_downbeat_prob(tid):
    return np.load(str(BTRANS_DIR / f"{tid}.npz"))["downbeat_act"].astype(np.float64)

def btrans_tempo(tid):
    z = np.load(str(BTRANS_DIR / f"{tid}.npz"))
    return float(z["tempo_bpm"]), z["tempo_logits"].astype(np.float64)


def resample_to(prob, src_fps, dst_fps, dst_len):
    """Linear-interpolate a 1-D activation from src grid onto a dst grid."""
    src_t = np.arange(len(prob)) / src_fps
    dst_t = np.arange(dst_len) / dst_fps
    return np.interp(dst_t, src_t, prob)


def peakpick(prob, thresh, fps, width=7):
    """Local maxima (max-pool width `width`) above `thresh` -> beat times (s)."""
    t = torch.from_numpy(np.ascontiguousarray(prob)).float().unsqueeze(0)
    peaks = t.masked_fill(t != F.max_pool1d(t, width, 1, width // 2), -1000.0)
    mask = peaks.squeeze(0) > thresh
    fr = torch.nonzero(mask).cpu().numpy()[:, 0]
    if len(fr):
        keep = [fr[0]]
        for x in fr[1:]:
            if x - keep[-1] > 1:
                keep.append(x)
        fr = np.array(keep)
    return fr / fps


def make_dbn(min_bpm=30.0, max_bpm=215.0, fps=FPS, transition_lambda=100):
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    min_bpm = max(30.0, float(min_bpm))
    max_bpm = min(300.0, float(max_bpm))
    if max_bpm <= min_bpm + 5:
        max_bpm = min_bpm + 5
    return DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4], min_bpm=min_bpm, max_bpm=max_bpm,
        fps=fps, transition_lambda=transition_lambda)


def dbn_beats(beat_prob, downbeat_prob, dbn):
    """Run DBN on (beat, downbeat) probability pair -> beat times (s)."""
    eps = 1e-5
    bp = beat_prob * (1 - eps) + eps / 2
    dp = downbeat_prob * (1 - eps) + eps / 2
    combined = np.vstack((np.maximum(bp - dp, eps / 2), dp)).T
    out = dbn(combined)
    return out[:, 0]


def ev(ref, est):
    if len(ref) == 0 or len(est) == 0:
        return {"F": 0.0, "CMLt": 0.0, "AMLt": 0.0, "CMLc": 0.0, "AMLc": 0.0}
    r = mir_eval.beat.evaluate(ref, est)
    return {"F": r["F-measure"],
            "CMLt": r["Correct Metric Level Total"],
            "AMLt": r["Any Metric Level Total"],
            "CMLc": r["Correct Metric Level Continuous"],
            "AMLc": r["Any Metric Level Continuous"]}


def mean_scores(per_track):
    keys = ["F", "CMLt", "AMLt", "CMLc", "AMLc"]
    return {k: float(np.mean([d[k] for d in per_track])) for k in keys}
