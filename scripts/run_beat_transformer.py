"""
Run Beat Transformer (Zhao et al., ISMIR 2022) on SMC:
  - Use 8-fold cross-validation checkpoints, mapping each SMC track to the fold
    in which Beat Transformer held it out (reproducing the shuffle with seed 0).
  - Spleeter demixes audio into 5 stems → mel spectrograms (128 bins, 43.07 fps).
  - Extract per-track beat activation, downbeat activation, and tempo BPM
    (argmax over 300-dim tempo logits).
  - Additionally, use the tempo prediction to constrain madmom's DBN when
    decoding beat_this activations (mirroring the earlier TCN-tempo experiment).

Outputs:
  bt_transformer_cache/{track}.npz  — per-track activations and tempo
  bt_transformer_beats/{track}.beats — Beat Transformer's beats after DBN
  bt_transformer_tempo_constrained_results.csv — the final comparison CSV
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
BT_REPO = ROOT / "Beat-Transformer"
CKPT_DIR = BT_REPO / "checkpoint"
BT_AUDIO_LIST = BT_REPO / "data" / "audio_lists" / "smc.txt"

AUDIO_DIR = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
GT_DIR = ROOT / "beat_this_annotations" / "smc" / "annotations" / "beats"
BT_ACT_DIR = ROOT / "beat_this_activations_cache"   # beat_this (beat_this_net) activations
CACHE_DIR = ROOT / "bt_transformer_cache"
BT_BEATS_DIR = ROOT / "bt_transformer_beats"
OUT_CSV = DATA_DIR / "bt_transformer_tempo_constrained_results.csv"

# Make Beat Transformer code importable
sys.path.insert(0, str(BT_REPO / "code"))

# Beat Transformer FPS (Spleeter STFT: sr=44100, hop=1024)
BT_FPS = 44100 / 1024           # ≈ 43.066
BT_THIS_FPS = 50                # beat_this activations
SR = 44100
N_FFT = 4096
HOP = 1024

# Model hyperparams (must match pretrained checkpoints)
MODEL_KW = dict(
    attn_len=5, instr=5, ntoken=2, dmodel=256, nhead=8,
    d_hid=1024, nlayers=9, norm_first=True,
)
NUM_FOLDS = 8
# DEVICE is set at runtime after parsing args (so CUDA_VISIBLE_DEVICES takes effect)
DEVICE = None


# ─────────────────────────── fold mapping ────────────────────────────

def build_bt_fold_map() -> dict[str, int]:
    """Reproduce Beat Transformer's SMC fold assignment.
    audioDataset shuffles the audio list with np.random.seed(0) then splits
    into 8 consecutive folds; the last fold has the remainder.
    """
    with open(BT_AUDIO_LIST) as f:
        paths = [ln.strip() for ln in f if ln.strip()]
    np.random.seed(0)
    np.random.shuffle(paths)
    fold_size = len(paths) // NUM_FOLDS
    fmap = {}
    for i in range(NUM_FOLDS - 1):
        for p in paths[i * fold_size:(i + 1) * fold_size]:
            fmap[Path(p).stem.lower()] = i
    for p in paths[(NUM_FOLDS - 1) * fold_size:]:
        fmap[Path(p).stem.lower()] = NUM_FOLDS - 1
    return fmap


# ───────────────────── Spleeter + mel spectrogram ────────────────────
# We extract the masked STFTs directly from Spleeter's TensorFlow graph
# (bypassing ISTFT + re-STFT), matching the original training-time pipeline
# in Beat-Transformer/preprocessing/demixing.py.

_TF_EXTRACTOR = None
_MEL_FB = None
_INSTRUMENT_ORDER = ("vocals", "piano", "drums", "bass", "other")  # from 5stems.json


def _get_mel_filter():
    global _MEL_FB
    if _MEL_FB is None:
        import librosa
        _MEL_FB = librosa.filters.mel(
            sr=SR, n_fft=N_FFT, n_mels=128, fmin=30, fmax=11000
        ).T.astype(np.float32)
    return _MEL_FB


def _build_tf_extractor():
    """Build Spleeter's masked-STFT graph once; return (sess, features, masked_stfts)."""
    import json
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    try:
        tf.config.set_visible_devices([], "GPU")  # keep GPU free for torch
    except Exception:
        pass
    from spleeter.model import EstimatorSpecBuilder, InputProviderFactory
    from spleeter.model.provider import ModelProvider

    with open(
        "/home/sogang/mnt/db_2/anaconda3/envs/analyze-smc/lib/python3.10/"
        "site-packages/spleeter/resources/5stems.json"
    ) as f:
        params = json.load(f)
    params["MWF"] = False
    params["stft_backend"] = "tensorflow"

    provider = InputProviderFactory.get(params)
    features = provider.get_input_dict_placeholders()
    builder = EstimatorSpecBuilder(features, params)
    masked_stfts = builder.masked_stfts  # dict instr -> tensor (T, 2049, 2) complex

    mp = ModelProvider.default()
    model_dir = mp.get(params["model_dir"])
    latest = tf.train.latest_checkpoint(model_dir)
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, latest)
    return sess, features, masked_stfts


def _get_tf_extractor():
    global _TF_EXTRACTOR
    if _TF_EXTRACTOR is None:
        _TF_EXTRACTOR = _build_tf_extractor()
    return _TF_EXTRACTOR


def audio_to_demixed_mel(wav_path: Path) -> np.ndarray:
    """Load 44.1kHz stereo audio → Spleeter's masked STFTs (from TF graph)
    → mel spectrogram per stem. Returns shape (T, 5, 128).

    This matches Beat-Transformer's training-time preprocessing exactly,
    bypassing the ISTFT→re-STFT round-trip which loses small amounts of
    amplitude/alignment information.
    """
    from spleeter.audio.adapter import AudioAdapter
    sess, features, masked_stfts = _get_tf_extractor()
    mel_fb = _get_mel_filter()

    adapter = AudioAdapter.default()
    w, _ = adapter.load(str(wav_path), sample_rate=SR)  # (N, C)
    if w.shape[-1] == 1:
        w = np.concatenate([w, w], axis=-1)
    elif w.shape[-1] > 2:
        w = w[:, :2]

    results = sess.run(
        masked_stfts,
        feed_dict={
            features["waveform"]: w.astype(np.float32),
            features["audio_id"]: str(wav_path.stem),
        },
    )

    mel_per_stem = []
    for name in _INSTRUMENT_ORDER:
        stft = results[name]                     # (T, 2049, 2) complex
        complex_avg = np.mean(stft, axis=-1)     # (T, 2049)
        mag = np.abs(complex_avg)
        mel = (mag ** 2) @ mel_fb                # (T, 128)
        mel_per_stem.append(mel.astype(np.float32))

    T = min(m.shape[0] for m in mel_per_stem)
    specs = np.stack([m[:T] for m in mel_per_stem], axis=0)  # (5, T, 128)
    specs = specs.transpose(1, 0, 2)                         # (T, 5, 128)
    return specs


def mel_to_model_input(specs: np.ndarray) -> torch.Tensor:
    """(T, 5, 128) linear mel → log-compressed model input (1, 5, T, 128)."""
    import librosa
    x = specs.transpose(1, 2, 0)                 # (5, 128, T)
    x_db = np.stack([librosa.power_to_db(x[i], ref=np.max) for i in range(5)], axis=0)  # (5, 128, T)
    x_in = x_db.transpose(0, 2, 1)[None]         # (1, 5, T, 128)
    return torch.from_numpy(x_in.astype(np.float32))


# ──────────────────────── model loader + inference ───────────────────

_MODEL_CACHE: dict[int, torch.nn.Module] = {}


def get_bt_model(fold: int) -> torch.nn.Module:
    if fold not in _MODEL_CACHE:
        from DilatedTransformer import Demixed_DilatedTransformerModel
        model = Demixed_DilatedTransformerModel(**MODEL_KW)
        ckpt_path = CKPT_DIR / f"fold_{fold}_trf_param.pt"
        sd = torch.load(str(ckpt_path), map_location="cpu")["state_dict"]
        model.load_state_dict(sd)
        model.to(DEVICE)
        model.eval()
        _MODEL_CACHE[fold] = model
    return _MODEL_CACHE[fold]


@torch.no_grad()
def run_beat_transformer(wav_path: Path, fold: int):
    """Run Beat Transformer on a single track with its held-out fold checkpoint.
    Returns: beat_act (T,), downbeat_act (T,), tempo_logits (300,), tempo_bpm (float)
    """
    specs = audio_to_demixed_mel(wav_path)       # (T, 5, 128)
    x = mel_to_model_input(specs).to(DEVICE)     # (1, 5, T, 128)
    model = get_bt_model(fold)
    pred, tempo_logits = model(x)
    beat_act = torch.sigmoid(pred[0, :, 0]).detach().cpu().numpy()
    downbeat_act = torch.sigmoid(pred[0, :, 1]).detach().cpu().numpy()
    t_logits = tempo_logits[0].detach().cpu().numpy()  # (300,)
    tempo_bpm = float(int(np.argmax(t_logits)))  # index == BPM (integer)
    return beat_act, downbeat_act, t_logits, tempo_bpm


# ────────────────────────── DBN wrappers ─────────────────────────────

def bt_own_beats(beat_act: np.ndarray, downbeat_act: np.ndarray,
                 min_bpm: float = 30.0, max_bpm: float = 215.0) -> np.ndarray:
    """DBNBeatTrackingProcessor on Beat Transformer activations. We widen
    Beat Transformer's default [55, 215] range to [30, 215] because SMC
    contains many tracks with tempo <55 BPM (21% of tracks) which get
    forced into double-tempo by the default setting."""
    from madmom.features.beats import DBNBeatTrackingProcessor
    min_bpm = max(30.0, float(min_bpm))
    max_bpm = min(300.0, float(max_bpm))
    if max_bpm <= min_bpm + 5:
        max_bpm = min_bpm + 5
    tracker = DBNBeatTrackingProcessor(
        min_bpm=min_bpm, max_bpm=max_bpm, fps=BT_FPS,
        transition_lambda=100, observation_lambda=6,
        num_tempi=None, threshold=0.2,
    )
    return tracker(beat_act)


def bt_own_beats_viterbi_select(beat_act: np.ndarray, tempo_logits: np.ndarray,
                                 K: int = 5, pct: float = 0.20) -> np.ndarray:
    """Wide-DBN baseline + top-K tempo candidates; pick beats with highest
    Viterbi log-probability (user's idea: let the HMM tell us which
    candidate fits the activations best, like madmom's 3/4-vs-4/4 logic)."""
    from madmom.features.beats import DBNBeatTrackingProcessor, threshold_activations

    def decode(mn: float, mx: float):
        mn = max(30.0, float(mn)); mx = min(300.0, float(mx))
        if mx <= mn + 5:
            mx = mn + 5
        dbn = DBNBeatTrackingProcessor(
            min_bpm=mn, max_bpm=mx, fps=BT_FPS,
            transition_lambda=100, observation_lambda=6,
            num_tempi=None, threshold=0.2,
        )
        act, first = threshold_activations(beat_act, 0.2)
        if not act.any():
            return np.empty(0), -np.inf
        path, lp = dbn.hmm.viterbi(act)
        if not path.any():
            return np.empty(0), -np.inf
        br = dbn.om.pointers[path]
        idx = np.nonzero(np.diff(br))[0] + 1
        if br[0]: idx = np.r_[0, idx]
        if br[-1]: idx = np.r_[idx, br.size]
        peaks = []
        if idx.any():
            for l, r in idx.reshape((-1, 2)):
                peaks.append(np.argmax(act[l:r]) + l)
        return (np.array(peaks) + first) / BT_FPS, lp

    # NMS peaks from tempo logits (≥ 8 BPM apart, BPM ≥ 20)
    order = np.argsort(-tempo_logits)
    picked = []
    for i in order:
        if i < 20: continue
        if all(abs(i - j) >= 8 for j in picked):
            picked.append(int(i))
        if len(picked) >= K:
            break

    # Wide baseline + narrow candidates
    candidates = [decode(30.0, 215.0)]
    for b in picked:
        candidates.append(decode(b * (1 - pct), b * (1 + pct)))
    best = max(candidates, key=lambda x: x[1])
    return best[0]


def tempo_topk_range(tempo_logits: np.ndarray, K: int = 6, nms_dist: int = 8,
                      min_bpm_floor: int = 20) -> tuple[float, float] | None:
    """Return (min_bpm, max_bpm) = ±20% envelope around the union of top-K
    NMS-filtered tempo-head peaks. Used for 'top-K union' constraint."""
    order = np.argsort(-tempo_logits)
    picked = []
    for i in order:
        if i < min_bpm_floor: continue
        if all(abs(i - j) >= nms_dist for j in picked):
            picked.append(int(i))
        if len(picked) >= K:
            break
    if not picked:
        return None
    return (min(picked) * 0.8, max(picked) * 1.2)


def bt_this_constrained_dbn(beat_logits: torch.Tensor, downbeat_logits: torch.Tensor,
                              min_bpm: float, max_bpm: float) -> np.ndarray:
    """Apply madmom's DBN (constrained) to beat_this activations, mirroring
    the pipeline used in run_madmom_tcn_baseline.py for consistency."""
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    min_bpm = max(30.0, float(min_bpm))
    max_bpm = min(300.0, float(max_bpm))
    if max_bpm <= min_bpm + 5:
        max_bpm = min_bpm + 5
    dbn = DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4],
        min_bpm=min_bpm, max_bpm=max_bpm, fps=BT_THIS_FPS,
        transition_lambda=100,
    )
    beat_prob = beat_logits.double().sigmoid().cpu().numpy()
    downbeat_prob = downbeat_logits.double().sigmoid().cpu().numpy()
    eps = 1e-5
    beat_prob = beat_prob * (1 - eps) + eps / 2
    downbeat_prob = downbeat_prob * (1 - eps) + eps / 2
    combined = np.vstack((
        np.maximum(beat_prob - downbeat_prob, eps / 2),
        downbeat_prob,
    )).T
    out = dbn(combined)
    return out[:, 0]  # beat times


# ────────────────────── evaluation helpers ───────────────────────────

import mir_eval

def load_gt(path: Path) -> np.ndarray:
    return np.array([float(line.strip()) for line in open(path) if line.strip()])


def evaluate_beats(ref: np.ndarray, est: np.ndarray) -> dict:
    keys = ["F-measure", "Correct Metric Level Total", "Any Metric Level Total"]
    if len(ref) == 0 or len(est) == 0:
        return {k: 0.0 for k in keys}
    res = mir_eval.beat.evaluate(ref, est)
    return {k: res[k] for k in keys}


def gt_tempo_from_beats(ref: np.ndarray) -> float:
    if len(ref) < 2:
        return 0.0
    return 60.0 / float(np.median(np.diff(ref)))


def classify_tempo(ratio: float) -> str:
    if 0.875 <= ratio <= 1.125:
        return "correct"
    if 1.75 <= ratio <= 2.25:
        return "double"
    if 0.4375 <= ratio <= 0.5625:
        return "half"
    return "other"


# ───────────────────────────── main ──────────────────────────────────

def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0,
                        help="shard index (for multi-GPU parallelism)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="total number of shards")
    parser.add_argument("--inference-only", action="store_true",
                        help="run BT inference on this shard and exit (skip eval)")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip inference; only run DBN decoding + evaluation")
    args = parser.parse_args()

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    CACHE_DIR.mkdir(exist_ok=True)
    BT_BEATS_DIR.mkdir(exist_ok=True)

    fmap = build_bt_fold_map()

    # Collect all SMC tracks that have GT
    track_ids = sorted(p.stem for p in GT_DIR.glob("*.beats"))
    track_ids = [t for t in track_ids if t.lower() in fmap]

    # Apply shard filter to inference list
    shard_track_ids = [t for i, t in enumerate(track_ids)
                       if i % args.num_shards == args.shard]

    # ── Step 1: Beat Transformer inference (cached) ──
    if args.eval_only:
        print(f"[Step 1] Skipped (eval-only mode)")
        to_process = []
    else:
        print(f"[Step 1] Beat Transformer inference on shard {args.shard}/{args.num_shards} "
              f"({len(shard_track_ids)} tracks)...")
        to_process = []
        for tid in shard_track_ids:
            cache = CACHE_DIR / f"{tid}.npz"
            if not cache.exists():
                to_process.append(tid)
        print(f"  {len(shard_track_ids) - len(to_process)} cached in shard, "
              f"{len(to_process)} to process")

    if to_process:
        # Group by fold to reuse loaded checkpoints
        by_fold = defaultdict(list)
        for tid in to_process:
            by_fold[fmap[tid.lower()]].append(tid)

        from tqdm import tqdm
        for fold in sorted(by_fold.keys()):
            tids = by_fold[fold]
            print(f"  Fold {fold}: {len(tids)} tracks")
            _ = get_bt_model(fold)  # load once
            for tid in tqdm(tids, desc=f"fold{fold}"):
                wav_path = AUDIO_DIR / f"{tid.upper()}.wav"
                if not wav_path.exists():
                    print(f"    MISSING audio: {wav_path}")
                    continue
                try:
                    beat_act, db_act, t_logits, t_bpm = run_beat_transformer(wav_path, fold)
                except Exception as e:
                    print(f"    ERROR on {tid}: {e}")
                    continue
                np.savez(
                    str(CACHE_DIR / f"{tid}.npz"),
                    beat_act=beat_act.astype(np.float32),
                    downbeat_act=db_act.astype(np.float32),
                    tempo_logits=t_logits.astype(np.float32),
                    tempo_bpm=np.float32(t_bpm),
                    fold=np.int32(fold),
                )
            # Free model after fold is done to save GPU memory
            del _MODEL_CACHE[fold]
            torch.cuda.empty_cache()

    if args.inference_only:
        print(f"\nInference-only mode: finished shard {args.shard}.")
        return

    # ── Step 2: Evaluate all variants ──
    print(f"\n[Step 2] Decoding DBNs and evaluating...")
    from tqdm import tqdm
    rows = []
    for tid in tqdm(track_ids):
        cache = CACHE_DIR / f"{tid}.npz"
        if not cache.exists():
            continue
        d = np.load(str(cache))
        beat_act = d["beat_act"]
        db_act = d["downbeat_act"]
        bt_bpm = float(d["tempo_bpm"])
        # ground truth
        ref = load_gt(GT_DIR / f"{tid}.beats")
        gt_bpm = gt_tempo_from_beats(ref)

        # beat_this activations (from cache)
        bt_cache = BT_ACT_DIR / f"{tid}.npz"
        bt_this = np.load(str(bt_cache))
        beat_logits = torch.from_numpy(bt_this["beat"])
        downbeat_logits = torch.from_numpy(bt_this["downbeat"])

        # Beat Transformer's own beats — paper default DBN [55, 215]
        bt_own_paper = bt_own_beats(beat_act, db_act, min_bpm=55.0, max_bpm=215.0)
        # Beat Transformer's own beats — WIDE DBN [30, 215] (our recommendation)
        bt_own = bt_own_beats(beat_act, db_act)  # defaults to [30, 215]
        np.savetxt(str(BT_BEATS_DIR / f"{tid}.beats"), bt_own, fmt="%.4f")
        # Viterbi-selected narrow candidates (user's idea)
        t_logits = d["tempo_logits"]
        bt_own_viterbi = bt_own_beats_viterbi_select(beat_act, t_logits, K=5, pct=0.20)
        # BT activations + BT-tempo-constrained DBN (top-1 ±20%)
        if bt_bpm > 0:
            bt_own_btconstr = bt_own_beats(beat_act, db_act, bt_bpm * 0.8, bt_bpm * 1.2)
        else:
            bt_own_btconstr = bt_own
        # BT activations + oracle-tempo-constrained DBN
        if gt_bpm > 0:
            bt_own_oracle = bt_own_beats(beat_act, db_act, gt_bpm * 0.8, gt_bpm * 1.2)
        else:
            bt_own_oracle = bt_own

        # Beat_this raw (from existing output files) and + DBN (existing) — read cached outputs
        from_bt_raw = ROOT / "beat_this_output" / f"{tid.upper()}.beats"
        from_bt_dbn = ROOT / "beat_this_output_dbn" / f"{tid.upper()}.beats"
        bt_raw = np.array([float(l.split("\t")[0]) for l in open(from_bt_raw) if l.strip()]) if from_bt_raw.exists() else np.array([])
        bt_dbn = np.array([float(l.split("\t")[0]) for l in open(from_bt_dbn) if l.strip()]) if from_bt_dbn.exists() else np.array([])

        # beat_this + wide DBN [30, 215] (same fix as BT)
        bt_wide = bt_this_constrained_dbn(beat_logits, downbeat_logits, 30.0, 215.0)

        # BT-tempo-constrained DBN on beat_this (top-1 ±20%)
        if bt_bpm > 0:
            bt_btconstr = bt_this_constrained_dbn(
                beat_logits, downbeat_logits, bt_bpm * 0.8, bt_bpm * 1.2,
            )
        else:
            bt_btconstr = bt_dbn

        # Top-K union BT-tempo-constrained DBN on beat_this (K=6)
        topk_rng = tempo_topk_range(t_logits, K=6)
        if topk_rng is not None:
            bt_topk = bt_this_constrained_dbn(
                beat_logits, downbeat_logits, topk_rng[0], topk_rng[1],
            )
        else:
            bt_topk = bt_dbn

        # Oracle (ground-truth-tempo-constrained) DBN on beat_this
        if gt_bpm > 0:
            bt_oracle = bt_this_constrained_dbn(
                beat_logits, downbeat_logits, gt_bpm * 0.8, gt_bpm * 1.2,
            )
        else:
            bt_oracle = bt_dbn

        # Evaluate
        s_bt_transformer_paper = evaluate_beats(ref, bt_own_paper)
        s_bt_transformer = evaluate_beats(ref, bt_own)
        s_bt_transformer_viterbi = evaluate_beats(ref, bt_own_viterbi)
        s_bt_transformer_btc = evaluate_beats(ref, bt_own_btconstr)
        s_bt_transformer_orc = evaluate_beats(ref, bt_own_oracle)
        s_bt_topk = evaluate_beats(ref, bt_topk)
        s_bt_raw = evaluate_beats(ref, bt_raw)
        s_bt_dbn = evaluate_beats(ref, bt_dbn)
        s_bt_wide = evaluate_beats(ref, bt_wide)
        s_bt_btconstr = evaluate_beats(ref, bt_btconstr)
        s_bt_oracle = evaluate_beats(ref, bt_oracle)

        ratio = bt_bpm / gt_bpm if gt_bpm > 0 else 0.0
        rows.append({
            "track_id": tid,
            "gt_bpm": round(gt_bpm, 2),
            "bt_transformer_bpm": round(bt_bpm, 2),
            "tempo_ratio": round(ratio, 4),
            "tempo_class": classify_tempo(ratio),
            # Beat Transformer — paper default DBN [55, 215] (for reference)
            "bt_transformer_paper_F": round(s_bt_transformer_paper["F-measure"], 4),
            "bt_transformer_paper_CMLt": round(s_bt_transformer_paper["Correct Metric Level Total"], 4),
            "bt_transformer_paper_AMLt": round(s_bt_transformer_paper["Any Metric Level Total"], 4),
            # Beat Transformer — wide DBN [30, 215] (our recommendation)
            "bt_transformer_F": round(s_bt_transformer["F-measure"], 4),
            "bt_transformer_CMLt": round(s_bt_transformer["Correct Metric Level Total"], 4),
            "bt_transformer_AMLt": round(s_bt_transformer["Any Metric Level Total"], 4),
            # Beat Transformer — wide + Viterbi-selected top-5 narrow (user's idea)
            "bt_transformer_viterbi_F": round(s_bt_transformer_viterbi["F-measure"], 4),
            "bt_transformer_viterbi_CMLt": round(s_bt_transformer_viterbi["Correct Metric Level Total"], 4),
            "bt_transformer_viterbi_AMLt": round(s_bt_transformer_viterbi["Any Metric Level Total"], 4),
            # Beat Transformer activations + BT-tempo-constrained DBN
            "bt_transformer_btconstr_F": round(s_bt_transformer_btc["F-measure"], 4),
            "bt_transformer_btconstr_CMLt": round(s_bt_transformer_btc["Correct Metric Level Total"], 4),
            "bt_transformer_btconstr_AMLt": round(s_bt_transformer_btc["Any Metric Level Total"], 4),
            # Beat Transformer activations + oracle-tempo-constrained DBN
            "bt_transformer_oracle_F": round(s_bt_transformer_orc["F-measure"], 4),
            "bt_transformer_oracle_CMLt": round(s_bt_transformer_orc["Correct Metric Level Total"], 4),
            "bt_transformer_oracle_AMLt": round(s_bt_transformer_orc["Any Metric Level Total"], 4),
            # beat_this raw
            "bt_F": round(s_bt_raw["F-measure"], 4),
            "bt_CMLt": round(s_bt_raw["Correct Metric Level Total"], 4),
            "bt_AMLt": round(s_bt_raw["Any Metric Level Total"], 4),
            # beat_this + unconstrained DBN [55, 215] (default)
            "bt_dbn_F": round(s_bt_dbn["F-measure"], 4),
            "bt_dbn_CMLt": round(s_bt_dbn["Correct Metric Level Total"], 4),
            "bt_dbn_AMLt": round(s_bt_dbn["Any Metric Level Total"], 4),
            # beat_this + wide DBN [30, 215]
            "bt_wide_F": round(s_bt_wide["F-measure"], 4),
            "bt_wide_CMLt": round(s_bt_wide["Correct Metric Level Total"], 4),
            "bt_wide_AMLt": round(s_bt_wide["Any Metric Level Total"], 4),
            # beat_this + Beat Transformer tempo-constrained DBN (top-1 ±20%)
            "bt_btconstr_F": round(s_bt_btconstr["F-measure"], 4),
            "bt_btconstr_CMLt": round(s_bt_btconstr["Correct Metric Level Total"], 4),
            "bt_btconstr_AMLt": round(s_bt_btconstr["Any Metric Level Total"], 4),
            # beat_this + top-K=6 tempo-union DBN
            "bt_topk_F": round(s_bt_topk["F-measure"], 4),
            "bt_topk_CMLt": round(s_bt_topk["Correct Metric Level Total"], 4),
            "bt_topk_AMLt": round(s_bt_topk["Any Metric Level Total"], 4),
            # beat_this + oracle-tempo-constrained DBN
            "bt_oracle_F": round(s_bt_oracle["F-measure"], 4),
            "bt_oracle_CMLt": round(s_bt_oracle["Correct Metric Level Total"], 4),
            "bt_oracle_AMLt": round(s_bt_oracle["Any Metric Level Total"], 4),
        })

    # ── Write CSV ──
    if not rows:
        print("No rows produced; exiting.")
        return
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print("TEMPO ACCURACY (Beat Transformer vs ground truth)")
    print(f"{'='*80}")
    cnt = Counter(r["tempo_class"] for r in rows)
    total = len(rows)
    for cls in ("correct", "double", "half", "other"):
        n = cnt.get(cls, 0)
        print(f"  {cls:<8} {n:>4} ({100*n/total:>5.1f}%)")
    print(f"  (reference: madmom TCN on SMC was 'correct' on 22% of tracks)")

    print(f"\n{'='*80}")
    print("AGGREGATE PERFORMANCE")
    print(f"{'='*80}")
    print(f"  {'System':<32} {'Mean F':>8} {'Med F':>8} {'CMLt':>8} {'AMLt':>8}")
    print("  " + "-" * 70)
    for name, fk, ck, ak in [
        ("BT acts + DBN[55,215] (paper)",   "bt_transformer_paper_F",    "bt_transformer_paper_CMLt",    "bt_transformer_paper_AMLt"),
        ("BT acts + DBN[30,215] (wide)",    "bt_transformer_F",          "bt_transformer_CMLt",          "bt_transformer_AMLt"),
        ("BT acts + Viterbi-select",        "bt_transformer_viterbi_F",  "bt_transformer_viterbi_CMLt",  "bt_transformer_viterbi_AMLt"),
        ("BT acts + BT-tempo top-1 ±20%",   "bt_transformer_btconstr_F", "bt_transformer_btconstr_CMLt", "bt_transformer_btconstr_AMLt"),
        ("BT acts + oracle GT-tempo",       "bt_transformer_oracle_F",   "bt_transformer_oracle_CMLt",   "bt_transformer_oracle_AMLt"),
        ("beat_this (no DBN)",              "bt_F",                      "bt_CMLt",                      "bt_AMLt"),
        ("beat_this + DBN [55,215] (def)",  "bt_dbn_F",                  "bt_dbn_CMLt",                  "bt_dbn_AMLt"),
        ("beat_this + DBN [30,215] (wide)", "bt_wide_F",                 "bt_wide_CMLt",                 "bt_wide_AMLt"),
        ("beat_this + BT-tempo top-1 ±20%", "bt_btconstr_F",             "bt_btconstr_CMLt",             "bt_btconstr_AMLt"),
        ("beat_this + BT-tempo top-6 union","bt_topk_F",                 "bt_topk_CMLt",                 "bt_topk_AMLt"),
        ("beat_this + oracle GT-tempo",     "bt_oracle_F",               "bt_oracle_CMLt",               "bt_oracle_AMLt"),
    ]:
        fs = [r[fk] for r in rows]
        cs = [r[ck] for r in rows]
        a_s = [r[ak] for r in rows]
        print(f"  {name:<32} {np.mean(fs):>8.4f} {np.median(fs):>8.4f} {np.mean(cs):>8.4f} {np.mean(a_s):>8.4f}")

    print(f"\n{'='*80}")
    print("CONSTRAINT EFFECT CONDITIONED ON TEMPO ACCURACY")
    print(f"{'='*80}")
    print(f"  {'Tempo class':<12} {'N':>4} {'bt_F':>8} {'bt_dbn_F':>10} {'bt_btconstr_F':>14} {'bt_oracle_F':>12}")
    print("  " + "-" * 66)
    for cls in ("correct", "double", "half", "other"):
        subset = [r for r in rows if r["tempo_class"] == cls]
        if not subset:
            continue
        print(f"  {cls:<12} {len(subset):>4} "
              f"{np.mean([r['bt_F'] for r in subset]):>8.4f} "
              f"{np.mean([r['bt_dbn_F'] for r in subset]):>10.4f} "
              f"{np.mean([r['bt_btconstr_F'] for r in subset]):>14.4f} "
              f"{np.mean([r['bt_oracle_F'] for r in subset]):>12.4f}")

    # Compare to TCN experiment if CSV exists
    tcn_csv = DATA_DIR / "tcn_tempo_constrained_results.csv"
    if tcn_csv.exists():
        print(f"\n{'='*80}")
        print("THREE-POINT CURVE: TEMPO ACCURACY → CONSTRAINT EFFECTIVENESS")
        print(f"{'='*80}")
        tcn_rows = list(csv.DictReader(open(tcn_csv)))
        tcn_correct = sum(1 for r in tcn_rows if r["tempo_class"] == "correct")
        print(f"  {'Estimator':<24} {'Correct':>12} {'bt+constr mean F':>20} {'bt+DBN mean F':>18}")
        print("  " + "-" * 76)
        tcn_constr = np.mean([float(r["bt_tcnconstr_F"]) for r in tcn_rows])
        tcn_dbn = np.mean([float(r["bt_dbn_F"]) for r in tcn_rows])
        print(f"  {'madmom TCN':<24} {tcn_correct}/{len(tcn_rows)} ({100*tcn_correct/len(tcn_rows):.0f}%)      "
              f"{tcn_constr:>20.4f} {tcn_dbn:>18.4f}")
        bt_correct = cnt.get("correct", 0)
        bt_constr_mean = np.mean([r["bt_btconstr_F"] for r in rows])
        bt_dbn_mean = np.mean([r["bt_dbn_F"] for r in rows])
        print(f"  {'Beat Transformer':<24} {bt_correct}/{len(rows)} ({100*bt_correct/len(rows):.0f}%)      "
              f"{bt_constr_mean:>20.4f} {bt_dbn_mean:>18.4f}")
        bt_oracle_mean = np.mean([r["bt_oracle_F"] for r in rows])
        print(f"  {'oracle (GT tempo)':<24} {'217/217 (100%)':>14}      {bt_oracle_mean:>20.4f} {bt_dbn_mean:>18.4f}")


if __name__ == "__main__":
    main()
