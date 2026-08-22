"""Beat Transformer frontend (Zhao, Xia & Wang, ISMIR 2022; external/beat_transformer submodule).

Why this frontend matters for the ladder: Beat Transformer was DESIGNED around the madmom DBN --
its published results are activation + DBN -- whereas Beat This performs best without one. Pairing
the DBN-family rungs with the frontend that was co-designed with the DBN is the fairer baseline
comparison.

Wraps the OFFICIAL code: Demixed_DilatedTransformerModel (code/DilatedTransformer.py) with their
released fold checkpoints, fed their exact input -- Spleeter 5-stem demixed, log-compressed mel
spectrograms. Spleeter needs TensorFlow, which does not coexist with this env's torch stack, so
demixing runs in a SUBPROCESS under a Spleeter-equipped interpreter (see
frontends/beat_transformer_demix.py; default: the analyze-smc env, whose 5-stem weights are
already cached). The torch model itself runs in-process.

Properties of what this emits:
  * native grid is 44100/1024 ~= 43.066 fps (Spleeter's STFT hop); pass target_fps to interpolate.
  * activations are LOGITS (ACTIVATION_FORM="logit"); Beat Transformer's own pipeline applies
    sigmoid and feeds madmom with NO bounding (BOUNDING="none") and decorrelation floor 0
    (eight_fold_test.py: np.maximum(beat - downbeat, 0)).
  * their shipped decode differs from Beat This's madmom call: observation_lambda=6,
    num_tempi=None, threshold=0.2 (and correct=True, madmom's default). To run R0/R1 as
    Beat-Transformer-ships-it, pass those explicitly -- our shipped defaults are Beat This's.
  * the model takes the WHOLE piece in one forward (dilated self-attention; no chunking in their
    inference code).

Checkpoints: "fold_0".."fold_7" -- their 8-fold split over ballroom/hainsworth/carnatic/
harmonix/smc (seed-0 shuffle; NOT Beat This's folds). GTZAN was held out of ALL folds, so any
fold is honest on GTZAN; on the training datasets, fold-honesty needs their fold mapping.
"""
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile
import torch

from frontends import Frontend

_BEAT_TRANSFORMER_ROOT = Path(__file__).resolve().parent.parent / "external" / "beat_transformer"
_DEMIX_SCRIPT = Path(__file__).resolve().parent / "beat_transformer_demix.py"
_DEMIX_PYTHON = "/home/sogang/mnt/db_2/anaconda3/envs/analyze-smc/bin/python"

# Their eight_fold_test.py hyperparameters -- must match the released checkpoints.
_MODEL_KWARGS = dict(attn_len=5, instr=5, ntoken=2, dmodel=256, nhead=8,
                     d_hid=1024, nlayers=9, norm_first=True)


class BeatTransformerFrontend(Frontend):
    OUTPUT_MODES = {"activations": 2}

    ACTIVATION_FORM = "logit"
    BOUNDING = "none"          # their pipeline feeds madmom raw sigmoids, no clip/squeeze,
                               # decorrelation floor exactly 0 (ours: 1e-12 for log safety)
    NATIVE_FPS = 44100 / 1024  # ~= 43.066

    def __init__(self, checkpoint: str = "fold_0", device: str = "cuda",
                 target_fps: Optional[float] = None, output: str = "activations",
                 demix_python: str = _DEMIX_PYTHON, spleeter_model_path: Optional[str] = None):
        if output not in self.OUTPUT_MODES:
            raise KeyError(f"unknown output mode {output!r} for {self.name} "
                           f"(have: {sorted(self.OUTPUT_MODES)})")
        self.output = output

        code_dir = str(_BEAT_TRANSFORMER_ROOT / "code")
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from DilatedTransformer import Demixed_DilatedTransformerModel

        checkpoint_path = _BEAT_TRANSFORMER_ROOT / "checkpoint" / f"{checkpoint}_trf_param.pt"
        if not checkpoint_path.exists():
            available = sorted(p.name.split("_trf_")[0]
                               for p in (_BEAT_TRANSFORMER_ROOT / "checkpoint").glob("*.pt"))
            raise KeyError(f"unknown checkpoint {checkpoint!r} (have: {available})")

        self._model = Demixed_DilatedTransformerModel(**_MODEL_KWARGS)
        state = torch.load(str(checkpoint_path), map_location="cpu")["state_dict"]
        self._model.load_state_dict(state)
        self._model.to(device).eval()

        self.checkpoint = checkpoint
        self.device = device
        self.demix_python = demix_python
        self.spleeter_model_path = spleeter_model_path
        self.fps = target_fps if target_fps is not None else self.NATIVE_FPS

    @torch.no_grad()
    def get_features(self, signal, sample_rate: int) -> torch.Tensor:
        """[num_samples] mono audio -> [num_frames, 2] (beat, downbeat) LOGITS at self.fps."""
        x = self._demix(signal, sample_rate)                                  # [5, T, 128]
        pred, _tempo_logits = self._model(
            torch.from_numpy(x).unsqueeze(0).to(self.device))                 # whole piece
        out = pred[0, :, :2].float().cpu()                                    # [T@43fps, 2]
        if self.fps != self.NATIVE_FPS:
            num_target = int(round(out.shape[0] * self.fps / self.NATIVE_FPS))
            out = torch.nn.functional.interpolate(
                out.t().unsqueeze(0), size=num_target, mode="linear", align_corners=False
            ).squeeze(0).t()
        return out

    def _demix(self, signal, sample_rate: int) -> np.ndarray:
        """[5, T, 128] log-compressed demixed mel via the Spleeter subprocess (see module doc)."""
        with tempfile.TemporaryDirectory(prefix="bt_demix_") as tmp:
            wav_path = Path(tmp) / "in.wav"
            npz_path = Path(tmp) / "out.npz"
            soundfile.write(str(wav_path), np.asarray(signal, dtype=np.float32), sample_rate)
            command = [self.demix_python, str(_DEMIX_SCRIPT), str(wav_path), str(npz_path)]
            if self.spleeter_model_path:
                command += ["--model-path", self.spleeter_model_path]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"spleeter demixing failed "
                                   f"({self.demix_python}):\n{result.stderr[-2000:]}")
            return np.load(str(npz_path))["x"]


if __name__ == "__main__":
    # Smoke test on a real catalog song: track with Beat Transformer's OWN shipped decode
    # (observation_lambda=6, num_tempi=None, threshold=0.2) through R0 and R1.
    from data.songs import iter_songs
    from tracker import Tracker

    song = next(s for s in iter_songs() if s.audio_path is not None)
    signal, sr = soundfile.read(str(song.audio_path), dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frontend = BeatTransformerFrontend(device=device)
    print(f"frontend: {frontend.name} ckpt={frontend.checkpoint} fps={frontend.fps:.3f} "
          f"form={frontend.ACTIVATION_FORM} bounding={frontend.BOUNDING}")
    print(f"song: {song.dataset}/{song.audio_path.name}")

    shipped = dict(observation_lambda=6, num_tempi=None, threshold=0.2)
    for bar_pointer in ("madmom_dbn", "2016_dbn"):
        kwargs = dict(shipped)
        if bar_pointer != "madmom_dbn":
            kwargs["device"] = device
        events = Tracker(frontend, bar_pointer, **kwargs).track(signal, sr)
        n_beats = len(events["beats"])
        ibi = float(np.diff(events["beats"]).mean()) if n_beats > 1 else float("nan")
        print(f"  {bar_pointer:12s}: {n_beats:3d} beats, {len(events['downbeats']):2d} downbeats, "
              f"mean IBI {ibi:.3f}s -> {60.0 / ibi:.1f} BPM")
