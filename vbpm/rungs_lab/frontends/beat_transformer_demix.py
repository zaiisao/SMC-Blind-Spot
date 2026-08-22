"""Spleeter 5-stem demixing for the Beat Transformer frontend. RUNS IN A SEPARATE ENV.

Beat Transformer consumes Spleeter-demixed mel spectrograms, and Spleeter needs TensorFlow --
which does not coexist with the chart env's torch stack. So the frontend shells out to this script
under a Spleeter-equipped interpreter (default: the analyze-smc env, which has spleeter + the
5-stem weights already cached from the SMC-paper analysis).

    <spleeter-env-python> beat_transformer_demix.py in.wav out.npz [--model-path DIR]

Writes out.npz with x: [5, T, 128] float32 -- log-compressed (power_to_db, ref=max, per stem)
demixed mel spectrograms, EXACTLY the Demixed_DilatedTransformerModel input.

The recipe is behavior-copied from the proven extractor in
Analyze-SMC/scripts/run_beat_transformer.py (the SMC blind-spot paper's Beat Transformer runs),
which itself mirrors Beat-Transformer/preprocessing/demixing.py: the masked STFTs are taken
DIRECTLY from Spleeter's TensorFlow graph -- no ISTFT -> re-STFT round-trip -- then
complex-averaged over the stereo channels, magnitude-squared, and mel-projected
(sr 44100, n_fft 4096, hop 1024 -> fps 44100/1024, 128 mels, 30-11000 Hz).
(The upstream preprocessing/demixing.py file itself has a dict-vs-append bug and cannot run
as published; this is the working form of the same math.)
"""
import argparse
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

DEFAULT_MODEL_PATH = "/home/sogang/jaehoon/Analyze-SMC/pretrained_models"
SR, N_FFT = 44100, 4096
INSTRUMENT_ORDER = ("vocals", "piano", "drums", "bass", "other")   # 5stems.json order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="input audio file (anything ffmpeg reads), or with "
                                      "--batch: a list file of '<audio_path>\\t<out_path>' lines")
    parser.add_argument("out", nargs="?", default=None,
                        help="output .npz path (single-file mode)")
    parser.add_argument("--batch", action="store_true",
                        help="batch mode: reuse one TF session across the whole list "
                             "(per-process startup costs ~10s; do not pay it per song)")
    parser.add_argument("--model-path", default=os.environ.get("SPLEETER_MODEL_PATH",
                                                               DEFAULT_MODEL_PATH),
                        help="directory containing spleeter's 5stems weights")
    args = parser.parse_args()
    os.environ["MODEL_PATH"] = args.model_path   # spleeter's ModelProvider reads this

    import librosa
    import numpy as np
    import spleeter
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    try:
        tf.config.set_visible_devices([], "GPU")   # demixing on CPU; GPU stays free for torch
    except Exception:
        pass
    from spleeter.audio.adapter import AudioAdapter
    from spleeter.model import EstimatorSpecBuilder, InputProviderFactory
    from spleeter.model.provider import ModelProvider

    with open(Path(spleeter.__file__).parent / "resources" / "5stems.json") as f:
        params = json.load(f)
    params["MWF"] = False
    params["stft_backend"] = "tensorflow"

    provider = InputProviderFactory.get(params)
    features = provider.get_input_dict_placeholders()
    builder = EstimatorSpecBuilder(features, params)
    masked_stfts = builder.masked_stfts                       # instr -> [T, 2049, 2] complex

    model_dir = ModelProvider.default().get(params["model_dir"])
    sess = tf.compat.v1.Session()
    tf.compat.v1.train.Saver().restore(sess, tf.train.latest_checkpoint(model_dir))

    mel_fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=128,
                                 fmin=30, fmax=11000).T.astype(np.float32)
    adapter = AudioAdapter.default()

    def demix_one(audio_path: str, out_path: str):
        waveform, _ = adapter.load(audio_path, sample_rate=SR)              # [N, C] float32
        if waveform.shape[-1] == 1:
            waveform = np.concatenate([waveform, waveform], axis=-1)
        elif waveform.shape[-1] > 2:
            waveform = waveform[:, :2]
        results = sess.run(masked_stfts, feed_dict={
            features["waveform"]: waveform.astype(np.float32),
            features["audio_id"]: Path(audio_path).stem,
        })
        mel_per_stem = []
        for name in INSTRUMENT_ORDER:
            magnitude = np.abs(np.mean(results[name], axis=-1))   # complex channel average
            mel_per_stem.append(((magnitude ** 2) @ mel_fb).astype(np.float32))   # [T, 128]
        num_frames = min(m.shape[0] for m in mel_per_stem)
        linear = np.stack([m[:num_frames] for m in mel_per_stem])             # [5, T, 128]
        # The model input's hidden log-compression (what spectrogram_dataset does at load time):
        # power_to_db per stem, referenced to that stem's max.
        x = np.stack([librosa.power_to_db(stem.T, ref=np.max).T for stem in linear])
        np.savez(out_path, x=x.astype(np.float32))

    if args.batch:
        pairs = [line.rstrip("\n").split("\t") for line in open(args.audio) if line.strip()]
        for i, (audio_path, out_path) in enumerate(pairs):
            if Path(out_path).exists():
                continue
            try:
                demix_one(audio_path, out_path)
            except Exception as e:                                # keep the batch alive
                print(f"FAILED {audio_path}: {e}", flush=True)
                continue
            if (i + 1) % 10 == 0:
                print(f"[{i + 1}/{len(pairs)}]", flush=True)
        print(f"done: {len(pairs)} entries", flush=True)
    else:
        demix_one(args.audio, args.out)


if __name__ == "__main__":
    main()
