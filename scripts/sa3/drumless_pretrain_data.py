"""Build self-supervised beat data: drumless audio in, drum onsets as targets.

The oracle decomposition (2026-08-24) showed 92.4% of SMC's error survives perfect
level and phase correction, so the bottleneck is the frame activations -- the model
is not detecting beats, not merely mis-decoding them. Better activations need more
training signal, and beat-annotated audio is capped at ~4,500 tracks.

Drum stems are free supervision at much larger scale. Separate a drum-bearing
corpus, feed the model the DRUMLESS mix, and ask it to predict where the drums
were. That forces pulse inference from harmony, melody and articulation alone --
which is exactly SMC's task, where beats are acoustically marked but not by drums.

WHAT COULD KILL THIS, and it is measured: the archived percussion work found
Spleeter-drumless training gave +0.108 in-distribution and ZERO transfer to
naturally drumless music, with the two measured as acoustically distinct
(salience 1.81 vs SMC's 2.12). If separation artifacts define a private domain,
this pretraining teaches a skill that does not transfer. Checked by
`drumless_domain_gap.py` before any training is launched.

SMC and ASAP are excluded: SMC is the evaluation set, ASAP is drumless piano with
no drums to learn from.
"""
import argparse, glob, json, os, sys
import numpy as np
import soundfile as sf

CORPORA = ["gtzan", "ballroom", "hjdb", "hainsworth", "beatles", "rwc"]
AUDIO = "/disk1/jaehoon/dataset_store/audio_by_stem"
SR_OUT = 22050


def drum_beats(drums, sr, teacher):
    """Pseudo-beat labels: run the tracker on the ISOLATED DRUM STEM.

    Raw drum onsets are the wrong target -- measured 128-174 per 30 s on ballroom,
    i.e. every hi-hat and snare, 2-3x the beat rate. What we want is the pulse the
    drums imply.

    Teacher and student see DIFFERENT inputs, so this is not the self-training
    trap already refuted here: the teacher's task (beats from isolated drums) is
    near-trivial, the student's (same beats from the drumless mix) is exactly
    SMC's problem. The student cannot copy, it has to infer.
    """
    import soxr, torch
    y = drums.mean(axis=1) if drums.ndim > 1 else drums
    y = soxr.resample(y, sr, SR_OUT) if sr != SR_OUT else y
    energy = float(np.mean(np.abs(y)))
    if energy < 1e-4:
        return np.array([]), energy
    net, spect, post = teacher
    from beat_this.inference import split_predict_aggregate
    with torch.inference_mode():
        pr = split_predict_aggregate(
            spect(torch.from_numpy(np.ascontiguousarray(y)).to(next(net.parameters()).device)),
            1500, 6, "keep_first", net)
        est, _ = post(pr["beat"].float(), pr["downbeat"].float())
    return np.asarray(est), energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/disk1/jaehoon/drumless")
    ap.add_argument("--corpora", nargs="+", default=CORPORA)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--teacher-gpu", type=int, default=0)
    ap.add_argument("--jamendo", action="store_true",
                    help="use the 55,701 UNANNOTATED MTG-Jamendo tracks: the only\n                         corpus where a drum-stem teacher earns its keep")
    ap.add_argument("--seconds-excerpt", type=float, default=0.0,
                    help="take one excerpt of this length instead of the whole track")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.jamendo:
        # 55,701 unannotated tracks. This is the whole point: the annotated
        # corpora already HAVE beat labels, so training on their drumless
        # versions is the archived experiment that transferred at zero. Only
        # unlabelled music makes the drum-stem teacher worth having.
        files = sorted(glob.glob("/disk1/jaehoon/dataset_store/mtg_jamendo/*/*.mp3"))
    else:
        files = []
        for c in a.corpora:
            for ext in ("*.wav", "*.flac", "*.mp3"):
                files += sorted(glob.glob(os.path.join(AUDIO, c, ext)))
        files = sorted(files)
    if a.limit:
        files = files[: a.limit]
    mine = [f for i, f in enumerate(files) if i % a.shards == a.shard]
    print(f"[shard {a.shard}] {len(mine)} of {len(files)} files", flush=True)

    import soxr, torch
    from spleeter.separator import Separator
    from beat_this.inference import load_model
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect
    sep = Separator("spleeter:4stems")
    # 'final0' is fine as the teacher: it never sees SMC's fold 0 at student
    # training time, and it is only ever run on drum stems from OTHER corpora.
    tdev = f"cuda:{a.teacher_gpu}" if torch.cuda.is_available() else "cpu"
    teacher = (load_model("final0", device=tdev).eval(),
               LogMelSpect(device=tdev),
               Postprocessor(type="minimal", fps=50))

    done = 0
    for f in mine:
        stem = os.path.splitext(os.path.basename(f))[0]
        corpus = os.path.basename(os.path.dirname(f))
        base = os.path.join(a.out, f"{corpus}__{stem}")
        if os.path.exists(base + ".json") and os.path.getsize(base + ".json") > 0:
            done += 1
            continue
        try:
            y, sr = sf.read(f, always_2d=True, dtype="float32")
        except Exception:
            continue
        if y.shape[1] == 1:
            y = np.repeat(y, 2, axis=1)
        if len(y) < sr * 5:
            continue
        if a.seconds_excerpt and len(y) > sr * a.seconds_excerpt:
            # one excerpt from the middle: full Jamendo tracks are ~4 min and we
            # want breadth of material over depth per track
            st = int((len(y) - sr * a.seconds_excerpt) // 2)
            y = y[st: st + int(sr * a.seconds_excerpt)]
        try:
            out = sep.separate(y)
        except Exception as e:
            print(f"  [skip] {stem[:40]}: {type(e).__name__}", flush=True)
            continue

        drums = out["drums"]
        rest = out["vocals"] + out["bass"] + out["other"]      # the drumless mix
        times, drum_energy = drum_beats(drums, sr, teacher)
        if len(times) < 8:
            continue

        m = rest.mean(axis=1)
        m = soxr.resample(m, sr, SR_OUT) if sr != SR_OUT else m
        peak = np.abs(m).max()
        if peak > 0:
            m = m / peak * 0.9
        sf.write(base + ".wav.tmp", m, SR_OUT, format="WAV")
        os.replace(base + ".wav.tmp", base + ".wav")
        np.savetxt(base + ".beats", times, fmt="%.6f")
        json.dump({"source": f, "corpus": corpus, "n_beats": int(len(times)),
                   "drum_energy": drum_energy, "seconds": len(m) / SR_OUT},
                  open(base + ".json", "w"))
        done += 1
        if done % 25 == 0:
            print(f"  [shard {a.shard}] {done}/{len(mine)}", flush=True)
    print(f"[shard {a.shard}] DONE {done}", flush=True)


if __name__ == "__main__":
    main()
