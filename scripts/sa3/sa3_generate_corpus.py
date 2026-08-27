"""Generate a synthetic beat-annotated corpus for tracker fine-tuning.

Conditioning grids are REAL human annotations from the generator's own training
folds -- not synthetic curves. That gives real tempo and rubato distributions,
avoids testing compliance on the same curve family we trained with, and makes
the labels human annotations rather than something we invented. Only the audio
is new.

Fold discipline: grids come from folds != --eval-fold, and the generator itself
never saw --eval-fold. So nothing derived from the evaluation tracks reaches the
tracker through any path.
"""
import argparse, json, os, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CACHE = Path("/disk1/jaehoon/sa3_beat_cache")
VARIANT = re.compile(r"__(w|tempo)\d+$")

SHARP_PROMPTS = [
    # v2 bank: clear attacks and varied instrumentation. Measured basis: with the
    # soft bank only 21% of generated beats coincided with a detectable onset
    # (real SMC: 37%), and 6 prompts made the least-diverse corpus we measured.
    # Training data needs legible beats; "sound like SMC" was the wrong goal.
    "solo piano, staccato, crisp articulation, clear rhythmic pulse",
    "acoustic guitar, plucked, bright attack, steady rhythm",
    "marimba solo, woody attack, precise mallet strikes",
    "harpsichord, sharp plucked notes, baroque, steady",
    "jazz trio, walking bass, brushed drums, clear beat",
    "folk band, acoustic, strummed guitar, tambourine on the beat",
    "classical string quartet, spiccato bowing, rhythmic drive",
    "solo cello, pizzicato, defined note onsets",
    "flamenco guitar, percussive strumming, palmas claps",
    "ragtime piano, stride left hand, crisp swing",
    "bluegrass, banjo rolls, mandolin chop on the offbeat",
    "accordion, polka, bright bellows accents",
    "vibraphone, clear mallet attack, jazz ballad",
    "upright bass and piano duo, defined walking pulse",
    "handpan, resonant strikes, clear meter",
    "acoustic drum kit, brushes, soft dynamics, precise timing",
    "chamber orchestra, staccato winds, marked downbeats",
    "salsa piano montuno, syncopated, sharp chords",
    "afrobeat guitar, muted strokes, tight groove",
    "reggae piano skank, offbeat chops, clear pulse",
    "celtic fiddle, driving reels, accented bowing",
    "tango ensemble, bandoneon, sharp marcato accents",
    "boogie woogie piano, driving left hand, percussive",
    "gamelan, metallic percussion, interlocking rhythm",
]
PERCUSSIVE_PROMPTS = [
    # Control experiment (2026-08-24). Everything else we generate is slow, rubato
    # and weakly percussive, so a null result cannot distinguish "the ControlNet
    # cannot place beats" from "this music has no beats to place". These prompts
    # ask for the easiest possible beat-tracking material. If the grid is not
    # followed HERE, the failure is the adapter, not the genre.
    "electronic dance track, four on the floor kick, strong steady beat",
    "drum kit groove, loud snare backbeat, hi-hat eighths, tight",
    "funk band, slap bass, punchy drums, tight pocket groove",
    "rock drums, heavy kick and snare, driving straight beat",
    "techno, punchy kick drum, metronomic, relentless pulse",
    "hip hop beat, hard drum machine, boom bap, crisp snare",
    "disco groove, four on the floor, hi-hat offbeats, bass guitar",
    "marching band snare drum, sharp rudiments, strict tempo",
    "latin percussion ensemble, congas and timbales, clave, tight",
    "house music, drum machine, clean kick every beat, steady",
]
PROMPTS = [
    "solo piano, quiet, expressive, intimate recording",
    "ambient piano, soft, no percussion, distant room",
    "slow classical piano, gentle, sparse, unaccented",
    "quiet acoustic guitar, fingerpicked, soft attack",
    "string quartet, slow, legato, chamber recording",
    "solo piano, romantic, rubato, warm reverb",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--cfg-scale", type=float, default=6.0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--match-smc-tempo", action="store_true",
                    help="rescale each conditioning grid toward SMC's tempo "
                         "distribution. Free and label-exact: the grid is warped "
                         "in closed form and the model generates to match, so "
                         "no audio is stretched and no artifacts are introduced. "
                         "Tempo scaling does not change IBI cv, so it costs "
                         "nothing in adherence -- unlike raising rubato depth.")
    ap.add_argument("--smc-bpm-target", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="uniform target range; omit to sample from SMC's own "
                         "empirical BPM distribution, which is what actually "
                         "matches it (median 70.9, p10 44, p90 114)")
    ap.add_argument("--sharp-prompts", action="store_true",
                    help="v2 bank: clear-attack, varied instrumentation")
    ap.add_argument("--percussive-prompts", action="store_true",
                    help="control bank: drums/EDM, the easiest possible beat "
                         "tracking material. Isolates 'can the adapter place "
                         "beats' from 'does this music have beats'.")
    ap.add_argument("--trim-tail", action="store_true",
                    help="cut audio at last beat + 0.3 s: no unlabelled music")
    ap.add_argument("--preroll", action="store_true",
                    help="random 0.2..1.2 s before the first beat, so beat 1 is "
                         "never at t=0 (was an exploitable cue: 100%% of v1 clips)")
    args = ap.parse_args()

    import soundfile as sf, torch
    from sa3_beat.build import load_beat_model
    from sa3_beat.generate import generate, load_adapter, _patch_conditioning_inputs
    from sa3_beat.data import load_beats, warp_times

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}"
    model, cfg = load_beat_model(device=dev)
    load_adapter(model, args.adapter, dev)
    _patch_conditioning_inputs(model)
    sr = cfg["sample_rate"]

    # real grids from the generator's OWN training folds
    pool = [l.strip()[:-4] for l in open(CACHE / f"train_excl_fold{args.eval_fold}.txt")
            if l.strip() and not VARIANT.search(l.strip()[:-4])]
    pool = [u for u in pool if u.startswith(("smc__", "asap__", "rwc__", "hainsworth__"))]
    pool_all_smc = [u for u in pool if u.startswith("smc__")]
    rng = np.random.RandomState(1234 + args.shard)

    # Empirical SMC tempo distribution, from the training-fold SMC grids only.
    smc_bpm = []
    for u in pool_all_smc:
        bb, _ = load_beats(CACHE / f"{u}.beats")
        bb = bb[bb < 120]
        if len(bb) >= 8:
            smc_bpm.append(60.0 / float(np.median(np.diff(bb))))
    smc_bpm = np.array(smc_bpm)
    print(f"[gen] SMC tempo prior from {len(smc_bpm)} grids: "
          f"median {np.median(smc_bpm):.1f} p10 {np.percentile(smc_bpm,10):.1f} "
          f"p90 {np.percentile(smc_bpm,90):.1f}", flush=True)
    print(f"[gen] {len(pool)} candidate grid sources; shard {args.shard}/{args.shards}", flush=True)

    made = 0
    for i in range(args.n):
        if i % args.shards != args.shard:
            continue
        uid = pool[rng.randint(len(pool))]
        beats, db = load_beats(CACHE / f"{uid}.beats")
        if len(beats) < 12:
            continue
        # random window, then shift so the first beat sits at t=0 -- generation
        # anchors phase to the audio start, so the conditioning must too.
        start = rng.uniform(0, max(0.1, beats[-1] - args.seconds - 1))
        sel = beats[(beats >= start) & (beats < start + args.seconds)]
        if len(sel) < 8:
            continue
        g = sel - sel[0]
        d = None if db is None else db[(db >= start) & (db < start + args.seconds)] - sel[0]
        if d is not None:
            d = d[d >= 0]
        if args.match_smc_tempo and len(g) > 2:
            cur = 60.0 / float(np.median(np.diff(g)))
            want = (rng.uniform(*args.smc_bpm_target) if args.smc_bpm_target
                    else float(smc_bpm[rng.randint(len(smc_bpm))]))
            # generous clip: the grid is warped in closed form and the audio is
            # generated to match, so a large stretch costs nothing but coverage
            rate = np.clip(cur / want, 0.35, 3.0)      # >1 slows the grid down
            g = warp_times(g, 1.0 / rate)
            if d is not None and len(d):
                d = warp_times(d, 1.0 / rate)
            keep = g < args.seconds
            g = g[keep]
            if d is not None:
                d = d[d < args.seconds]
            if len(g) < 8:
                continue
        bank = (PERCUSSIVE_PROMPTS if args.percussive_prompts
                else SHARP_PROMPTS if args.sharp_prompts else PROMPTS)
        prompt = bank[rng.randint(len(bank))]
        if args.preroll and len(g):
            off = rng.uniform(0.2, 1.2)
            g = g + off
            if d is not None and len(d):
                d = d + off
            keep = g < args.seconds
            g = g[keep]
            if d is not None:
                d = d[d < args.seconds]
            if len(g) < 8:
                continue
        # Draw the seed BEFORE the resume check, not in the generate() call:
        # skipping must consume exactly as much of the RNG stream as generating,
        # or every clip after the first skip diverges from a clean run.
        seed = int(rng.randint(1 << 30))
        name = f"gen_{args.shard:02d}_{i:05d}"
        # A triple counts as done only if all three files exist AND are non-empty.
        # The 2026-08-23 crash left 22 zero-byte sidecars, so an exists() check
        # alone would silently keep truncated labels.
        if all((out / f"{name}.{e}").exists() and (out / f"{name}.{e}").stat().st_size > 0
               for e in ("wav", "beats", "json")):
            made += 1
            continue
        wav, _ = generate(model, cfg, prompt, g, d, args.seconds,
                          args.steps, args.cfg_scale, seed=seed, device=dev)
        wav = wav / (np.abs(wav).max() + 1e-9)
        if args.trim_tail and len(g):
            end = int(min(wav.shape[-1], (g[-1] + 0.3) * sr))
            wav = wav[..., :end] if wav.ndim > 1 else wav[:end]

        def _atomic(ext, writer):
            tmp = out / f".{name}.{ext}.tmp"
            writer(tmp)
            os.replace(tmp, out / f"{name}.{ext}")

        _atomic("beats", lambda p: p.write_text(
            "\n".join(f"{t:.6f}\t{1 if (d is not None and len(d) and np.min(np.abs(d-t))<1e-6) else 2}"
                      for t in g) + "\n"))
        _atomic("json", lambda p: json.dump(
            {"source_grid": uid, "prompt": prompt, "window_start": float(start)}, open(p, "w")))
        # format= is required: the temp name ends in .tmp, so soundfile cannot
        # infer the container from the extension.
        _atomic("wav", lambda p: sf.write(str(p), wav.T, sr, format="WAV"))
        made += 1
        if made % 25 == 0:
            print(f"  [{made}] {name} <- {uid[:36]}", flush=True)
    print(f"[done] shard {args.shard}: {made} clips", flush=True)


if __name__ == "__main__":
    main()
