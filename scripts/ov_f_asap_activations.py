"""
EXPERIMENT F (overnight batch): FOLD-HONEST Beat This activations for asap.  ONLY GPU JOB.

Why asap: 473 expressive piano performances with an 8-folds.split, and (experiment D) an
oracle constant-period metronome ceilings there at F ~ 0.164. It is therefore the corpus
that can actually DISCRIMINATE a latent-tempo model -- unlike ballroom, where a metronome
already scores ~0.984 and every method saturates.

Method: exactly as scripts/exp_tta_full_worker.py does it -- one Audio2Frames /
Spect2Frames checkpoint per fold (`checkpoint_path=f"fold{fold}"`), each track decoded
ONLY by the checkpoint whose fold it is held out of, so the activations are contamination-
free. Input is the precomputed log-mel spectrogram cache
/disk1/jaehoon/beat_this_spect/done/asap.npz, entry "<stem>/track.npy" ([T,128] float16,
50 fps) which is exactly what Spect2Frames consumes.

Output: one .npz per track under /disk1/jaehoon/asap_bt_fold_acts/ with keys
'beat','downbeat' (LOGITS at 50 fps) plus 'fold' -- same layout as
beat_this_activations_cache/, so the SMC harness loaders work unchanged.
Incremental: each track is written as it finishes and existing files are skipped, so the
job is restartable and a crash loses at most one track.

VALIDATION: before touching the GPU, assert that (a) the split file lists 473 stems,
(b) every stem has a "<stem>/track.npy" entry in the spectrogram archive, and (c) the
first spectrogram loads with shape [T,128]. Aborts loudly otherwise.
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch

SPLIT = Path("/disk1/jaehoon/dataset_store/beat_this_annotations/asap/8-folds.split")
SPECT = Path("/disk1/jaehoon/beat_this_spect/done/asap.npz")
OUTDIR_ACT = Path("/disk1/jaehoon/asap_bt_fold_acts")
OUTDIR = Path("/tmp/overnight"); OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "F_asap_activations.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:3")
    args = ap.parse_args()
    t0 = time.time()
    OUTDIR_ACT.mkdir(parents=True, exist_ok=True)

    fold_map = {}
    for line in open(SPLIT):
        line = line.strip()
        if line:
            stem, f = line.split("\t")
            fold_map[stem] = int(f)
    stems = sorted(fold_map)
    print(f"[F] {len(stems)} stems, folds {sorted(set(fold_map.values()))}", flush=True)

    import zipfile
    names = set(zipfile.ZipFile(SPECT).namelist())
    missing = [s for s in stems if f"{s}/track.npy" not in names]
    blob = np.load(SPECT, mmap_mode="r")
    try:
        probe = blob[f"{stems[0]}/track.npy"]
        shape_ok = probe.ndim == 2 and probe.shape[1] == 128
        probe_shape = list(probe.shape)
    except Exception as e:
        shape_ok, probe_shape = False, str(e)

    bad = []
    if len(stems) != 473:
        bad.append(f"split lists {len(stems)} stems, expected 473")
    if missing:
        bad.append(f"{len(missing)} stems missing from spect archive, e.g. {missing[:3]}")
    if not shape_ok:
        bad.append(f"probe spect shape {probe_shape} is not [T,128]")
    state = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_stems": len(stems),
             "device": args.device, "out_dir": str(OUTDIR_ACT),
             "probe_shape": probe_shape,
             "validation": {"ok": not bad, "problems": bad},
             "done": 0, "failed": []}
    OUT.write_text(json.dumps(state, indent=2))
    if bad:
        print("[F] *** VALIDATION FAILED ***\n  " + "\n  ".join(bad), flush=True)
        print("[F] INPUT FORMAT UNUSABLE -- skipping rather than improvising.", flush=True)
        sys.exit(2)
    print("[F] validation OK", flush=True)

    from beat_this.inference import Spect2Frames

    by_fold = {}
    for s in stems:
        by_fold.setdefault(fold_map[s], []).append(s)

    done = 0
    for fold in sorted(by_fold):
        todo = [s for s in sorted(by_fold[fold]) if not (OUTDIR_ACT / f"{s}.npz").exists()]
        print(f"[F] fold {fold}: {len(todo)} to do (of {len(by_fold[fold])})", flush=True)
        if not todo:
            continue
        s2f = Spect2Frames(checkpoint_path=f"fold{fold}", device=args.device)
        for s in todo:
            try:
                spect = torch.from_numpy(
                    np.asarray(blob[f"{s}/track.npy"], dtype=np.float32)).to(args.device)
                b, d = s2f(spect)
                np.savez_compressed(
                    OUTDIR_ACT / f"{s}.npz",
                    beat=b.detach().cpu().numpy().astype(np.float32),
                    downbeat=d.detach().cpu().numpy().astype(np.float32),
                    fold=np.int32(fold))
                done += 1
            except Exception as e:
                state["failed"].append({"stem": s, "fold": fold, "error": repr(e)})
                print(f"[F] FAIL {s}: {e!r}", flush=True)
            if done % 20 == 0:
                state["done"] = done
                OUT.write_text(json.dumps(state, indent=2))
                print(f"[F]  {done} done [{time.time()-t0:.0f}s]", flush=True)
        del s2f
        torch.cuda.empty_cache()

    state["done"] = done
    state["n_cached_total"] = len(list(OUTDIR_ACT.glob("*.npz")))
    OUT.write_text(json.dumps(state, indent=2))
    print(f"[F] DONE {done} new, {state['n_cached_total']} cached total, "
          f"{time.time()-t0:.0f}s -> {OUTDIR_ACT}", flush=True)


if __name__ == "__main__":
    main()
