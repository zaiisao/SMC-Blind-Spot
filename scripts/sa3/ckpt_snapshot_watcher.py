"""Snapshot each training checkpoint whenever it advances an epoch.

train.py's ModelCheckpoint writes one file per run with no {epoch} in the name,
so every epoch overwrites the last. With --logger none there is also no
metrics.csv, which together means a finished run leaves nothing to plot. This
watcher copies each checkpoint as it changes, stamped with the epoch read from
the file itself, so the arms can be scored per-epoch afterwards.

Snapshots go to /disk1 -- home has only ~80G free and a full run is ~24G.
A checkpoint is only copied once its mtime has been stable for STABLE_S, so we
never copy a file mid-write.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

import torch

CKPT_DIR = Path("/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this/checkpoints")
OUT_DIR = Path("/disk1/jaehoon/ckpt_snapshots")
POLL_S = 45
STABLE_S = 30          # mtime must be this old before we trust the file
MIN_FREE_GB = 120      # stop snapshotting rather than fill a shared disk

RUNS = {
    "vanilla_f0": "vanilla_f0 S0 fold0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt",
    "gated05_f0": "gated05_f0 S0 fold0 gated_phase_bce-h512-augTrueTrueTrue.ckpt",
    "gated10_f0": "gated10_f0 S0 fold0 gated_phase_bce-h512-augTrueTrueTrue.ckpt",
    # second vanilla seed: the null distribution for every arm comparison
    "vanillaS1_f0": "vanillaS1_f0 S1 fold0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt",
    # generated-data arms: stock loss, +471 evidence-filtered / +999 raw sdedit clips
    "sdfilt_f0": "sdfilt_f0 S0 fold0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt",
    "sdall_f0": "sdall_f0 S0 fold0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt",
    # seeds 2-4: with 0 and 1 these give five draws of the same config, so the
    # seed spread can be reported with an interval instead of asserted from one pair
    "vanillaS2_f0": "vanillaS2_f0 S2 fold0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt",
    "vanillaS3_f0": "vanillaS3_f0 S3 fold0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt",
    "vanillaS4_f0": "vanillaS4_f0 S4 fold0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt",
    # the improved generated corpus, two seeds -- matched in dose and sources to sdall
    "r3corpus_s0": "r3corpus_s0 S0 fold0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt",
    "r3corpus_s1": "r3corpus_s1 S1 fold0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt",
    # tracker-tuned corpus: the first generator whose output does not lose F
    # against its own source audio (+0.028 vs r3's -0.141)
    "tf1corpus_s0": "tf1corpus_s0 S0 fold0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt",
    "tf1corpus_s1": "tf1corpus_s1 S1 fold0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt",
    # U-shaped tempo pool at ~21% dose -- the two axes never varied before
    "ucorpus_s0": "ucorpus_s0 S0 fold0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt",
    "ucorpus_s1": "ucorpus_s1 S1 fold0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt",
    "ucorpus_s2": "ucorpus_s2 S2 fold0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt",
}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def read_epoch(path):
    try:
        d = torch.load(path, map_location="meta", weights_only=False, mmap=True)
        return int(d.get("epoch", -1))
    except Exception as e:
        log(f"  could not read epoch from {path.name}: {e}")
        return None


def trainers_alive():
    out = subprocess.run(
        ["pgrep", "-af", "launch_scripts/train.py"], capture_output=True, text=True
    ).stdout
    return sum(1 for line in out.splitlines() if "--loss" in line)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"watching {len(RUNS)} runs -> {OUT_DIR}")
    seen = {}
    idle_rounds = 0

    while True:
        alive = trainers_alive()
        for name, fname in RUNS.items():
            src = CKPT_DIR / fname
            if not src.exists():
                continue
            mtime = src.stat().st_mtime
            if time.time() - mtime < STABLE_S:
                continue                       # still being written
            if seen.get(name) == mtime:
                continue                       # already handled this epoch

            ep = read_epoch(src)
            if ep is None:
                continue
            dst = OUT_DIR / f"{name}_ep{ep:03d}.ckpt"
            seen[name] = mtime
            if dst.exists():
                continue
            if free_gb(OUT_DIR) < MIN_FREE_GB:
                log(f"  SKIP {dst.name}: only {free_gb(OUT_DIR):.0f}G free on /disk1")
                continue
            tmp = dst.with_suffix(".partial")
            shutil.copy2(src, tmp)
            tmp.rename(dst)
            log(f"  saved {dst.name}  ({dst.stat().st_size/1e9:.2f} GB, "
                f"{free_gb(OUT_DIR):.0f}G free)")

        if alive == 0:
            idle_rounds += 1
            if idle_rounds >= 3:
                log("no trainers left, exiting")
                return
        else:
            idle_rounds = 0
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
