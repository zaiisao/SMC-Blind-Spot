"""
Run beat_this inference per fold: for each fold N, run the held-out songs
using the fold{N} checkpoint so every song is predicted by a model that
never saw it during training.
"""
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
SPLIT_FILE = ROOT / "beat_this_annotations" / "smc" / "8-folds.split"
AUDIO_DIR = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
OUTPUT_DIR = ROOT / "beat_this_output"

OUTPUT_DIR.mkdir(exist_ok=True)

# Parse fold assignments
folds = defaultdict(list)
with open(SPLIT_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        track_id, fold_num = line.split("\t")
        # smc_001 -> SMC_001.wav
        wav_name = track_id.upper().replace("SMC_", "SMC_") + ".wav"
        wav_path = AUDIO_DIR / wav_name
        if wav_path.exists():
            folds[int(fold_num)].append(str(wav_path))
        else:
            print(f"WARNING: {wav_path} not found, skipping")

for fold_num in sorted(folds.keys()):
    files = folds[fold_num]
    model = f"fold{fold_num}"
    print(f"\n{'='*60}")
    print(f"Fold {fold_num}: running {len(files)} files with checkpoint {model}")
    print(f"{'='*60}")
    cmd = [
        "beat_this",
        *files,
        "-o", str(OUTPUT_DIR),
        "--model", model,
        "--gpu", "0",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: fold {fold_num} failed with return code {result.returncode}")
        sys.exit(1)

print(f"\nDone! {sum(len(v) for v in folds.values())} predictions saved to {OUTPUT_DIR}")
