"""SMC loader (rungs_lab-only extension): annotations + official 8-folds.split + demix cache.
SMC is BEAT-ONLY. No audio needed -- the BT frontend consumes the era demix mel cache."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
ANN = ROOT / "dataset_store" / "beat_this_annotations" / "smc"
CACHE = ROOT / "cache" / "bt_demix" / "smc"

def load_smc():
    folds = {}
    for line in (ANN / "8-folds.split").read_text().splitlines():
        stem, fold = line.split()
        folds[stem] = int(fold)
    entries = []
    for beats_path in sorted((ANN / "annotations" / "beats").glob("*.beats")):
        stem = beats_path.stem                          # e.g. smc_001
        mel = CACHE / f"SMC_{stem.split('_')[1]}.npz"
        if not mel.exists() or stem not in folds:
            continue
        beat_times = np.loadtxt(beats_path, ndmin=2)[:, 0]
        entries.append({"stem": stem, "fold": folds[stem], "mel_path": mel,
                        "beat_times": beat_times})
    return entries
