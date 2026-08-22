"""Per-frame TEMPO-BIN labels on the DBN's own tempo grid.

The 2016 emission is tempo-BLIND: a state's class depends only on `state_positions` (position
within the bar), never on `state_interval_frames`. So every tempo hypothesis receives identical
observation evidence at every frame, and all tempo inference comes from the transition prior plus
the timing of beat evidence. This builds the targets for a direct tempo observation channel.

Target = index into the DBN's interval grid (frames-per-beat) nearest the local inter-beat
interval. Frames whose local IBI falls outside the grid are marked -1 and excluded from training.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from build_emission_labels import load_beats, FPS
from rungs.bar_pointer.state_space import BarPointerStateSpace
EMB = Path("/disk1/jaehoon/bt_emb")

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", default="gtzan,ballroom,hainsworth,beatles,asap")
ap.add_argument("--out", default="/disk1/jaehoon/bt_labels_tempo")
a = ap.parse_args()
grid = BarPointerStateSpace(fps=FPS, min_bpm=30., max_bpm=215., beats_per_bar=4,
                            observation_lambda=16, num_tempi=60).interval_frames.astype(float)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
print(f"tempo grid: {len(grid)} bins, {grid.min():.0f}-{grid.max():.0f} frames "
      f"({FPS*60/grid.max():.0f}-{FPS*60/grid.min():.0f} BPM)")

hist = np.zeros(len(grid), np.int64); n_out = n_tot = 0
for ds in [d.strip() for d in a.datasets.split(",")]:
    d = EMB / ds
    if not d.exists(): continue
    od = out / ds; od.mkdir(parents=True, exist_ok=True)
    for f in sorted(d.glob("*.npz")):
        z = np.load(f); T = z["emb"].shape[0]; sf = int(z["start_frame"])
        bt, bn = load_beats(ds, f.stem)
        if bt is None or len(bt) < 3: continue
        fr = bt * FPS - sf; ibi_f = np.diff(bt) * FPS      # local IBI in FRAMES
        lab = np.full(T, -1, np.int64)
        for k in range(len(fr) - 1):
            lo, hi = max(0, int(np.ceil(fr[k]))), min(T, int(np.ceil(fr[k + 1])))
            if hi <= lo: continue
            v = ibi_f[k]
            if v < grid.min() * 0.97 or v > grid.max() * 1.03:
                n_out += hi - lo; n_tot += hi - lo; continue
            lab[lo:hi] = int(np.argmin(np.abs(grid - v))); n_tot += hi - lo
        m = lab >= 0
        hist += np.bincount(lab[m], minlength=len(grid))
        np.savez_compressed(od / f"{f.stem}.npz", label=lab.astype(np.int16), fold=int(z["fold"]))
print(f"labelled frames {int(hist.sum()):,}; outside grid {n_out:,} ({100*n_out/max(n_tot,1):.2f}%)")
nz = (hist > 0).sum()
print(f"occupied bins {nz}/{len(grid)}; top-5 bins hold {100*np.sort(hist)[-5:].sum()/hist.sum():.1f}% of frames")
(out / "_info.json").write_text(json.dumps({"grid": grid.tolist(), "hist": hist.tolist()}, indent=1))
