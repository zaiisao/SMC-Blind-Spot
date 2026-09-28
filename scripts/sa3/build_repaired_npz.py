"""Apply the deterministic spectral repair to EVERY spectrogram in sdedit.npz
(the base track plus all 21 pitch/tempo augmentation variants -- the training
config requires all 22 or the loader silently skips the clip) and write an
uncompressed npz the MemmappedNpzFile loader can read.

The repair itself is scripts/sa3/spectral_repair.py's repair(): per-bin affine
matching to a real-slow-music reference, dynamics expansion, flux-ranked
transient sharpening. Reference is fitted GLOBALLY on real corpora, never
per-clip, so no per-clip information leaks in.

Why this run exists even though the onset-z gate failed: the ballroom control
showed onset-z does not predict trackability (ballroom: z 1.9, F 0.96), so the
gate was not a valid reason to skip training. This settles it empirically.
"""

import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts/sa3")
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
from spectral_repair import fit_reference, repair
from beat_this.dataset.mmnpz import MemmappedNpzFile

BT = Path("/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
SRC = BT / "data/audio/spectrograms/sdedit.npz"
DST = BT / "data/audio/spectrograms/sdedit_rep.npz"


def main():
    print("  fitting reference...", flush=True)
    ref = fit_reference()
    src = MemmappedNpzFile(SRC)
    keys = sorted(src.files)
    print(f"  repairing {len(keys)} spectrograms -> {DST}", flush=True)
    # uncompressed zip of .npy members, which is exactly what numpy's npz is
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_STORED) as z:
        for i, k in enumerate(keys):
            sp = np.asarray(src[k], dtype=np.float32)
            out = repair(sp, ref).astype(np.float16)   # source npz stores fp16
            import io
            buf = io.BytesIO()
            np.save(buf, out)
            z.writestr(k + ".npy", buf.getvalue())
            if i % 2000 == 0:
                print(f"    {i}/{len(keys)}", flush=True)
    print("  verifying...", flush=True)
    chk = MemmappedNpzFile(DST)
    assert set(chk.files) == set(src.files), "key mismatch"
    a = np.asarray(chk[keys[0]])
    assert a.shape == np.asarray(src[keys[0]]).shape and np.isfinite(a).all()
    print(f"  done: {len(chk.files)} entries, shapes match, finite")


if __name__ == "__main__":
    main()
