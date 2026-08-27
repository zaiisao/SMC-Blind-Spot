"""What does the count penalty actually read on a REAL trained model?

The term compares widened prediction mass to widened target mass. A model whose
natural response is wider than one frame per beat inflates the reading, so the
penalty could be dominated by response WIDTH rather than by beat COUNT. Unit
tests showed a 7-frame-wide-but-correct prediction scoring 0.864, versus 1.022
for a genuinely doubled one -- too close for comfort if real models are wide.

Measures, per tempo band, on real SMC audio:
  width     mean frames with sigma>0.5 per annotated beat
  penalty   the count term as the loss would compute it
If penalty tracks the doubling fractions we measured (0.51 at <55, 0.05 at >100)
the term is reading count. If it is uniformly high, it is reading width and
lambda must stay small.
"""
import glob, os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
ANN = "/disk1/jaehoon/dataset_store/beat_this_annotations/smc"
AUDIO = "/disk1/jaehoon/dataset_store/audio_by_stem/smc"
FPS = 50

import soundfile as sf, soxr
from beat_this.inference import split_predict_aggregate
from beat_this.model.loss import CountRegularizedShiftTolerantBCELoss as CR
from beat_this.model.pl_module import PLBeatThis
from beat_this.preprocessing import LogMelSpect

ck_path = sys.argv[1]
gpu = sys.argv[2] if len(sys.argv) > 2 else "0"
dev = f"cuda:{gpu}"
ck = torch.load(ck_path, map_location=dev, weights_only=False)
pl = PLBeatThis(**dict(ck["hyper_parameters"]))
pl.load_state_dict(ck["state_dict"])
net = pl.model.to(dev).eval()
spect = LogMelSpect(device=dev)
crit = CR(pos_weight=32, count_lambda=1.0)

rows = []
for bt in sorted(glob.glob(f"{ANN}/annotations/beats/*.beats")):
    stem = os.path.basename(bt)[:-6]
    wav = None
    for e in (".wav", ".flac", ".mp3"):
        p = os.path.join(AUDIO, stem + e)
        if os.path.exists(p):
            wav = p
            break
    if wav is None:
        continue
    truth = np.loadtxt(bt, ndmin=2)
    if truth.size == 0 or len(truth) < 8:
        continue
    truth = truth[:, 0]
    y, sr = sf.read(wav, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    y22 = soxr.resample(y, sr, 22050) if sr != 22050 else y
    with torch.inference_mode():
        pr = split_predict_aggregate(
            spect(torch.from_numpy(np.ascontiguousarray(y22)).to(dev)),
            1500, 6, "keep_first", net)
        logit = pr["beat"].float().reshape(1, -1)
    n = logit.shape[-1]
    tgt = torch.zeros(1, n, device=dev)
    idx = np.clip((truth * FPS).astype(int), 0, n - 1)
    tgt[0, torch.from_numpy(idx).to(dev)] = 1.0
    m = torch.ones_like(tgt)
    with torch.inference_mode():
        pen = crit._count_penalty(logit, tgt, m).item()
        width = (torch.sigmoid(logit) > 0.5).sum().item() / max(len(truth), 1)
    rows.append((60.0 / float(np.median(np.diff(truth))), pen, width))

b = np.array([r[0] for r in rows]); p = np.array([r[1] for r in rows])
w = np.array([r[2] for r in rows])
print(f"\n  {os.path.basename(ck_path)[:44]}   n={len(rows)}")
print(f"  {'band':>8}{'n':>5}{'width':>9}{'count penalty':>15}")
for lo, hi, tag in [(0, 55, "<55"), (55, 70, "55-70"), (70, 100, "70-100"),
                    (100, 1e9, ">100")]:
    mm = (b >= lo) & (b < hi)
    if mm.sum():
        print(f"  {tag:>8}{mm.sum():5d}{w[mm].mean():9.2f}{p[mm].mean():15.3f}")
print(f"  {'ALL':>8}{len(b):5d}{w.mean():9.2f}{p.mean():15.3f}")
print("\n  width ~1 frame/beat  -> term reads COUNT, lambda can be moderate")
print("  width >>1            -> term reads WIDTH, lambda must be small")
