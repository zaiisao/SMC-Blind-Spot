"""Does a coarser stretch grid over the new -60..-24 range cost tempo coverage?

The audio stage is measured at ~106 wav/min, so each extra stretch value costs
real hours. Beat This ships stride 4 over +/-20; the question is whether the
EXTENSION needs the same density or whether stride 8/12 covers the 45-70 bpm
band just as well. Coverage is what matters, not grid density -- the marginal
is over tracks x stretches, and neighbouring stretch values land ~4% apart in
bpm, well inside any tempo bin we would report.
"""
import glob, os, warnings
import numpy as np
warnings.filterwarnings("ignore")

ANN = "/disk1/jaehoon/dataset_store/beat_this_annotations"
AUD = "/disk1/jaehoon/dataset_store/audio_by_stem"
HAVE = {d for d in os.listdir(AUD) if not d.startswith(".")}
LO, HI, MX = 45.0, 70.0, 1.75

def stretch_needed(b):
    return b / (min(max(b, LO), HI) if b < HI else HI)

base, sel = [], []
for d in sorted(os.listdir(ANN)):
    bd = f"{ANN}/{d}/annotations/beats"
    if not os.path.isdir(bd):
        continue
    for f in glob.glob(bd + "/*.beats"):
        try:
            t = np.loadtxt(f, ndmin=2)
        except Exception:
            continue
        if t.size == 0 or len(t) < 8:
            continue
        ibi = float(np.median(np.diff(t[:, 0])))
        if ibi <= 0:
            continue
        b = 60.0 / ibi
        base.append(b)
        if d in HAVE and stretch_needed(b) <= MX:
            sel.append(b)

base, sel = np.array(base), np.array(sel)
def marg(bpms, ps):
    return np.concatenate([bpms * (1 + p / 100.0) for p in ps])

ship = list(range(-20, 21, 4))
m0 = marg(base, ship)
print(f"base tracks {len(base)}   selected {len(sel)}\n")
print(f"{'extra stretches':<26}{'new files':>10}{'hours@106':>10}{'<70':>8}{'<64':>8}{'<55':>8}")
print(f"{'none (shipped only)':<26}{0:10d}{0:10.1f}{(m0<70).mean():8.3f}{(m0<64).mean():8.3f}{(m0<55).mean():8.3f}")
for stride in (4, 8, 12):
    ex = list(range(-60, -20, stride))
    m = np.concatenate([m0, marg(sel, ex)])
    n = len(sel) * len(ex)
    print(f"{'-60..-24 step %d' % stride:<26}{n:10d}{n/106/60:10.1f}"
          f"{(m<70).mean():8.3f}{(m<64).mean():8.3f}{(m<55).mean():8.3f}")
print(f"\n{'SMC (the target)':<26}{'':>10}{'':>10}{0.475:8.3f}{0.378:8.3f}{0.207:8.3f}")
