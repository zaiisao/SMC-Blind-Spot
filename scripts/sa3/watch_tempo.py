"""Live per-tempo F for every arm. Reads the per-track CSV dumps, not the log text."""
import csv, glob, re, sys
import numpy as np
from pathlib import Path
ANN = Path('/disk1/jaehoon/dataset_store/beat_this_annotations/smc/annotations/beats')
_bpm = {}
def bpm(k):
    if k not in _bpm:
        t = np.loadtxt(ANN / f'{k}.beats', ndmin=2)[:, 0]
        _bpm[k] = 60.0 / np.median(np.diff(t))
    return _bpm[k]

roots = sys.argv[1:] or ['/disk1/jaehoon/bt_scratch', '/disk1/jaehoon/bt_scratch2',
                         '/disk1/jaehoon/bt_v2runs']
for root in roots:
    for arm in sorted(glob.glob(f'{root}/*/')):
        fs = sorted(glob.glob(f'{arm}/pertrack_step*.csv'),
                    key=lambda p: int(re.search(r'step(\d+)', p).group(1)))
        if not fs:
            continue
        print(f"== {Path(arm.rstrip('/')).name}  ({Path(root).name})")
        print(f"   {'step':>5s} {'ALL':>7s} {'slow':>7s} {'mid':>7s} {'fast':>7s}  {'lvl_slow':>8s}")
        for f in fs:
            s = int(re.search(r'step(\d+)', f).group(1))
            r = list(csv.DictReader(open(f)))
            b = np.array([bpm(x['track']) for x in r])
            F = np.array([float(x['F']) for x in r])
            L = np.array([int(x['level_ok']) for x in r])
            q = np.quantile(b, [1/3, 2/3])
            m = [b <= q[0], (b > q[0]) & (b <= q[1]), b > q[1]]
            print(f"   {s:5d} {F.mean():7.4f} " + " ".join(f"{F[x].mean():7.4f}" for x in m)
                  + f"  {100*L[m[0]].mean():7.0f}%")
        print()
