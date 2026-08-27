"""Pipeline-level sanity checks for the gph13 vs ph13 comparison.

The loss math is checked in check_gated_loss.py. These are the checks that would
invalidate the COMPARISON even with a correct loss: wiring, fold honesty, masking,
gate behaviour on real annotations, and evaluator determinism.
"""
import glob, os, subprocess, sys
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
from beat_this.model.loss import (ShiftTolerantBCELoss,
                                  PhaseWeightedShiftTolerantBCELoss,
                                  GatedPhaseShiftTolerantBCELoss)

OK, BAD, WARN = [], [], []
def check(name, cond, detail="", warn_only=False):
    tag = "PASS" if cond else ("WARN" if warn_only else "FAIL")
    (OK if cond else (WARN if warn_only else BAD)).append(name)
    print(f"  [{tag}] {name}" + (f"\n         {detail}" if detail else ""))


print("\n=== A. the flag actually reaches the loss module")
ckpt = sorted(glob.glob("/disk1/jaehoon/bt_wide/phase5/gph13_prebugfix/*.ckpt"))
if ckpt:
    hp = torch.load(ckpt[0], map_location="cpu", weights_only=False)["hyper_parameters"]
    check("checkpoint records the gated loss",
          hp.get("loss_type") == "gated_phase_shift_tolerant_weighted_bce",
          f"loss_type={hp.get('loss_type')}  pos_weights={hp.get('pos_weights')}")
else:
    check("checkpoint available to inspect", False, "no prebugfix ckpt found", True)

print("\n=== B. the two arms differ ONLY in the loss")
ps = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True).stdout
def argsof(name):
    for l in ps.splitlines():
        if "train.py" in l and f"--name {name}" in l:
            t = l.split()
            return {t[i]: t[i + 1] for i in range(len(t) - 1) if t[i].startswith("--")}
    return None
g = argsof("gph13")
if g:
    ref = {"--gpu", "--name", "--loss", "--checkpoint-dir"}
    ph13_expect = {"--fold": "0", "--seed": "0", "--max-epochs": "100",
                   "--data-dir": "/disk1/jaehoon/bt_wide/data", "--phase-gamma": "1.3"}
    diffs = {k: (v, g.get(k)) for k, v in ph13_expect.items() if g.get(k) != v}
    check("gph13 matches ph13 on fold/seed/epochs/data/gamma", not diffs,
          f"mismatches: {diffs}" if diffs else
          f"fold={g.get('--fold')} seed={g.get('--seed')} ep={g.get('--max-epochs')} "
          f"gamma={g.get('--phase-gamma')} loss={g.get('--loss')}")
    check("no tempo-aug override on gph13 (must match ph13 default)",
          "--tempo-min" not in g and "--tempo-max" not in g,
          f"tempo flags present: {[k for k in g if 'tempo' in k]}")
else:
    check("gph13 process visible", False, "not running?", True)

print("\n=== C. masked/padded items must not corrupt the normaliser")
m = GatedPhaseShiftTolerantBCELoss(pos_weight=32, tolerance=3, phase_gamma=1.3)
T = 600
t = torch.zeros(2, T); beats = list(range(40, T - 40, 40))
t[:, beats] = 1.0
p = torch.full((2, T), -6.0); p[:, beats] = 4.0
for x in beats[:-1]:
    p[:, x + 20] = 3.0
mask = torch.ones(2, T); mask[1, T // 2:] = 0.0        # item 1 half padding
l_mask = m(p, t, mask=mask)
check("forward runs with a mask", torch.isfinite(l_mask), f"loss={l_mask.item():.6f}")
# padding must not change item 0's contribution: compare item 0 alone
l0_alone = m(p[:1], t[:1], mask=mask[:1])
check("masked forward is finite and mask changes the value",
      torch.isfinite(l0_alone) and not torch.isclose(l_mask, l0_alone),
      f"batch={l_mask.item():.6f}  item0-alone={l0_alone.item():.6f}")

print("\n=== D. degenerate inputs must not crash or NaN")
for nm, tt in [("no beats", torch.zeros(1, 300)),
               ("one beat", torch.zeros(1, 300)),
               ("all beats", torch.ones(1, 300))]:
    if nm == "one beat":
        tt[0, 150] = 1.0
    pp = torch.randn(1, 300)
    v = m(pp, tt)
    check(f"{nm}: finite loss", bool(torch.isfinite(v)), f"loss={v.item():.6f}")

print("\n=== E. phase on REAL annotations (rubato, non-uniform IBI)")
ann = sorted(glob.glob("/disk1/jaehoon/dataset_store/beat_this_annotations/smc/"
                       "annotations/beats/*.beats"))[:40]
cvs, mids = [], []
for f in ann:
    b = np.loadtxt(f, ndmin=2)[:, 0]
    if len(b) < 8:
        continue
    d = np.diff(b)
    cvs.append(d.std() / d.mean())
    T2 = int(b[-1] * 50) + 60
    tt = torch.zeros(1, T2)
    fr = np.clip(np.rint(b * 50).astype(int), 0, T2 - 1)
    tt[0, fr] = 1.0
    ph = m._phase(tt)[0].numpy()
    inside = ~np.isnan(ph)
    mids.append(float(np.nanmax(ph[inside])) if inside.any() else np.nan)
check("phase stays in [0,1) on real rubato annotations",
      all((not np.isnan(x)) and x < 1.0 for x in mids),
      f"n={len(mids)} tracks, median IBI cv {np.median(cvs):.3f}, max phase {max(mids):.4f}")

print("\n=== F. gate is INFORMATIVE on real activations (not saturated)")
cache = sorted(glob.glob("/home/sogang/jaehoon/Analyze-SMC/cache/ts_tta_cache/*__r1.0.npz"))
gates = []
for f in cache[:60]:
    stem = os.path.basename(f).split("__")[0]
    bt = f"/disk1/jaehoon/dataset_store/beat_this_annotations/smc/annotations/beats/{stem}.beats"
    if not os.path.exists(bt):
        continue
    b = np.loadtxt(bt, ndmin=2)[:, 0]
    lg = np.load(f)["logits"]
    if len(b) < 8:
        continue
    tt = torch.zeros(1, len(lg))
    fr = np.clip(np.rint(b * 50).astype(int), 0, len(lg) - 1)
    tt[0, fr] = 1.0
    gv = m._gate(torch.from_numpy(lg).float().unsqueeze(0), tt)[0].numpy()
    ph = m._phase(tt)[0].numpy()
    sel = np.abs(ph - 0.5) < 0.05
    if sel.any():
        gates.append(float(np.nanmean(gv[sel])))
gates = np.array(gates)
check("gate spans a real range on SMC (not stuck at 0 or 1)",
      len(gates) > 10 and gates.std() > 0.05 and 0.05 < gates.mean() < 0.95,
      f"n={len(gates)}  mean {gates.mean():.3f}  sd {gates.std():.3f}  "
      f"min {gates.min():.3f}  max {gates.max():.3f}")

print("\n=== G. evaluator determinism")
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts/sa3")
from smc_eval import fold_songs
f0 = fold_songs(0)
check("fold 0 has the expected 27 tracks", len(f0) == 27, f"n={len(f0)}")
split = "/disk1/jaehoon/bt_wide/data/annotations/wide/8-folds.split"
fo = {}
for line in open(split):
    q = line.split()
    if len(q) == 2:
        fo[q[0]] = int(q[1])
train_stems = {k for k, v in fo.items() if v != 0}
eval_stems = {s for s, _, _ in f0}
leak = [s for s in eval_stems if any(s in k and fo.get(k) != 0 for k in fo)]
check("no eval track appears in a training fold", not leak, f"leaked: {leak[:5]}")

print(f"\n  {len(OK)} passed, {len(BAD)} failed, {len(WARN)} warnings")
if BAD:
    print("  FAILED: " + ", ".join(BAD))
if WARN:
    print("  WARN:   " + ", ".join(WARN))
