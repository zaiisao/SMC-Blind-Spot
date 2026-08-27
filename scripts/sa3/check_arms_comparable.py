"""Are the running arms actually comparable to ph13 and to each other?

Checks the things that would silently invalidate the comparison: differing corpus,
fold, seed, epochs, augmentation, pos_weight, or a loss that is not the one the
flag claims. Reads the LIVE process command lines and the checkpoints on disk
rather than trusting the launch scripts.
"""
import glob, os, re, subprocess, sys
from collections import defaultdict

import torch

REF = "ph13"
ARMS = ["ph13", "bidir13", "peritem", "sig20"]
BASE = "/disk1/jaehoon/bt_wide/phase5"

OK, BAD, WARN = [], [], []
def check(name, cond, detail="", warn=False):
    tag = "PASS" if cond else ("WARN" if warn else "FAIL")
    (OK if cond else (WARN if warn else BAD)).append(name)
    print(f"  [{tag}] {name}" + (f"\n         {detail}" if detail else ""))


ps = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True).stdout
live = {}
for line in ps.splitlines():
    if "train.py" not in line or "--name" not in line:
        continue
    tok = line.split()
    d = {tok[i]: tok[i + 1] for i in range(len(tok) - 1) if tok[i].startswith("--")}
    live[d.get("--name")] = d

print("\n=== 1. which arms are actually running")
for a in ARMS:
    if a == REF:
        continue
    check(f"{a} has a live process", a in live,
          f"gpu {live[a]['--gpu']}  loss {live[a]['--loss']}" if a in live
          else "not running")

print("\n=== 2. shared settings must be IDENTICAL across arms")
keys = ["--fold", "--seed", "--max-epochs", "--data-dir"]
vals = defaultdict(set)
for a, d in live.items():
    for k in keys:
        vals[k].add(d.get(k))
for k in keys:
    check(f"{k} identical across live arms", len(vals[k]) == 1,
          f"{k} = {vals[k]}")

print("\n=== 3. augmentation flags must match ph13 (i.e. all defaults)")
aug = ["--tempo-min", "--tempo-max", "--tempo-stride",
       "--no-tempo-augmentation", "--no-pitch-augmentation", "--no-mask-augmentation"]
for a, d in live.items():
    present = [k for k in aug if k in d or k in " ".join(d)]
    check(f"{a}: no augmentation override", not present,
          f"found {present}" if present else "all defaults")

print("\n=== 4. each arm's loss flag differs, everything else does not")
losses = {a: d.get("--loss") for a, d in live.items()}
check("each live arm uses a DISTINCT loss", len(set(losses.values())) == len(losses),
      str(losses))

print("\n=== 5. the checkpoint records the loss the flag asked for")
want = {"bidir13": "bidirectional_phase_shift_tolerant_weighted_bce",
        "peritem": "per_item_shift_tolerant_weighted_bce",
        "sig20": "phase_shift_tolerant_weighted_bce",
        "ph13": "phase_shift_tolerant_weighted_bce"}
for a in ARMS:
    cks = sorted(glob.glob(f"{BASE}/{a}/*epoch=*.ckpt"))
    if not cks:
        check(f"{a}: checkpoint available", False, "none yet", warn=True)
        continue
    hp = torch.load(cks[0], map_location="cpu", weights_only=False)["hyper_parameters"]
    check(f"{a}: ckpt loss_type == flag", hp.get("loss_type") == want[a],
          f"ckpt says {hp.get('loss_type')}, expected {want[a]}")
    extra = {k: hp.get(k) for k in ("phase_gamma", "phase_sigma", "halving_gamma",
                                    "pos_weights") if k in hp}
    print(f"         {a}: {extra}")

print("\n=== 6. sig20 must differ from ph13 ONLY in sigma/gamma")
if "sig20" in live:
    d = live["sig20"]
    check("sig20 uses the same loss class as ph13",
          d.get("--loss") == "phase_shift_tolerant_weighted_bce", d.get("--loss"))
    check("sig20 sigma is 0.20 and gamma 0.424",
          d.get("--phase-sigma") == "0.20" and d.get("--phase-gamma") == "0.424",
          f"sigma={d.get('--phase-sigma')} gamma={d.get('--phase-gamma')}")

print("\n=== 7. pos_weight consistency (all arms inherit the same value)")
pws = {}
for a in ARMS:
    cks = sorted(glob.glob(f"{BASE}/{a}/*epoch=*.ckpt"))
    if cks:
        hp = torch.load(cks[0], map_location="cpu", weights_only=False)["hyper_parameters"]
        pws[a] = hp.get("pos_weights")
check("pos_weights identical across arms", len(set(map(str, pws.values()))) <= 1,
      str(pws))

print("\n=== 8. evaluation is fold-honest for every arm")
sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/scripts/sa3")
from smc_eval import fold_songs
f0 = {s for s, _, _ in fold_songs(0)}
split = "/disk1/jaehoon/bt_wide/data/annotations/wide/8-folds.split"
fo = {}
for line in open(split):
    p = line.split()
    if len(p) == 2:
        fo[p[0]] = int(p[1])
train_smc0 = [k for k, v in fo.items() if v != 0 and any(s in k for s in f0)]
check("no eval track is in any training fold", not train_smc0,
      f"leaked: {train_smc0[:5]}")
check("fold 0 = 27 tracks", len(f0) == 27, f"n={len(f0)}")

print("\n=== 9. all arms read the SAME corpus (same file count)")
n = len(glob.glob("/disk1/jaehoon/bt_wide/data/annotations/wide/annotations/beats/*.beats"))
check("corpus has 2678 annotation files", n == 2678, f"n={n}")

print(f"\n  {len(OK)} passed, {len(BAD)} failed, {len(WARN)} warnings")
if BAD:
    print("  FAILED: " + ", ".join(BAD))
if WARN:
    print("  WARN:   " + ", ".join(WARN))
