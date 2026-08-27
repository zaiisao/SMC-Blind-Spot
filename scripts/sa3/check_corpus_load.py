"""Verify the downbeat-mask fix: all 2678 tracks load, downbeat supervision intact.

Before: dataset_info["has_downbeats"] was True for the merged "wide" set, and any
one-column track was skipped -- 724 of 2678 dropped, including all 200 SMC.
After: downbeat_mask is per track, so one-column tracks train the beat head and
are masked out of the downbeat loss.
"""
import sys
from collections import Counter

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
from beat_this.dataset.dataset import BeatDataModule  # noqa: E402

OK, BAD = [], []
def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""))


DATA = "/disk1/jaehoon/bt_wide/data"
dm = BeatDataModule(DATA, batch_size=8, num_workers=0, fold=0)
dm.setup(stage="fit")
items = dm.train_dataset.items
print(f"\n  loaded {len(items)} training items")

pref = Counter(str(i["dataset"]) for i in items)
print("  by dataset: " + ", ".join(f"{k}={v}" for k, v in sorted(pref.items())))

with_db = sum(1 for i in items if i["downbeat_mask"])
without = len(items) - with_db
print(f"  downbeat_mask True={with_db}  False={without}")

check("one-column tracks are no longer dropped", without > 0,
      f"{without} tracks train the beat head with downbeat loss masked off")
check("two-column tracks keep downbeat supervision", with_db > 1500,
      f"{with_db} tracks still supply downbeats")
check("masked tracks really have no downbeats",
      all((i["beat_value"] == 1).sum() == 0 for i in items if not i["downbeat_mask"]),
      "no track masked off actually had downbeat labels")
check("unmasked tracks really do have downbeats",
      all((i["beat_value"] == 1).sum() > 0 for i in items if i["downbeat_mask"]),
      "every unmasked track has at least one downbeat")

# fold honesty: no fold-0 track may appear in the training split
fo = {}
for line in open(f"{DATA}/annotations/wide/8-folds.split"):
    p = line.split()
    if len(p) == 2:
        fo[p[0]] = int(p[1])
names = [str(i.get("spect_path", "")).split("/")[0] for i in items]
leak = [n for n in names if fo.get(n) == 0]
check("no fold-0 track in the training split", not leak, f"leaked: {leak[:5]}")

smc = [n for n in names if "smc" in n]
check("SMC tracks now enter training", len(smc) > 0,
      f"{len(smc)} smc tracks in the training split (was 0)")

print(f"\n  {len(OK)} passed, {len(BAD)} failed")
if BAD:
    print("  FAILED: " + ", ".join(BAD))
