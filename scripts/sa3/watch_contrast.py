"""Live view of contrastive adapter training.

Written after an inverted loss mask went unnoticed for 2,250 steps: the progress
bar did not flush, and a single noisy step read as a trend. This prints the
numbers that would have caught it -- dead steps, and the gap broken out by
corruption kind. The kinds matter because half / double / offbeat are regular
grids that differ from the truth only in phase or level, so they cannot be
separated without relating the grid to the audio. If those three lag while
jitter and dropout race ahead, the model found a shortcut.
"""
import csv, glob, sys, time

RUNS = sys.argv[1:] or ["contrastA", "contrastB"]
KEY = ["half", "double", "offbeat", "ratio", "jitter", "drift", "dropout", "other"]


def load(tag):
    fs = glob.glob(f"/disk1/jaehoon/sa3_beat_runs/cn_{tag}/lightning_logs/*/metrics.csv")
    if not fs:
        return None
    try:
        rows = list(csv.DictReader(open(sorted(fs)[-1])))
    except Exception:
        return None
    return [r for r in rows if r.get("step")]


def f(r, c):
    v = r.get(c, "")
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None


def trend(rows, col, n=40):
    vals = [f(r, col) for r in rows]
    vals = [v for v in vals if v is not None]
    if len(vals) < 4:
        return None, None
    h = len(vals) // 2
    return sum(vals[:h]) / h, sum(vals[h:]) / len(vals[h:])


while True:
    print("\033[2J\033[H" + time.strftime("%H:%M:%S"))
    for tag in RUNS:
        rows = load(tag)
        if not rows:
            print(f"== {tag}: no metrics yet"); continue
        last = rows[-1]
        step = int(float(last["step"]))
        dead = next((f(r, "train/dead_steps") for r in reversed(rows)
                     if f(r, "train/dead_steps") is not None), 0)
        a, b = trend(rows, "train/margin_gap")
        arrow = "" if a is None else ("UP  " if b > a * 1.15 else
                                      "flat" if b > a * 0.85 else "DOWN")
        print(f"== {tag}  step {step}/8000   dead={int(dead or 0)}")
        if a is not None:
            print(f"   margin_gap  first-half {a:+.6f} -> second-half {b:+.6f}  [{arrow}]")
        for c in ("train/mse_loss", "train/hinge"):
            v = next((f(r, c) for r in reversed(rows) if f(r, c) is not None), None)
            if v is not None:
                print(f"   {c.split('/')[1]:<10} {v:.6f}")
        gaps = {k: next((f(r, f"gap/{k}") for r in reversed(rows)
                         if f(r, f"gap/{k}") is not None), None) for k in KEY}
        gaps = {k: v for k, v in gaps.items() if v is not None}
        if gaps:
            print("   gap by corruption (want all > 0; half/double/offbeat are the honest ones):")
            for k in KEY:
                if k in gaps:
                    bar = "#" * min(30, max(0, int(gaps[k] * 4000)))
                    print(f"     {k:<8} {gaps[k]:+.6f} {bar}")
    sys.stdout.flush()
    time.sleep(20)
