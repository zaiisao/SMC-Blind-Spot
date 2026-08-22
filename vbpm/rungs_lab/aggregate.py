import json, sys, numpy as np, collections
rows = json.load(open(sys.argv[1]))
by = collections.defaultdict(list)
for r in rows: by[r["dataset"]].append(r)
metrics = ["beatF","CMLt","AMLt","downbeatF","dbCMLt","dbAMLt"]
hdr = f"{'dataset':12s} {'n':>5s} rung " + " ".join(f"{m:>10s}" for m in metrics)
print(hdr)
allrows = {"ALL(pooled)": rows}
for d in sorted(by): allrows[d] = by[d]
for d, rs in allrows.items():
    for rung in ("R0","R1"):
        vals = [f"{np.mean([r[rung][m] for r in rs]):10.4f}" for m in metrics]
        print(f"{d:12s} {len(rs):5d} {rung:4s} " + " ".join(vals))
