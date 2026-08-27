"""Assign the REMAINING tracks between the two machines by measured throughput.

Local rendered 15747 files before the rebalance, spread over all 18 shards, and
titan rendered shards 12-17 independently. So "who still owes what" is no longer
a clean function of shard id -- it has to be counted from what is actually on
disk on each machine.

A track counts as done only when all 32 of its files exist (the same rule the
resume check uses), so partial renders are re-queued rather than silently kept.
"""
import os, subprocess, sys

ROOT = "/disk1/jaehoon/bt_wide"
SRC = "/disk3/jaehoon/bt_wide/audio_src"
NFILES = 32

names = sorted(os.path.splitext(f)[0] for f in os.listdir(SRC))
print(f"corpus: {len(names)} tracks")

def done_local():
    base = f"{ROOT}/data/audio/mono_tracks/wide"
    out = set()
    if not os.path.isdir(base):
        return out
    for d in os.listdir(base):
        p = os.path.join(base, d)
        try:
            if len(os.listdir(p)) >= NFILES:
                out.add(d)
        except OSError:
            pass
    return out

def done_titan():
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "titan",
         "for d in /mount/btwide/data/audio/mono_tracks/wide/*/; do "
         "n=$(ls \"$d\" | wc -l); [ \"$n\" -ge 32 ] && basename \"$d\"; done"],
        capture_output=True, text=True, timeout=600)
    return set(x for x in r.stdout.split() if x)

L, T = done_local(), done_titan()
print(f"  complete on local {len(L)}   on titan {len(T)}   overlap {len(L & T)}")
rest = [n for n in names if n not in L and n not in T]
print(f"  still to render: {len(rest)} tracks ({len(rest)*NFILES} files)")

# split by measured rate: titan 484 wav/min vs local ~170
LR, TR = 7774.0, 13975.0  # measured audio-sec/min over a steady window
ttake = len(rest) - int(round(len(rest) * LR / (LR + TR)))

# titan can only render tracks whose SOURCE audio was rsynced to it, so fill its
# quota from those first; anything short is a transfer we have to make.
r = subprocess.run(["ssh", "-o", "BatchMode=yes", "titan",
                    "ls /mount/btwide/audio_src"],
                   capture_output=True, text=True, timeout=300)
have = set(os.path.splitext(x)[0] for x in r.stdout.split() if x)
cand = [n for n in rest if n in have]
print(f"  titan has source for {len(cand)} of the {len(rest)} remaining")

# Balance by AUDIO DURATION, not track count. What is left now is 3.3x longer
# on average than what is already done (median 167 s vs 30 s) and spans 10x
# within itself, so slicing a sorted list can hand one machine every long ASAP
# piano piece. Longest-first greedy assignment to whichever box has the least
# work relative to its speed keeps both finishing together.
import soundfile as sf

def dur(n):
    for e in (".wav", ".flac", ".mp3"):
        p = os.path.join(SRC, n + e)
        if os.path.exists(p):
            try:
                return sf.info(p).duration
            except Exception:
                return 60.0
    return 60.0

durs = {n: dur(n) for n in rest}
order = sorted(rest, key=lambda n: -durs[n])
loc, tit = [], []
lsec = tsec = 0.0
for n in order:
    # cost in "machine-seconds": same audio costs less on the faster box
    if (lsec / LR) <= (tsec / TR) or n not in have:
        loc.append(n); lsec += durs[n]
    else:
        tit.append(n); tsec += durs[n]
print(f"  by duration: local {lsec/3600:.1f} audio-h, titan {tsec/3600:.1f} audio-h")
print(f"  -> local {len(loc)} tracks   titan {len(tit)} tracks")
print(f"     projected local {len(loc)*NFILES/LR:.0f} min, titan {len(tit)*NFILES/TR:.0f} min")

open("/tmp/local_tracks.txt", "w").write("\n".join(loc) + "\n")
open("/tmp/titan_tracks.txt", "w").write("\n".join(tit) + "\n")
print("wrote /tmp/local_tracks.txt and /tmp/titan_tracks.txt")
