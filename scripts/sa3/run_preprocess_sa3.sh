#!/bin/bash
# Three-stage sharded preprocessing of sa3gen into Beat This spectrogram bundles.
# 8x8 measured fastest (34s/64 clips vs 51s at 16x4 and ~57s vanilla) -- process
# startup dominates past 8 shards.
set -u
cd "$(dirname "$0")/../.."
PY=/disk1/jaehoon/sa3env/bin/python
ROOT=${ROOT:-/disk1/jaehoon/bt_sa3}
N=${N:-8}; TH=${TH:-8}

for stage in audio spect; do
  echo "[pp] stage=$stage  $N shards x $TH threads  $(date '+%H:%M:%S')"
  pids=()
  for s in $(seq 0 $((N-1))); do
    "$PY" -u scripts/sa3/sa3_preprocess_bt.py --root "$ROOT" --stage $stage \
      --shard "$s" --shards "$N" --threads "$TH" >> "cache/pp_${stage}_${s}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "[pp] stage=$stage done $(date '+%H:%M:%S')"
done

echo "[pp] stage=npz $(date '+%H:%M:%S')"
"$PY" -u scripts/sa3/sa3_preprocess_bt.py --root "$ROOT" --stage npz >> cache/pp_npz.log 2>&1
echo "[pp] ALL DONE $(date '+%H:%M:%S')"
ls -la "$ROOT"/data/audio/spectrograms/*.npz
