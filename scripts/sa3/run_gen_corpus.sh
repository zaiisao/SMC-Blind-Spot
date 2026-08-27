#!/bin/bash
# Launch sharded corpus generation on healthy GPUs only, with an AER watchdog.
#
# Replaces the ad-hoc `CUDA_VISIBLE_DEVICES=$i ... &` loop that was running when
# the machine hard-locked on 2026-08-23. Three changes: faulty GPUs are excluded
# up front, a watchdog halts the run if any link starts erroring mid-flight, and
# the generator now resumes instead of restarting from zero.
set -u
cd "$(dirname "$0")/../.."

PY=/disk1/jaehoon/sa3env/bin/python
ADAPTER=${ADAPTER:-/disk1/jaehoon/sa3_beat_runs/cn_fold0_rw/beat_adapter_final.safetensors}
OUT=${OUT:-/disk1/jaehoon/sa3_gen_corpus_fold0}
N=${N:-4000}
SECONDS_PER=${SECONDS_PER:-20}
EXTRA=${EXTRA:---match-smc-tempo}

mapfile -t GPUS < <(python3 scripts/sa3/gpu_health.py | sed -n 's/^gpu \([0-9]\+\).*OK$/\1/p')
if [ ${#GPUS[@]} -eq 0 ]; then echo "no healthy GPUs; refusing to launch" >&2; exit 1; fi

# SHARDS is decoupled from the GPU count on purpose. Output names are
# gen_<shard>_<index> and a shard owns index i iff i % SHARDS == shard, so
# changing SHARDS renames every clip and defeats resume. The existing corpus was
# written with 4 shards, so 4 it stays; shards are dealt round-robin over
# whatever healthy GPUs remain, two to a card if need be.
SHARDS=${SHARDS:-4}
echo "[launch] healthy GPUs: ${GPUS[*]}  shards=$SHARDS  n=$N  out=$OUT"

PIDS=()
for ((s=0; s<SHARDS; s++)); do
  g=${GPUS[$((s % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" scripts/sa3/sa3_generate_corpus.py \
    --adapter "$ADAPTER" --out "$OUT" --n "$N" --shard "$s" --shards "$SHARDS" \
    --seconds "$SECONDS_PER" --gpu 0 $EXTRA \
    >> "cache/gencorpus_${s}.log" 2>&1 &
  PIDS+=($!)
  echo "  shard $s -> physical gpu $g (pid ${PIDS[-1]})"
done

# PCIe watchdog: monitors the GPUs this run actually launched on, by PCI
# address, and halts on link-integrity errors. See aer_watchdog.sh for why it
# tracks the launched set rather than the currently-healthy set.
ADDRS=$(for g in "${GPUS[@]}"; do
  nvidia-smi -i "$g" --query-gpu=pci.bus_id --format=csv,noheader | sed 's/^....//' | tr 'A-F' 'a-f'
done | paste -sd,)
setsid bash scripts/sa3/aer_watchdog.sh "$ADDRS" "$(IFS=,; echo "${PIDS[*]}")" \
  > cache/aer_watchdog.log 2>&1 < /dev/null &
WATCHDOG=$!
echo "[launch] watchdog pid $WATCHDOG on $ADDRS"
wait "${PIDS[@]}"
kill "$WATCHDOG" 2>/dev/null
echo "[launch] all shards exited"
