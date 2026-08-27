#!/bin/bash
# Watch PCIe AER counters on a fixed set of GPUs and halt a run if a link starts
# failing. Usage: aer_watchdog.sh "<pci_addr,...>" "<pid,...>"
#
# Two lessons are baked in. First, monitor the GPUs the run was LAUNCHED on, by
# PCI address -- an earlier version summed only GPUs still classified healthy,
# so a card that degraded dropped off the watch list and its errors became
# invisible, disabling the watchdog precisely when it should fire. Second,
# distinguish error classes: RxErr/BadTLP/BadDLLP are link-integrity failures
# (GPU 1's signature before the 2026-08-23 hard lock) and halt immediately,
# while an occasional Timeout under full load is common and only halts if it
# keeps climbing.
set -u
ADDRS=${1:?pci addrs}; PIDS=${2:?pids}
IFS=',' read -ra A <<< "$ADDRS"
IFS=',' read -ra P <<< "$PIDS"
LINK_RE='^(RxErr|BadTLP|BadDLLP|Rollover|CorrIntErr|HeaderOF) '
TIMEOUT_LIMIT=${TIMEOUT_LIMIT:-5}

count() {  # $1=addr $2=regex -> summed count
  local f=/sys/bus/pci/devices/$1/aer_dev_correctable n=0 k v
  [ -r "$f" ] || { echo 0; return; }
  while read -r k v; do [[ "$k $v" =~ $2 ]] && n=$((n + v)); done < "$f"
  echo "$n"
}
nonfatal() {
  local f=/sys/bus/pci/devices/$1/aer_dev_nonfatal
  [ -r "$f" ] && sed -n 's/^TOTAL_ERR_NONFATAL \([0-9]*\)/\1/p' "$f" || echo 0
}

declare -A L0 T0 N0
for a in "${A[@]}"; do
  L0[$a]=$(count "$a" "$LINK_RE"); T0[$a]=$(count "$a" '^Timeout '); N0[$a]=$(nonfatal "$a")
done
echo "[aer] watching ${A[*]} (link/timeout/nonfatal baselines recorded)"

halt() {
  echo "[aer] HALT: $1" >&2
  kill "${P[@]}" 2>/dev/null
  exit 1
}
# watch ALL pids, not just the first -- an earlier invocation exited the moment its
# first job finished, leaving every later job unprotected.
alive(){ for q in "${P[@]}"; do kill -0 "$q" 2>/dev/null && return 0; done; return 1; }
while alive; do
  sleep 60
  for a in "${A[@]}"; do
    l=$(count "$a" "$LINK_RE"); t=$(count "$a" '^Timeout '); n=$(nonfatal "$a")
    [ "$l" -gt "${L0[$a]}" ] && halt "$a link-integrity errors ${L0[$a]} -> $l"
    [ "$n" -gt "${N0[$a]}" ] && halt "$a non-fatal AER ${N0[$a]} -> $n"
    if [ "$t" -gt "${T0[$a]}" ]; then
      d=$((t - T0[$a]))
      echo "[aer] $a Timeout +$d (limit $TIMEOUT_LIMIT) -- link intact, continuing"
      [ "$d" -ge "$TIMEOUT_LIMIT" ] && halt "$a Timeout errors +$d >= $TIMEOUT_LIMIT"
    fi
  done
done
echo "[aer] run finished; watchdog exiting"
