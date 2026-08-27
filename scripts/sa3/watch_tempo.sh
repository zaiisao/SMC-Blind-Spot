#!/bin/bash
# Live per-tempo F for every running/finished arm. Usage: bash scripts/sa3/watch_tempo.sh [logdir]
cd "$(dirname "$0")/../.."
for f in cache/s2_*.log cache/v2t_*.log; do
  [ -s "$f" ] || continue
  a=$(basename "$f" .log); a=${a#s2_}; a=${a#v2t_}
  n=$(grep -ac 'by tempo' "$f" 2>/dev/null)
  [ "$n" -eq 0 ] && continue
  echo "== $a"
  paste -d'|' <(grep -aoE '^\[smc-eval\] step [0-9]+' "$f" | grep -oE '[0-9]+$') \
              <(grep -aoE 'F=[0-9.]+ ' "$f" | head -"$n" | tr -d 'F= ') \
              <(grep -a 'by tempo' "$f" | sed 's/.*by tempo: //') \
  | awk -F'|' '{printf "   step %-5s ALL %-7s %s\n",$1,$2,$3}'
done
