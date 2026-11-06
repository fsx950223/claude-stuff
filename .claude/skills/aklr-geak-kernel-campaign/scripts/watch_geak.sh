#!/usr/bin/env bash
# Block until a GEAK run's log shows progress, the run completes, or the run dies; give up after CAP seconds.
# Meant for a background shell (run_in_background) so the agent is woken on change instead of polling.
#
# Usage: watch_geak.sh <RUN_DIR>/geak.log [CAP_SECONDS=2700]
# Env:   CONTAINER   if set, check liveness with `docker exec $CONTAINER kill -0` (GEAK runs in a container)
#        WATCH_PAT   extended regex of log lines that count as progress (case-insensitive)
#                    default: phase changes, rounds, verification, winners, errors, the hung-agent guard
LOG="${1:?usage: watch_geak.sh <geak.log> [cap_seconds]}"
CAP="${2:-2700}"
PAT="${WATCH_PAT:-=== PHASE|round [0-9]|done\.|verif|winner|commit|error|guard|speedup}"
PGID="$(cat "$(dirname "$LOG")/geak.pgid" 2>/dev/null)"

alive() {
    [ -z "$PGID" ] && return 0
    if [ -n "${CONTAINER:-}" ]; then docker exec "$CONTAINER" kill -0 "$PGID" 2>/dev/null; else kill -0 "$PGID" 2>/dev/null; fi
}

n0=$(grep -ciE "$PAT" "$LOG" 2>/dev/null); n0=${n0:-0}
t=0
while [ "$t" -lt "$CAP" ]; do
    sleep 30; t=$((t + 30))
    if grep -qE "COMPLETE\.|WORKFLOW_RESULT" "$LOG" 2>/dev/null; then echo "STATUS: complete"; break; fi
    if ! alive; then echo "STATUS: process gone"; break; fi
    n=$(grep -ciE "$PAT" "$LOG" 2>/dev/null); n=${n:-0}
    if [ "$n" != "$n0" ]; then echo "STATUS: progressed"; break; fi
done
[ "$t" -ge "$CAP" ] && echo "STATUS: still running (cap)"
echo "waited=${t}s"
tail -12 "$LOG" | cut -c1-300
