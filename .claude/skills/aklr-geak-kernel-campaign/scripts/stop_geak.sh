#!/usr/bin/env bash
# Stop a GEAK campaign: the waiter (if any), the workflow's process group (node + every claude agent it spawned),
# and any measurement still running out of the run's experiment dir.
#
# Usage: stop_geak.sh <RUN_DIR> [EXP_ROOT]
# Env:   CONTAINER   if set, run the kills inside that container (where GEAK was launched)
set -u
RUN_DIR="${1:?usage: stop_geak.sh <RUN_DIR> [EXP_ROOT]}"
EXP_ROOT="${2:-}"
PGID="$(cat "$RUN_DIR/geak.pgid" 2>/dev/null || true)"

run() { if [ -n "${CONTAINER:-}" ]; then docker exec "$CONTAINER" bash -c "$1"; else bash -c "$1"; fi; }

run "pkill -TERM -f wait_and_launch.sh 2>/dev/null || true"
if [ -n "$PGID" ]; then
    run "kill -TERM -- -$PGID 2>/dev/null || true; sleep 5; kill -KILL -- -$PGID 2>/dev/null || true"
fi
# Agents' tool subprocesses can leave the group; anything still running out of the experiment dir is ours.
if [ -n "$EXP_ROOT" ]; then
    run "for p in \$(pgrep -f '$EXP_ROOT'); do [ \"\$p\" != \"\$\$\" ] && kill -KILL \"\$p\" 2>/dev/null; done; true"
fi
sleep 2
echo "left over (should be empty):"
run "ps -eo pid,etime,cmd | grep -E 'run_workflow|claude -p|task_runner|wait_and_launch' | grep -v grep | cut -c1-160 || true"
