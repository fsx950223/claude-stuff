#!/usr/bin/env bash
# Wait for an idle GPU, re-check the pristine task on it, then launch GEAK's kernel_workflow pinned to that GPU.
#
# Run it where the GPU stack lives (inside the dev container), detached, so it outlives the Claude session:
#   docker exec -d -e ANTHROPIC_API_KEY -e ANTHROPIC_BASE_URL -e ANTHROPIC_CUSTOM_HEADERS \
#     -e RUN_DIR=... -e TASK_DIR=... -e GEAK_DIR=... -e ARGS_TEMPLATE=... <container> \
#     bash -c 'exec setsid bash /path/to/wait_and_launch.sh'
#
# Required environment:
#   RUN_DIR        output dir (waiter.log, geak.log, geak_args.json, geak.pgid, gpu, geak_result.json)
#   TASK_DIR       the GEAK task dir (kernel_path)
#   GEAK_DIR       GEAK checkout
#   ARGS_TEMPLATE  kernel_workflow args JSON; every "__GPU__" is replaced by the chosen GPU id
# Optional:
#   GPU_PREF       preference order (default "0 1 2 3")
#   IDLE_CHECKS    consecutive idle minutes required (default 3)
#   PRECHECK_CMD   run in TASK_DIR with HIP_VISIBLE_DEVICES=<gpu>; must exit 0 (default: correctness mode)
#   MODEL          GEAK_CLAUDE_MODEL (default claude-opus-5-5)
#   EXTRA_PATH     prepended to PATH (venv bin, node bin, claude bin)
#
# Idle = the rule GEAK's gpu_lock.sh uses: gpu_busy_percent <= 5 and VRAM used <= 1024 MB, read from
# /sys/class/drm/renderD$((128 + 8 * id)). Check that this mapping matches `amd-smi list` on your machine.
set -u
: "${RUN_DIR:?}" "${TASK_DIR:?}" "${GEAK_DIR:?}" "${ARGS_TEMPLATE:?}"
GPU_PREF="${GPU_PREF:-0 1 2 3}"
IDLE_CHECKS="${IDLE_CHECKS:-3}"
PRECHECK_CMD="${PRECHECK_CMD:-python3 scripts/task_runner.py correctness}"
MODEL="${MODEL:-claude-opus-5-5}"
[ -n "${EXTRA_PATH:-}" ] && export PATH="$EXTRA_PATH:$PATH"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/waiter.log"
log() { echo "[$(date -u +%F' '%T)] $*" >> "$LOG"; }

idle() {
    local dev busy vram
    dev="$(readlink -f "/sys/class/drm/renderD$((128 + 8 * $1))/device")" || return 1
    busy=$(cat "$dev/gpu_busy_percent" 2>/dev/null || echo 100)
    vram=$(( $(cat "$dev/mem_info_vram_used" 2>/dev/null || echo 99999999999) / 1048576 ))
    [ "$busy" -le 5 ] && [ "$vram" -le 1024 ]
}

declare -A streak
for g in $GPU_PREF; do streak[$g]=0; done
log "waiting for an idle GPU (preference: $GPU_PREF)"
n=0
while true; do
    chosen=""; status=""
    for g in $GPU_PREF; do
        if idle "$g"; then streak[$g]=$(( ${streak[$g]} + 1 )); else streak[$g]=0; fi
        status+="g$g=${streak[$g]} "
    done
    for g in $GPU_PREF; do [ "${streak[$g]}" -ge "$IDLE_CHECKS" ] && { chosen=$g; break; }; done
    n=$((n + 1)); [ $((n % 15)) -eq 1 ] && log "idle streaks: $status"
    if [ -n "$chosen" ]; then
        log "GPU $chosen idle for $IDLE_CHECKS checks; running the pristine pre-check on it"
        rm -rf "$TASK_DIR/build"
        if (cd "$TASK_DIR" && HIP_VISIBLE_DEVICES=$chosen bash -c "$PRECHECK_CMD") > "$RUN_DIR/precheck_gpu$chosen.log" 2>&1; then
            rm -rf "$TASK_DIR/build"
            # Our own pre-check leaves gpu_busy_percent elevated for a few seconds; do not mistake it for a tenant.
            sleep 15
            if idle "$chosen"; then break; fi
            log "GPU $chosen got busy right after the pre-check; waiting again"
        else
            rm -rf "$TASK_DIR/build"
            log "pre-check FAILED on GPU $chosen (see precheck_gpu$chosen.log); waiting again"
        fi
        streak[$chosen]=0
    fi
    sleep 60
done

sed "s/__GPU__/$chosen/g" "$ARGS_TEMPLATE" > "$RUN_DIR/geak_args.json"
python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$RUN_DIR/geak_args.json" || { log "STATUS: ABORTED bad args json"; exit 1; }
log "launching GEAK on GPU $chosen"
cd "$GEAK_DIR"
GEAK_CLAUDE_MODEL="$MODEL" GEAK_FREEZE_HARNESS=1 GEAK_GPU_ALLOWED="$chosen" HIP_VISIBLE_DEVICES="$chosen" \
    setsid node interface/runtime/engine/run_workflow.mjs kernel_workflow/kernel_workflow.js \
    --args-file "$RUN_DIR/geak_args.json" --result-file "$RUN_DIR/geak_result.json" > "$RUN_DIR/geak.log" 2>&1 &
echo $! > "$RUN_DIR/geak.pgid"
echo "$chosen" > "$RUN_DIR/gpu"
log "STATUS: LAUNCHED pgid=$(cat "$RUN_DIR/geak.pgid") gpu=$chosen"
wait
log "STATUS: GEAK exited rc=$?"
