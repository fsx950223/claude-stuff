---
name: capture-kernel-trace
description: >
  Capture GPU kernel ATT (Advanced Thread Trace) via rocprofv3 on a remote Docker
  container. Discovers kernel names, configures input.yaml with the target
  kernel_include_regex, runs rocprofv3 -i input.yaml with FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1,
  and downloads the latest ui_output_agent_* directory to local for analysis.
  Usage: /capture-kernel-trace <test_script.py> [kernel_name_pattern]
tools: Bash,Read,Write,Edit,Grep,Glob
---

# Capture Kernel Trace

Capture rocprofv3 ATT traces from a remote GPU Docker container, then download
the trace output locally for analysis.

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<test_script>` | Yes | Python test/bench script to profile, e.g. `bench_ps_pingpong.py` |
| `[kernel_pattern]` | No | Kernel name regex. If omitted, discover via `--stats` first |

If no test script is provided, ask the user.

## Connection Info

Read from MEMORY.md. Defaults:

```
HOST=10.67.77.162
USER=sixifang
SSH_KEY=/home/dladmin/Downloads/id_rsa
CONTAINER=hungry_dijkstra
WORKDIR=/home/dladmin/claude-stuff
REMOTE_WORKDIR=/FlyDSL
```

SSH command pattern:
```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no $USER@$HOST \
  "docker exec -e LD_LIBRARY_PATH=/FlyDSL/build-fly-v23/python_packages/flydsl/_mlir/_mlir_libs \
   -e PYTHONPATH=/FlyDSL/python:/FlyDSL/tests:/aiter \
   -e FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 \
   $CONTAINER bash -c '<CMD>'"
```

---

## Workflow

```
Step 1: Deploy test script to remote container
Step 2: Discover kernel names (if pattern not provided)
Step 3: Configure input.yaml with kernel_include_regex
Step 4: Run rocprofv3 -i input.yaml to collect ATT trace
Step 5: Find and download latest ui_output_agent_* to local
```

---

## Step 1: Deploy Test Script

Copy the test script to the remote container if it's a local file:

```bash
# Copy local file to container via SSH + docker cp
scp -i $SSH_KEY $TEST_SCRIPT $USER@$HOST:/tmp/
ssh -i $SSH_KEY $USER@$HOST "docker cp /tmp/$TEST_SCRIPT $CONTAINER:/tmp/"
```

If the test script is already on the remote (e.g., `/FlyDSL/tests/...`), skip this step.

---

## Step 2: Kernel Discovery (if no pattern provided)

Run rocprofv3 in stats mode to list kernel names:

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no $USER@$HOST \
  "docker exec -e LD_LIBRARY_PATH=/FlyDSL/build-fly-v23/python_packages/flydsl/_mlir/_mlir_libs \
   -e PYTHONPATH=/FlyDSL/python:/FlyDSL/tests:/aiter \
   $CONTAINER bash -c \
   'cd /tmp && rocprofv3 --stats --kernel-trace -f csv -o /tmp/discover -- python $TEST_SCRIPT 2>&1'"
```

Parse output to find kernel names:

```bash
ssh ... "docker exec $CONTAINER bash -c 'cat /tmp/discover_kernel_stats.csv'"
```

Present the kernel list and let the user pick, or auto-select the FlyDSL/target kernel
(typically contains `pa_decode`, `kernel_0`, or the function name from the test script).

---

## Step 3: Configure input.yaml

Create (or edit) the input.yaml on the remote with the target `kernel_include_regex`:

```yaml
jobs:
   -
       kernel_include_regex: <KERNEL_PATTERN>
       kernel_iteration_range: "[1, [2-4]]"
       output_file: out
       output_directory: /tmp/kernel_trace_output
       output_format: [csv]
       truncate_kernels: true
       sys_trace: true
       advanced_thread_trace: true
       att_target_cu: 1
       att_shader_engine_mask: "0xf"
       att_simd_select: "0xf"
       att_buffer_size: "0x6000000"
```

Write this to the remote:

```bash
ssh -i $SSH_KEY $USER@$HOST "docker exec $CONTAINER bash -c 'cat > /tmp/input_trace.yaml << \"YAMLEOF\"
jobs:
   -
       kernel_include_regex: \"$KERNEL_PATTERN\"
       kernel_iteration_range: \"[1, [2-4]]\"
       output_file: out
       output_directory: /tmp/kernel_trace_output
       output_format: [csv]
       truncate_kernels: true
       sys_trace: true
       advanced_thread_trace: true
       att_target_cu: 1
       att_shader_engine_mask: \"0xf\"
       att_simd_select: \"0xf\"
       att_buffer_size: \"0x6000000\"
YAMLEOF
'"
```

Key configuration:
- `kernel_include_regex`: Exact name or regex from Step 2
- `kernel_iteration_range`: `"[1, [2-4]]"` skips warmup (iteration 0), traces iterations 2-4
- `att_target_cu: 1`: Single CU for manageable output
- `att_buffer_size: "0x6000000"`: 96MB per SE (increase to `0xC000000` if truncated)

---

## Step 4: Run rocprofv3 with ATT

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no $USER@$HOST \
  "docker exec -e LD_LIBRARY_PATH=/FlyDSL/build-fly-v23/python_packages/flydsl/_mlir/_mlir_libs \
   -e PYTHONPATH=/FlyDSL/python:/FlyDSL/tests:/aiter \
   -e FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 \
   $CONTAINER bash -c \
   'cd /tmp && rm -rf /tmp/kernel_trace_output && rocprofv3 -i /tmp/input_trace.yaml -- python $TEST_SCRIPT 2>&1'"
```

**IMPORTANT**: Set `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` to get source-to-assembly mapping
in the trace output. This enables DWARF debug info in the compiled HSACO, so `code.json`
will contain source file:line annotations for each ISA instruction.

Timeout: allow 3-5 minutes for JIT compilation + trace collection.

---

## Step 5: Download Trace Output

### 5.1 Find the latest ui_output_agent_* directory

```bash
ssh -i $SSH_KEY $USER@$HOST \
  "docker exec $CONTAINER bash -c \
   'ls -td /tmp/kernel_trace_output/ui_output_agent_* 2>/dev/null | head -5'"
```

The output directories are named `ui_output_agent_<PID>_dispatch_<N>`. Pick the latest
(most recent by modification time). If multiple dispatches exist, download the one
corresponding to the target iteration.

### 5.2 Download to local

Use `docker cp` + `scp` to get the trace directory to the local machine:

```bash
# Create local destination
LOCAL_TRACE_DIR=$WORKDIR/trace_data/$(date +%Y%m%d_%H%M%S)_$KERNEL_SHORT_NAME
mkdir -p $LOCAL_TRACE_DIR

# Copy from container to host, then to local
UI_OUTPUT_DIR=<latest ui_output_agent_* path>

ssh -i $SSH_KEY $USER@$HOST \
  "docker cp $CONTAINER:$UI_OUTPUT_DIR /tmp/ui_trace_download"

scp -r -i $SSH_KEY $USER@$HOST:/tmp/ui_trace_download/* $LOCAL_TRACE_DIR/
```

Also download supporting files:

```bash
# Kernel trace CSV (timing, VGPR info)
ssh -i $SSH_KEY $USER@$HOST \
  "docker cp $CONTAINER:/tmp/kernel_trace_output/out_kernel_trace.csv /tmp/"
scp -i $SSH_KEY $USER@$HOST:/tmp/out_kernel_trace.csv $LOCAL_TRACE_DIR/

# Stats CSV if present
ssh -i $SSH_KEY $USER@$HOST \
  "docker cp $CONTAINER:/tmp/kernel_trace_output/stats_*.csv /tmp/ 2>/dev/null"
scp -i $SSH_KEY $USER@$HOST:/tmp/stats_*.csv $LOCAL_TRACE_DIR/ 2>/dev/null
```

### 5.3 Verify download

```bash
ls -la $LOCAL_TRACE_DIR/
# Should contain: code.json, occupancy.json, filenames.json, wstates*.json, se*_*.json
# Plus: out_kernel_trace.csv, stats_*.csv

# Quick validation
python3 -c "
import json, sys
with open('$LOCAL_TRACE_DIR/code.json') as f:
    data = json.load(f)
n = len(data.get('code', []))
has_src = sum(1 for i in data.get('code', []) if i[3])
print(f'Instructions: {n}, with source mapping: {has_src} ({100*has_src//max(n,1)}%)')
"
```

---

## Output

After capture, report:

1. **Trace location**: Local path to the downloaded trace directory
2. **Kernel info**: Name, VGPR/AGPR counts, grid size, duration (from out_kernel_trace.csv)
3. **Source mapping**: Whether debug info is present (% of instructions with source annotations)
4. **Instruction count**: Total instructions in code.json
5. **Next step**: Suggest running `/kernel-trace-analysis` on the downloaded trace for bottleneck analysis

Example output:
```
Trace captured: ~/claude-stuff/trace_data/20260325_153000_pa_decode/
  Kernel: pa_decode_sw_kernel_0
  Duration: 208.3 us
  arch_vgpr=96, accum_vgpr=128, SGPR=80
  Instructions: 2692, source-mapped: 2105 (78%)

Run /kernel-trace-analysis to analyze bottlenecks.
```

---

## Error Handling

| Error | Fix |
|-------|-----|
| `rocprof-trace-decoder library path not found` | Install decoder: see kernel-trace-analysis skill Step 3 |
| `INVALID_SHADER_DATA` | aqlprofile/decoder version mismatch, update both |
| Empty ui_output_agent_* | kernel_include_regex didn't match — re-check kernel name from Step 2 |
| No source mapping in code.json | Ensure `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` is set |
| Trace truncated (missing instructions) | Increase `att_buffer_size` to `0xC000000` (192MB) |
| `libmlir_float16_utils.so` missing | Symlink: `_mlir -> build-fly-v23`, set LD_LIBRARY_PATH |
| SSH timeout | Increase timeout, check host connectivity |
| `kernel_iteration_range` mismatch | Test runs fewer iterations than expected — use `"[0, [1-2]]"` |

---

## Quick Reference

### One-liner: discover + capture + download

```bash
# 1. Discover
ssh -i $KEY $U@$H "docker exec -e LD_LIBRARY_PATH=$LIB -e PYTHONPATH=$PP $C bash -c \
  'rocprofv3 --stats --kernel-trace -f csv -o /tmp/disc -- python /tmp/$SCRIPT 2>&1'" | \
  grep -v "^W2" | tail -10

# 2. Capture (after setting KERNEL_PATTERN)
ssh -i $KEY $U@$H "docker exec -e LD_LIBRARY_PATH=$LIB -e PYTHONPATH=$PP \
  -e FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 $C bash -c \
  'rm -rf /tmp/kto && cat > /tmp/it.yaml << EOF
jobs:
   -
       kernel_include_regex: \"$KP\"
       kernel_iteration_range: \"[1, [2-4]]\"
       output_file: out
       output_directory: /tmp/kto
       output_format: [csv]
       truncate_kernels: true
       sys_trace: true
       advanced_thread_trace: true
       att_target_cu: 1
       att_shader_engine_mask: \"0xf\"
       att_simd_select: \"0xf\"
       att_buffer_size: \"0x6000000\"
EOF
rocprofv3 -i /tmp/it.yaml -- python /tmp/$SCRIPT 2>&1'"

# 3. Download
LATEST=$(ssh -i $KEY $U@$H "docker exec $C bash -c 'ls -td /tmp/kto/ui_output_agent_* | head -1'")
mkdir -p trace_data/latest
ssh -i $KEY $U@$H "docker cp $C:$LATEST /tmp/ui_dl"
scp -r -i $KEY $U@$H:/tmp/ui_dl/* trace_data/latest/
```
