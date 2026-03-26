---
name: benchmark-kernel
description: >
  Benchmark GPU kernels using rocprofv3 --stats --kernel-trace to collect accurate
  GPU-only kernel execution times. Runs a command under rocprofv3, parses the CSV
  output for kernel stats (name, calls, avg/min/max duration, percentage), and
  presents a clean comparison table. Use when: measuring kernel performance, comparing
  before/after optimization, or validating that a code change didn't regress performance.
  Supports running on remote Docker containers via SSH.
  Usage: /benchmark-kernel <cmd>
tools: Bash,Read,Grep,Glob,Write
---

# Benchmark GPU Kernel

Measure GPU kernel execution time using `rocprofv3 --stats --kernel-trace -f csv`.
This gives **GPU-only** kernel duration — no Python launch overhead, no CUDA event
measurement error.

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<CMD>` | Yes | The command to benchmark, e.g. `python test_pa.py` |

If no command is provided, ask the user.

## Why rocprofv3 (not CUDA events)

| Method | Measures | Problem |
|--------|----------|---------|
| `torch.cuda.Event` | GPU + Python dispatch | FlyDSL ~800us overhead, Gluon ~100us |
| `triton.testing.do_bench` | GPU + JIT dispatch | CPU overhead masks kernel changes |
| `rocprofv3 --kernel-trace` | **GPU kernel only** | Ground truth |

A kernel change that adds 50us GPU time can appear neutral in CUDA event benchmarks
because Python dispatch overhead (800us) dominates. Always use rocprofv3 for A/B comparisons.

## Workflow

```
Step 1: Determine execution environment (local, SSH, Docker)
Step 2: Run rocprofv3 --stats --kernel-trace -f csv -- <CMD>
Step 3: Parse the kernel_stats.csv output
Step 4: Present results table filtered to kernels of interest
Step 5: (Optional) Compare with a baseline
```

---

## Step 1: Determine Execution Environment

Check the user's context from MEMORY.md and conversation:

### Local execution
```bash
cd <working_dir> && rocprofv3 --stats --kernel-trace -f csv -o /tmp/bench_result -- <CMD> 2>&1
```

### Remote Docker (e.g., FlyDSL on hungry_dijkstra)
```bash
ssh -i <SSH_KEY> -o StrictHostKeyChecking=no <USER>@<HOST> \
  "docker exec -e LD_LIBRARY_PATH=<LIB_PATH> <CONTAINER> bash -c \
  'cd <WORKDIR> && rocprofv3 --stats --kernel-trace -f csv -o /tmp/bench_result -- <CMD> 2>&1'"
```

Key environment variables to set:
- `LD_LIBRARY_PATH`: for FlyDSL, point to the built `_mlir_libs` directory
- `FLYDSL_RUNTIME_ENABLE_CACHE`: omit or set to 1 for benchmarking (0 causes recompilation)
- `HIP_VISIBLE_DEVICES`: set if needed to restrict to specific GPU

---

## Step 2: Run rocprofv3

The command:
```bash
rocprofv3 --stats --kernel-trace -f csv -o /tmp/bench_result -- <CMD>
```

This produces:
| File | Content |
|------|---------|
| `/tmp/bench_result_kernel_stats.csv` | **Aggregated stats**: name, calls, total/avg/min/max duration, stddev |
| `/tmp/bench_result_kernel_trace.csv` | Per-dispatch timing: timestamps, VGPR, grid size |
| `/tmp/bench_result_agent_info.csv` | GPU agent info |
| `/tmp/bench_result_domain_stats.csv` | Domain-level stats |

The `_kernel_stats.csv` is the primary output for benchmarking.

### Timeout

Set a reasonable timeout. If the command includes JIT compilation (FlyDSL/Triton first run),
allow 3-5 minutes. For cached runs, 1-2 minutes is usually sufficient.

---

## Step 3: Parse kernel_stats.csv

The CSV format:
```
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"kernel_name",N,total_ns,avg_ns,pct,min_ns,max_ns,stddev
```

Read the file and parse:
```bash
cat /tmp/bench_result_kernel_stats.csv
```

Or filter to kernels of interest:
```bash
grep -i '<pattern>' /tmp/bench_result_kernel_stats.csv
```

Common filter patterns:
| Target | Pattern |
|--------|---------|
| FlyDSL PA decode | `pa_decode` |
| Gluon PA decode | `paged_attention` |
| Triton kernels | custom kernel name |
| All non-PyTorch | exclude `at::native` |

---

## Step 4: Present Results

Format the results as a clean table:

```
| Kernel | Calls | Avg (us) | Min (us) | Max (us) | StdDev (us) | % GPU |
|--------|:-----:|:--------:|:--------:|:--------:|:-----------:|:-----:|
| pa_decode_sw_kernel_0 | 1 | 197.0 | 197.0 | 197.0 | 0.0 | 0.15 |
| paged_attention_decode_* | 104 | 106.5 | 99.7 | 116.0 | 1.7 | 8.31 |
```

### Conversion
- CSV values are in **nanoseconds** — divide by 1000 for microseconds
- `Percentage` is already a percentage of total GPU time

### Interpretation guidelines

| Metric | Meaning |
|--------|---------|
| Calls=1 | Single invocation — high variance, re-run for stability |
| StdDev > 10% of Avg | Noisy measurement, need more runs |
| Min vs Max spread > 20% | Possible thermal throttling or contention |

If the target kernel has `Calls=1`, recommend the user modify the test script to
call the kernel multiple times (e.g., warmup + 10 iterations) for stable measurements.

---

## Step 5: A/B Comparison (Optional)

When comparing two versions (before/after optimization):

### 5.1 Run baseline
```bash
# Restore baseline code
# Run benchmark
rocprofv3 --stats --kernel-trace -f csv -o /tmp/bench_baseline -- <CMD>
```

### 5.2 Apply change
```bash
# Apply the optimization
# Clear cache if needed (rm -rf ~/.flydsl/cache/* /tmp/flydsl_cache/*)
# Run benchmark
rocprofv3 --stats --kernel-trace -f csv -o /tmp/bench_optimized -- <CMD>
```

### 5.3 Present comparison

```
| Version | Kernel | Avg (us) | Min (us) | VGPRs | Speedup |
|---------|--------|:--------:|:--------:|:-----:|:-------:|
| Baseline | pa_decode_sw_kernel_0 | 243.0 | 240.0 | 188 | 1.00x |
| V prefetch | pa_decode_sw_kernel_0 | 201.0 | 198.0 | 120 | 1.21x |
| + LDS pad | pa_decode_sw_kernel_0 | 197.0 | 195.0 | 188 | 1.23x |
```

Speedup = baseline_avg / optimized_avg.

### 5.4 VGPR and resource info

The `kernel_trace.csv` (not stats) contains per-dispatch resource info. Parse it for
VGPR counts if needed:

```bash
head -1 /tmp/bench_result_kernel_trace.csv  # see column names
grep '<kernel_name>' /tmp/bench_result_kernel_trace.csv
```

Look for columns like `Arch_VGPR`, `Accum_VGPR`, `SGPR`, `LDS_Block_Size_v2`,
`Scratch_En`, `Private_Segment_Size`.

---

## Step 6: Stability Check

For reliable A/B comparisons, ensure:

1. **Cache warm**: Don't benchmark with `FLYDSL_RUNTIME_ENABLE_CACHE=0` — it causes
   recompilation every call (~1.3s overhead per call)
2. **Multiple calls**: Modify test script to call kernel 10+ times; use avg of last N
3. **Same GPU state**: No other workloads running on the GPU
4. **Same code path**: Both versions must execute the same kernel (e.g., same `one_shot`
   mode, same block size)

If the test script only calls the kernel once (Calls=1 in stats), warn the user:
```
WARNING: Target kernel only dispatched once. Results may have high variance.
Consider modifying the test to call the kernel multiple times for stable measurements.
```

---

## Error Handling

| Error | Fix |
|-------|-----|
| `rocprofv3: command not found` | `export PATH=/opt/rocm/bin:$PATH` |
| `unrecognized arguments: python ...` | Need `--` separator before the command |
| `No tracing options enabled` | Add `--kernel-trace` before `--stats` |
| Empty stats CSV | Kernel may not have been dispatched; check command output |
| `Permission denied` | May need to be in `video` group or run as root |
| SQLite DB instead of CSV | Forgot `-f csv`; re-run with `-f csv` flag |

---

## Quick Reference

### One-liner for local benchmarking
```bash
rocprofv3 --stats --kernel-trace -f csv -o /tmp/bench -- <CMD> 2>&1 && \
  head -1 /tmp/bench_kernel_stats.csv && grep '<pattern>' /tmp/bench_kernel_stats.csv
```

### One-liner for remote Docker benchmarking
```bash
ssh -i <KEY> <USER>@<HOST> "docker exec -e LD_LIBRARY_PATH=<LIB> <CONTAINER> bash -c \
  'cd <DIR> && rocprofv3 --stats --kernel-trace -f csv -o /tmp/bench -- <CMD> 2>&1'" \
  | tail -5 && \
ssh -i <KEY> <USER>@<HOST> "docker exec <CONTAINER> bash -c \
  'head -1 /tmp/bench_kernel_stats.csv && grep <pattern> /tmp/bench_kernel_stats.csv'"
```
