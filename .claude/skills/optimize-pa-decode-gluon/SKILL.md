---
name: optimize-pa-decode-gluon
description: >
  Optimize the pa_decode_gluon paged attention decode implementation in
  aiter/ops/triton/gluon/pa_decode_gluon.py from the ROCm/aiter repo.
  Covers both Gluon kernel optimizations and Python API-level optimizations.
  Use when user wants to improve performance of the paged attention decode
  kernels on AMD MI300X (CDNA3) or MI350 (CDNA4) GPUs.
  Usage: /optimize-pa-decode-gluon
tools: Read,Edit,Bash,Grep,Glob,Agent
---

# Optimize pa_decode_gluon

Optimize the paged attention decode implementation in `aiter/ops/triton/gluon/pa_decode_gluon.py`.

## Source Reference

- **Repository**: https://github.com/ROCm/aiter
- **File**: `aiter/ops/triton/gluon/pa_decode_gluon.py`
- **Components**: 5 Gluon/Triton kernels + 2 wrapper functions + 1 Python API function

## Architecture Overview

The file implements a two-phase paged attention decode:

1. **Phase 1 - Partitioned Attention**: One of 3 kernel variants computes partial attention per context partition
   - `paged_attention_decode_v2_gluon_dot_kernel` - standard blocks (KV_BLOCK_SIZE=16/64)
   - `paged_attention_decode_v2_gluon_large_block_dot_kernel` - large blocks (KV_BLOCK_SIZE=1024)
   - `paged_attention_decode_sliding_window` - sliding window / prefix sharing (PS) path
2. **Phase 2 - Reduction**: One of 2 kernels reduces partial results across partitions
   - `paged_attention_decode_v2_reduce_kernel` - standard reduction
   - `paged_attention_decode_ps_reduce_kernel` - PS-aware reduction
3. **API Function**: `pa_decode_gluon()` - orchestrates the two phases

## Optimization Checklist

When this skill is invoked, systematically analyze and optimize the following areas. **Always read the latest source code first** - do not rely on cached/stale versions.

### Part 1: Gluon Kernel Optimizations

#### 1.1 Memory Access Patterns
- [ ] **Coalesced global loads**: Ensure `gl.load` / `gl.amd.cdna3.buffer_load` access patterns produce coalesced 128-byte transactions. Check that innermost dimension strides are 1 (element-contiguous).
- [ ] **Shared memory bank conflicts**: When `gl.allocate_shared_memory` or `gl.SwizzledSharedLayout` is used, verify swizzle parameters eliminate bank conflicts for the actual access pattern. Current: `SwizzledSharedLayout(KV_16B_ELEMENT_COUNT, 1, 16, order=[1,0])` - verify vec/perPhase/maxPhase are optimal for the query tensor shape.
- [ ] **Vectorized loads**: Confirm `size_per_thread` in BlockedLayout uses maximum vectorization (8 for bf16/fp16, 16 for fp8) in the contiguous dimension. If a layout has `size_per_thread=[1,1]` in the inner dim, consider restructuring.
- [ ] **Redundant loads**: Query tensor is loaded once and reused across the KV loop - good. Check if block_tables lookup can be hoisted or prefetched.

#### 1.2 Compute Efficiency
- [ ] **MFMA instruction selection**: Verify `QK_PV_MFMA_INSTR_SHAPE` matches the optimal MFMA instruction for the target arch. For CDNA3 fp8: `mfma_f32_16x16x32_fp8`, for CDNA3 bf16: `mfma_f32_16x16x16_bf16`. For CDNA4: check if 32x32 variants or new instructions give better throughput.
- [ ] **Warp-level parallelism**: Current config uses 4 warps per CTA (`warps_per_cta=[1,4]` for MFMA). For CDNA4 with higher register file, consider `warps_per_cta=[2,4]` or `[1,8]` if register pressure allows.
- [ ] **waves_per_eu tuning**: The wrapper sets `waves_per_eu` to 3 or 4 based on `QUERY_GROUP_SIZE_POW2`. Profile to determine if this is optimal. Lower values reduce register spilling but may underutilize the CU.
- [ ] **Occupancy vs register pressure**: The large_block kernel has more live variables. Check if `maxnreg` pragma or register allocation hints could help.
- [ ] **exp2 vs exp**: Already using `tl.math.exp2` with `LOG2_E` multiplier - this is optimal for AMD GPUs. Confirm no inadvertent `tl.exp` calls.

#### 1.3 Loop Structure & Prefetch
- [ ] **K-cache prefetch (double-buffer)**: The main KV loop issues 8× `global_load_dwordx4` for K-cache tiles, then hits `s_waitcnt vmcnt(0)` with **~33K stall cycles** before the first MFMA. The 126 instructions between load and consume are not enough to hide global memory latency. Apply double-buffering: pre-load the first iteration's K-cache before the loop, issue the next iteration's K-cache loads right after the swap, then compute MFMA on the current buffer while the next buffer is in flight.
  - **Register budget**: Trace shows VGPR=256, AGPR=55, total=311/512. There is ~201 regs headroom — enough for a second set of 8× dwordx4 buffers (~128 VGPRs).
  - **Pattern**: See `/prefetch-data-load` skill for the mechanical transformation.
- [ ] **V-value load hoisting**: V-value loads (8× `global_load_dwordx4` at source :1669) currently start after the softmax reduce phase. Their addresses can be computed before the reduce. **Hoist V-value load issuance into the softmax reduce barrier-wait region** (L606-L770) to overlap V-value fetch latency (~17K idle cycles) with barrier stalls (~96K stall cycles). This turns wasted barrier-wait time into useful prefetch.
- [ ] **Overlap value load with QK MFMA** in `paged_attention_decode_v2_gluon_dot_kernel`: Value is loaded after QK MFMA, so value load latency is fully exposed. Restructure to issue value load before or concurrently with QK MFMA (mirrors existing pattern in `paged_attention_decode_sliding_window`).
- [ ] **Loop unrolling**: `KV_COMPUTE_BLOCK_COUNT` is typically 1-4. Check if explicit unrolling (via `tl.static_range` or compile-time loop) helps.
- [ ] **num_stages**: Currently hardcoded to `num_stages=1`. If Triton supports multi-stage pipelining for Gluon kernels, try `num_stages=2` to overlap next iteration's loads with current compute.

#### 1.3.1 Softmax Reduce Optimization (Highest Impact)
The softmax cross-wave reduce phase (source :189 and :291) is the **single largest bottleneck** at ~96K stall cycles (40.6% of all stalls). It currently performs two sequential reduce passes (max and sum) each with 3-4 barriers:

```
ds_bpermute → s_waitcnt lgkmcnt(0) → s_barrier → ds_swizzle(SWAP,16)
→ s_waitcnt → ds_write LDS → s_waitcnt → s_barrier → ds_read LDS → ...
```

Optimizations:
- [ ] **Merge max and sum reduce passes**: If online softmax computes max and sum in separate reduce trees, merge them into a single pass that reduces a `(max, sum)` tuple together. This halves the barrier count from ~7 to ~3-4.
- [ ] **Use continuous `ds_bpermute` tree reduce**: Replace the `ds_bpermute → barrier → ds_write LDS → barrier → ds_read LDS` pattern with consecutive `ds_bpermute` calls for each tree level. Each `ds_bpermute` is a cross-lane shuffle that doesn't require LDS write/read round-trips or inter-wave barriers.
- [ ] **Reduce LDS round-trip in reduce**: Each `ds_write → s_waitcnt → s_barrier → ds_read` sequence costs ~6-8K stall cycles. If the reduce can use only `ds_bpermute`/`ds_swizzle` (intra-wavefront), the LDS write/read + barrier steps can be eliminated entirely for within-wavefront stages.

#### 1.4 Data Type Optimizations
- [ ] **FP8 accumulation precision**: Currently accumulates in fp32, which is correct. However, the `probability_scale` path for per-token FP8 quantization does `value_scale_value * FP8_MAX_VALUE / (value_scale_max + 1e-8)` - the epsilon `1e-8` may be unnecessary if `value_scale_max > 0` is guaranteed. Removing it saves a floating-point add.
- [ ] **Output type conversion**: `attention_accumulator.to(OUTPUT_DTYPE)` happens after the loop. If intermediate results are only used in fp32 operations, this is fine. Verify no unnecessary intermediate conversions.
- [ ] **Cast placement**: `query_converted.to(COMPUTE_TYPE)` and `key_converted.to(COMPUTE_TYPE)` should happen as late as possible (right before MFMA) to minimize register pressure from wider types.

#### 1.5 Masking Optimizations
- [ ] **Compile-time mask elimination**: When `IS_CAUSAL=False` and the partition is fully within context length, the boundary mask is all-true. Add a fast path that skips `tl.where` for fully-valid partitions.
- [ ] **Mask sentinel value**: Currently uses `float(-3.4e38)` instead of `-inf` to avoid NaN. This is correct but slightly less precise. Verify this doesn't affect output quality for very long sequences.

#### 1.6 CDNA4-Specific Optimizations
- [ ] **New MFMA instructions**: CDNA4 (gfx950) may support larger MFMA shapes or new data types. Check if `mfma_f32_32x32x32` variants are available and beneficial.
- [ ] **Increased register file**: CDNA4 has more VGPRs per CU. Increase `KV_COMPUTE_BLOCK_SIZE` or tile sizes to exploit this.
- [ ] **Matrix core improvements**: CDNA4 may have improved fp8 throughput. Verify `MFMA_INSTR_K=32` is still optimal.

### Part 2: Python API Optimizations

#### 2.1 Tensor Allocation
- [ ] **Duplicate allocation**: `exp_sums`, `max_logits`, and `temporary_output` are conditionally allocated TWICE in `pa_decode_gluon()` - once before assertions and once after. Remove the first allocation block (lines before assertions) since the second one always executes.
- [ ] **Pre-allocation / caching**: These intermediate tensors are allocated every call. Consider accepting them as pre-allocated buffers (which the API already supports via optional params) and document this as the recommended usage pattern.
- [ ] **Memory pool**: Use `torch.cuda.caching_allocator` or a persistent buffer pool to avoid per-call allocation overhead for `exp_sums`, `max_logits`, `temporary_output`.

#### 2.2 Kernel Dispatch
- [ ] **Grid calculation**: `max_context_partition_num` is passed as an argument. For `one_shot` cases (1 partition), the reduction kernel is skipped - this is already optimized.
- [ ] **PS path short-circuit**: When `PS=True`, the sliding_window kernel is launched inside the wrapper and returns early. But the wrapper is still called through an extra function indirection. Consider inlining or using `@torch.compile` for the dispatch logic.
- [ ] **Stride computation**: Multiple `.stride()` calls on reshaped tensors. These are cheap but could be computed once and cached for hot paths.

#### 2.3 One-Shot Path
- [ ] **Direct output write**: When `one_shot=True`, the kernel writes directly to `output_5d`, skipping the reduction kernel. But `temporary_output` may still be allocated. Add early return before allocation when `one_shot=True`.
- [ ] **one_shot detection**: `one_shot = max_context_partition_num <= 1` is set twice. Clean up to compute once.

#### 2.4 Validation Overhead
- [ ] **Assert cost**: Multiple `assert` statements run on every call. In production, these should be compiled out (`python -O`) or guarded behind a debug flag.
- [ ] **Type checking**: `query.dtype in [...]` checks run every call. Consider a `@lru_cache` or one-time validation approach.

#### 2.5 Recommended Splits
- [ ] **get_recommended_splits**: Uses `torch.cuda.get_device_properties()` which may not be cached. Wrap in `@lru_cache` (partially done already for `get_cdna_version`).
- [ ] **Occupancy-aware splitting**: `get_occupancy()` returns hardcoded `2`. This should be dynamic based on kernel register usage and shared memory. Use Triton's occupancy calculator if available.

### Part 3: Integration-Level Optimizations

#### 3.1 Kernel Fusion
- [ ] **Fuse reduction into attention kernel**: For small `max_context_partition_num` (<=4), the reduction could be done within the attention kernel using cross-CTA synchronization or within the same CTA if partitions fit.
- [ ] **ONE_SHOT in sliding_window**: The `paged_attention_decode_sliding_window` kernel already supports `ONE_SHOT` mode which fuses attention+reduction. Verify this path is always taken when possible.

#### 3.2 Autotuning
- [ ] **Uncomment autotune**: The `paged_attention_decode_v2_gluon_dot_kernel` has a commented-out `@triton.autotune` decorator. Enable it with a focused config space to find optimal `waves_per_eu` and `num_stages` per hardware.
- [ ] **Config key**: Autotune key should include `HEAD_SIZE`, `KV_BLOCK_SIZE`, `QUERY_GROUP_SIZE`, and `CONTEXT_PARTITION_SIZE` for representative coverage.

## How To Apply

1. **Read the source**: Always start by reading the latest version of `pa_decode_gluon.py`.
2. **Profile first**: Before making changes, establish a baseline using `rocprofv3` or the aiter benchmarks.
3. **Apply one optimization at a time**: Make a single change, benchmark, and verify correctness before moving to the next.
4. **Verify correctness**: Use the existing test suite (`test_pa_decode_gluon.py` or equivalent) to ensure output matches within tolerance.
5. **Benchmark**: Compare kernel duration (us), memory bandwidth utilization (GB/s), and MFMA utilization (%) before and after.

## Quick Wins (Highest Impact, Lowest Risk)

1. **Merge softmax max/sum reduce passes** — reduces ~7 barriers to ~3-4, saving ~40K+ stall cycles (estimated 15-20% kernel speedup). Modifies reduce logic only.
2. **K-cache prefetch double-buffering** — eliminates ~33K stall at `s_waitcnt vmcnt(0)` before MFMA. Register headroom is sufficient (201 spare regs). See `/prefetch-data-load`.
3. **Hoist V-value loads into barrier-wait region** — overlaps ~17K idle with ~96K barrier stall, nearly free latency hiding.
4. **Overlap value load with QK MFMA** in `dot_kernel` — mirrors existing pattern in sliding_window kernel.
5. **Fix duplicate tensor allocation** in `pa_decode_gluon()` API — pure cleanup, zero risk.

## Trace-Based Evidence (MI308 gfx942, dispatch 4025)

Reference trace: `~/Documents/ui_output_agent_12034_dispatch_4025/`

| Region | Stall Cycles | % of Total Stall | Root Cause |
|--------|-------------|------------------|------------|
| Softmax reduce (:189 + :291) | 96,112 | 40.6% | 7 barriers + serial LDS reduce |
| QK pre-MFMA wait (L397 vmcnt) | 32,740 | 13.8% | K-cache load latency exposed |
| Prologue lgkmcnt waits | 17,960 | 7.6% | Kernel arg scalar loads |
| K-cache global_load stalls | 38,036 | 16.1% | TA pressure from 8 concurrent loads |
| V-value region idle | 17,212 idle | — | Address computation bubbles |

MFMA utilization: **1.4%** (32 MFMA instructions, 10,280 / 735,416 total cycles) — severely memory/sync bound.

## Output

After optimization, report:
- Which optimizations were applied
- Before/after kernel duration (if benchmarked)
- Any correctness concerns or trade-offs
- Remaining optimization opportunities
