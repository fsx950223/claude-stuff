---
name: check-kernel-resources
description: >
  Extract GPU kernel resource usage (arch VGPR, accum VGPR, SGPR, shared memory,
  scratch/spill) from exported .amdgcn assembly files. Parses the .amdhsa_kernel
  descriptor and metadata sections to report register allocation, LDS usage,
  spill counts, and occupancy. Works with both FlyDSL and Gluon/Triton amdgcn dumps.
  Usage: /check-kernel-resources <file.amdgcn>
tools: Read,Grep,Bash,Glob
---

# Check Kernel Resources

Parse `.amdgcn` assembly files to extract GPU kernel resource usage: VGPR (arch/accum),
SGPR, shared memory (LDS), scratch, spill counts, and compute occupancy.

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<file>` | Yes | Path to `.amdgcn` file, or a glob pattern to check multiple files |

If no file is provided, search for `*.amdgcn` files in the current directory.

## How to Get .amdgcn Files

### FlyDSL (recommended: FLYDSL_DUMP_IR=1)

Set `FLYDSL_DUMP_IR=1` to dump all compilation stages including the final ISA:

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/flydsl_dump python test_kernel.py
```

Output: `/tmp/flydsl_dump/<kernel_name>/15_final_isa.s` — this is the `.amdgcn` assembly.

```
/tmp/flydsl_dump/<kernel_name>/
├── 00_origin.mlir                        # Original MLIR
├── 01_fly_rewrite_func_signature.mlir    # After signature rewrite
├── ...                                   # Intermediate passes
├── 13_reconcile_unrealized_casts.mlir    # Pre-backend MLIR
├── 14_gpu_module_to_binary.mlir          # Contains embedded ELF binary
└── 15_final_isa.s                        # ← AMDGCN assembly (target file)
```

Remote Docker example:
```bash
ssh -i $SSH_KEY $USER@$HOST \
  "docker exec -e LD_LIBRARY_PATH=$LIB_PATH \
   -e PYTHONPATH=/FlyDSL/python:/FlyDSL/tests:/aiter \
   -e FLYDSL_DUMP_IR=1 \
   -e FLYDSL_DUMP_DIR=/tmp/flydsl_dump \
   $CONTAINER bash -c 'rm -rf /tmp/flydsl_dump && python /tmp/test_kernel.py 2>&1'"
```

Then download the `.s` file:
```bash
ssh -i $SSH_KEY $USER@$HOST "docker cp $CONTAINER:/tmp/flydsl_dump/ /tmp/"
scp -r -i $SSH_KEY $USER@$HOST:/tmp/flydsl_dump/ ./
```

**Note**: `FLYDSL_DUMP_IR=1` disables the compilation cache (`_mem_cache` bypass),
so each run recompiles. This is fine for resource checking but not for benchmarking.

---

## Parsing Rules

The `.amdgcn` file contains two sources of resource info:

### 1. Kernel Descriptor (`.amdhsa_kernel` block) — PRIMARY

```asm
.amdhsa_kernel kernel_name
    .amdhsa_group_segment_fixed_size 6784      ← LDS (shared memory) in bytes
    .amdhsa_private_segment_fixed_size 0        ← scratch memory in bytes (0 = no spill)
    .amdhsa_next_free_vgpr 104                  ← total VGPRs allocated
    .amdhsa_next_free_sgpr 36                   ← SGPRs used (before system SGPRs)
    .amdhsa_accum_offset 104                    ← arch/accum VGPR boundary
    .amdhsa_reserve_vcc 1                       ← +2 SGPRs for VCC if 1
.end_amdhsa_kernel
```

**Derived values:**
```
arch_vgpr  = amdhsa_accum_offset                (rounded up to granularity)
accum_vgpr = amdhsa_next_free_vgpr - amdhsa_accum_offset
             (0 if accum_offset == next_free_vgpr, meaning no MFMA accumulators)
total_sgpr = amdhsa_next_free_sgpr + (2 if reserve_vcc) + (2 if reserve_xnack)
             + system SGPRs (workgroup_id x/y/z = up to 3)
lds_bytes  = amdhsa_group_segment_fixed_size
scratch    = amdhsa_private_segment_fixed_size
```

### 2. Metadata Section (`.amdgpu_metadata`) — SECONDARY

```yaml
.sgpr_count:     42
.sgpr_spill_count: 0
.vgpr_count:     104
.vgpr_spill_count: 0
```

These values confirm the kernel descriptor. `spill_count > 0` indicates register pressure.

### 3. Symbol Attributes — TERTIARY

```asm
.set kernel_name.num_vgpr, 104
.set kernel_name.numbered_sgpr, 36
```

---

## Occupancy Calculation (gfx942 / MI300X)

```
Each SIMD has:
  - 256 arch VGPRs (v registers)
  - 256 accum VGPRs (a registers, for MFMA)

Granularity: 8 (VGPRs allocated in blocks of 8)

arch_vgpr_alloc  = ceil(arch_vgpr / 8) * 8
accum_vgpr_alloc = ceil(accum_vgpr / 8) * 8   (0 if no accumulators)

waves_by_arch  = 256 / arch_vgpr_alloc     (if arch > 0)
waves_by_accum = 256 / accum_vgpr_alloc    (if accum > 0)
waves_per_simd = min(waves_by_arch, waves_by_accum, 8)

Typical targets:
  ≤ 64 VGPRs (either file)  → 4 waves/SIMD (good)
  ≤ 128 VGPRs               → 2 waves/SIMD (acceptable for MFMA-heavy)
  ≤ 256 VGPRs               → 1 wave/SIMD  (minimum)
  > 256 in either file      → SPILL (critical regression)
```

---

## Workflow

```
Step 1: Locate .amdgcn file(s)
Step 2: Parse .amdhsa_kernel block for resource values
Step 3: Parse .amdgpu_metadata for spill counts
Step 4: Compute occupancy
Step 5: Present results table
Step 6: Flag warnings (spills, low occupancy, high LDS)
```

### Step 1: Locate Files

```bash
# Find .amdgcn files in current directory
ls *.amdgcn 2>/dev/null

# Or search recursively
find . -name "*.amdgcn" -type f
```

### Step 2-4: Parse and Compute

Use Grep to extract values from the `.amdhsa_kernel` block:

```bash
grep -E "amdhsa_next_free_vgpr|amdhsa_next_free_sgpr|amdhsa_accum_offset|amdhsa_group_segment|amdhsa_private_segment_fixed|amdhsa_kernel |amdhsa_reserve_vcc|amdhsa_reserve_xnack|spill_count|\.vgpr_count|\.sgpr_count" file.amdgcn
```

For multi-kernel files, parse each `.amdhsa_kernel ... .end_amdhsa_kernel` block separately.

### Step 5: Present Results

Format results as a table:

```
| Kernel | arch VGPR | accum VGPR | SGPR | LDS (bytes) | Scratch | Spills | Waves/SIMD |
|--------|:---------:|:----------:|:----:|:-----------:|:-------:|:------:|:----------:|
| pa_decode_kernel_0 | 104 | 0 | 42 | 6784 | 0 | 0 | 2 |
| gluon_pa_decode     | 120 | 12 | 102 | 0 | 0 | 0 | 2 |
```

### Step 6: Warnings

Flag these conditions:

| Condition | Severity | Meaning |
|-----------|----------|---------|
| `scratch > 0` | CRITICAL | Register spilling to memory — major perf regression |
| `vgpr_spill_count > 0` | CRITICAL | Confirmed VGPR spills |
| `accum_vgpr = 0` with MFMA instructions | WARNING | MFMA results in arch VGPRs (check `maxnreg` or `waves_per_eu`) |
| `waves_per_simd = 1` | WARNING | Minimum occupancy — memory latency fully exposed |
| `LDS > 32768` | INFO | Using >50% of available LDS (65536 bytes per workgroup on gfx942) |
| `arch_vgpr > 128` | WARNING | High register pressure — consider reducing live values |

---

## Multiple Kernels

Some `.amdgcn` files contain multiple kernels (e.g., main kernel + reduce kernel).
Parse each `.amdhsa_kernel` block separately and present all kernels in the table.

Detect kernel names from:
```asm
.amdhsa_kernel kernel_name_here
```

---

## Comparison Mode

When given two `.amdgcn` files (before/after optimization), present a comparison:

```
| Metric | Before | After | Delta |
|--------|:------:|:-----:|:-----:|
| arch VGPR | 148 | 104 | -44 (good) |
| accum VGPR | 0 | 132 | +132 (MFMA using AGPRs now) |
| SGPR | 80 | 42 | -38 |
| LDS | 8704 | 6784 | -1920 |
| Scratch | 356 | 0 | -356 (spill eliminated!) |
| Waves/SIMD | 1 | 2 | +1 (2x occupancy) |
```

---

## Quick Reference

### One-liner: extract key values from .amdgcn
```bash
grep -E "amdhsa_kernel |next_free_vgpr|next_free_sgpr|accum_offset|group_segment_fixed|private_segment_fixed" file.amdgcn
```

### Compute arch/accum from grep output
```
arch_vgpr  = accum_offset value
accum_vgpr = next_free_vgpr - accum_offset
             (if accum_offset == next_free_vgpr → accum = 0, all VGPRs are arch)
```

### ISA instruction count summary
```bash
# Count key instruction types
for p in v_mfma s_barrier s_waitcnt buffer_load ds_read ds_write ds_bpermute global_load; do
  echo "$p: $(grep -c "$p" file.amdgcn)"
done
```
