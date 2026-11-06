# Speed up aiter's FlyDSL paged-attention decode on <arch>: <model> dense decode and verify

<!-- Example brief for a GEAK task. Every agent (tech lead, engineers, verifiers) reads it. Fill the <...> parts from
     your own baseline run; keep the sections: operator, scored workloads with baseline, correctness gate (with the
     exact limits), rules, where to look. -->

## The operator

`pa_src/pa_decode.py::pa_decode` is aiter's FlyDSL paged-attention decode (`aiter.ops.flydsl.pa_decode`, ROCm/aiter
`<branch>` at `<commit>`). The task optimizes its `<arch>` path for one model's attention shape on one GPU:

- `<Hq>` query heads, `<Hkv>` KV heads, head dim `<D>`, softmax scale `D**-0.5`.
- Q and output BF16. KV either BF16, or FP8 E4M3FN with FP32 per-token K and V scales.
- Dense paged KV, page `<page>`; key cache `[pages, Hkv, D/x, page, x]`, value cache `[pages, Hkv, page/x, D, x]`.
- Decode: query_length 1. MTP verify: query_length 4, causally masked queries sharing one request's table.

## Scored workloads (performance)

The score is the geometric mean over these workloads of baseline latency / candidate latency, with the caller's real
calling convention (`<how the plan / tables / bounds are built>`), planning done before timing.

| Workload | BF16 us | FP8 us |
|---|---:|---:|
| `<case id>` | `<from your baseline run>` | `<...>` |

Timing (unchanged from the benchmark): a CUDA graph of N complete calls, M replays per GPU-event sample, the minimum
of S samples per round, the median of R rounds. Run-to-run noise is about `<x>`% per workload: confirm a small gain
by repeating.

Observations to start from (verify them, do not trust them): `<output mode / launch count per call, which cases
are latency-bound, which are at the measured bandwidth roof>`.

## Correctness gate (all workloads must pass)

`python3 scripts/task_runner.py correctness` runs every benchmark workload with both KV dtypes, including the suites
that are not scored but share the planner, tile kernel and reducer. A speedup that breaks or slows them is not
acceptable.

Each workload must pass:
1. The benchmark's own checks, eager and after CUDA-graph replay, then a replay with changed inputs and a
   NaN-filled output (the graph must read current inputs and write every output row).
2. Stricter limits (the benchmark's inputs give a nearly uniform softmax that hides softmax bugs):
   - BF16 worst per-row relative L2 vs the FP32 reference <= 2x the pristine worst;
   - the same workload with Q scaled by 32 (peaked softmax): BF16 <= 2x the pristine worst, FP8 <= 1.5x the
     pristine value of that workload;
   - FP8 output vs the pristine kernel's output on identical inputs: worst per-row relative L2 <= 0.035.

## Rules

- **Edit only `pa_src/`.** `scripts/`, `config.yaml`, this file, the immutable base checkout and the pristine-output
  store are the measurement. New modules go in `pa_src/` too; shared helpers outside it are immutable: copy what
  you need.
- **Only the GPU you were given.** Shared machine. Run every GPU command through the workflow's `gpu_lock.sh`.
- **Keep the API** the harness and real callers use (`<functions, plan fields, workspace shapes>`).
- **Other architectures must behave exactly as before**; this machine cannot test them. Gate changes on the arch.
- **CUDA-graph safe**: no host synchronization, `.item()`, value-dependent allocation or host reads of device data on
  the call path.
- **Deterministic**: keep a fixed reduction order.
- **No cross-workgroup spin-waits** except between workgroups guaranteed co-resident.
- **Generalize**: no special-casing of the exact benchmark shapes.

## Where to look

- **Start with `notes/README.md`** if present (carried-over analysis/profile from an earlier run).
- Code map: `<host wrapper>`, `<planner>`, `<tile kernel>`, `<pipeline>`, `<reducer>`.
- History: `git -C <base checkout> log --oneline -- <kernel paths>`.
- Tools: `FLYDSL_DUMP_IR=1` dumps IR and ISA; prefer `rocprofv3 --kernel-trace` over PMC counters (PMC collection
  can be slow or hang on new arches). Subset runs: `python3 scripts/task_runner.py performance --case <substring>`.
- AKLR's knowledge base (gfx950-centric; mechanisms carry, numbers do not): `<AKLR>/harness/skills/aklr-kb/SKILL.md`,
  `attention/INDEX.md`, `attention/decode_split_kv.md`, `attention/grid_dispatch.md`, `attention/softmax.md`.

Known issues: `<pre-existing flaky failures, and what the runner does about them>`.
