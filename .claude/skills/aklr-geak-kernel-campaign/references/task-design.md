# Designing the GEAK task for a kernel that lives in another repository

The task directory is what GEAK copies into every workspace (canonical, each engineer's, each verifier's). It must
be small, self-describing and must measure **this workspace's** candidate, nothing else.

## Layout

```
<task>/
├── config.yaml            # source_file_path (editable files), compile/correctness/performance commands
├── program.md             # the brief every agent reads (assets/program.example.md)
├── <candidate>/           # editable copies of the kernel files, laid out as in the package; NOT named aiter/
├── scripts/task_runner.py # the only measurement entry point (scripts/example_pa_decode_task_runner.py)
└── notes/                 # optional: carried-over analysis/profile from earlier runs (README.md first)
```

- **Never name the candidate directory `aiter/`.** GEAK's `materialize_workspace.sh --link-aiter` replaces a
  top-level `aiter/` (or `aiter/aiter/`) with a symlink to one shared read-only copy, so edits would leak between
  workspaces or vanish. It also excludes `build/`, `aiter/jit`, `*.so`, `*.o`, `__pycache__` and `.git` when copying.
- **Keep the measurement under `scripts/task_runner.py`.** With `GEAK_FREEZE_HARNESS=1`, GEAK freezes exactly
  `scripts/harness_run.py`, `scripts/task_runner.py`, `scripts/_runtime.py` and `test_cases.json` (plus
  `config.yaml`, `program.md`) and restores them before every measurement. A helper under another name is not
  protected.

## Runner contract (what GEAK's agents parse)

| Mode | Must print | Exit |
|---|---|---|
| `compile` | `Compilation: PASS` or `Compilation: FAIL` + `Error: ...` | 0 / 1 |
| `correctness` | `Correctness: PASS` or `Correctness: FAIL`, one line per failing case with the reason | 0 / 1 |
| `performance` | one `Perf: <ms> ms (<case id>)` line per case; optionally a geomean line | 0 / 1 |

Also write `build/<mode>_report.json` (the AKLR/GEAK runners do). Accept `--case <substring>` so agents can time a
subset quickly.

## Bind the candidate to the measurement (overlay)

Benchmarks import the kernel from a package (`aiter.ops.flydsl.pa_decode`). Do not pip-install per workspace, and
do not edit a shared checkout. On every invocation:

1. Rebuild `build/<pkg>_src/` from scratch: real directories along the path to the candidate files, **symlinks**
   to the immutable base checkout for everything else (fast: a few hundred links), and **real copies** of every
   candidate file (a symlink would make `Path(module.__file__).resolve()` point outside the overlay).
2. Refuse to write into a parent that resolves outside the overlay (a symlinked directory would write into the base).
3. Delete the overlay copy of any original candidate file the workspace deleted.
4. Import with `sys.path.insert(0, overlay)` and assert `module.__file__` resolves inside the overlay.
5. Derive everything from the runner's own location (`Path(__file__).parent.parent`), never from a path baked in
   when the task was written: GEAK copies the task around and checks source binding.

Keep the JIT artifacts of the base (e.g. `aiter/jit/module_*.so`) built once in the base checkout; the overlay
links to them.

### Compile caches

Check how the DSL caches compiled kernels before trusting any timing. FlyDSL keys its cache on the function
source, its closures and (for aiter's PA) an `implementation_cache_tag()` hash of the PA files; helpers in other
files may not be covered. Set the cache dir per candidate content:
`FLYDSL_RUNTIME_CACHE_DIR=<root>/<sha256 of all candidate files>`. Identical sources share binaries; any edit gets a
fresh cache. Clear `FLYDSL_AUTOTUNE`-style knobs that could make runs differ.

### Run cases in a child process

Run the cases in a subprocess and parse one JSON line per case. A GPU memory fault kills the process: mark the case
it was on, retry the remaining cases once, and fail a case that crashes twice. Pre-existing flaky faults then cost
one retry instead of rejecting a correct candidate.

### Reuse the benchmark, do not rewrite it

Import the benchmark script as a module (`importlib.util.spec_from_file_location`) and call its own functions
(case matrix, input builder, reference, checks, timer) instead of its `main()`. You get the exact inputs and timing
of the published numbers, and you avoid `main()` side effects such as `git rev-parse` on the source tree. Wrap its
functions (monkeypatch module attributes for the duration of a call) to add checks.

## Harden the correctness oracle (do this before any agent runs)

An agent loop optimizes against whatever the oracle accepts. Prove the oracle rejects wrong kernels:

1. **Probe it.** In a throwaway copy of the task, break something small and real (for attention: the `LOG2E`
   constant by 10% and by 3%; for GEMM: one K-tile skipped; a scale index off by one) and run `correctness`. If it
   still passes, the oracle is too weak. In the PA example, the benchmark's BF16 check (`rtol=atol=5e-3`) passed a
   10% softmax-temperature error: its inputs give logits with std ~0.09, a nearly uniform softmax, outputs ~3e-3,
   below the absolute tolerance.
2. **Add an input regime that exposes the bug class.** For attention, re-run each case with Q scaled (x32 gave
   logits std ~3, a peaked softmax). Reuse the harness's own measure path by wrapping its input builder.
3. **Use relative per-row error limits calibrated on the pristine kernel**, not absolute tolerances: record the
   pristine worst case, allow ~2x for BF16, ~1.5x per case for FP8.
4. **Differential check against stored pristine outputs** for low-precision paths whose error vs an FP32 reference is
   dominated by quantization noise (FP8 here: 2-3% row error vs FP32 on the benchmark inputs, 7-27% with peaked inputs). First confirm the pristine kernel is
   bitwise deterministic (diff 0.0 across runs). Then measure how far *harmless* changes move it: in the PA case a
   0.5% softmax-scale change already moved FP8 outputs by up to ~0.026 through P-rounding flips, so the FP8
   differential limit (0.035) only catches structural bugs; the BF16 limits are the fine-grained guard.
5. **Re-run the probes** after adding the checks and record which probe each check catches.
6. Make the pristine-output store read-only (`chmod -R a-w`) and keep the "record" switch maintainer-only.

## Scope and scoring

- **One regime per run.** The score is a mean over the scored cases; mixing latency-bound and bandwidth-bound cases
  tunes for whichever is more numerous. Score one regime; keep every case that shares the code in correctness.
- **Measure noise first.** Run `performance` three times on the pristine kernel. GEAK commits a round winner only
  above `min_improve` (default 2% geomean); if per-case noise is larger, tell the agents in `program.md`.
- **Check the timer floor.** If the harness cannot resolve below some latency (AKLR's timer reads ~4 us per dispatch),
  gains below it are invisible; list such cases in `program.md`.
- **Record the baseline yourself** (full matrix, saved JSON) before launching; it is the reference for the final A/B.
