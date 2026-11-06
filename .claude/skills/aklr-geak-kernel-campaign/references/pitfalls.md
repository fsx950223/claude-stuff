# Pitfalls seen in a real campaign (aiter FlyDSL PA decode on a new AMD arch, shared 4-GPU box)

| Symptom | Cause | Fix |
|---|---|---|
| `forge_task` does not apply | AKLR's `forge_task` wraps AKLR schemas (`aklr/ops/<op>/impls/...json`) and drives KernelForge, which lives in a private Hyperloom branch; its flow and docs target gfx950 | For a kernel in another repo, or without KernelForge, wrap it as a GEAK task (this skill). Use AKLR for its knowledge base (`harness/skills/aklr-kb`) and its `geak-campaign` skill as the template |
| Agents' edits leak between workspaces or vanish | candidate dir named `aiter/`; GEAK's `--link-aiter` swaps it for a shared symlink | name it e.g. `pa_src/` and overlay it in the runner |
| A deliberately broken softmax passes correctness | benchmark inputs give a near-uniform softmax; BF16 outputs (~3e-3) sit below `atol=5e-3` | probe the oracle; add a peaked-input pass and relative per-row limits calibrated on pristine (`task-design.md`) |
| FP8 checks cannot tell a 3% softmax-scale bug from noise | FP8 Q/P quantization dominates the error vs FP32 (2-3% per row on the benchmark inputs, 7-27% with peaked inputs); harmless rounding changes move outputs ~1-2.6% | differential vs stored pristine outputs with a structural-bug limit; rely on BF16 for fine numerics |
| Edits seem to have no effect / stale timings | DSL compile cache keyed on a subset of sources | per-content cache dir (`FLYDSL_RUNTIME_CACHE_DIR=<root>/<sha256 of candidate files>`) |
| `py_compile.compile(..., cfile=os.devnull)` raises `FileExistsError` | py_compile refuses a non-regular cfile | syntax-check with `compile(source, path, "exec")` |
| Profile phase runs 60 min, then `[hung-agent guard] ... resolving null` | rocprofv3 PMC counter passes were slow, refused (co-tenant) or hung on the new arch | say in `program.md`: use `--kernel-trace` and ISA dumps, no PMC sweeps; seed `notes/` with an existing profile |
| Tokens keep being spent after the guard fired | the guard resolves the promise but leaves the `claude -p` process running | kill that agent process and its children |
| `gpu_lock.sh`: "GPU N still shows foreign work after 30s ... refusing to measure" | another user's job on the card (busy > 5% or VRAM > 1 GB) | wait for an idle card; do not set `GEAK_GPU_REQUIRE_IDLE=0` for timing you intend to trust |
| The run disappeared mid-round; `geak.log` stops; container `Exited (255)` with "error gathering device information ... /dev/kfd" | host reboot; the container's restart raced the GPU driver | `docker start <container>`; rerun (no headless resume); copy the dead run's analysis into `notes/` |
| Background `watch` shells vanished | they belong to the Claude session; a new session starts without them | run long-lived waiters inside the container with `docker exec -d ... setsid`; re-arm watchers each session |
| Waiter found an idle GPU, ran the pre-check, then "GPU got busy" forever | it re-tested idleness right after its own pre-check; `gpu_busy_percent` still showed the pre-check | sleep ~15 s after the pre-check before re-testing (`wait_and_launch.sh` does) |
| `python3 -m venv` fails: ensurepip not available | minimal container | `python3 -m venv --without-pip --system-site-packages`, then `<venv>/bin/python -m pip ...` |
| `node: command not found` | GEAK's headless runtime needs Node >= 20; not in ROCm images | `scripts/install_node.sh` (user-level, checksum-verified) |
| Intermittent GPU memory fault on one long-context case, also on main | pre-existing, cause unknown | runner retries a crashed case once; document it in `program.md` |
| Wrong GPU measured | `HIP_VISIBLE_DEVICES` index vs `amd-smi` index vs `/sys/class/drm/renderD(128+8*id)` assumed equal | confirm the mapping once (BDF in `amd-smi list` vs `readlink -f /sys/class/drm/renderD*/device`) |
| Budget used up by round 1 | the planner may spend 3 of budget 4 on round 1, leaving one direction | pick `budget` for the number of directions you want in total, not per round |
| No cost figure at the end | GEAK records no token/cost numbers | track wall time; estimate from your gateway's usage page |
