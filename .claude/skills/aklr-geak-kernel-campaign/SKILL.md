---
name: aklr-geak-kernel-campaign
description: >
  Optimize an AMD GPU kernel with AKLR's agent tooling: wrap the kernel and its benchmark as a GEAK kernel_workflow
  task (or use AKLR's forge_task when the kernel is an AKLR op and KernelForge is installed), harden the correctness
  oracle before any agent runs, launch and watch the multi-agent campaign on a shared GPU machine, stop or rerun it,
  and verify the result yourself before delivering. Use when asked to optimize a kernel "with AKLR", "with GEAK",
  "with the tool", to set up such a task for a kernel in aiter or another repo, or to check on, stop or rerun such a
  campaign.
tools: Read,Write,Edit,Bash,Grep,Glob,AskUserQuestion
---

# Optimize a kernel with AKLR / GEAK

An agent loop optimizes against whatever the measurement accepts. Most of the work, and most of the value, is in
building a measurement that binds to the candidate, scores the right regime, and rejects wrong kernels. Launching
the agents is the easy part.

Reference files:
- [task-design.md](references/task-design.md): task layout, runner contract, overlay, oracle hardening, scope.
- [operations.md](references/operations.md): install, pre-flight, launch, watch, stop, rerun, deliver.
- [pitfalls.md](references/pitfalls.md): failures seen in a real campaign and their fixes.
- [scripts/example_pa_decode_task_runner.py](scripts/example_pa_decode_task_runner.py): a complete, tested runner
  (aiter FlyDSL paged-attention decode + its benchmark); copy it and change the `ADAPT` parts.
- [scripts/](scripts/): `install_node.sh`, `wait_and_launch.sh`, `watch_geak.sh`, `stop_geak.sh`.
- [assets/](assets/): `config.example.yaml`, `program.example.md`, `geak_args.template.json`.

## 1. Choose the engine

| Situation | Use |
|---|---|
| The kernel is an AKLR op (`aklr/ops/<op>/impls/...json` schema) and KernelForge (Hyperloom) is installed | AKLR's own flow: `docs/forge_task.md`, skill `forge-loop-campaign` in the AKLR repo |
| AKLR op, no KernelForge | AKLR skill `geak-campaign` (forge_task prepare + GEAK) |
| Kernel in another repo (aiter, FlyDSL, ...) with its own benchmark | **this skill**: a custom GEAK task around that benchmark |
| A few hours of budget is not justified, or the arch is too new for agents to profile | optimize directly, still with the hardened runner below as the A/B oracle |

Always use AKLR's knowledge base as reading material for the agents: `harness/skills/aklr-kb/SKILL.md` and its
op folders (gfx950-centric: mechanisms carry to other arches, numbers do not).

Agents spend money from launch on. Before launching, settle with the user: engine, the scored regime, and the budget
(GEAK `budget` = total optimization directions over all rounds; expect hours of wall time).

## 2. Scope and baseline

1. Get the benchmark (branch/commit) into an **immutable base checkout** on a path the GPU container sees.
2. Run its full matrix once on an idle GPU and save the JSON: this is the baseline for the final A/B.
3. Pick **one regime** to score (e.g. dense decode+verify, not also prefill); keep every case that shares the code
   in correctness.
4. Measure noise: three runs of the scored set on the pristine kernel.

## 3. Build the task (references/task-design.md)

- `config.yaml` (assets/config.example.yaml): editable files, the three commands.
- Candidate files copied into `<task>/<candidate>/` (never a top-level `aiter/`).
- `scripts/task_runner.py` from the example runner: overlay rebuilt per call, benchmark imported as a module,
  per-content compile cache, cases in a child process with one crash retry, GEAK's output formats.
- `program.md` (assets/program.example.md): operator, scored workloads with baseline, the exact correctness limits,
  rules (edit only the candidate, only the given GPU, keep the API, keep other arches unchanged, graph-safe,
  deterministic, no spin-waits, no shape special-casing), where to look.

## 4. Harden the oracle, then prove it

1. Probe the benchmark's checks with deliberately broken copies (e.g. softmax scale off by 10% and 3%).
2. Add what the probes show is missing: a peaked-input pass, per-row relative limits calibrated on pristine, and
   for low-precision paths a differential check against stored pristine outputs (after confirming the pristine
   kernel is bitwise deterministic). With the example runner:
   `PA_RECORD_PRISTINE=1 python3 scripts/task_runner.py correctness` on the pristine sources, then `chmod -R a-w`
   the store.
3. Re-run the probes: each must fail with a clear message; the pristine kernel must pass with zero diff.

## 5. Install, pre-flight, launch (references/operations.md)

1. GEAK cloned, Node >= 20 (`scripts/install_node.sh`), a `--without-pip --system-site-packages` venv with GEAK
   installed, the Claude CLI authenticated inside the container, API endpoint reachable from it.
2. Pre-flight: `compile`, `correctness`, `performance` on the chosen GPU; one run through GEAK's
   `materialize_workspace.sh` + `gpu_lock.sh`.
3. Launch detached inside the container with `GEAK_FREEZE_HARNESS=1`, `GEAK_GPU_ALLOWED=<g>`,
   `HIP_VISIBLE_DEVICES=<g>`, `GEAK_CLAUDE_MODEL`, credentials forwarded by name. On a shared machine use
   `scripts/wait_and_launch.sh`, which waits for an idle GPU and re-runs the pristine pre-check there.
4. Record the run's PGID and tell the user the run dir, the GPU, and how to stop it.

## 6. Watch

- Arm `scripts/watch_geak.sh` as a background shell; re-arm it after every wake-up and in every new session.
- At each check-in: phase/round, `analysis.json`/`roadmap.md` sanity, harness files identical to the task's
  (`cmp`), base checkout clean, GPU still yours.
- After a `[hung-agent guard]` line, kill the orphaned agent process.
- Report to the user at phase changes and round results; do not relay GEAK's speedups as conclusions.

## 7. Stop or rerun

- Stop: `scripts/stop_geak.sh <run_dir> <exp_root>` (with `CONTAINER=<c>`), then confirm no process of yours is
  left on the GPU.
- A run killed by a reboot or crash cannot be resumed headless. Restart the container if needed
  (`docker start`), copy the dead run's analysis and profile into `<task>/notes/`, point `program.md` at them, and
  relaunch (ask the user first: it spends the budget again).

## 8. Verify and deliver

Only after `geak_result.json` says `accepted`:
1. Extract the winning diff (candidate dir only) and review it by hand.
2. Re-run full correctness and an alternating pristine/candidate A/B (>= 3 each) on an idle GPU, plus the
   non-scored suites for regressions.
3. Port it to a branch of the kernel's repo, run its tests and formatters, open the PR with the A/B table. Report
   numbers you measured, not GEAK's.

## Rules

- Never measure on a GPU another user is using; never override GEAK's idle check for numbers you will report.
- Never edit the base checkout, the benchmark, or the pristine-output store during a campaign.
- Keep credentials out of commands, files and logs (`docker exec -e NAME` forwards by name).
- Ask before relaunching or extending a budget; the user pays for every agent.
