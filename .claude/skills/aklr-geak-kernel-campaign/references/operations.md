# Operating a GEAK campaign: install, launch, watch, stop, deliver

Paths below: `$WORK` is a directory visible both on the host and inside the dev container (e.g. a bind-mounted home
directory); `$C` is the container that has the GPU stack (torch, triton/FlyDSL, the kernel's repo deps).

## Install (no root needed)

```bash
# GEAK (public). Keep it on a path the container sees.
git clone https://github.com/AMD-AGI/GEAK.git $WORK/GEAK

# Node.js >= 20 for GEAK's headless runtime, verified against nodejs.org checksums
bash scripts/install_node.sh                      # -> ~/.local/opt/node

# A venv that sees the container's torch/flydsl but isolates GEAK's deps.
# Containers often lack ensurepip: create it without pip and use the system pip through it.
docker exec $C bash -lc "python3 -m venv --without-pip --system-site-packages $WORK/venv &&
  $WORK/venv/bin/python -m pip install -q -e $WORK/GEAK"   # GEAK_SKIP_BOOTSTRAP=1 avoids a 2nd clone + CLI install

docker exec $C bash -lc "export PATH=$HOME/.local/opt/node/bin:\$PATH; node $WORK/GEAK/interface/runtime/engine/selftest.mjs | tail -1"
```

The Claude Code CLI must be on PATH inside the container and authenticated. GEAK spawns
`claude -p --dangerously-skip-permissions --allowedTools Bash Read Write --model $GEAK_CLAUDE_MODEL`. Pass the
credentials by name only, so they never appear in commands or logs:
`docker exec -e ANTHROPIC_API_KEY -e ANTHROPIC_BASE_URL -e ANTHROPIC_CUSTOM_HEADERS ...` (forward whichever your
setup uses). If both OpenAI-side and Anthropic-side keys are set, GEAK picks claude; see GEAK
`interface/runtime/SETUP.md`.

Check the container can reach the API endpoint (`curl -sS -o /dev/null -w '%{http_code}' $ANTHROPIC_BASE_URL`).

## Pre-flight on the chosen GPU

```bash
docker exec -e HIP_VISIBLE_DEVICES=<g> -w <task> $C bash -lc '. $WORK/venv/bin/activate
  python3 scripts/task_runner.py compile && python3 scripts/task_runner.py correctness | tail -1 &&
  python3 scripts/task_runner.py performance | tail -1'
```

Then simulate GEAK's copy + lock path once (catches source-binding mistakes):

```bash
bash $WORK/GEAK/kernel_workflow/scripts/materialize_workspace.sh --src <task> --dst /tmp/probe_ws \
  --shared-root /tmp/probe_shared --link-aiter
cd /tmp/probe_ws && GEAK_GPU_ALLOWED=<g> bash $WORK/GEAK/kernel_workflow/scripts/gpu_lock.sh <g> \
  python3 scripts/task_runner.py performance --case <one case>
```

## Launch

Direct, when a GPU is idle now (detached, own process group, survives the Claude session):

```bash
docker exec -d -e ANTHROPIC_API_KEY -e ANTHROPIC_BASE_URL -e ANTHROPIC_CUSTOM_HEADERS \
  -e GEAK_CLAUDE_MODEL=claude-opus-5-5 -e GEAK_FREEZE_HARNESS=1 -e GEAK_GPU_ALLOWED=<g> -e HIP_VISIBLE_DEVICES=<g> \
  -w $WORK/GEAK $C bash -lc "export PATH=$WORK/venv/bin:$HOME/.local/opt/node/bin:$HOME/.local/bin:\$PATH;
  exec setsid node interface/runtime/engine/run_workflow.mjs kernel_workflow/kernel_workflow.js \
    --args-file $RUN/geak_args.json --result-file $RUN/geak_result.json > $RUN/geak.log 2>&1"
docker exec $C bash -c 'ps -eo pid,pgid,cmd | grep run_workflow | grep -v grep'   # record the PGID in $RUN/geak.pgid
```

On a shared machine, prefer `scripts/wait_and_launch.sh` (waits for an idle GPU, re-runs the pristine pre-check on
it, fills `__GPU__` in `assets/geak_args.template.json`, launches, records `geak.pgid` and `gpu`).

`geak_args.json` keys that matter: `kernel_path` (task dir), `workflow_dir` (`GEAK/kernel_workflow`), `budget`
(total optimization directions over all rounds; a `deep_explore` direction costs `deep_cost`, default 2),
`gpu_ids`, `exp_root`, `task` (a steer every planner sees), `min_improve` (default 0.02), `agent_timeout_ms`
(default 3600000, applies to every agent, engineers included).

## Watch

Run `scripts/watch_geak.sh $RUN/geak.log 2700` as a background shell (with `CONTAINER=$C`); it returns on
progress, completion, death or the cap. Phases and what to check:

| Phase | Typical time seen | Check |
|---|---|---|
| Setup | 1-2 min | `EVAL_DIR` printed |
| Analyze | 6-9 min | `$EVAL_DIR/analysis.json`, `roadmap.md` make sense (code paths of the scored cases) |
| Benchmark | 7-10 min | "baseline geomean ... reliable=true" matches your own baseline |
| Profile | up to the 60-min agent guard | if it hits the guard, kill the orphaned agent (below) |
| Optimize rounds | 45+ min per round | `round_N/engineer_*/workspace` diffs; verified speedups in the log |
| Final report + Director validation | | `geak_result.json`: `validation_status`, `eval_dir` |

Every check-in, verify the harness was not touched and the base checkout is clean:

```bash
for f in scripts/task_runner.py config.yaml program.md; do cmp -s <task>/$f $EVAL_DIR/workspace/$f || echo "CHANGED $f"; done
git -C <base checkout> status --porcelain
```

After a `[hung-agent guard]` line, GEAK resolves that agent to null **but does not kill it**: it keeps spending
tokens and can hold the GPU lock. Find it (`ps -eo pid,ppid,etime,cmd | grep 'claude -p'`, the long-running one)
and kill it with its children.

## Stop

`scripts/stop_geak.sh $RUN $EXP_ROOT` (with `CONTAINER=$C`): kills the waiter, the workflow's process group (node and
every agent) and anything still running out of the experiment dir. Confirm the GPU shows no process from you.

There is no resume for a headless run that died (the engine's resume works only for the Claude Code `Workflow` tool
within one session; `state_dir` resume needs a committed cumulative best). Before rerunning, copy what the dead run
learned (`analysis.json`, `roadmap.md`, kernel-trace profile tables) into the task's `notes/` and point `program.md`
at it, so the new run does not pay for it again.

## Deliver (never on GEAK's word alone)

1. `geak_result.json`: `validation_status == accepted`; read `tech_lead_report.md` and each round's report.
2. Take the winning diff (`git -C $EVAL_DIR/workspace diff <baseline commit>` limited to the candidate dir).
3. Re-verify it yourself on an idle GPU: full `correctness`; `performance` alternating pristine / candidate at least
   3 times each; the non-scored suites' performance for regressions; the original benchmark's own `--compare-to`
   flow if it has one.
4. Review the diff by hand: arch gating, graph safety, determinism, no shape special-casing, no edits outside the
   candidate files.
5. Port the diff to a branch of the kernel's repository, run its unit tests and formatters, open the PR with the
   A/B table. If the change came through AKLR's `forge_task`, follow AKLR's `publish` and `aklr validate` instead.
