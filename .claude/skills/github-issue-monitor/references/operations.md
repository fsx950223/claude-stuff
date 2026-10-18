# Watcher operations

## Runtime and delivery adapter

The helper supports Linux/POSIX, Python 3.10+, `gh`, and an existing authenticated
GitHub session. It uses `fcntl` locks and `/proc` to verify the process it stops.
Each instance monitors one issue. A detached process runs only while its host or
container remains alive; it has no automatic reboot restart or cloud schedule.
If the user wants a persistent service, integrate with the environment's
supported scheduler and verify that service separately.

Delivery is a configured argument array containing one standalone `{message}`
argument. The helper calls it with `shell=False`. Comment text must be treated as
data, never shell code, `eval`, or an inline program. An adapter must acknowledge
acceptance with exit code 0 and preserve the destination session's context.

For Codex, first verify the installed `codex queue --help` supports an existing
session. Its documented form is:

```text
codex queue --thread SESSION_UUID --message MESSAGE
```

For Claude Code or another agent, use the environment's actual supported inbox,
session tool, or a previously validated adapter executable. There is no assumed
`claude queue` command. If no existing-session interface is available, report that
delivery is unavailable; do not replace it with fresh autonomous sessions and
pretend conversation context was retained. An adapter JSON file looks like:

```json
["/absolute/path/to/session-adapter", "--session", "SESSION_ID", "--message", "{message}"]
```

Keep credentials in the existing authenticated tools or environment, not in the
argument file. `--thread` supplies the Codex argument array; use
`--delivery-command-file` for other adapters. Neither is a request to create a
new session, alter its model, or bypass its sandbox/approvals.

## Initialize without starting

Use an absolute private runtime directory distinct from the skill checkout. Copy
the script there so upgrading or deleting a checkout cannot change a running
instance. The example values below are placeholders, not session authorization.
Use the actual user-approved values.

```bash
install -d -m 700 /absolute/runtime/issue-monitor
install -m 700 scripts/watch_issue.py /absolute/runtime/issue-monitor/watch_issue.py

python3 /absolute/runtime/issue-monitor/watch_issue.py \
  --directory /absolute/runtime/issue-monitor init \
  --repository OWNER/REPO --issue 1 \
  --author REQUEST_AUTHOR --mention @REQUEST_AGENT \
  --interval-seconds 900 --cwd /absolute/workspace \
  --thread SESSION_UUID \
  --scope 'Authorized development tasks in this workspace; report results to this issue.' \
  --authorize-execution --authorize-results
```

Use `--hostname HOST` for GitHub Enterprise, `--gh /path/to/gh` when needed, and
`--codex /path/to/codex` with `--thread`. The result author defaults to the verified
`gh api user` login. Add `--result-author LOGIN` only for identities already
authorized to provide completion results. A new setup must not overwrite an
existing config or state. Use `--include-existing` only when the user explicitly
requests historical tasks. Initialization validates issue access and establishes
the cursor immediately, so requests arriving between init and start are retained.

If execution is not authorized, don't initialize an executing watcher. If result
posting is not authorized, omit `--authorize-results`: the agent must report in
chat and the watcher deliberately waits at that task until the user resolves the
result-posting requirement. Do not invent posting permission to clear the queue.

The runtime contains private `config.json`, `state.json`, PID/lock files, and
rotating logs. `status` prints a summary, not complete configuration, session ID,
command arguments, comment bodies, or delivery receipts. Don't paste those
private files wholesale into reports.

## Inspect, start, and poll

```bash
python3 /absolute/runtime/issue-monitor/watch_issue.py --directory /absolute/runtime/issue-monitor status
python3 /absolute/runtime/issue-monitor/watch_issue.py --directory /absolute/runtime/issue-monitor once --dry-run
python3 /absolute/runtime/issue-monitor/watch_issue.py --directory /absolute/runtime/issue-monitor preview COMMENT_ID
python3 /absolute/runtime/issue-monitor/watch_issue.py --directory /absolute/runtime/issue-monitor start
```

`once --dry-run` reads GitHub and previews eligible IDs without changing the
cursor or delivering messages. `preview` intentionally prints a task payload;
keep it out of public reports. `start` verifies that the daemon acquired its lock
and wrote a matching PID record. Don't claim monitoring is active from a Popen
PID alone. `once` performs one real authorized poll when no daemon holds the
lock; running it is an execution/delivery action.

First startup preserves the initialization baseline. A subsequent `start` skips
new comments and cancels undelivered pending work from the pause by default. Use
`start --include-paused` only when the user explicitly requests the backlog.
Delivered tasks retain their state so they cannot be silently replayed.

The poll reads every GitHub page, discovers new requests, checks completion
markers, and delivers at most one task. One unresolved delivery gates all later
tasks. Every task is re-read before it is queued and again by the receiving
agent before execution. A temporary API error preserves the cursor; a daemon
retries fetches after up to 60 seconds. Delivery errors are not automatically
retried because acceptance may be ambiguous.

## Execute and post a task result

```bash
python3 /absolute/runtime/issue-monitor/watch_issue.py --directory /absolute/runtime/issue-monitor inspect COMMENT_ID
```

This re-reads the live trigger, checks its eligibility, finds matching result
comments from configured result authors, and verifies each match using its exact
comment endpoint. A 404 means the trigger is deleted; other errors propagate.
The watcher ignores result comments containing the trigger mention, preventing
an author from quoting a result marker in a request to suppress execution.

Write a result body outside the repository. It should contain:

```markdown
Handled [the request](https://github.com/OWNER/REPO/issues/1#issuecomment-COMMENT_ID).

What changed or was measured, how the outcome was verified, and any real blocker.

<!-- issue-agent-task:OWNER/REPO#1:COMMENT_ID -->
```

Then use:

```bash
python3 /absolute/runtime/issue-monitor/watch_issue.py \
  --directory /absolute/runtime/issue-monitor post-result COMMENT_ID \
  --body-file /absolute/path/result.md
```

The helper requires recorded posting authorization, verifies the current GitHub
login, checks for an existing completion result, posts using
`gh issue comment --body-file`, re-reads the returned comment ID, and compares the
entire body, author and marker. It does not execute the task for the agent. It
also permits reporting a deleted/withdrawn trigger as skipped. The marker marks
the request handled; describe unresolved underlying work honestly.

If writing directly with `gh`, perform the same deduplication and readback checks.
Don't add a marker to an unhandled task solely to release the queue.

## Stop, change cadence, and resume

```bash
python3 /absolute/runtime/issue-monitor/watch_issue.py --directory /absolute/runtime/issue-monitor stop
python3 /absolute/runtime/issue-monitor/watch_issue.py --directory /absolute/runtime/issue-monitor status
```

Stop disables future polls/deliveries before signalling the verified PID. It
waits for exit and the released daemon lock, and only then reports `running:
false`. It cancels undelivered pending work; it cannot retract an accepted agent
message. An in-flight delivery may finish before shutdown. If the stop times out,
inspect again and wait for that command to finish; do not claim exit or signal an
unverified PID. Do not hard-kill an adapter after an ambiguous acceptance.

For an existing custom watcher, inspect its own stop interface, process and
scheduler; don't install a second instance to stop the first. Use filtered
status fields and verify lock release without printing private config. When an
authorized trigger asks to stop its own monitor, stop first, verify, then post
its result and exact completion marker. Do not restart to record the marker:
the result remains verifiable on GitHub.

To change the interval, stop and verify, use
`set-interval --seconds N`, then start if the user requested continued monitoring.
Stopping alone withdraws monitoring authorization; it does not implicitly
authorize a later restart. After reboot, restart only according to the user's
explicit persistence instruction.

## Recover delivery problems

| Status | Meaning | Next action |
|---|---|---|
| `pending` | Discovered and not delivered | Wait for the current task or resume only if authorized. |
| `delivering` | Intent persisted; adapter outcome not recorded | Inspect session/receipt and GitHub marker. Treat as uncertain after a crash. |
| `queued` | Adapter accepted the message | Wait for the task result; do not deliver again. |
| `completed` | Result author and live marker verified | Retain the result URL; no replay. |
| `delivery_failed` / `delivery_uncertain` | Adapter failed or acceptance is unknown | Stop, investigate, and resolve before later deliveries. |
| `skipped` / `cancelled_on_stop` / `cancelled_on_resume` | No task was delivered | Retain the reason; post a new request if needed. |

An exact result marker resolves a delivered task even after a crash. If there is
no result, verify whether the adapter accepted the message. Only when the user
authorizes retry and delivery is confirmed absent, stop the daemon and run:

```bash
python3 /absolute/runtime/issue-monitor/watch_issue.py \
  --directory /absolute/runtime/issue-monitor retry COMMENT_ID \
  --confirmed-not-delivered
```

Resume with `start --include-paused` to retain that pending retry. Retry creates a
recorded pending attempt; it never erases the cursor or completed tasks. Unknown
acceptance remains blocked until the user or session state resolves it. Don't
automatically replay it, hand-edit state to `completed`, change the destination
thread, or reset the baseline.
