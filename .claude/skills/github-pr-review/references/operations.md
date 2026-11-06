# Setup and operations

## Runtime and prerequisites

The bundled helpers use Linux/POSIX, Python 3.10+, jsonschema, Git, gh and either
Codex CLI or Claude Code CLI. Scheduling uses a user systemd timer; one-shot
reviews can run without systemd. The skill instructions can also guide a review
in either agent. An installed skill is not a cloud scheduler.

Keep runtime configuration, state and reports in a private directory separate
from the skill checkout. Each scheduled run starts a fresh review CLI process
using its existing account/model settings. It does not preserve an interactive
conversation. Persisted configuration specifies the recurring task.

Codex uses a workspace-write sandbox with network access and no interactive
approvals. Claude Code uses dontAsk and its existing configured tool permissions;
unpermitted tool calls are denied. Verify required GitHub, checkout and test
commands can run unattended within those permissions. Do not bypass permissions
or alter global settings to make a test pass.

## Prepare a configuration

Copy [config.example.json](../assets/config.example.json) outside the repository.
Replace github_login with the verified identity from gh api user --jq .login.
The example repositories setting limits reviews to ROCm/aiter. Set a list of
OWNER/REPO names from the user's requested scope; an empty list permits all
repositories accessible to the account. This filter applies to fresh discovery
and incomplete reviews in the retry queue.
Set execution_authorized, publication_authorized and publish_reviews from the
user's existing instructions. Set review_backend to codex or claude; the helper
does not pass a model override. Optional gh, codex and claude fields specify
executable paths when they are absent from PATH.

The example cadence is 09:00 Asia/Shanghai. Preserve the user's actual timezone
and time. Configure authorized environments using [rocm-testing.md](rocm-testing.md),
or leave the object empty for changes that do not need remote testing. Configs
contain no passwords or tokens; use existing SSH and CLI authentication.

Inspect any existing reviewer first. Use its config, timer, state and reports
rather than making a second instance. Older runtimes named run.py can be operated
in place; migration is a separate requested update. Record authorization fields
if migrating to this runner, using consent already given in the conversation.

## Prepare and activate

From the skill directory:

~~~bash
python3 scripts/install.py --config /absolute/private/config.json \
  --directory /absolute/private/github-pr-review
python3 /absolute/private/github-pr-review/review_prs.py \
  --directory /absolute/private/github-pr-review --dry-run
~~~

Preparation copies the runner, prompt and schema to the runtime and renders
service/timer files under its systemd directory. It does not enable a timer,
contact GitHub, submit reviews, or overwrite an existing runtime. A dry run
verifies identity and discovers PRs without reviewing or publishing.

When scheduling is authorized, activate on the initial installation:

~~~bash
python3 scripts/install.py --config /absolute/private/config.json \
  --directory /absolute/private/new-runtime --activate
~~~

For an already prepared runtime, inspect rendered files, verify the active
GitHub identity and that destination unit names are unused, then install them:

~~~bash
install -m 600 /absolute/private/github-pr-review/systemd/github-pr-review.service ~/.config/systemd/user/
install -m 600 /absolute/private/github-pr-review/systemd/github-pr-review.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now github-pr-review.timer
systemctl --user show github-pr-review.timer -p ActiveState -p NextElapseUSecRealtime
~~~

Create the user unit directory if absent. Use --unit-name for a user-requested
separate instance. The installer refuses existing runtime files and destination
units. It validates configuration and, before activation, identity, dependencies
and recorded execution authorization.

Verify the timer's actual next execution. Persistent=true runs a missed schedule
when the user manager next starts. The computer needs power and network access,
and remote allocations/reservations must remain valid. Inspect loginctl
show-user USER -p Linger; if execution after logout is requested and lingering is
absent, configure it using the host's permitted method. The installer does not
enable lingering.

## Run and inspect

~~~bash
python3 /absolute/private/github-pr-review/review_prs.py \
  --directory /absolute/private/github-pr-review
python3 /absolute/private/github-pr-review/review_prs.py \
  --directory /absolute/private/github-pr-review --pr https://github.com/OWNER/REPO/pull/NUMBER
systemctl --user list-timers github-pr-review.timer
systemctl --user status github-pr-review.service
journalctl --user -u github-pr-review.service -n 50 --no-pager
~~~

The --pr selector must belong to the selected PR set or incomplete retry queue.
publish_reviews=false saves review.md and review-payload.json as drafts; true
allows the runner's sole GitHub mutation, POSTing a PR review. Open, non-draft
PRs are selected from the union of requests and assignments. Overlapping runs
exit after observing the runtime lock. GitHub's 1000-result search cap raises an
error rather than silently truncating candidates.

Reports contain task SHAs, result.json, events.jsonl, agent.stderr.log, review
text/payload and test evidence. latest.json points to the newest report;
state.json records submitted review IDs and their commit/verdict. Logs remain
private. Publish only needed findings and sanitized validation evidence.

An active terminal review suppresses repeat review of its unchanged head.
Dismissed reviews can be retried. Incomplete COMMENT reviews remain queued after
GitHub removes a review request. Closed PRs leave the candidate set.
Before posting, the runner checks head/base and acting identity and supplies
the reviewed commit_id. GitHub offers no atomic head/base-and-review operation:
a PR can advance between the check and POST. The explicit commit ID attaches
the review to the commit actually examined.

On an ambiguous POST failure, do not immediately repost: the next authorized
run checks GitHub's marker and active verdict before publishing. Preserve state
for recovery. Missing essential tests produce COMMENT; agent crashes, invalid
JSON, schema failures or a changed account cannot approve.

## Stop, resume and change configuration

~~~bash
systemctl --user disable --now github-pr-review.timer
systemctl --user is-active github-pr-review.timer
~~~

Stopping a timer prevents new runs without cancelling an active service.
When cancellation is requested, stop its service and verify it exited too.
Keep state/reports. Resume only when requested with systemctl --user enable
--now github-pr-review.timer, then verify the next run.

Configuration is read on each run. Apply scope/publication changes when no review
is active so an active review cannot retain obsolete consent. For a cadence
change, stop the timer, update both config and installed OnCalendar, daemon-reload,
then resume when continued work is authorized. Update installed scripts
separately when requested; changing the skill checkout does not silently change
an installed automation.
