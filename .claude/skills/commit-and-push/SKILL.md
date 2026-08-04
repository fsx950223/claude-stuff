---
name: commit-and-push
description: Safely format, commit, and push already staged changes while matching the repository's exact CI formatter versions. Use whenever the user asks to commit, commit and push, push staged changes, fix code style before pushing, or update a PR branch.
---

# Commit and Push Staged Changes

Commit only what the user staged, preserve the intended diff, and prove that the
committed tree—not merely the pre-commit working tree—passes the same style tools
as CI before pushing.

## 1. Snapshot the staged scope

Run:

```bash
git status --short --branch
git diff --cached --name-only
git diff --name-only
```

- Stop if nothing is staged.
- Record the exact staged file list.
- Stop if a staged file also has pre-existing unstaged edits (`MM file`); re-adding
  it would mix unrelated work into the commit.
- Never add untracked or previously unstaged files automatically.

## 2. Discover CI's formatter versions

Do not assume the locally installed formatter matches CI. Inspect repository
wrappers and style workflows before formatting.

- Prefer the repository's style wrapper when it exactly mirrors CI.
- Read workflow pins such as `ruff==X.Y.Z`.
- Run that exact Ruff release; an older local Ruff can miss CI-only rules such
  as `PIE810` even when its check passes.
- If CI uses `psf/black@stable` without a version, use the current stable Black
  (`uvx black`) and compare it with the `black` executable that commit hooks call.
- If a failed run is available, its log is authoritative; for example,
  `gh run view <run> --job <job> --log-failed` reports the installed Black version.

Example checks:

```bash
black --version
uvx black --version
uvx --from ruff==<CI_VERSION> ruff --version
```

When the hook's formatter differs from CI, align the active environment before
committing (without editing project dependency files). Otherwise the hook can
silently undo CI-compatible formatting. For example, install the exact version
reported by CI:

```bash
python -m pip install "black==<CI_VERSION>"
```

## 3. Format only the recorded files

Run `/format-code`, then run the exact CI versions on the recorded Python files:

```bash
uvx --from ruff==<CI_VERSION> ruff check --fix <staged-python-files>
uvx black <staged-python-files>
```

Use the repository's configured clang-format for staged C/C++ files.

After formatting:

1. Inspect `git status --short`.
2. Re-stage only files recorded in step 1.
3. Run the CI-equivalent checks in check-only mode.
4. Run `git diff --cached --check`.

## 4. Review content and scope

Inspect:

```bash
git diff --cached
git diff --cached --stat
```

For a PR or a merge from the base branch, also inspect the effective PR scope:

```bash
git diff --name-only origin/main...HEAD
gh pr diff --name-only
```

Do not push formatter- or hook-induced changes in unrelated files. Pay special
attention to comment-only churn, Unicode-to-ASCII transliteration, and unrelated
directories introduced while resolving a merge.

## 5. Commit without bypassing hooks

Build the message from the staged diff. Use an imperative subject no longer than
72 characters and pass it through a heredoc:

```bash
git commit -s -m "$(cat <<'EOF'
type(scope): Description
EOF
)"
```

Never use `--no-verify`. If a hook creates unrelated changes or expands scope,
stop and report the hook behavior instead of bypassing it.

## 6. Validate the committed tree

Hooks may rewrite and stage files during `git commit`, so pre-commit checks alone
are insufficient. Immediately run:

```bash
git status --short --branch
git show --stat --oneline HEAD
git show --format= --name-only HEAD
```

Then rerun the exact CI Black/Ruff checks against the Python files committed in
`HEAD`. If the hook changed CI-formatted code back to an older style, do not push:

1. Align the hook's formatter with CI.
2. Reformat using CI's version.
3. Create a follow-up commit (never amend a pushed commit).

Recheck the effective PR file list after merge commits; hooks often process every
file changed relative to the first parent, including upstream files outside the
feature.

## 7. Push safely

Record the branch with `git branch --show-current`.

- With an upstream: `git push`
- Without an upstream: `git push -u origin HEAD`
- Never force-push unless the user explicitly requests it.
- If rejected because the remote advanced, fetch and replay/merge the remote
  changes, then retry once. Do not overwrite another contributor's update.

After pushing a PR, inspect style checks. If CI still fails, read the exact job
log, reproduce that pinned rule locally, and fix it in a new commit.
