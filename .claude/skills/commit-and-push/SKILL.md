# Commit and Push Staged Changes

## Overview

Create a commit from files that are already staged with `git add`, then push it to the current branch with `git push`. Always run `/format-code` before `git commit`. If formatting modifies files that were already staged, re-stage only those originally staged files before continuing; never add other unstaged or untracked files.

## Steps

1. **Check staged state**
   - Run `git status` and confirm that at least one file is staged.
   - If there are no staged files, tell the user to run `git add` first, then stop.
   - Record the current staged file list with `git diff --cached --name-only`.
   - If any staged file already has unstaged changes before formatting, stop and ask the user to sort out staging manually. For example, `git status --short` showing `MM <file>` means a later `git add <file>` would mix pre-existing unstaged edits into the commit.

2. **Run `/format-code` before committing**
   - Invoke `/format-code` to format the currently staged/modified code.
   - After formatting, run `git status` again.
   - If formatting made originally staged files modified/unstaged, run `git add <file>` only for files recorded in step 1, then continue.
   - Do not stage files that were not in the step 1 staged list, even if `/format-code` changed them.

3. **Review the staged changes**
   - Run `git diff --cached` to inspect the staged diff.
   - Generate a concise, accurate commit message from the actual staged changes.

4. **Create the commit**
   - Generate the commit message from the staged diff.
   - Use this format: `git commit -s -m "type(scope): Description"` (`-s` adds `Signed-off-by`).
   - Example: `git commit -s -m "fix(pa): Correct sliding window mask in decode kernel"`
   - Message rules: 72 characters or fewer, imperative mood (`Fix` / `Add` / `Update`), capitalized description, no trailing period.

5. **Push to the current branch with `git`**
   - Record the branch name with `git branch --show-current`.
   - If the current branch already has an upstream, run `git push`.
   - If the branch has no upstream, run `git push -u origin HEAD`.
   - If the push is rejected because the remote branch has new commits, run `git pull --rebase`, then retry the same `git push` command once.

## Rules

- **Limit `git add` scope**: By default, commit only files the user already staged with `git add`. The only allowed automatic `git add` is re-staging files that were recorded as staged in step 1 and then changed by `/format-code`. Never add new files, untracked files, or modified files outside the original staged list.
- **Format before commit**: Always run `/format-code` before generating `git diff --cached` and the commit message.
- **Avoid mixing old unstaged edits**: If a staged file already has unstaged changes before `/format-code`, stop and ask the user to resolve staging manually because automatic `git add` cannot distinguish old unstaged edits from formatting edits.
- **Commit message**: Base the message on the actual staged diff and describe the purpose of the change.
- **Push**: Push to the current branch with `git push`; use `git push -u origin HEAD` when the branch has no upstream. Do not require GitHub CLI for pushing.
