# Commit and Push Staged Changes

## Overview

为已 `git add` 暂存的文件生成 commit，并 push 到当前分支。在 `git commit` 之前必须先执行 `/format-code`。格式化后如果原本已 staged 的文件被 formatter 改动，可以只对这些原本 staged 的文件重新执行 `git add` 后继续 commit/push；不得添加其它未 staged 或 untracked 文件。

## Steps

1. **检查暂存状态**
   - 运行 `git status` 确认有已暂存文件
   - 若无 staged 文件，提示用户先执行 `git add` 再运行此命令
   - 记录当前 staged 文件列表：`git diff --cached --name-only`
   - 如果 staged 文件在格式化前已经同时存在 unstaged 改动（例如 `git status --short` 显示 `MM <file>`），停止并提示用户先手动整理/暂存这些文件；否则后续 `git add <file>` 会把格式化前已有的未暂存改动也带入 commit

2. **在 commit 前执行 `/format-code`**
   - 调用 `/format-code`，先格式化当前 staged/modified 的代码
   - 格式化完成后重新运行 `git status`
   - 如果格式化导致原本 staged 的文件重新变成 modified/unstaged，只对步骤 1 记录的原本 staged 文件执行 `git add <file>`，然后继续流程
   - 不要 stage 不在步骤 1 staged 列表中的文件，即使它们也被 `/format-code` 改动

3. **查看变更内容**
   - 运行 `git diff --cached` 查看 staged 的 diff
   - 根据实际变更生成简洁、准确的 commit message

4. **生成并执行 commit**
   - 基于 diff 内容生成 commit message
   - 格式：`git commit -s -m "type(scope): description"`（`-s` 添加 Signed-off-by）
   - 示例：`git commit -s -m "fix(pa): correct sliding window mask in decode kernel"`
   - 规则：≤72 字符、祈使语气（fix/add/update）、首字母大写、句末无句号

5. **Push 到当前分支**
   - 运行 `git push -u origin HEAD` 或 `git push -u origin $(git branch --show-current)`
   - 若 push 被拒绝（远程有新提交），执行 `git pull --rebase && git push`

## Rules

- **限制 `git add` 范围**：默认只提交用户已经 `git add` 暂存的文件。唯一允许自动执行 `git add` 的情况是：`/format-code` 改动了步骤 1 记录的原本 staged 文件；此时只能 `git add` 这些原本 staged 的文件。绝对不要添加新文件、untracked 文件、或任何不在原 staged 列表中的 modified 文件
- **先格式化再 commit**：在生成 `git diff --cached` 和 commit message 之前，必须先执行 `/format-code`
- **防止混入旧的 unstaged 改动**：如果某个 staged 文件在运行 `/format-code` 前已经有 unstaged 改动，停止并提示用户先手动整理，因为自动 `git add` 无法区分旧改动和格式化改动
- **Commit message**：基于实际 diff，描述做了什么以及原因
- **Push**：推送到当前分支对应的远程分支
