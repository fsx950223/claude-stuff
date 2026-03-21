# Feature Worktree

## Overview

基于 repo 主分支创建 git worktree + 新 feature branch 来开发新功能。支持本地 repo 和远程容器上的 repo。开发完成后可提交并清理 worktree。

## Usage

```
/feature-worktree <branch-name> [repo-path] [--container <name>] [--host <host>]
```

示例：
- `/feature-worktree feat/my-feature` — 本地 repo
- `/feature-worktree feat/shared-bt /FlyDSL --container hungry_dijkstra` — 远程容器

## Steps

### 1. 确定 repo 位置和主分支

- 解析参数：`branch-name`（必填）、`repo-path`（默认 cwd）、`--container`、`--host`
- 如果指定了 `--container`，通过 SSH + docker exec 操作远程 repo
  - SSH 命令模式：`ssh -i /home/dladmin/Downloads/id_rsa sixifang@<host> "docker exec <container> bash -c '<cmd>'"`
  - 默认 host：`10.67.77.162`
- 检测主分支名：优先 `origin/main`，其次 `origin/master`，通过 `git remote show origin | grep 'HEAD branch'` 确认

### 2. Fetch 并创建 worktree

```bash
cd <repo-path>
git fetch origin <main-branch>
git worktree add <worktree-path> origin/<main-branch> -b <branch-name>
```

- worktree 路径规则：`<repo-path>-<短分支名>`
  - 例如 repo 在 `/FlyDSL`，branch 为 `feat/shared-bt` → worktree 在 `/FlyDSL-shared-bt`
  - 短分支名：取 `/` 后的部分，`-` 连接
- 如果 branch 已存在，提示用户选择：覆盖（删除重建）或使用已有 branch

### 3. 报告结果

输出：
- Worktree 路径
- Branch 名
- 基于的 commit（main 的 HEAD）
- 后续操作提示

### 4. 开发完成后提交（用户触发）

当用户说"提交"或"commit"时：
1. 将工作目录的改动文件复制到 worktree（如果用户直接在原 repo 工作目录修改了文件）
2. 在 worktree 中 `git add` + `git commit`
3. Commit message 格式：`type(scope): description`

### 5. 清理 worktree（用户触发）

当用户说"清理 worktree"时：
```bash
cd <repo-path>
git worktree remove <worktree-path>
# 如果需要也删除 branch：
git branch -d <branch-name>
```

## Rules

- **永远基于最新的 origin 主分支**：创建前必须 `git fetch origin`
- **不修改原 repo 的工作目录状态**：所有 commit 操作在 worktree 中完成
- **branch 命名**：用户提供的名字原样使用，不自动添加前缀
- **远程操作**：所有 git 命令通过 SSH + docker exec 执行，不在本地执行
- **冲突处理**：如果 worktree 路径已存在，先确认是否要删除
- **不自动 push**：提交后不自动 push，等用户明确要求
