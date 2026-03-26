# Diagnose and Fix Bug

Systematic workflow for diagnosing crashes, segfaults, and logic bugs in complex codebases. Derived from real-world debugging of use-after-free, ABI mismatch, memory corruption, and cache invalidation bugs in GPU compiler stacks.

## When to Use

- User reports a crash (SIGSEGV, SIGABRT, exit code 139/134/137)
- User reports incorrect output or silent data corruption
- User reports "works once, crashes on second call" patterns
- User says "it crashes when I set ENV_VAR=X"
- Any bug that isn't immediately obvious from the error message

## Workflow

### Phase 1: Reproduce

**Goal**: Get a reliable, minimal reproduction of the bug.

1. **Run the failing command exactly as the user describes**. Capture full output including exit code.
2. **Record the crash signature**:
   - Exit code 139 = SIGSEGV (segfault — dangling pointer, use-after-free, buffer overflow)
   - Exit code 134 = SIGABRT (assertion failure, double-free, heap corruption)
   - Exit code 137 = SIGKILL (OOM, timeout)
   - Python traceback = logic error, type error, missing attribute
   - Hang (no output) = deadlock, infinite loop, blocked I/O
3. **Identify the last successful output line** — the crash happened between that line and the next expected line.
4. **Check if it's deterministic**: run 2-3 times. If intermittent, suspect race condition or uninitialized memory.

### Phase 2: Narrow the Scope

**Goal**: Find the exact function/module where the crash occurs.

1. **Trace the call chain** from the crash point:
   - Find the function being called at the crash (from the last output line or traceback)
   - Read that function's source code
   - Follow the code path to identify what state it depends on
2. **Binary search with print statements** if needed:
   - Add prints at entry/exit of suspected functions
   - Narrow to the exact line that crashes
3. **For "works once, crashes on second call" patterns**, focus on:
   - Caching/memoization code (`lru_cache`, dict caches, class-level caches)
   - State that persists between calls (class attributes, global variables, module-level dicts)
   - Resources that get freed/closed after first use (file handles, GPU contexts, native memory)

### Phase 3: Root Cause Analysis

**Goal**: Understand WHY the crash happens, not just WHERE.

Common bug patterns and their signatures:

#### Use-After-Free (SIGSEGV on 2nd+ call)
- **Pattern**: Object A holds a raw pointer/reference to Object B's memory. B gets garbage collected. A dereferences dangling pointer.
- **How to find**: Look for code that:
  1. Extracts a raw pointer (ctypes, `data_ptr()`, `ctypes.addressof`)
  2. Stores it in a cache/dict that outlives the source object
  3. The source object is NOT stored in the same or longer-lived cache
- **Key question**: "Who owns the memory that this pointer points to, and when does that owner get destroyed?"
- **Example**: `CallState` caches `func_exe` (ctypes pointer into `ExecutionEngine`) but `CompiledArtifact` (which owns `ExecutionEngine`) is not cached when `enable_cache=False`.

#### ABI/Version Mismatch (wrong output, unreachable, 0-instruction kernels)
- **Pattern**: Headers from version X compiled against libraries from version Y.
- **How to find**: Check `cmake` cache files for version info, compare header paths vs library paths.
- **Key question**: "Were all components built from the same source tree?"

#### Cache Invalidation Bug (stale results, wrong config applied)
- **Pattern**: Cache key doesn't include all relevant parameters.
- **How to find**: Read the cache key construction code, compare against all parameters that affect output.
- **Key question**: "If I change parameter X, does the cache key change too?"

#### Resource Leak (OOM, gradual slowdown, SIGKILL)
- **Pattern**: GPU memory, file descriptors, or native objects allocated but never freed.
- **How to find**: Add destructor logging, check `nvidia-smi`/`rocm-smi` memory between calls.

### Phase 4: Verify Theory Before Fixing

**Goal**: Confirm the root cause before writing any fix.

1. **Trace the exact execution path** that leads to the crash:
   - For the working case (1st call): what code path runs, what gets cached
   - For the failing case (2nd call): what code path runs, what stale state does it hit
2. **Write down the causal chain**: "A happens, which causes B, which causes C (crash)"
3. **Predict what would happen with the fix**: "If we do X, then B won't happen, so C won't happen"
4. **If possible, write a minimal reproducer** that isolates just the bug mechanism

### Phase 5: Apply Minimal Fix

**Goal**: Fix the bug with the smallest possible change.

Principles:
- **Fix the root cause, not the symptom**. Don't add try/except around a segfault.
- **Minimal diff**: Change as few lines as possible. Don't refactor surrounding code.
- **Preserve behavior for the working case**: The fix should be a no-op when the bug condition isn't triggered.
- **Add a comment explaining WHY**: Future readers need to understand why this guard exists.

Common fix patterns:
- **Use-after-free**: Either (a) don't cache the dangling reference, or (b) also cache the owner object to prevent GC.
- **Cache invalidation**: Add the missing parameter to the cache key.
- **ABI mismatch**: Rebuild all components from the same source.
- **Resource leak**: Add cleanup in destructor/finally/context-manager.

### Phase 6: Test the Fix

1. **Run the original failing command** — must pass now
2. **Run the original passing case** (e.g., with cache enabled) — must still pass
3. **Run existing tests** if available — no regressions
4. **Edge cases**: Try the boundary conditions (0 iterations, 1 iteration, many iterations)

### Phase 7: Commit on Clean Branch

1. **Create a new branch from the upstream base** (main/master), not from the feature branch
2. **Cherry-pick or apply only the fix** — no unrelated changes
3. **Write a commit message** that includes:
   - What was happening (the symptom)
   - Why it was happening (the root cause mechanism)
   - What the fix does (the change)
   - Reference to the failing test/command

Use git worktree for clean branch creation:
```bash
git fetch origin main
git worktree add /path/to/worktree origin/main -b fix/descriptive-name
cd /path/to/worktree
git cherry-pick <commit-hash>
```

## Anti-Patterns to Avoid

- **Don't guess-and-check**: Read the code, understand the mechanism, THEN fix
- **Don't add defensive null checks without understanding why the value is null**
- **Don't disable caching/optimization as a "fix"** — find why it breaks
- **Don't blame the compiler/runtime without evidence** — it's almost always your code
- **Don't add `try/except: pass` around crashes** — that hides bugs, doesn't fix them
- **Don't fix multiple bugs in one commit** — each fix should be independently reviewable

## Debugging Tools Reference

| Tool | When to Use |
|------|------------|
| `grep -rn "pattern"` | Find where a variable/function is defined/used |
| `git log --oneline -20` | Recent changes that might have introduced the bug |
| `git diff branch1..branch2 -- file` | What changed between working/broken versions |
| Exit code 139 | SIGSEGV — focus on pointer/memory bugs |
| Exit code 134 | SIGABRT — focus on assertions, double-free |
| `strace -f -e trace=memory` | Track mmap/munmap for memory lifecycle |
| `gdb -batch -ex run -ex bt` | Get native stack trace on crash |
| `PYTHONFAULTHANDLER=1` | Get Python stack trace on SIGSEGV |
