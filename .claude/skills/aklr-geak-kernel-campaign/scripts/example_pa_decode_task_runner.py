#!/usr/bin/env python3
"""Example GEAK task runner: aiter's FlyDSL paged-attention decode (PA), scored by a benchmark script.

Worked example for the aklr-geak-kernel-campaign skill. It wraps
op_tests/op_benchmarks/flydsl/bench_pa_minimax_m3.py from ROCm/aiter branch zhiding512/pa-benchmark; to wrap another
kernel, keep the structure (overlay, child process, hardened checks, report format) and replace the parts marked ADAPT.

Modes:
  compile      overlay pa_src/ onto the immutable aiter checkout and import the PA module
  correctness  every MiniMax-M3 case (dense, sparse, prefill, cluster boundary; ATOM profile,
               plus the dense cases under the PR profile), BF16 and FP8 KV, checked against
               FP32 attention, eagerly and through a CUDA graph replay with changed queries
  performance  the 20 scored cases: ATOM profile, dense decode + dense MTP4 verify, BF16 and
               FP8 KV, timed exactly as bench_pa_minimax_m3.py times them

The candidate is pa_src/ of THIS workspace (the directory above scripts/). Each invocation
rebuilds build/aiter_src/: real directories down to the PA files, real copies of every pa_src
file, and symlinks to the immutable aiter checkout for everything else. The benchmark harness
(input construction, FP32 reference, checks, CUDA-graph timer) is imported unchanged from the
zhiding512/pa-benchmark branch checkout. Reports go to build/.

Maintainer-only: `PA_RECORD_PRISTINE=1 python3 scripts/task_runner.py correctness` on the PRISTINE sources records the
pristine outputs and calibration.json into PA_PRISTINE_OUT; make that directory read-only afterwards.

Environment: PA_WORK (default ~/pa_opt), PA_AITER_BASE, PA_PRISTINE_OUT, PA_FLYDSL_CACHE_ROOT, PA_ARCH (default gfx1250).
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
PA_SRC = TASK_DIR / "pa_src"
BUILD = TASK_DIR / "build"
OVERLAY = BUILD / "aiter_src"
WORK = Path(os.environ.get("PA_WORK", "~/pa_opt")).expanduser()
# ADAPT: the immutable checkout the candidate is overlaid onto, and the benchmark harness inside it.
AITER_BASE = Path(os.environ.get("PA_AITER_BASE", str(WORK / "aiter-base"))).expanduser()
BENCH = AITER_BASE / "op_tests/op_benchmarks/flydsl/bench_pa_minimax_m3.py"
FLYDSL_CACHE_ROOT = Path(os.environ.get("PA_FLYDSL_CACHE_ROOT", str(WORK / "flydsl_cache"))).expanduser()
PA_PKG = "aiter/ops/flydsl"  # ADAPT: package directory the candidate files belong to
PRISTINE_OUT = Path(os.environ.get("PA_PRISTINE_OUT", str(WORK / "pristine_outputs"))).expanduser()
ARCH = os.environ.get("PA_ARCH", "gfx1250")
# ADAPT: the candidate files of the pristine checkout; one missing from pa_src is deleted in the overlay.
ORIGINAL_PA_FILES = (
    "pa_decode.py",
    "kernels/pa_decode_kernel.py",
    "kernels/pa_decode_plan.py",
    "kernels/pa_decode_reduce.py",
    "kernels/pa_decode/__init__.py",
    "kernels/pa_decode/context.py",
    "kernels/pa_decode/op_epilog.py",
    "kernels/pa_decode/op_gemm.py",
    "kernels/pa_decode/op_lds.py",
    "kernels/pa_decode/op_merge.py",
    "kernels/pa_decode/op_softmax.py",
    "kernels/pa_decode/pipeline.py",
    "kernels/pa_decode/traits.py",
)
FIXED_ENV = {
    "AITER_RUNTIME_GPU_ARCH": ARCH,
    "FLYDSL_GPU_ARCH": ARCH,
    "AITER_USE_SYSTEM_TRITON": "1",
}
DENSE_SUITES = ("dense-decode", "dense-verify")  # ADAPT: the scored regime (one regime per run)
CORRECTNESS_TIMING = dict(graph_calls=1, replays=1, samples=1, rounds=1, warmup=1)
PERF_TIMING = dict(graph_calls=10, replays=5, samples=5, rounds=3, warmup=5)


def candidate_files():
    return sorted(p.relative_to(PA_SRC).as_posix() for p in PA_SRC.rglob("*.py"))


def source_digest():
    digest = hashlib.sha256()
    for rel in candidate_files():
        digest.update(rel.encode())
        digest.update((PA_SRC / rel).read_bytes())
    return digest.hexdigest()


def build_overlay():
    """Rebuild build/aiter_src from this workspace's pa_src on every call."""
    if not (BENCH.is_file() and (AITER_BASE / PA_PKG / "pa_decode.py").is_file()):
        raise RuntimeError(f"immutable aiter checkout missing: {AITER_BASE}")
    files = candidate_files()
    if "pa_decode.py" not in files:
        raise RuntimeError("pa_src/pa_decode.py is missing")
    targets = [f"{PA_PKG}/{rel}" for rel in files]
    real_dirs = {""}
    for target in targets:
        parts = target.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            real_dirs.add("/".join(parts[:i]))
    if OVERLAY.exists() or OVERLAY.is_symlink():
        if OVERLAY.is_symlink() or OVERLAY.is_file():
            OVERLAY.unlink()
        else:
            shutil.rmtree(OVERLAY)

    def mirror(rel):
        src = AITER_BASE / rel if rel else AITER_BASE
        dst = OVERLAY / rel if rel else OVERLAY
        dst.mkdir(parents=True, exist_ok=True)
        if not src.is_dir():
            return
        for entry in os.scandir(src):
            if entry.name in (".git", "__pycache__"):
                continue
            child = f"{rel}/{entry.name}" if rel else entry.name
            if child in real_dirs and entry.is_dir(follow_symlinks=False):
                mirror(child)
            else:
                os.symlink(entry.path, dst / entry.name)

    mirror("")
    for rel in ORIGINAL_PA_FILES:
        if rel not in files:
            stale = OVERLAY / PA_PKG / rel
            if stale.is_symlink() or stale.exists():
                stale.unlink()
    for rel, target in zip(files, targets):
        dst = OVERLAY / target
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.parent.is_symlink() or not dst.parent.resolve().is_relative_to(OVERLAY.resolve()):
            raise RuntimeError(f"overlay parent escapes the workspace: {dst.parent}")
        shutil.copyfile(PA_SRC / rel, dst)
    return files


def child_env():
    env = dict(os.environ)
    env.update(FIXED_ENV)
    # Content-addressed compile cache: identical sources share binaries, any edit gets a fresh dir.
    env["FLYDSL_RUNTIME_CACHE_DIR"] = str(FLYDSL_CACHE_ROOT / source_digest()[:24])
    for key in ("FLYDSL_AUTOTUNE", "AITER_FLYDSL_KERNARG_PRELOAD", "FLYDSL_RUNTIME_RUN_ONLY"):
        env.pop(key, None)
    return env


def case_keys(mode):  # ADAPT: scored cases for performance; every case that shares the code for correctness
    bench = load_bench()
    keys = []
    if mode == "performance":
        for dtype in ("bf16", "fp8"):
            for case in bench.case_matrix():
                if case["suite"] in DENSE_SUITES:
                    keys.append(("atom", dtype, case["id"]))
        return keys
    for dtype in ("bf16", "fp8"):
        for case in bench.case_matrix():
            keys.append(("atom", dtype, case["id"]))
    for dtype in ("bf16", "fp8"):
        for case in bench.case_matrix():
            if case["suite"] in DENSE_SUITES:
                keys.append(("pr", dtype, case["id"]))
    return keys


def load_bench():
    spec = importlib.util.spec_from_file_location("pa_minimax_m3_bench", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_child(keys, check):
    """Run cases in a subprocess; return {key: result}. One retry after an abnormal exit."""
    results = {}
    log_tail = ""
    pending = list(keys)
    for attempt in range(2):
        if not pending:
            break
        record = os.environ.get("PA_RECORD_PRISTINE") == "1"  # maintainer only; the store is read-only
        spec = json.dumps({"keys": pending, "check": check, "record": record})
        done = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "_child", spec],
            cwd=TASK_DIR,
            env=child_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        for line in done.stdout.splitlines():
            if line.startswith("PA_RESULT "):
                row = json.loads(line[len("PA_RESULT ") :])
                results[tuple(row["key"])] = row
        log_tail = (done.stdout + done.stderr)[-4000:]
        pending = [k for k in pending if tuple(k) not in results]
        if done.returncode == 0 or not pending:
            break
        # A crash (e.g. a GPU memory fault) kills the process; mark the case it was on and retry
        # the rest once. A case that crashes twice fails.
        crashed = pending[0]
        if attempt == 1:
            results[tuple(crashed)] = {
                "key": crashed,
                "ok": False,
                "error": f"process exited {done.returncode} on this case twice",
            }
            pending = pending[1:]
        print(f"note: runner exited {done.returncode} at {crashed}; retrying remaining cases", flush=True)
    for key in pending:
        results.setdefault(tuple(key), {"key": key, "ok": False, "error": "not run (process crashed)"})
    return results, log_tail


SHARP_Q_SCALE = 32.0  # logits std ~2.9 instead of ~0.09: a peaked softmax exposes softmax/merge bugs


def sharp_error(bench, case, dtype, profile, args, pa, torch):
    """Re-run the harness's own measure() with queries scaled by SHARP_Q_SCALE.

    The harness's inputs give a near-uniform softmax whose BF16 output (~3e-3) is below its
    5e-3 absolute tolerance, so a wrong softmax scale passes. Here the same inputs, plan, eager
    call, graph capture and changed-query replay run with peaked attention; every check records
    the worst per-row relative L2 error (and fails on NaN/inf). Thresholds are applied by the caller.
    """
    seen, captured = [], []
    build_inputs, check_output = bench.build_inputs, bench.check_output

    def build(case_, dtype_, seed, torch_):
        inputs = build_inputs(case_, dtype_, seed, torch_)
        inputs["query"].mul_(SHARP_Q_SCALE)
        return inputs

    def check(output, expected, torch_, dtype_):
        actual = output.float()
        if not torch_.isfinite(actual).all():
            raise AssertionError("sharp-softmax output contains NaN or infinity")
        if not captured:
            captured.append(output.detach().clone())
        delta = actual - expected
        row = delta.flatten(1).norm(dim=1) / expected.flatten(1).norm(dim=1).clamp_min(1e-12)
        seen.append(row.max().item())
        return {
            "max_abs_error": delta.abs().max().item(),
            "relative_l2_error": (delta.norm() / expected.norm().clamp_min(1e-12)).item(),
            "max_row_relative_l2_error": row.max().item(),
        }

    bench.build_inputs, bench.check_output = build, check
    try:
        bench.measure(case, dtype, profile, args, pa, torch)
    finally:
        bench.build_inputs, bench.check_output = build_inputs, check_output
    return max(seen), captured[0]


def measure_capturing(bench, case, dtype, profile, args, pa, torch):
    """The harness's measure(), keeping its checks, plus the eager output of the first call."""
    captured = []
    check_output = bench.check_output

    def check(output, expected, torch_, dtype_):
        if not captured:
            captured.append(output.detach().clone())
        return check_output(output, expected, torch_, dtype_)

    bench.check_output = check
    try:
        row = bench.measure(case, dtype, profile, args, pa, torch)
    finally:
        bench.check_output = check_output
    return row, (captured[0] if captured else None)


def row_rel_l2(a, b):
    a, b = a.float().flatten(1), b.float().flatten(1)
    return ((a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-12)).max().item()


def pristine_diff(torch, key, variant, output, record):
    path = PRISTINE_OUT / ("__".join(key) + f"__{variant}.pt")
    if record:
        PRISTINE_OUT.mkdir(parents=True, exist_ok=True)
        torch.save(output.cpu(), path)
        return 0.0
    if not path.is_file():
        return None
    return row_rel_l2(output, torch.load(path).to(output.device))


def child(spec_json):
    spec = json.loads(spec_json)
    sys.path.insert(0, str(OVERLAY))
    bench = load_bench()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no ROCm GPU visible")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != ARCH.split("-")[0]:
        raise RuntimeError(f"expected {ARCH}, got {arch}")
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    if not Path(pa.__file__).resolve().is_relative_to(OVERLAY.resolve()):
        raise RuntimeError(f"PA imported from {pa.__file__}, not the workspace overlay")
    timing = CORRECTNESS_TIMING if spec["check"] else PERF_TIMING
    args = argparse.Namespace(seed=37, workgroup_budget=None, check=spec["check"], **timing)
    cases = {case["id"]: case for case in bench.case_matrix()}
    with torch.inference_mode():
        for profile, dtype, case_id in spec["keys"]:
            key = [profile, dtype, case_id]
            try:
                if spec["check"]:
                    row, eager = measure_capturing(bench, cases[case_id], dtype, profile, args, pa, torch)
                else:
                    row, eager = bench.measure(cases[case_id], dtype, profile, args, pa, torch), None
                out = {
                    "key": key,
                    "ok": True,
                    "us": row["median_us"],
                    "output_mode": row["output_mode"],
                    "max_row_rel_l2": row.get("max_row_relative_l2_error"),
                    "max_abs": row.get("max_abs_error"),
                }
                if spec["check"]:
                    sharp, sharp_out = sharp_error(bench, cases[case_id], dtype, profile, args, pa, torch)
                    out["sharp_max_row_rel_l2"] = sharp
                    record = spec.get("record", False)
                    out["pristine_diff"] = pristine_diff(torch, key, "std", eager, record)
                    out["pristine_diff_sharp"] = pristine_diff(torch, key, "sharp", sharp_out, record)
            except AssertionError as exc:
                out = {"key": key, "ok": False, "error": f"mismatch: {str(exc)[:600]}"}
            except Exception as exc:  # the candidate failed to compile or launch
                out = {"key": key, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:600]}"}
            print("PA_RESULT " + json.dumps(out), flush=True)
            torch.cuda.synchronize()


# Extra correctness limits on the worst per-row relative L2 error, on top of the harness's checks. The benchmark's
# inputs give a near-uniform softmax whose BF16 output (~3e-3) sits below its 5e-3 absolute tolerance, so it cannot
# see a wrong softmax scale. Limits are derived from calibration.json, recorded on the pristine kernel.
BF16_MARGIN = 2.0  # BF16 limits: 2x the pristine worst case over all workloads (standard and sharp inputs)
FP8_SHARP_MARGIN = 1.5  # FP8 sharp error may grow to 1.5x the pristine value of its workload
# FP8 output vs the pristine kernel on identical inputs (the pristine kernel must be bitwise deterministic). A 0.5%
# softmax-scale change already moves it to ~0.026 through FP8 P rounding flips, so this catches structural bugs
# (scale indexing, lost partitions, masking), not rounding changes; BF16 is the fine-grained guard.
FP8_PRISTINE_DIFF_LIMIT = 0.035


def load_calibration():
    path = PRISTINE_OUT / "calibration.json"
    if not path.is_file():
        raise RuntimeError(f"missing {path}: record it on the pristine sources with PA_RECORD_PRISTINE=1")
    return json.loads(path.read_text())


def write_calibration(rows):
    bf16 = [r for r in rows if r["ok"] and r["key"][1] == "bf16"]
    fp8 = [r for r in rows if r["ok"] and r["key"][1] == "fp8"]
    cal = {
        "bf16_row_l2_worst": max(r["max_row_rel_l2"] for r in bf16),
        "bf16_sharp_row_l2_worst": max(r["sharp_max_row_rel_l2"] for r in bf16),
        "fp8_sharp_row_l2": {},
    }
    for r in fp8:
        case = r["key"][2]
        cal["fp8_sharp_row_l2"][case] = max(cal["fp8_sharp_row_l2"].get(case, 0.0), r["sharp_max_row_rel_l2"])
    (PRISTINE_OUT / "calibration.json").write_text(json.dumps(cal, indent=1) + "\n")
    return cal


def extra_check(row, cal):
    """Return an error string when a passing harness result breaks the stricter limits."""
    dtype, case = row["key"][1], row["key"][2]
    std, sharp = row.get("max_row_rel_l2"), row.get("sharp_max_row_rel_l2")
    if dtype == "bf16":
        limit = round(BF16_MARGIN * cal["bf16_row_l2_worst"], 4)
        if std is not None and std > limit:
            return f"BF16 row relative L2 {std:.5f} > {limit}"
        limit = round(BF16_MARGIN * cal["bf16_sharp_row_l2_worst"], 4)
        if sharp is not None and sharp > limit:
            return f"BF16 sharp-softmax row relative L2 {sharp:.5f} > {limit}"
    elif sharp is not None:
        limit = FP8_SHARP_MARGIN * cal["fp8_sharp_row_l2"][case]
        if sharp > limit:
            return f"FP8 sharp-softmax row relative L2 {sharp:.5f} > {limit:.5f} (1.5x pristine)"
        for variant in ("pristine_diff", "pristine_diff_sharp"):
            diff = row.get(variant)
            if diff is not None and diff > FP8_PRISTINE_DIFF_LIMIT:
                return f"FP8 {variant} (row relative L2 vs pristine kernel) {diff:.5f} > {FP8_PRISTINE_DIFF_LIMIT}"
    return None


def report(name, data):
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / f"{name}_report.json").write_text(json.dumps(data, indent=2) + "\n")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "_child":
        child(sys.argv[2])
        return
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    parser.add_argument("--case", action="append", help="limit to case ids containing this text")
    cli = parser.parse_args()
    mode, case_filter = cli.mode, cli.case
    BUILD.mkdir(parents=True, exist_ok=True)
    try:
        files = build_overlay()
    except Exception as exc:
        report(mode, {"status": "fail", "error": f"overlay: {exc}"})
        print(f"Compilation: FAIL\nError: overlay: {exc}")
        sys.exit(1)

    if mode == "compile":
        errors = []
        for rel in files:
            try:
                compile((PA_SRC / rel).read_text(), str(PA_SRC / rel), "exec")
            except SyntaxError as exc:
                errors.append(f"{rel}: {exc}")
        if not errors:
            probe = (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "import importlib, pathlib; pa = importlib.import_module('aiter.ops.flydsl.pa_decode'); "  # ADAPT
                "assert pathlib.Path(pa.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve()), pa.__file__; "
                "[getattr(pa, n) for n in ('pa_decode', 'plan_pa_decode', 'compile_pa_decode_tile')]; print('import ok')"
            )
            done = subprocess.run(
                [sys.executable, "-c", probe, str(OVERLAY)],
                cwd=TASK_DIR,
                env=child_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            if done.returncode != 0:
                errors.append((done.stdout + done.stderr)[-3000:])
        report("compile", {"status": "fail" if errors else "ok", "errors": errors, "files": files})
        print(f"Compilation: {'FAIL' if errors else 'PASS'}")
        for error in errors:
            print(f"Error: {error}")
        sys.exit(1 if errors else 0)

    keys = case_keys(mode)
    if case_filter:
        keys = [k for k in keys if any(f in "/".join(k) for f in case_filter)]
    results, tail = run_child(keys, check=(mode == "correctness"))
    rows = [results[tuple(k)] for k in keys]
    if mode == "correctness":
        if os.environ.get("PA_RECORD_PRISTINE") == "1":
            if not all(r["ok"] for r in rows):
                print("Correctness: FAIL (pristine run did not pass the harness; nothing calibrated)")
                sys.exit(1)
            print(f"calibration: {json.dumps(write_calibration(rows))[:300]}")
        cal = load_calibration()
        for r in rows:
            if r["ok"]:
                problem = extra_check(r, cal)
                if problem:
                    r["ok"], r["error"] = False, f"mismatch: {problem}"
        failed = [r for r in rows if not r["ok"]]
        report("correctness", {"status": "fail" if failed else "ok", "cases": rows})
        for r in rows:
            label = "/".join(r["key"])
            if r["ok"]:
                err = r.get("max_row_rel_l2")
                diff = r.get("pristine_diff")
                diff_s = r.get("pristine_diff_sharp")
                print(
                    f"PASS {label} mode={r['output_mode']} max_row_rel_l2={err:.5f} "
                    f"sharp={r['sharp_max_row_rel_l2']:.5f} "
                    f"vs_pristine={'n/a' if diff is None else f'{diff:.5f}'}/"
                    f"{'n/a' if diff_s is None else f'{diff_s:.5f}'}"
                )
            else:
                print(f"FAIL {label}: {r['error']}")
        print(f"Correctness: {'FAIL' if failed else 'PASS'} ({len(rows) - len(failed)}/{len(rows)} cases)")
        if failed and all("not run" in r["error"] for r in failed):
            print(tail)
        sys.exit(1 if failed else 0)

    failed = [r for r in rows if not r["ok"]]
    cases = [
        {"test_case_id": "/".join(r["key"]), "execution_time_ms": r["us"] / 1000.0, "output_mode": r["output_mode"]}
        for r in rows
        if r["ok"]
    ]
    geo_us = math.exp(sum(math.log(c["execution_time_ms"] * 1000) for c in cases) / len(cases)) if cases else None
    report("performance", {"test_cases": cases, "failed": failed, "geomean_us": geo_us})
    for c in cases:
        print(f"Perf: {c['execution_time_ms']:.6f} ms ({c['test_case_id']})")
    for r in failed:
        print(f"FAIL {'/'.join(r['key'])}: {r['error']}")
    if geo_us is not None:
        print(f"Geomean latency: {geo_us:.4f} us over {len(cases)} cases")
    if failed or not cases:
        print(tail)
        sys.exit(1)


if __name__ == "__main__":
    main()
