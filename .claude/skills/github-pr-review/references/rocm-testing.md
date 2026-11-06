# ROCm test environments

Use actual user-supplied hosts and containers. Architecture labels choose where
tests run; verify the actual architecture inside the container. Python, Torch
and FlyDSL versions can differ between containers on the same host.

## Configuration

Merge only environments authorized for this instance into test_environments:

~~~json
{
  "gfx950": {
    "ssh_alias": "YOUR_GFX950_SSH_ALIAS",
    "access": "slurm-podman",
    "slurm_user": "YOUR_SLURM_USER",
    "slurm_node": "YOUR_GPU_NODE",
    "container": "YOUR_DEDICATED_REVIEW_CONTAINER",
    "workspace": "/workspace",
    "gpu_lock": "/workspace/gpu.lock",
    "restart_authorized": false,
    "owner_label": {"io.codex.pr-review.owner": "YOUR_GITHUB_LOGIN"},
    "note": "Discover the active allocation on every run."
  },
  "gfx942": {
    "ssh_alias": "YOUR_GFX942_SSH_ALIAS",
    "access": "docker",
    "container": "YOUR_TEST_CONTAINER",
    "python": "python3",
    "workspace": "/tmp",
    "note": "Use Python and GPU dependencies inside the container."
  }
}
~~~

Do not persist current job IDs. A container restart requires recorded restart
authorization and a matching owner label. Missing environments or allocations
are validation blockers, not permission to acquire a new allocation or substitute
another user's container.

## SSH and allocation discovery

Use BatchMode=yes, ConnectTimeout=10, RequestTTY=no, ControlMaster=no and
ControlPath=none. SSH aliases can contain obsolete RemoteCommand settings;
override them for discovery and execution.

Construct remote argument arrays with Python shlex.join rather than shell
interpolation. For SLURM, query the login host with squeue -h -u USER
-o '%i|%N|%t' using RemoteCommand=none. Select a RUNNING allocation containing
the configured node. Expand compressed node lists with scontrol show hostnames.

Run srun --overlap --jobid=ACTUAL_ID --ntasks=1 --cpus-per-task=1 followed by
podman exec -i CONTAINER python3 -. For Docker, use docker exec -i CONTAINER
python3 - through SSH. Stream scripts through stdin. OpenSSH expands percent
tokens in RemoteCommand; if embedding a script containing percent signs there,
escape each literal percent as %% before passing that option.

## Isolate and verify the workload

Create a unique source/cache directory per PR and reviewed SHA. Transfer an
archive of a verified exact-SHA checkout and verify its checksum. Preserve
submodule content needed by builds; do not archive a moving branch or reuse
an unrelated installed aiter module.

Verify torch.cuda.get_device_properties(0).gcnArchName, dependency versions and
imports such as aiter.fused_moe.__file__. Query rocm-smi for activity and memory;
choose an idle device according to the host's usage conventions. Set
HIP_VISIBLE_DEVICES before Torch import. If no device is idle, record the
missing validation instead of running on a busy device.

Use the configured lock with flock -w and bound tests with timeout. Set
PYTHONPATH to the exact checkout and isolate AITER_JIT_DIR,
FLYDSL_RUNTIME_CACHE_DIR, TORCH_EXTENSIONS_DIR and TRITON_CACHE_DIR. Keep logs,
distinguish executed tests from skips, and run regressions appropriate to the
change. A smoke kernel alone does not validate an entire PR.

Tests on gfx942 or gfx950 cannot establish gfx1250 correctness. When an
essential target is unavailable, keep the review incomplete with a limitation.

## Provision a dedicated container when requested

Provisioning belongs to an explicit setup request rather than an unattended
review. Inspect cached images, dates, versions and runtime requirements. When
the user requests the latest existing ROCm image, choose the newest suitable
cached image, pin its ID/digest and use --pull=never.

Create a distinct container and task-owned workspace, preserving other
containers. On rootless Podman with crun, a working starting configuration uses
/dev/kfd and /dev/dri, --group-add keep-groups, private shared memory and a
read-only /sys bind mount. The /sys mount can be needed for driver discovery:
rocm-smi may exit zero with empty stdout and "amdgpu not found" on stderr when
/sys/class/drm or /sys/module/amdgpu is hidden. Inspect output instead of assuming
success from the exit code. Use supported host runtime options; do not default
to --privileged or copy another container's home mounts.

Validate architecture, GPU execution, compiler/runtime imports and meaningful
targeted tests. Record image, container ID, mounts and creation arguments
privately. Update test configuration after validation; the reviewer re-reads
it before remote tests.
