---
name: build-rocm-image
description: Connect to a remote host via SSH and build a Docker image with rocprofv3, vllm, aiter and FlyDSL. Use when user wants to build/rebuild the ROCm development image on a remote host. Usage: /build-rocm-image <hostname>
allowed-tools: Bash
---

# Build ROCm Development Image

Build a Docker image on a remote host with rocm gpu access based on `rocm/vllm-dev:nightly`.
## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<HOST>` | Yes | The remote hostname to SSH into and build the image on. Example: `hjbog-srdc-39.amd.com` |

When this skill is invoked, the argument passed in is the target hostname. Replace all occurrences of `<HOST>` below with the provided hostname. If no hostname is provided, ask the user for it before proceeding.

## Target Host

- **Host**: `<HOST>` (provided as argument)
- **Access**: SSH (key-based authentication)

## Base Image

- **Image**: `rocm/vllm-dev:nightly`
- **Included**: rocprofv3, PyTorch (versions determined by the nightly base image)

## Customization

- **vLLM**: Replace pre-installed version with https://github.com/ROCm/vllm branch `ps_pa`
- **aiter**: Replace pre-installed version with https://github.com/ROCm/aiter branch `main`
- **FlyDSL**: Build and install from https://github.com/ROCm/FlyDSL branch `main` using the same LLVM/MLIR + FlyDSL build sequence documented in the `build-flydsl` skill

## Authentication

Do not put GitHub tokens in clone URLs, Dockerfiles, or command lines. Pass the token to Docker BuildKit as a secret:

- The remote host must have `GITHUB_TOKEN` set in the environment used by the `docker build` command.
- The Dockerfile reads the token only through `/run/secrets/github_token` during the specific `RUN` steps that need GitHub access.
- The token is not written to `/tmp/Dockerfile.rocm-custom`, Docker image layers, or the final image.

## Build Steps

### Step 1: Generate Dockerfile on remote host

```bash
ssh -o ConnectTimeout=30 <HOST> 'cat > /tmp/Dockerfile.rocm-custom' <<'DOCKERFILE'
# syntax=docker/dockerfile:1.7
FROM rocm/vllm-dev:nightly

# Uninstall existing, vllm, aiter
RUN pip uninstall -y vllm aiter 2>/dev/null; true

# Install build dependencies
RUN pip install ninja cmake pybind11

# Clone and install aiter from main branch
RUN --mount=type=secret,id=github_token \
    GITHUB_TOKEN="$(cat /run/secrets/github_token)" && \
    cd /tmp && \
    git -c http.extraHeader="Authorization: Bearer ${GITHUB_TOKEN}" clone --depth 1 --branch main https://github.com/ROCm/aiter.git && \
    cd aiter && \
    pip install -e . && \
    cd / && rm -rf /tmp/aiter

# Clone, build, and install FlyDSL from main branch.
# This follows the build-flydsl skill: build LLVM/MLIR first, build FlyDSL,
# then install editable from the persistent checkout.
RUN --mount=type=secret,id=github_token \
    GITHUB_TOKEN="$(cat /run/secrets/github_token)" && \
    cd /opt && \
    git -c http.extraHeader="Authorization: Bearer ${GITHUB_TOKEN}" clone --depth 1 --branch main https://github.com/ROCm/FlyDSL.git && \
    cd FlyDSL && \
    bash scripts/build_llvm.sh -j$(nproc) && \
    bash scripts/build.sh -j$(nproc) && \
    pip install -e .

# Clone and install vllm from ROCm/vllm ps_pa branch
RUN --mount=type=secret,id=github_token \
    GITHUB_TOKEN="$(cat /run/secrets/github_token)" && \
    cd /tmp && \
    git -c http.extraHeader="Authorization: Bearer ${GITHUB_TOKEN}" clone --depth 1 --branch ps_pa https://github.com/ROCm/vllm.git && \
    cd vllm && \
    pip install -e . && \
    cd / && rm -rf /tmp/vllm

# Install rocprof-trace-decoder: download installer, extract .so, copy to /opt/rocm/lib
RUN --mount=type=secret,id=github_token \
    GITHUB_TOKEN="$(cat /run/secrets/github_token)" && \
    cd /tmp && \
    wget -q --header="Authorization: Bearer ${GITHUB_TOKEN}" https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh && \
    chmod +x rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh && \
    ./rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh --skip-license --prefix=/tmp/rtd-install && \
    find /tmp/rtd-install -name '*.so*' -exec cp -a {} /opt/rocm/lib/ \; && \
    ldconfig && \
    rm -rf /tmp/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh /tmp/rtd-install

# Verify installations
RUN python3 -c 'import triton; print(f"triton version: {triton.__version__}")' && \
    python3 -c 'import vllm; print(f"vllm version: {vllm.__version__}")' && \
    python3 -c 'import aiter; print("aiter OK")' && \
    python3 -c 'import flydsl; print("FlyDSL OK")' && \
    which rocprofv3 && echo 'rocprofv3 OK' && \
    ls /opt/rocm/lib/librocprof*decoder* && echo 'rocprof-trace-decoder OK'

LABEL description="ROCm dev image with vllm(ROCm/ps_pa), aiter(main), FlyDSL(main), rocprofv3, rocprof-trace-decoder"
DOCKERFILE
```

### Step 2: Build the image

Build the image with BuildKit enabled and pass `GITHUB_TOKEN` as a secret. Use `--network=host` to ensure git clone works.

```bash
ssh -o ConnectTimeout=30 <HOST> 'if [ -z "$GITHUB_TOKEN" ]; then echo "GITHUB_TOKEN is not set on the remote host" >&2; exit 1; fi; DOCKER_BUILDKIT=1 docker build --network=host --secret id=github_token,env=GITHUB_TOKEN -t rocm-dev-custom:flydsl -f /tmp/Dockerfile.rocm-custom /tmp'
```

### Step 3: Verify the built image

```bash
ssh -o ConnectTimeout=30 <HOST> "docker run --rm rocm-dev-custom:flydsl bash -c '
echo \"=== Triton ===\"
python3 -c \"import triton; print(triton.__version__)\"
echo \"=== vLLM ===\"
python3 -c \"import vllm; print(vllm.__version__)\"
echo \"=== aiter ===\"
python3 -c \"import aiter; print(aiter.__version__)\" 2>/dev/null || python3 -c \"import aiter; print(\\\"aiter OK\\\")\"
echo \"=== FlyDSL ===\"
python3 -c \"import flydsl; print(flydsl.__version__)\" 2>/dev/null || python3 -c \"import flydsl; print(\\\"FlyDSL OK\\\")\"
echo \"=== rocprofv3 ===\"
rocprofv3 --version 2>/dev/null || which rocprofv3
echo \"=== ROCm ===\"
cat /opt/rocm/.info/version
'"
```

### Step 4: Clean up

```bash
ssh -o ConnectTimeout=30 <HOST> "rm -f /tmp/Dockerfile.rocm-custom"
```

## Output

Report to the user:
- The image name and tag
- Versions of triton, vllm, aiter, and ROCm inside the image
- Any build warnings or errors

## Error Handling

- If SSH connection fails, inform the user they need a valid SSH key and Conductor reservation
- If disk space is insufficient, explain that the FlyDSL LLVM/MLIR build needs about 50GB and suggest cleaning unused images with `docker image prune`

## Example Usage

To start a container from the built image with GPU access:

```bash
ssh <HOST> "docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --shm-size=64g rocm-dev-custom:flydsl bash"
```
