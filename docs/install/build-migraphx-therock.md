# Building MIGraphX with TheRock

This document describes how to build MIGraphX in a [TheRock](https://github.com/ROCm/TheRock)-style environment using the `therock` package backend. That backend is intended for ROCm layouts where dependency package names differ from the default ROCm stack (see [PR #4714](https://github.com/ROCm/AMDMIGraphX/pull/4714) for background and CMake details, including `MIGRAPHX_PACKAGE_BACKEND`).

The workflow below (and [`tools/docker/dockerfile.therock`](../../tools/docker/dockerfile.therock)) is derived from TheRock’s ROCm Python packaging guidance in [“Using ROCm Python packages”](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#using-rocm-python-packages).

## Prerequisites

- A machine with AMD GPU access (`/dev/kfd`, `/dev/dri`) if you intend to run on hardware.
- Docker (or another OCI runtime) for the recommended workflow below.

## Docker image (recommended starting point)

Use [`tools/docker/dockerfile.therock`](../../tools/docker/dockerfile.therock) as the basis for your build image. It creates a Python virtualenv at `/opt/rocm-7.13`, installs the ROCm SDK from AMD nightlies via pip, wires `/opt/rocm` to the SDK, and installs `rbuild`.

### `ROCM_PACKAGE` (required)

When you **build** the image, you must set **`ROCM_PACKAGE`** to the nightly **folder / ASIC** name that matches your target hardware. The allowed values are listed under the ROCm nightlies v2 index:

**<https://rocm.nightlies.amd.com/v2>**

Examples include `gfx120X-all`, `gfx110X-dgpu`, `gfx103X-dgpu`, and others. Pick the entry that corresponds to the GPU family you are building for.

Build example (from the repository root, adjusting `ROCM_PACKAGE` as needed):

```bash
docker build -f tools/docker/dockerfile.therock \
  --build-arg ROCM_PACKAGE=gfx120X-all \
  -t migraphx-therock:latest .
```

The Dockerfile default is `ROCM_PACKAGE=gfx120X-all` if you omit `--build-arg`.

## Run the container

Launch an interactive container with GPU device access and a workspace mount (adjust paths and image name to match your setup):

```bash
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video \
  -v $HOME/code:/workspace \
  migraphx-therock:latest
```

Inside the container, work from your MIGraphX source tree (for example under `/workspace/AMDMIGraphX` if you cloned into `$HOME/code` on the host).

## Build MIGraphX

1. Activate the TheRock / ROCm SDK environment:

   ```bash
   source /opt/rocm-7.13/bin/activate
   ```

2. Produce packages (dependencies under `deps`, with the `therock` packaging backend):

   ```bash
   rbuild package -d deps -DGPU_TARGETS=gfx1201 -DMIGRAPHX_PACKAGE_BACKEND=therock
   ```

Replace `gfx1201` with the GPU target that matches your hardware or CI image. The cache variable is **`GPU_TARGETS`** (see [PR #4714](https://github.com/ROCm/AMDMIGraphX/pull/4714) and the main build docs).

### Notes

- **`MIGRAPHX_PACKAGE_BACKEND=therock`** selects dependency naming and packaging behavior aligned with TheRock-style stacks, as described in the PR above.
- If ROCm libraries are present under `/opt/rocm` but not registered as system packages, `apt install` of generated `.deb` files may still report missing dependencies; in that situation use **`pre-installed`** for packaging or install in an environment where the declared dependencies exist—again, see [PR #4714](https://github.com/ROCm/AMDMIGraphX/pull/4714) for the intended options.
