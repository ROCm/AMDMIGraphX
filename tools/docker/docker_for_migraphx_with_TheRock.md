## Building MIGraphX with TheRock

### Prerequisites

- Docker with GPU device access
- Target GPU architecture family that TheRock hub supports

### 1. Build the Docker Image

The Dockerfile accepts the following build arguments:

| Argument | Description |
|----------|-------------|
| `GPU_ARCH` | Target GPU architecture family. Determines which TheRock ROCm packages (`amdrocm-core-*`, `amdrocm-blas-*`, `amdrocm-dnn-*`, etc.) are installed, and sets the `GPU_ARCH_FOR_THEROCK` environment variable for CMake packaging. |
| `ROCM_VERSION` | ROCm version identifier. |
| `ROCM_RELEASE_URL` | URL of the TheRock ROCm apt repository to install packages from. |

```bash
docker build \
    --build-arg GPU_ARCH=<gpu_arch> \
    --build-arg ROCM_VERSION=<rocm_version> \
    --build-arg ROCM_RELEASE_URL=<rocm_release_url> \
    -t migraphx-therock:<gpu_arch> \
    -f tools/docker/therock_deb.docker \
    .
```

### 2. Launch the Container

```bash
docker run -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -v $(pwd):/code/AMDMIGraphX \
    migraphx-therock:<gpu_arch>
```

### 3. Build MIGraphX inside the Container

```bash
cd /code/AMDMIGraphX
rbuild build -d depend -B build -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')
```

If the build fails due to GPU architecture detection issues, retry with the explicit TheRock GPU arch flag:

```bash
rbuild build -d depend -B build \
    -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*') \
    -DMIGRAPHX_PACKAGE_BACKEND=therock -DGPU_ARCH_FOR_THEROCK=$GPU_ARCH_FOR_THEROCK
```

### 4. Package (optional)

```bash
cd build
make package
dpkg -i migraphx*.deb
```
