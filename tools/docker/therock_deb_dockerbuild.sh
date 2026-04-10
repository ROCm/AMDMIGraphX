#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============ Configurable Parameters ============
if [ -z "${GPU_ARCH}" ]; then
    echo "ERROR: GPU_ARCH environment variable is not set."
    echo "Please export GPU_ARCH to your target arch that therock hub supports (e.g. export GPU_ARCH=gfx120x) before running this script."
    exit 1
fi
ROCM_VERSION="${ROCM_VERSION:-7.13}"
ROCM_NIGHTLY_URL="${ROCM_NIGHTLY_URL:-https://rocm.nightlies.amd.com/deb/20260401-23832802691}"

IMAGE_NAME="${IMAGE_NAME:-migraphx-therock}"
IMAGE_TAG="${IMAGE_TAG:-${GPU_ARCH}}"
CONTAINER_NAME="${CONTAINER_NAME:-migraphx-therock-${GPU_ARCH}}"
# ==================================================

echo "=== MIGraphX TheRock Docker Build ==="
echo "  GPU_ARCH:         ${GPU_ARCH}"
echo "  ROCM_VERSION:     ${ROCM_VERSION}"
echo "  ROCM_NIGHTLY_URL: ${ROCM_NIGHTLY_URL}"
echo "  IMAGE:            ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  CONTAINER:        ${CONTAINER_NAME}"
echo "======================================"

# Build
docker build \
    --build-arg GPU_ARCH="${GPU_ARCH}" \
    --build-arg ROCM_VERSION="${ROCM_VERSION}" \
    --build-arg ROCM_NIGHTLY_URL="${ROCM_NIGHTLY_URL}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "${SCRIPT_DIR}/therock_deb.dockerfile" \
    "${PROJECT_DIR}"

# Run
docker run -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -v "${PROJECT_DIR}:/code/AMDMIGraphX" \
    --name "${CONTAINER_NAME}" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    /bin/bash