#!/usr/bin/env bash
# Verify Linux hardening flags on shipped MIGraphX binaries (SEC-00802 / SEC-00469).
set -euo pipefail

build_dir="${BUILD_DIR:-build}"
require_checksec="${CHECKSEC_REQUIRED:-0}"

if ! command -v checksec >/dev/null 2>&1; then
    if [[ "${require_checksec}" == "1" ]]; then
        echo "checksec is required but not installed"
        exit 1
    fi
    echo "checksec not installed; skipping hardening gate"
    exit 0
fi

shopt -s nullglob
binaries=(
    "${build_dir}/bin/migraphx-driver"
    "${build_dir}/lib/libmigraphx.so"*
)

found=0
failed=0
for bin in "${binaries[@]}"; do
    [[ -f "${bin}" ]] || continue
    found=1
    echo "Checking ${bin}"
    output="$(checksec --file="${bin}")"
    echo "${output}"
    if ! grep -Eq 'Stack:.*(Canary found|canary found)' <<<"${output}"; then
        echo "Missing stack canary for ${bin}"
        failed=1
    fi
    if ! grep -Eq 'RELRO:.*(Full RELRO|full relro)' <<<"${output}"; then
        echo "Missing full RELRO for ${bin}"
        failed=1
    fi
    if ! grep -Eq 'PIE:.*(PIE enabled|pie enabled|Yes)' <<<"${output}"; then
        echo "Missing PIE for ${bin}"
        failed=1
    fi
done

if ((found == 0)); then
    echo "No MIGraphX binaries found under ${build_dir}; skipping"
    exit 0
fi

exit "${failed}"
