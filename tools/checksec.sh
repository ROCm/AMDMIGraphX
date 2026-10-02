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

if ! command -v readelf >/dev/null 2>&1; then
    echo "readelf is required for hardening verification"
    exit 1
fi

shopt -s nullglob
binaries=(
    "${build_dir}/bin/migraphx-driver"
    "${build_dir}/lib/libmigraphx.so"*
)

run_checksec() {
    local bin="$1"
    if checksec file "${bin}" --output json >/dev/null 2>&1; then
        checksec file "${bin}" --output json
        return 0
    fi
    if checksec --file="${bin}" --output json >/dev/null 2>&1; then
        checksec --file="${bin}" --output json
        return 0
    fi
    return 1
}

verify_with_readelf() {
    local bin="$1"
    local is_shared="$2"
    local failed=0

    if ! readelf -s "${bin}" 2>/dev/null | grep -q '__stack_chk_fail'; then
        echo "Missing stack canary for ${bin}"
        failed=1
    fi

    if ! readelf -l "${bin}" 2>/dev/null | grep -q 'GNU_RELRO'; then
        echo "Missing GNU_RELRO for ${bin}"
        failed=1
    elif ! readelf -d "${bin}" 2>/dev/null | grep -q 'BIND_NOW'; then
        echo "Missing BIND_NOW (full RELRO) for ${bin}"
        failed=1
    fi

    local elf_type
    elf_type="$(readelf -h "${bin}" 2>/dev/null | awk '/Type:/ {print $2}')"
    if [[ "${elf_type}" != "DYN" ]]; then
        if [[ "${is_shared}" == "1" ]]; then
            echo "Expected shared object (Type DYN) for ${bin}, got ${elf_type}"
        else
            echo "Missing PIE (Type DYN) for ${bin}, got ${elf_type}"
        fi
        failed=1
    fi

    return "${failed}"
}

verify_with_checksec_json() {
    local bin="$1"
    local is_shared="$2"
    local json="$3"

    python3 - "${json}" "${is_shared}" "${bin}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
is_shared = sys.argv[2] == "1"
path = sys.argv[3]

if isinstance(payload, list):
    entry = next((item for item in payload if item.get("name") == path), payload[0])
else:
    entry = payload

checks = entry.get("checks", entry)
canary = checks.get("canary", "")
relro = checks.get("relro", "")
pie = checks.get("pie", "")

failed = False
if "Canary Found" not in canary:
    print(f"Missing stack canary for {path}: {canary!r}")
    failed = True
if "Full RELRO" not in relro:
    print(f"Missing full RELRO for {path}: {relro!r}")
    failed = True
if is_shared:
    if "DSO" not in pie.upper():
        print(f"Expected DSO for shared library {path}, got {pie!r}")
        failed = True
elif "PIE Enabled" not in pie:
    print(f"Missing PIE for {path}: {pie!r}")
    failed = True

sys.exit(1 if failed else 0)
PY
}

found=0
failed=0
for bin in "${binaries[@]}"; do
    [[ -f "${bin}" ]] || continue
    found=1

    is_shared=0
    if [[ "${bin}" == *.so* ]]; then
        is_shared=1
    fi

    echo "Checking ${bin}"
    json=""
    if json="$(run_checksec "${bin}" 2>/dev/null)"; then
        echo "${json}"
        if ! verify_with_checksec_json "${bin}" "${is_shared}" "${json}"; then
            failed=1
        fi
    else
        echo "checksec JSON unavailable; falling back to readelf for ${bin}"
        if ! verify_with_readelf "${bin}" "${is_shared}"; then
            failed=1
        fi
    fi
done

if ((found == 0)); then
    echo "No MIGraphX binaries found under ${build_dir}; skipping"
    exit 0
fi

exit "${failed}"
