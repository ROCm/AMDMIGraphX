#!/bin/bash

#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#####################################################################################

set -e

WORK_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TESTER_SCRIPT="${TESTER:-$WORK_DIR/zoo_check.py}"
ATOL="${ATOL:-0.001}"
RTOL="${RTOL:-0.001}"
FP16_ATOL="${FP16_ATOL:-0.04}"
FP16_RTOL="${FP16_RTOL:-0.04}"
TARGET="${TARGET:-gpu}"
USE_LOCAL="${USE_LOCAL:-0}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-20m}"
DRIVER="${DRIVER:-migraphx-driver}"
PERF_ITERATIONS="${PERF_ITERATIONS:-10}"
KINDS="${KINDS:-accuracy perf}"

if [[ "${DEBUG:-0}" -eq 1 ]]; then
    PIPE=/dev/stdout
else
    PIPE=/dev/null
fi

if [[ "${VERBOSE:-0}" -eq 1 ]]; then
    set -x
fi

function iterate() {
  local dir="$1"

  for file in "$dir"/*; do
    if [ -f "$file" ]; then
      if [[ $file = *.tar.gz ]]; then
        process "$file"
      fi
    fi

    if [ -d "$file" ]; then
      iterate "$file"
    fi
  done
}

function run_name() {
    local base
    base="$(basename "$1")"
    base="${base%.tar.gz}"
    [[ "$2" == "fp16" ]] && base="${base}_fp16"
    echo "$base"
}

function log_stem() {
    echo "$WORK_DIR/logs/$2/$(run_name "$1" "$3")"
}

function mark_skipped() {
    local file="$1" reason="$2" dtypes="$3"
    local kind dt
    for kind in $KINDS; do
        for dt in $dtypes; do
            echo "SKIPPED: $reason" > "$(log_stem "$file" "$kind" "$dt").out"
        done
    done
    echo "WARNING: ${file}: $reason"
}

function process() {
    local file="$1"
    # int8/qdq archives are already quantized in-graph, so skip the fp16 pass.
    local dtypes="fp32 fp16"
    case "$(basename "$file")" in
        *int8* | *qdq*) dtypes="int8" ;;
    esac

    echo "INFO: process $file started"
    run_archive "$file" "$dtypes"
    cleanup "$file"
    echo "INFO: process $file finished"
}

function run_archive() {
    local file="$1" dtypes="$2"
    if ! setup "$file"; then
        mark_skipped "$file" "could not extract archive" "$dtypes"
        return 0
    fi

    local model_file
    model_file="$(find "$WORK_DIR/tmp_model" -type f -name '*.onnx' ! -name '._*' 2>/dev/null | sort | head -1)"
    if [[ -z "$model_file" ]]; then
        mark_skipped "$file" "no .onnx model found in archive" "$dtypes"
        return 0
    fi

    local kind dt
    for kind in $KINDS; do
        for dt in $dtypes; do
            "run_$kind" "$file" "$dt" "$model_file"
        done
    done
}

function setup() {
    local file="$1"
    echo "INFO: setup $file"
    if [[ "$USE_LOCAL" -ne 1 ]]; then
        local folder
        folder="$(cd -P -- "$(dirname -- "$file")" && pwd -P)" || return 1
        (cd "$folder" && git lfs pull --include="$(basename "$file")" --exclude="") \
            &> "${PIPE}" || return 1
    fi
    tar xzf "$file" -C "$WORK_DIR/tmp_model" &> "${PIPE}"
}

function cleanup() {
    local file="$1"
    echo "INFO: cleanup $file"
    if [[ "$USE_LOCAL" -ne 1 ]]; then
        local folder
        folder="$(cd -P -- "$(dirname -- "$file")" && pwd -P)" || return 0
        (cd "$folder" && git lfs prune) &> "${PIPE}" || true
    fi
    rm -rf "${WORK_DIR:?}/tmp_model/"* &> "${PIPE}"
}

function run_logged() {
    local file="$1" kind="$2" dtype="$3"
    shift 3
    local stem
    stem="$(log_stem "$file" "$kind" "$dtype")"
    echo "INFO: $kind $file ($dtype)"
    if ! timeout --kill-after=30s "$MODEL_TIMEOUT" "$@" > "$stem.out" 2> "$stem.err"; then
        echo "WARNING: $kind failed for ${file} ($dtype)"
    fi
    [[ -s "$stem.err" ]] || rm -f "$stem.err"
}

function run_accuracy() {
    local file="$1" dtype="$2" model_file="$3"
    local args_file="$WORK_DIR/tmp_model/driver-args-$dtype"
    local -a flag
    if [[ "$dtype" = "fp16" ]]; then
        flag=(--atol "$FP16_ATOL" --rtol "$FP16_RTOL" --target "$TARGET" --fp16)
    else
        flag=(--atol "$ATOL" --rtol "$RTOL" --target "$TARGET")
    fi
    rm -f "$args_file"
    run_logged "$file" accuracy "$dtype" \
        python3 "$TESTER_SCRIPT" "${flag[@]}" --emit-driver-args "$args_file" \
            "$(dirname "$model_file")"
}

function run_perf() {
    local file="$1" dtype="$2" model_file="$3"
    local args_file="$WORK_DIR/tmp_model/driver-args-$dtype"
    local -a driver_args=()
    local -a flag=(--onnx)
    if [[ -s "$args_file" ]]; then
        mapfile -t driver_args < "$args_file"
        flag+=("${driver_args[@]}")
    else
        flag+=("--$TARGET")
        [[ "$dtype" = "fp16" ]] && flag+=(--fp16)
    fi
    flag+=(-n "$PERF_ITERATIONS")
    run_logged "$file" perf "$dtype" \
        "$DRIVER" perf "$model_file" "${flag[@]}"
}

if [[ "$#" -eq 0 ]]; then
    echo "usage: $(basename "$0") <model-dir> [model-dir ...]" >&2
    exit 2
fi

for arg in "$@"; do
    if [[ ! -d "$arg" ]]; then
        echo "ERROR: '$arg' is not a directory" >&2
        exit 2
    fi
done

if [[ ! -f "$TESTER_SCRIPT" ]]; then
    echo "ERROR: tester not found: $TESTER_SCRIPT" >&2
    exit 2
fi
if ! python3 -c 'import migraphx' 2> /dev/null; then
    echo "ERROR: cannot import migraphx; add the build lib directory to PYTHONPATH" >&2
    exit 1
fi
if [[ " $KINDS " == *" perf "* ]] && ! command -v "$DRIVER" &> /dev/null; then
    echo "ERROR: migraphx-driver not found: $DRIVER" >&2
    exit 1
fi

mkdir -p "$WORK_DIR/tmp_model"
for kind in $KINDS; do
    mkdir -p "$WORK_DIR/logs/$kind"
done
rm -rf "${WORK_DIR:?}/tmp_model/"*

for arg in "$@"; do
    iterate "$(readlink -e "$arg")"
done
