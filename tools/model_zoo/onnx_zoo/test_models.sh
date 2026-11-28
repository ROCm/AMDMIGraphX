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
COMPARE="${COMPARE:-allclose}"
RMS_TOL="${RMS_TOL:-0.001}"
FP16_RMS_TOL="${FP16_RMS_TOL:-0.08}"
TARGET="${TARGET:-gpu}"
USE_LOCAL="${USE_LOCAL:-0}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-20m}"
DRIVER="${DRIVER:-migraphx-driver}"
KINDS="${KINDS:-accuracy perf}"
MODEL_LIST="${MODEL_LIST:-}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs}"
TOTAL_CHECKS=0
FAILED_CHECKS=0
SKIPPED_CHECKS=0
ROOT=""

if [[ "${DEBUG:-0}" -eq 1 ]]; then
    PIPE=/dev/stdout
else
    PIPE=/dev/null
fi

if [[ "${VERBOSE:-0}" -eq 1 ]]; then
    set -x
fi

# A model can be a .tar.gz archive or a model folder that is used in place.
function first_model() {
    find -L "$1" -maxdepth 1 -type f -name '*.onnx' ! -name '._*' | LC_ALL=C sort | head -1
}

# A model folder holds a model with its .onnx.json or test_data_set_*/ beside it.
function is_model_dir() {
    local model
    model="$(first_model "$1")"
    [[ -n "$model" ]] || return 1
    [[ -f "$model.json" ]] \
        || [[ -n "$(find -L "$1" -maxdepth 1 -type d -name 'test_data_set_*' -print -quit)" ]]
}

function iterate() {
  local dir="$1"

  if is_model_dir "$dir"; then
    process "$dir"
  fi

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

function run_list() {
    local root="$1" entry model setting i
    local -a models=() settings=() words
    while IFS= read -r entry || [[ -n "$entry" ]]; do
        [[ -z "$entry" || "$entry" = \#* ]] && continue
        # Trailing KEY=value words set a tolerance for this model only.
        setting=""
        while [[ ! -e "$root/$entry" \
            && "$entry" =~ ^(.*[^[:space:]])[[:space:]]+([A-Z0-9_]+=[^[:space:]]+)$ ]]; do
            entry="${BASH_REMATCH[1]}"
            case "${BASH_REMATCH[2]%%=*}" in
                ATOL | RTOL | FP16_ATOL | FP16_RTOL | RMS_TOL | FP16_RMS_TOL) ;;
                *)
                    echo "ERROR: unknown setting '${BASH_REMATCH[2]}' for listed model: $entry" >&2
                    exit 1
                    ;;
            esac
            setting="${BASH_REMATCH[2]} $setting"
        done
        model="$root/${entry%/}"
        if [[ ! -f "$model" && ! -d "$model" ]]; then
            echo "ERROR: listed model not found: $model" >&2
            exit 1
        fi
        models+=("$model")
        settings+=("$setting")
    done < "$MODEL_LIST"

    if [[ "${#models[@]}" -eq 0 ]]; then
        echo "ERROR: no models listed in '$MODEL_LIST'" >&2
        exit 1
    fi
    for i in "${!models[@]}"; do
        read -ra words <<< "${settings[$i]}"
        process "${models[$i]}" "${words[@]}"
    done
}

# Name shown in the summary: the archive name, or the path of a model folder
# below the model root.
function display_name() {
    local name
    if [[ -d "$1" && "$1" != "$ROOT" ]]; then
        echo "${1#"$ROOT"/}"
    else
        name="$(basename "$1")"
        echo "${name%.tar.gz}"
    fi
}

function run_name() {
    local base
    base="$(display_name "$1")"
    base="${base//\//__}"
    [[ "$2" == "fp16" ]] && base="${base}_fp16"
    echo "$base"
}

function log_stem() {
    echo "$LOG_DIR/$2/$(run_name "$1" "$3")"
}

function mark_skipped() {
    local file="$1" reason="$2" dtypes="$3"
    local kind dt
    for kind in $KINDS; do
        for dt in $dtypes; do
            TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            echo "SKIPPED: $reason" > "$(log_stem "$file" "$kind" "$dt").out"
        done
    done
    echo "WARNING: ${file}: $reason"
}

function process() {
    local file="$1"
    shift
    # Settings from the model list replace the global ones for this model.
    local ATOL="$ATOL" RTOL="$RTOL" FP16_ATOL="$FP16_ATOL" FP16_RTOL="$FP16_RTOL"
    local RMS_TOL="$RMS_TOL" FP16_RMS_TOL="$FP16_RMS_TOL"
    local setting dt
    for setting in "$@"; do
        printf -v "${setting%%=*}" '%s' "${setting#*=}"
    done

    # Quantized models are already quantized in-graph, so skip the fp16 pass.
    local dtypes="fp32 fp16"
    if [[ -d "$file" ]]; then
        # The name of a model folder need not say so: look at the graph.
        dtypes="$(python3 "$TESTER_SCRIPT" --precisions "$file" 2> /dev/null)" || true
        [[ "$dtypes" == "int8" ]] || dtypes="fp32 fp16"
    else
        case "$(basename "$file")" in
            *int8* | *qdq*) dtypes="int8" ;;
        esac
    fi
    # Tell summarize.py which model and precision each log belongs to.
    for dt in $dtypes; do
        printf '%s\t%s\t%s\n' "$(run_name "$file" "$dt")" "$(display_name "$file")" "$dt" \
            >> "$LOG_DIR/checks.tsv"
    done

    echo "INFO: process $file started"
    run_model "$file" "$dtypes"
    cleanup "$file"
    echo "INFO: process $file finished"
}

function run_model() {
    local file="$1" dtypes="$2"
    local model_file
    if [[ -d "$file" ]]; then
        model_file="$(first_model "$file")"
        if [[ -z "$model_file" ]]; then
            mark_skipped "$file" "no .onnx model found in folder" "$dtypes"
            return 0
        fi
    else
        if ! setup "$file"; then
            mark_skipped "$file" "could not extract archive" "$dtypes"
            return 0
        fi

        model_file="$(find "$WORK_DIR/tmp_model" -type f -name '*.onnx' ! -name '._*' 2>/dev/null | sort | head -1)"
        if [[ -z "$model_file" ]]; then
            mark_skipped "$file" "no .onnx model found in archive" "$dtypes"
            return 0
        fi
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
    if [[ "$USE_LOCAL" -ne 1 && ! -d "$file" ]]; then
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
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if ! timeout --kill-after=30s "$MODEL_TIMEOUT" "$@" > "$stem.out" 2> "$stem.err"; then
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        echo "WARNING: $kind failed for ${file} ($dtype)"
    elif grep -q '^SKIPPED:' "$stem.out"; then
        # Nothing to check, such as no golden data: not a pass and not a failure.
        TOTAL_CHECKS=$((TOTAL_CHECKS - 1))
        SKIPPED_CHECKS=$((SKIPPED_CHECKS + 1))
        echo "INFO: $kind skipped for ${file} ($dtype)"
    fi
    [[ -s "$stem.err" ]] || rm -f "$stem.err"
}

function run_accuracy() {
    local file="$1" dtype="$2" model_file="$3"
    local args_file="$WORK_DIR/tmp_model/driver-args-$dtype"
    local -a flag
    if [[ "$dtype" = "fp16" ]]; then
        flag=(--atol "$FP16_ATOL" --rtol "$FP16_RTOL" --rms-tol "$FP16_RMS_TOL"
            --target "$TARGET" --fp16)
    else
        flag=(--atol "$ATOL" --rtol "$RTOL" --rms-tol "$RMS_TOL" --target "$TARGET")
    fi
    flag+=(--compare "$COMPARE")
    if [[ -d "$file" ]]; then
        flag+=(--name "$(display_name "$file")")
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
    local -a flag=(--onnx --log-stdout)
    if [[ -s "$args_file" ]]; then
        mapfile -t driver_args < "$args_file"
        flag+=("${driver_args[@]}")
    else
        flag+=("--$TARGET")
        [[ "$dtype" = "fp16" ]] && flag+=(--fp16)
    fi
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
if [[ -n "$MODEL_LIST" && "$#" -ne 1 ]]; then
    echo "ERROR: MODEL_LIST requires exactly one model root" >&2
    exit 2
fi
if [[ -n "$MODEL_LIST" && ! -f "$MODEL_LIST" ]]; then
    echo "ERROR: model list not found: '$MODEL_LIST'" >&2
    exit 2
fi

if [[ "$COMPARE" != "allclose" && "$COMPARE" != "rms" ]]; then
    echo "ERROR: COMPARE must be allclose or rms, not '$COMPARE'" >&2
    exit 2
fi

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

mkdir -p "$WORK_DIR/tmp_model" "$LOG_DIR"
for kind in $KINDS; do
    mkdir -p "$LOG_DIR/$kind"
done
rm -rf "${WORK_DIR:?}/tmp_model/"*
: > "$LOG_DIR/checks.tsv"

if [[ -n "$MODEL_LIST" ]]; then
    ROOT="$(readlink -e "$1")"
    run_list "$ROOT"
else
    for arg in "$@"; do
        ROOT="$(readlink -e "$arg")"
        iterate "$ROOT"
    done
fi

PASSED_CHECKS=$((TOTAL_CHECKS - FAILED_CHECKS))
SKIPPED_NOTE=""
if (( SKIPPED_CHECKS > 0 )); then
    SKIPPED_NOTE=", ${SKIPPED_CHECKS} skipped"
fi
echo "INFO: health check: ${PASSED_CHECKS}/${TOTAL_CHECKS} checks passed${SKIPPED_NOTE}"
# Do not publish a baseline when no checks ran or at least half failed.
if (( TOTAL_CHECKS == 0 || FAILED_CHECKS * 2 >= TOTAL_CHECKS )); then
    echo "ERROR: unhealthy run: ${FAILED_CHECKS}/${TOTAL_CHECKS} checks failed" >&2
    exit 1
fi
