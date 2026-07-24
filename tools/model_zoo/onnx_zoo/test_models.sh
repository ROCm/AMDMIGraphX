#!/bin/bash

#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
SCRIPT_PATH=$(dirname $(dirname $(dirname $(readlink -f "$0"))))/test_runner.py
TESTER_SCRIPT="${TESTER:-$SCRIPT_PATH}"
ATOL="${ATOL:-0.001}"
RTOL="${RTOL:-0.001}"
FP16_ATOL="${FP16_ATOL:-0.04}"
FP16_RTOL="${FP16_RTOL:-0.04}"
TARGET="${TARGET:-gpu}"
USE_LOCAL="${USE_LOCAL:-0}"

if [[ "${DEBUG:-0}" -eq 1 ]]; then
    PIPE=/dev/stdout
else
    PIPE=/dev/null
fi

if [[ "${VERBOSE:-0}" -eq 1 ]]; then
    set -x
fi

# Iterate through input recursively, process any tar.gz file
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

# A previous run passed if its log shows zero failed cases
function passed() {
    [[ -f "$1" ]] && grep -qE 'Failed: 0$' "$1"
}

# Log file path for a given tar.gz and dtype
function log_file() {
    local base="$(basename "$1")"
    echo "$WORK_DIR/logs/$2/${base//\//_}.log"
}

# Process will download the lfs file, extract model and test data
# Test it with test_runner.py, then cleanup
function process() {
    local file="$1"
    # skip quantizing int8 models
    local dt dtypes="fp32 fp16"
    case "$(basename "$file")" in
        *int8* | *qdq*) dtypes="int8" ;;
    esac

    # skip when every precision we run for this model already passed
    local all_passed=1
    for dt in $dtypes; do
        passed "$(log_file "$file" "$dt")" || all_passed=0
    done
    if [[ "$all_passed" -eq 1 ]]; then
        echo "INFO: skip $file - already passed"
        return
    fi

    echo "INFO: process $file started"
    setup $file
    TEST_DIR="$(find "$WORK_DIR/tmp_model" -type f -name '*.onnx' ! -name '._*' -printf '%h\n' 2>/dev/null | sort -u | head -1)"
    for dt in $dtypes; do
        test $file "$dt"
    done
    cleanup $file
    echo "INFO: process $file finished"
}

# Download and extract files
function setup() {
    local file="$1"
    echo "INFO: setup $file"
    local_file="$(basename $file)"
    if [[ "$USE_LOCAL" -ne 1 ]]; then
        # We need to change the folder to pull the file
        folder="$(cd -P -- "$(dirname -- "$file")" && pwd -P)"
        cd $folder &> "${PIPE}" && git lfs pull --include="$local_file" --exclude="" &> "${PIPE}"; cd - &> "${PIPE}"
    fi
    tar xzf $file -C $WORK_DIR/tmp_model &> "${PIPE}"
}

# Remove tmp files and prune models
function cleanup() {
    local file="$1"
    echo "INFO: cleanup $file"
    if [[ "$USE_LOCAL" -ne 1 ]]; then
        # We need to change the folder to prune the file
        folder="$(cd -P -- "$(dirname -- "$file")" && pwd -P)"
        cd $folder &> "${PIPE}" && git lfs prune &> "${PIPE}"; cd - &> "${PIPE}"
    fi
    rm -r $WORK_DIR/tmp_model/* &> "${PIPE}"
}

# Run test_runner.py and log if something goes wrong
function test() {
    local file="$1"
    local dtype="$2"
    local log="$(log_file "$file" "$dtype")"
    # skip this precision if it already passed
    if passed "$log"; then
        echo "INFO: skip $file ($dtype) - already passed"
        return
    fi
    echo "INFO: test $file ($dtype)"
    if [[ -z "$TEST_DIR" ]]; then
        echo "SKIPPED: no .onnx model found in archive" > "$log"
        echo "WARNING: ${file} ($dtype) has no model to test"
        return
    fi
    local flag
    if [[ "$dtype" = "fp16" ]]; then
        flag="--atol $FP16_ATOL --rtol $FP16_RTOL --target $TARGET --fp16"
    else
        flag="--atol $ATOL --rtol $RTOL --target $TARGET"
    fi
    EXIT_CODE=0
    python3 $TESTER_SCRIPT ${flag} "$TEST_DIR" &> "$log" || EXIT_CODE=$?
    if [[ "${EXIT_CODE:-0}" -ne 0 ]]; then
        echo "WARNING: ${file} failed ($dtype)"
    fi
}

mkdir -p $WORK_DIR/logs/fp32/ $WORK_DIR/logs/fp16/ $WORK_DIR/logs/int8/ $WORK_DIR/tmp_model
rm -fr $WORK_DIR/tmp_model/*

for arg in "$@"; do
    iterate "$(dirname $(readlink -e $arg))/$(basename $arg)"
done
