#!/bin/bash

set -e

WORK_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH=$(dirname $(dirname $(readlink -f "$0")))/test_runner.py
TESTER_SCRIPT="${TESTER:-$SCRIPT_PATH}"
ATOL="${ATOL:-0.001}"
RTOL="${RTOL:-0.001}"
TARGET="${TARGET:-gpu}"

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

# Process will download the lfs file, extract model and test data
# Test it with test_runner.py, then cleanup
function process() {
    local file="$1"
    echo "INFO: process $file started"
    setup $file
    test $file fp32
    test $file fp16
    cleanup $file
    echo "INFO: process $file finished"
}

# Download and extract files
function setup() {
    local file="$1"
    echo "INFO: setup $file"
    local_file="$(basename $file)"
    # We need to change the folder to pull the file
    folder="$(cd -P -- "$(dirname -- "$file")" && pwd -P)"
    cd $folder &> "${PIPE}" && git lfs pull --include="$local_file" --exclude="" &> "${PIPE}"; cd - &> "${PIPE}"
    tar xzf $file -C $WORK_DIR/tmp_model &> "${PIPE}"
}

# Remove tmp files and prune models
function cleanup() {
    local file="$1"
    echo "INFO: cleanup $file"
    # We need to change the folder to pull the file
    folder="$(cd -P -- "$(dirname -- "$file")" && pwd -P)"
    cd $folder &> "${PIPE}" && git lfs prune &> "${PIPE}"; cd - &> "${PIPE}"
    rm -r $WORK_DIR/tmp_model/* &> "${PIPE}"
}

# Run test_runner.py and log if something goes wrong
function test() {
    local file="$1"
    echo "INFO: test $file ($2)"
    local_file="$(basename $file)"
    flag="--atol $ATOL --rtol $RTOL --target $TARGET"
    if [[ "$2" = "fp16" ]]; then
        flag="$flag --fp16"
    fi
    EXIT_CODE=0
    python3 $TESTER_SCRIPT ${flag} $WORK_DIR/tmp_model/*/ &> "$WORK_DIR/logs/$2/${local_file//\//_}.log" || EXIT_CODE=$?
    if [[ "${EXIT_CODE:-0}" -ne 0 ]]; then
        echo "WARNING: ${file} failed ($2)"
    fi
}

mkdir -p $WORK_DIR/logs/fp32/ $WORK_DIR/logs/fp16/ $WORK_DIR/tmp_model
rm -fr $WORK_DIR/tmp_model/*

for arg in "$@"; do
    iterate "$(dirname $(readlink -e $arg))/$(basename $arg)"
done
