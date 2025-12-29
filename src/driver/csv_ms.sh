#!/bin/bash

input_file="$1"
output_file="output.csv"

echo "subgraph name,time (ms)" > "$output_file"

awk '
/^Reading:/ {
    # Remove "Reading: Placeholder_Identity/" prefix and ".pb" suffix
    sub(/^Reading: Placeholder_Identity\//, "", $0)
    sub(/\.pb$/, "", $0)
    subgraph=$0
}
/^Total time:/ {
    match($0, /Total time: ([0-9.]+)ms/, arr)
    if (arr[1] != "" && subgraph != "") {
        print subgraph "," arr[1]
    }
}
' "$input_file" >> "$output_file"