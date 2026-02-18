#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
DRIVER="${BUILD_DIR}/bin/driver"
CK_DIR="${BUILD_DIR}/saved_models/ck_models"
MLIR_DIR="${BUILD_DIR}/saved_models/mlir_models"
CSV="${SCRIPT_DIR}/benchmark_ck_vs_mlir.csv"
NITER=1000

echo "batch,nhead,M,N,K,O,ck_time_ms,mlir_time_ms,faster,delta_ms,speedup_pct" > "$CSV"

BATCH=2
NHEAD=4

for M in 512 1024; do
    for N in 512 1024; do
        for K in 32 64 96; do
            for O in 32 64 96; do
                TAG="${BATCH}_${NHEAD}_${M}_${N}_${K}_${O}"

                CK_MODEL="${CK_DIR}/ck_${TAG}.mxr"
                MLIR_MODEL="${MLIR_DIR}/mlir_${TAG}.mxr"

                if [[ ! -f "$CK_MODEL" ]]; then
                    echo "SKIP (missing): $CK_MODEL"
                    continue
                fi
                if [[ ! -f "$MLIR_MODEL" ]]; then
                    echo "SKIP (missing): $MLIR_MODEL"
                    continue
                fi

                echo -n "Timing ${TAG} ... "

                CK_TIME=$($DRIVER time -n "$NITER" "$CK_MODEL" 2>&1 | grep "Total time:" | awk '{print $3}' | sed 's/ms//')
                MLIR_TIME=$($DRIVER time -n "$NITER" "$MLIR_MODEL" 2>&1 | grep "Total time:" | awk '{print $3}' | sed 's/ms//')

                FASTER=$(awk "BEGIN {print ($CK_TIME < $MLIR_TIME) ? \"ck\" : \"mlir\"}")
                DELTA=$(awk "BEGIN {printf \"%.6f\", $MLIR_TIME - $CK_TIME}")
                SPEEDUP=$(awk "BEGIN {printf \"%.2f\", ($MLIR_TIME - $CK_TIME) / $MLIR_TIME * 100}")
                echo "ck=${CK_TIME}ms  mlir=${MLIR_TIME}ms  faster=${FASTER}  delta=${DELTA}ms  speedup=${SPEEDUP}%"
                echo "${BATCH},${NHEAD},${M},${N},${K},${O},${CK_TIME},${MLIR_TIME},${FASTER},${DELTA},${SPEEDUP}" >> "$CSV"
            done
        done
    done
done

echo ""
echo "Results written to $CSV"
echo ""

awk -F',' 'NR > 1 {
    if ($9 == "ck") {
        ck_count++
        ck_speedups[ck_count] = $11
        ck_sum += $11
    } else {
        mlir_count++
        mlir_speedups[mlir_count] = -$11
        mlir_sum += -$11
    }
}
function median(arr, n,    i, tmp, j) {
    for (i = 1; i <= n; i++)
        for (j = i + 1; j <= n; j++)
            if (arr[i] > arr[j]) { tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp }
    if (n % 2 == 1) return arr[int(n/2) + 1]
    return (arr[n/2] + arr[n/2 + 1]) / 2
}
END {
    total = ck_count + mlir_count
    printf "=== Summary ===\n"
    printf "Total comparisons: %d\n\n", total
    printf "CK faster:   %d/%d\n", ck_count, total
    if (ck_count > 0) {
        printf "  avg speedup: %.2f%%\n", ck_sum / ck_count
        printf "  median speedup: %.2f%%\n", median(ck_speedups, ck_count)
    }
    printf "\nMLIR faster: %d/%d\n", mlir_count, total
    if (mlir_count > 0) {
        printf "  avg speedup: %.2f%%\n", mlir_sum / mlir_count
        printf "  median speedup: %.2f%%\n", median(mlir_speedups, mlir_count)
    }
}' "$CSV"
