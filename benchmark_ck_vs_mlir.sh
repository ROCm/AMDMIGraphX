#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
DRIVER="${BUILD_DIR}/bin/driver"
CK_DIR="${BUILD_DIR}/saved_models/ck_full_models"
MLIR_DIR="${BUILD_DIR}/saved_models/mlir_models"
NITER=1000

print_stats() {
    local filter_m="${1:-}" filter_n="${2:-}" filter_k="${3:-}" filter_o="${4:-}" filter_mode="${5:-and}"
    awk -F',' -v fm="$filter_m" -v fn="$filter_n" -v fk="$filter_k" -v fo="$filter_o" -v fmode="$filter_mode" '
    function matches(val, filter,    i, n, parts) {
        if (filter == "") return 1
        n = split(filter, parts, ",")
        for (i = 1; i <= n; i++) {
            gsub(/ /, "", parts[i])
            if (val == parts[i]) return 1
        }
        return 0
    }
    NR > 1 {
        mm = matches($4, fm); mn = matches($5, fn); mk = matches($6, fk); mo = matches($7, fo)
        if (fmode == "and") {
            if (!mm || !mn || !mk || !mo) next
        } else {
            hit = 0
            if (fm != "" && mm) hit = 1
            if (fn != "" && mn) hit = 1
            if (fk != "" && mk) hit = 1
            if (fo != "" && mo) hit = 1
            has_filter = (fm != "" || fn != "" || fk != "" || fo != "")
            if (has_filter && !hit) next
        }
        if ($10 == "ck") {
            ck_count++
            ck_speedups[ck_count] = $12
            ck_sum += $12
            if (ck_count == 1 || $12 < ck_min) ck_min = $12
            if (ck_count == 1 || $12 > ck_max) ck_max = $12
        } else {
            mlir_count++
            mlir_speedups[mlir_count] = -$12
            mlir_sum += -$12
            if (mlir_count == 1 || -$12 < mlir_min) mlir_min = -$12
            if (mlir_count == 1 || -$12 > mlir_max) mlir_max = -$12
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
            printf "  min speedup: %.2f%%\n", ck_min
            printf "  max speedup: %.2f%%\n", ck_max
        }
        printf "\nMLIR faster: %d/%d\n", mlir_count, total
        if (mlir_count > 0) {
            printf "  avg speedup: %.2f%%\n", mlir_sum / mlir_count
            printf "  median speedup: %.2f%%\n", median(mlir_speedups, mlir_count)
            printf "  min speedup: %.2f%%\n", mlir_min
            printf "  max speedup: %.2f%%\n", mlir_max
        }
    }' "$CSV"
}

if [[ "${1:-}" == "--stats" ]]; then
    shift
    FILTER_M="" FILTER_N="" FILTER_K="" FILTER_O="" FILTER_MODE="and"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -f) CSV="$2"; shift 2 ;;
            -m) FILTER_M="$2"; shift 2 ;;
            -n) FILTER_N="$2"; shift 2 ;;
            -k) FILTER_K="$2"; shift 2 ;;
            -o) FILTER_O="$2"; shift 2 ;;
            --and) FILTER_MODE="and"; shift ;;
            --or)  FILTER_MODE="or";  shift ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    print_stats "$FILTER_M" "$FILTER_N" "$FILTER_K" "$FILTER_O" "$FILTER_MODE"
    exit 0
fi

NUM_SPLITS=$1
CSV="${SCRIPT_DIR}/benchmark_splitkv_ck_full_vs_mlir_${NUM_SPLITS}_splits.csv"

echo "num_splits,batch,nhead,M,N,K,O,ck_time_ms,mlir_time_ms,faster,delta_ms,speedup_pct" > "$CSV"

BATCH=2
NHEAD=4

for M in 1 16 32; do
    for N in 1024 2048 4096; do
        for K in 32 48 64 80 96 128 192 256; do
            for O in 32 48 64 80 96 128 192 256; do
                TAG="${NUM_SPLITS}_${BATCH}_${NHEAD}_${M}_${N}_${K}_${O}"

                CK_MODEL="${CK_DIR}/ck_full_${TAG}.mxr"
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
                echo "${NUM_SPLITS},${BATCH},${NHEAD},${M},${N},${K},${O},${CK_TIME},${MLIR_TIME},${FASTER},${DELTA},${SPEEDUP}" >> "$CSV"
            done
        done
    done
done

echo ""
echo "Results written to $CSV"
echo ""
print_stats

