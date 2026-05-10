#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
DRIVER="${BUILD_DIR}/bin/driver"
CK_DIR="${BUILD_DIR}/saved_models/ck_full_models"
MLIR_DIR="${BUILD_DIR}/saved_models/mlir_models"
NITER=2500
MEASURE_PASSES=5
SCLK_MHZ="${SCLK_MHZ:-1900}"
DISCOVERY_MODE="${DISCOVERY_MODE:-0}"

# NUMA topology: Node 0 (CPUs 0-55,112-167) → GPU 0-3
#                Node 1 (CPUs 56-111,168-223) → GPU 4-7
declare -A NUMA_CPUS=(
    [0]="0-55,112-167" [1]="0-55,112-167" [2]="0-55,112-167" [3]="0-55,112-167"
    [4]="56-111,168-223" [5]="56-111,168-223" [6]="56-111,168-223" [7]="56-111,168-223"
)
declare -A NUMA_NODE=(
    [0]=0 [1]=0 [2]=0 [3]=0
    [4]=1 [5]=1 [6]=1 [7]=1
)
declare -A NUMA_CORE_LIST=(
    [0]="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167"
    [1]="56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223"
)

set_cpu_governor() {
    local governor="$1"
    local node="${NUMA_NODE[$GPU_ID]}"
    for cpu in ${NUMA_CORE_LIST[$node]}; do
        echo "$governor" | sudo tee "/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_governor" > /dev/null 2>&1 || true
    done
}

setup_perf_env() {
    local node="${NUMA_NODE[$GPU_ID]}"
    echo "=== Performance environment setup (GPU ${GPU_ID}, NUMA node ${node}) ==="

    if [[ "$DISCOVERY_MODE" == "1" ]]; then
        echo "  DISCOVERY MODE: skipping GPU clock pinning (will log natural clocks)"
    else
        echo "  Pinning GPU ${GPU_ID} clocks (setperfdeterminism ${SCLK_MHZ} MHz)..."
        sudo rocm-smi -d "$GPU_ID" --setperfdeterminism "$SCLK_MHZ" > /dev/null 2>&1
    fi

    local first_cpu
    first_cpu=$(echo "${NUMA_CORE_LIST[$node]}" | awk '{print $1}')
    ORIG_CPU_GOVERNOR=$(cat "/sys/devices/system/cpu/cpu${first_cpu}/cpufreq/scaling_governor" 2>/dev/null || echo "powersave")

    echo "  Setting CPU governor to performance (NUMA node ${node} cores only, was ${ORIG_CPU_GOVERNOR})..."
    set_cpu_governor performance

    export ROCR_VISIBLE_DEVICES="$GPU_ID"

    echo "  GPU ${GPU_ID} visible, NUMA CPUs: ${NUMA_CPUS[$GPU_ID]}"
    echo "=== Setup complete ==="
    echo ""
}

log_gpu_metrics() {
    local label="${1:-SNAPSHOT}"
    {
        echo "=== GPU ${GPU_ID} metrics [${label}] $(date '+%Y-%m-%d %H:%M:%S') ==="
        rocm-smi -d "$GPU_ID" -c -P -t -p 2>/dev/null | grep "GPU\[$GPU_ID\]" || true
        echo ""
    } >> "$METRICS_LOG"
}

start_gpu_sampler() {
    local label="$1"
    (
        while true; do
            local ts
            ts=$(date '+%H:%M:%S.%3N')
            local lines
            local sclk
            sclk=$(rocm-smi -d "$GPU_ID" -c 2>/dev/null | grep -i "sclk" || true)
            if [[ -n "$sclk" ]]; then
                echo "${ts} [${label}] ${sclk}"
            fi
            sleep 0.1
        done
    ) >> "$METRICS_LOG" &
    GPU_SAMPLER_PID=$!
}

stop_gpu_sampler() {
    if [[ -n "${GPU_SAMPLER_PID:-}" ]]; then
        kill "$GPU_SAMPLER_PID" 2>/dev/null || true
        wait "$GPU_SAMPLER_PID" 2>/dev/null || true
        unset GPU_SAMPLER_PID
    fi
}

teardown_perf_env() {
    if [[ "$DISCOVERY_MODE" == "1" ]]; then
        stop_gpu_sampler
        log_gpu_metrics "POST-BENCHMARK"
        echo "  Metrics log: $METRICS_LOG"
    fi
    echo ""
    echo "=== Restoring system defaults ==="
    if [[ "$DISCOVERY_MODE" != "1" ]]; then
        echo "  Resetting GPU ${GPU_ID} clocks (resetperfdeterminism)..."
        sudo rocm-smi -d "$GPU_ID" --resetperfdeterminism > /dev/null 2>&1
    fi
    echo "  Restoring CPU governor to ${ORIG_CPU_GOVERNOR} (NUMA node ${NUMA_NODE[$GPU_ID]} cores only)..."
    set_cpu_governor "$ORIG_CPU_GOVERNOR"
    echo "=== Restore complete ==="
}

get_median() {
    printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END {
        if (NR%2==1) print a[(NR+1)/2]
        else printf "%.7f\n", (a[NR/2]+a[NR/2+1])/2
    }'
}

measure_time() {
    local model="$1"
    local node="${NUMA_NODE[$GPU_ID]}"
    local output

    local times=()
    for _ in $(seq 1 "$MEASURE_PASSES"); do
        local t
        output=$(numactl --cpunodebind="$node" --membind="$node" \
            "$DRIVER" time -n "$NITER" "$model" 2>&1) || {
            echo "ERROR: measurement failed for $model" >&2
            echo "$output" >&2
            return 1
        }
        t=$(echo "$output" | grep "Total time:" | awk '{print $3}' | sed 's/ms//')
        if [[ -z "$t" ]]; then
            echo "ERROR: could not parse Total time from output:" >&2
            echo "$output" >&2
            return 1
        fi
        times+=("$t")
    done
    local median
    median=$(get_median "${times[@]}")
    echo "$median ${times[*]}"
}

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

if [[ -z "${GPU_ID:-}" ]]; then
    echo "ERROR: GPU_ID must be set. Example: GPU_ID=2 $0 $*" >&2
    exit 1
fi

NUM_SPLITS=$1

NO_PERF_SETUP="${NO_PERF_SETUP:-0}"
if [[ "$NO_PERF_SETUP" != "1" ]]; then
    setup_perf_env
    trap teardown_perf_env EXIT
fi

CSV="${SCRIPT_DIR}/benchmark_splitkv_ck_full_vs_mlir_${NUM_SPLITS}_splits.csv"
if [[ "$DISCOVERY_MODE" == "1" ]]; then
    METRICS_LOG="${SCRIPT_DIR}/gpu_metrics_${NUM_SPLITS}_splits_gpu${GPU_ID}_$(date '+%Y%m%d_%H%M%S').log"
    echo "GPU metrics log: $METRICS_LOG"
    log_gpu_metrics "PRE-BENCHMARK"
fi

RUN_COLS=""
for i in $(seq 0 $((MEASURE_PASSES - 1))); do
    RUN_COLS="${RUN_COLS},ck_run_${i}"
done
for i in $(seq 0 $((MEASURE_PASSES - 1))); do
    RUN_COLS="${RUN_COLS},mlir_run_${i}"
done
echo "num_splits,batch,nhead,M,N,K,O,ck_time_ms,mlir_time_ms,faster,delta_ms,speedup_pct${RUN_COLS}" > "$CSV"

BATCH=2
NHEAD=4

if [[ "$DISCOVERY_MODE" == "1" ]]; then
    echo "Benchmark config: NITER=${NITER}, MEASURE_PASSES=${MEASURE_PASSES} (median), DISCOVERY MODE (no clock pinning)"
else
    echo "Benchmark config: NITER=${NITER}, MEASURE_PASSES=${MEASURE_PASSES} (median), SCLK=${SCLK_MHZ} MHz"
fi
echo ""

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

                [[ "$DISCOVERY_MODE" == "1" ]] && start_gpu_sampler "${TAG}"
                read -r CK_TIME CK_RUNS <<< "$(measure_time "$CK_MODEL")"
                read -r MLIR_TIME MLIR_RUNS <<< "$(measure_time "$MLIR_MODEL")"
                [[ "$DISCOVERY_MODE" == "1" ]] && stop_gpu_sampler

                FASTER=$(awk "BEGIN {print ($CK_TIME < $MLIR_TIME) ? \"ck\" : \"mlir\"}")
                DELTA=$(awk "BEGIN {printf \"%.6f\", $MLIR_TIME - $CK_TIME}")
                SPEEDUP=$(awk "BEGIN {printf \"%.2f\", ($MLIR_TIME - $CK_TIME) / $MLIR_TIME * 100}")
                echo "ck=${CK_TIME}ms  mlir=${MLIR_TIME}ms  faster=${FASTER}  delta=${DELTA}ms  speedup=${SPEEDUP}%"
                echo "  ck_runs: ${CK_RUNS}  mlir_runs: ${MLIR_RUNS}"

                CK_RUN_COLS=$(echo "$CK_RUNS" | tr ' ' ',')
                MLIR_RUN_COLS=$(echo "$MLIR_RUNS" | tr ' ' ',')
                echo "${NUM_SPLITS},${BATCH},${NHEAD},${M},${N},${K},${O},${CK_TIME},${MLIR_TIME},${FASTER},${DELTA},${SPEEDUP},${CK_RUN_COLS},${MLIR_RUN_COLS}" >> "$CSV"
            done
        done
    done
done

echo ""
echo "Results written to $CSV"
echo ""
print_stats

