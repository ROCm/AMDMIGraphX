#!/usr/bin/env python3
"""
Simple concurrent MIGraphX test - tests batch size 1, 2, and 4 concurrently
"""

import os
import sys
import time
import threading
import argparse
import numpy as np
import onnxruntime as ort
from collections import defaultdict
from tqdm import tqdm


# Global storage for metrics
metrics_lock = threading.Lock()
all_latencies = defaultdict(list)  # {thread_id: [latencies]}
thread_results = {}  # {thread_id: {'batch_size': X, 'iterations': Y, ...}}


def run_inference_thread(thread_id, model_path, cache_dir, batch_size, num_iterations, max_dynamic_batch, 
                         verbose, optimization_level, warmup=3, show_io=False):
    """Run inference in a single thread"""
    if show_io:
        print(f"\n[Thread-{thread_id}] Starting with batch_size={batch_size}, max_dynamic_batch={max_dynamic_batch}")

    # Set environment variables
    os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH'] = cache_dir
    os.environ['ORT_MIGRAPHX_MAX_DYNAMIC_BATCH'] = str(max_dynamic_batch)

    # Create session
    session_options = ort.SessionOptions()
    
    # Set logging verbosity (default: verbose OFF)
    if verbose:
        session_options.log_severity_level = 0  # Verbose (0 = Verbose, 1 = Info, 2 = Warning, 3 = Error, 4 = Fatal)
        session_options.log_verbosity_level = 0  # Detailed logs
        if show_io:
            print(f"[Thread-{thread_id}] Verbose logging: ENABLED")
    else:
        session_options.log_severity_level = 2  # Warning level (less verbose)
        session_options.log_verbosity_level = 0  # Standard verbosity
        
    # Set graph optimization level
    opt_level_map = {
        0: ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        1: ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        2: ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        99: ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    }
    session_options.graph_optimization_level = opt_level_map.get(optimization_level, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    if show_io:
        opt_level_names = {
            0: "ORT_DISABLE_ALL",
            1: "ORT_ENABLE_BASIC",
            2: "ORT_ENABLE_EXTENDED",
            99: "ORT_ENABLE_ALL"
        }
        print(f"[Thread-{thread_id}] Graph optimization: {opt_level_names.get(optimization_level, 'ORT_DISABLE_ALL')}")

    # Configure MIGraphX provider with max_dynamic_batch
    migraphx_options = {
        "device_id": 0,
        "migraphx_fp16_enable": 0,
        "migraphx_int8_enable": 0,
        "migraphx_exhaustive_tune": 0,
        "migraphx_max_dynamic_batch": max_dynamic_batch,
    }

    providers = [
        ("MIGraphXExecutionProvider", migraphx_options)
    ]

    latencies = []
    successful_iterations = 0
    batch_mismatches = 0

    try:
        session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        if show_io:
            print(f"[Thread-{thread_id}] Session created. Providers: {session.get_providers()}")

        # Get ALL input info
        all_inputs = session.get_inputs()
        if show_io:
            print(f"[Thread-{thread_id}] Model has {len(all_inputs)} inputs:")
            for inp in all_inputs:
                print(f"  - {inp.name}: {inp.shape} ({inp.type})")

        # Helper function to generate input data
        def generate_input_data(batch_sz):
            feed = {}
            for inp in all_inputs:
                input_shape = list(inp.shape)
                if len(input_shape) > 0 and (isinstance(input_shape[0], str) or input_shape[0] < 0):
                    input_shape[0] = batch_sz
                for dim_idx in range(len(input_shape)):
                    if isinstance(input_shape[dim_idx], str) or input_shape[dim_idx] < 0:
                        input_shape[dim_idx] = 1
                if inp.type == 'tensor(float)':
                    feed[inp.name] = np.random.randn(*input_shape).astype(np.float32)
                elif inp.type == 'tensor(double)':
                    feed[inp.name] = np.random.randn(*input_shape).astype(np.float64)
                elif inp.type == 'tensor(int64)':
                    feed[inp.name] = np.random.randint(0, 100, size=input_shape).astype(np.int64)
                elif inp.type == 'tensor(int32)':
                    feed[inp.name] = np.random.randint(0, 100, size=input_shape).astype(np.int32)
                elif inp.type == 'tensor(int16)':
                    feed[inp.name] = np.random.randint(0, 100, size=input_shape).astype(np.int16)
                elif inp.type == 'tensor(int8)':
                    feed[inp.name] = np.random.randint(0, 100, size=input_shape).astype(np.int8)
                elif inp.type == 'tensor(bool)':
                    feed[inp.name] = np.random.randint(0, 2, size=input_shape).astype(np.bool_)
                else:
                    feed[inp.name] = np.random.randn(*input_shape).astype(np.float32)
            return feed

        # WARMUP: Run warmup iterations to exclude compilation time from measurements
        if warmup > 0:
            if show_io:
                print(f"[Thread-{thread_id}] Running {warmup} warmup iterations (not measured)...")
            for w in range(warmup):
                warmup_feed = generate_input_data(batch_size)
                warmup_start = time.perf_counter()
                _ = session.run(None, warmup_feed)
                warmup_time = (time.perf_counter() - warmup_start) * 1000
                if show_io:
                    print(f"[Thread-{thread_id}] Warmup {w+1}/{warmup}: {warmup_time:.2f}ms")
            if show_io:
                print(f"[Thread-{thread_id}] Warmup complete, starting measured iterations...")

        # Run iterations with progress bar (when show_io is off)
        if show_io:
            iter_range = range(num_iterations)
        else:
            iter_range = tqdm(range(num_iterations), 
                             desc=f"Thread-{thread_id} batch={batch_size}", 
                             position=thread_id-1, 
                             leave=True,
                             ncols=80)
        
        for i in iter_range:
            # Generate random input data for ALL inputs
            input_feed = generate_input_data(batch_size)
            
            # Log input shapes on first iteration (only if show_io enabled)
            if show_io and i == 0:
                print(f"[Thread-{thread_id}] Generated input shapes:")
                for name, data in input_feed.items():
                    print(f"    {name}: {data.shape}")

            # Run inference with ALL inputs - use perf_counter for accurate timing
            start = time.perf_counter()
            outputs = session.run(None, input_feed)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Track latency
            latencies.append(elapsed_ms)
            successful_iterations += 1

            # Check ALL output batch sizes
            all_outputs = session.get_outputs()
            batch_match = True
            output_details = []
            
            for output_idx, (output_info, output_data) in enumerate(zip(all_outputs, outputs)):
                output_name = output_info.name
                output_shape = output_data.shape
                
                # Check if output has batch dimension
                if len(output_shape) > 0:
                    output_batch = output_shape[0]
                    if output_batch != batch_size:
                        batch_match = False
                        output_details.append(f"❌ {output_name}: {output_shape} (batch={output_batch}, expected={batch_size})")
                    else:
                        output_details.append(f"✅ {output_name}: {output_shape} (batch={output_batch})")
                else:
                    # Scalar output (no batch dimension)
                    output_details.append(f"⚠️  {output_name}: {output_shape} (scalar, no batch dim)")
            
            match_symbol = "✅" if batch_match else "❌"
            
            # Track batch mismatches
            if not batch_match:
                batch_mismatches += 1
            
            # Update progress bar with latest latency (when not show_io)
            if not show_io and hasattr(iter_range, 'set_postfix'):
                iter_range.set_postfix({'ms': f'{elapsed_ms:.1f}', 'ok': match_symbol})
            
            # Log iteration details (only if show_io enabled)
            if show_io:
                print(f"[Thread-{thread_id}] Iter {i+1}: {elapsed_ms:.2f}ms, in_batch={batch_size} {match_symbol}")
                print(f"[Thread-{thread_id}] Output verification ({len(outputs)} outputs):")
                for detail in output_details:
                    print(f"[Thread-{thread_id}]   {detail}")

            time.sleep(0.05)  # Small delay

        if show_io:
            print(f"[Thread-{thread_id}] Completed {num_iterations} iterations")

        # Store results
        with metrics_lock:
            all_latencies[thread_id] = latencies
            thread_results[thread_id] = {
                'batch_size': batch_size,
                'iterations': successful_iterations,
                'batch_mismatches': batch_mismatches,
                'avg_latency': np.mean(latencies) if latencies else 0,
                'min_latency': np.min(latencies) if latencies else 0,
                'max_latency': np.max(latencies) if latencies else 0,
                'num_outputs': len(all_outputs) if 'all_outputs' in locals() else 0,
            }

    except Exception as e:
        print(f"[Thread-{thread_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        with metrics_lock:
            thread_results[thread_id] = {
                'batch_size': batch_size,
                'iterations': successful_iterations,
                'error': str(e)
            }


def main():
    parser = argparse.ArgumentParser(description='Test concurrent MIGraphX inference')
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--cache', default='/tmp/migraphx_cache', help='Cache directory')
    parser.add_argument('--iterations', type=int, default=3, help='Iterations per thread')
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 4],
                       help='Batch sizes to test (default: 1 4)')
    parser.add_argument('--threads', type=int, default=2,
                       help='Number of concurrent threads (default: 2)')
    parser.add_argument('--max-dynamic-batch', type=int, default=0,
                       help='Max dynamic batch size for pre-compilation (0=disabled, power of 2 recommended)')
    parser.add_argument('--warmup', type=int, default=3,
                       help='Number of warmup iterations to exclude compilation from timing (default: 3)')
    parser.add_argument('--show_io', action='store_true', default=False,
                       help='Show detailed input/output info for each iteration (default: OFF)')
    parser.add_argument('--thread-details', action='store_true', default=False,
                       help='Show per-thread statistics in final report (default: OFF)')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Enable verbose ONNX Runtime logging (default: OFF)')
    parser.add_argument('--optimization-level', type=int, choices=[0, 1, 2, 99], default=0,
                       help='ONNX Runtime graph optimization level: 0=DISABLE_ALL (default), 1=ENABLE_BASIC, 2=ENABLE_EXTENDED, 99=ENABLE_ALL')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        return 1

    # Validate threads
    if args.threads < 1:
        print(f"ERROR: Number of threads must be >= 1")
        return 1

    # Adjust batches list based on thread count
    if len(args.batches) < args.threads:
        print(f"INFO: Number of batches ({len(args.batches)}) < threads ({args.threads})")
        print(f"      Some threads will test the same batch sizes")
    elif len(args.batches) > args.threads:
        print(f"INFO: Number of batches ({len(args.batches)}) > threads ({args.threads})")
        print(f"      Only first {args.threads} batch sizes will be tested")
        args.batches = args.batches[:args.threads]

    # Validate max_dynamic_batch
    if args.max_dynamic_batch > 0 and args.max_dynamic_batch < max(args.batches):
        print(f"WARNING: max_dynamic_batch ({args.max_dynamic_batch}) is less than max test batch ({max(args.batches)})")
        print(f"         Increasing max_dynamic_batch to {max(args.batches)}")
        args.max_dynamic_batch = max(args.batches)

    # Create cache dir
    os.makedirs(args.cache, exist_ok=True)

    print("="*80)
    print("MIGraphX Concurrent Inference Test")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Cache: {args.cache}")
    print(f"Max Dynamic Batch: {args.max_dynamic_batch} {'(disabled)' if args.max_dynamic_batch == 0 else ''}")
    print(f"Concurrent Threads: {args.threads}")
    print(f"Iterations per thread: {args.iterations}")
    print(f"Batch sizes: {args.batches}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Show I/O details: {'ON' if args.show_io else 'OFF (only mismatches)'}")
    print(f"Verbose logging: {'ON' if args.verbose else 'OFF'}")
    opt_level_names = {0: "DISABLE_ALL", 1: "ENABLE_BASIC", 2: "ENABLE_EXTENDED", 99: "ENABLE_ALL"}
    print(f"Graph optimization: {opt_level_names.get(args.optimization_level, 'DISABLE_ALL')}")
    print("="*80)

    # Create threads - one per batch size up to thread limit
    threads = []
    for i in range(args.threads):
        # Cycle through batch sizes if we have more threads than batches
        batch_size = args.batches[i % len(args.batches)]
        thread = threading.Thread(
            target=run_inference_thread,
            args=(i+1, args.model, args.cache, batch_size, args.iterations, args.max_dynamic_batch,
                  args.verbose, args.optimization_level, args.warmup, args.show_io)
        )
        threads.append(thread)

    # Start all threads
    print("\nStarting concurrent inference...")
    start_time = time.time()

    for thread in threads:
        thread.start()
        time.sleep(0.5)  # Stagger starts

    # Wait for completion
    for thread in threads:
        thread.join()

    total_time = time.time() - start_time

    print("\n" + "="*80)
    print(f"All threads completed in {total_time:.2f}s")
    print("="*80)

    # Calculate and display metrics
    print_metrics(total_time, args.thread_details)

    return 0


def print_metrics(total_time, thread_details=False):
    """Calculate and print comprehensive metrics"""
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)

    # Collect all latencies across threads
    all_latencies_combined = []
    total_inferences = 0
    total_samples_processed = 0
    total_batch_mismatches = 0
    total_errors = 0

    # First pass: collect totals
    for thread_id in sorted(thread_results.keys()):
        result = thread_results[thread_id]
        if 'error' in result:
            total_errors += 1
            continue
        total_inferences += result['iterations']
        total_samples_processed += result['iterations'] * result['batch_size']
        total_batch_mismatches += result.get('batch_mismatches', 0)
        all_latencies_combined.extend(all_latencies[thread_id])

    # Per-thread statistics (only if thread_details is enabled)
    if thread_details:
        print("\n--- Per-Thread Statistics ---")
        for thread_id in sorted(thread_results.keys()):
            result = thread_results[thread_id]

            if 'error' in result:
                print(f"\nThread-{thread_id}: ERROR - {result['error']}")
                continue

            batch_size = result['batch_size']
            iterations = result['iterations']
            mismatches = result.get('batch_mismatches', 0)
            latencies = all_latencies[thread_id]

            if not latencies:
                print(f"\nThread-{thread_id}: No latencies recorded")
                continue

            # Calculate percentiles
            p50 = np.percentile(latencies, 50)
            p90 = np.percentile(latencies, 90)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            avg = np.mean(latencies)
            min_lat = np.min(latencies)
            max_lat = np.max(latencies)
            std = np.std(latencies)

            print(f"\nThread-{thread_id} (Batch Size: {batch_size}):")
            print(f"  Iterations: {iterations}")
            print(f"  Samples Processed: {iterations * batch_size}")
            if mismatches > 0:
                print(f"  ❌ Batch Mismatches: {mismatches}")
            if 'num_outputs' in result and result['num_outputs'] > 0:
                print(f"  Model Outputs Verified: {result['num_outputs']}")
            print(f"  Latency (ms):")
            print(f"    Min:    {min_lat:8.2f}")
            print(f"    Mean:   {avg:8.2f}")
            print(f"    Median: {p50:8.2f}")
            print(f"    P90:    {p90:8.2f}")
            print(f"    P95:    {p95:8.2f}")
            print(f"    P99:    {p99:8.2f}")
            print(f"    Max:    {max_lat:8.2f}")
            print(f"    StdDev: {std:8.2f}")

            # Single-thread throughput
            single_thread_throughput = (iterations * batch_size) / (sum(latencies) / 1000)
            print(f"  Single-Thread Throughput: {single_thread_throughput:.2f} inferences/sec")

    # Overall statistics
    if all_latencies_combined:
        print("\n--- Overall Statistics (All Threads Combined) ---")
        p50_all = np.percentile(all_latencies_combined, 50)
        p90_all = np.percentile(all_latencies_combined, 90)
        p95_all = np.percentile(all_latencies_combined, 95)
        p99_all = np.percentile(all_latencies_combined, 99)
        avg_all = np.mean(all_latencies_combined)
        min_all = np.min(all_latencies_combined)
        max_all = np.max(all_latencies_combined)
        std_all = np.std(all_latencies_combined)

        print(f"\nCombined Latency Across All Threads (ms):")
        print(f"  Min:    {min_all:8.2f}")
        print(f"  Mean:   {avg_all:8.2f}")
        print(f"  Median: {p50_all:8.2f}")
        print(f"  P90:    {p90_all:8.2f}")
        print(f"  P95:    {p95_all:8.2f}")
        print(f"  P99:    {p99_all:8.2f}")
        print(f"  Max:    {max_all:8.2f}")
        print(f"  StdDev: {std_all:8.2f}")

        print(f"\nTotal Inferences: {total_inferences}")
        print(f"Total Samples Processed: {total_samples_processed}")
        print(f"Total Wall-Clock Time: {total_time:.2f}s")

        # Concurrent throughput (total samples / wall clock time)
        concurrent_throughput = total_samples_processed / total_time
        print(f"\n🚀 Concurrent Throughput: {concurrent_throughput:.2f} inferences/sec")

        # Aggregate throughput (sum of all inferences / sum of all latencies)
        total_latency_seconds = sum(all_latencies_combined) / 1000
        aggregate_throughput = total_inferences / total_latency_seconds if total_latency_seconds > 0 else 0
        print(f"📊 Aggregate Throughput: {aggregate_throughput:.2f} inferences/sec")

        # Speedup from concurrency
        num_threads = len(thread_results)
        if num_threads > 1:
            speedup = concurrent_throughput / (aggregate_throughput / num_threads) if aggregate_throughput > 0 else 0
            efficiency = (speedup / num_threads) * 100 if num_threads > 0 else 0
            print(f"\n⚡ Concurrency Speedup: {speedup:.2f}x")
            print(f"📈 Parallel Efficiency: {efficiency:.1f}%")

        # Validation summary
        print(f"\n--- Validation Summary ---")
        print(f"Total Threads: {len(thread_results)}")
        if total_errors > 0:
            print(f"❌ Thread Errors: {total_errors}")
        if total_batch_mismatches > 0:
            print(f"❌ Batch Mismatches: {total_batch_mismatches}/{total_inferences} ({100*total_batch_mismatches/total_inferences:.1f}%)")
        else:
            print(f"✅ All {total_inferences} inferences passed batch validation")

    print("\n" + "="*80)


if __name__ == "__main__":
    sys.exit(main())
