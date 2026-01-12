#!/usr/bin/env python3
"""
Concurrent MIGraphX Execution Provider Test Script

This script tests the MIGraphX execution provider with concurrent inference
requests using different batch sizes to verify proper batch handling and caching.
"""

import os
import sys
import time
import threading
import argparse
import numpy as np
import onnxruntime as ort
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm


class ConcurrentInferenceTest:
    """Test harness for concurrent MIGraphX inference"""

    def __init__(self, model_path, cache_dir, max_dynamic_batch=0, verbose=False, optimization_level=0, warmup=3, show_io=False, thread_details=False, fp16_enable=False, bf16_enable=False, fp16_data=False, bf16_data=False):
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.max_dynamic_batch = max_dynamic_batch
        self.verbose = verbose
        self.optimization_level = optimization_level
        self.warmup = warmup
        self.show_io = show_io
        self.thread_details = thread_details
        self.fp16_enable = fp16_enable
        self.bf16_enable = bf16_enable
        self.fp16_data = fp16_data
        self.bf16_data = bf16_data
        self.sessions = {}
        self.results = defaultdict(list)
        self.lock = threading.Lock()

    def create_session(self, thread_id):
        """Create an ONNX Runtime session with MIGraphX EP"""
        if self.show_io:
            print(f"[Thread-{thread_id}] Creating session with MIGraphX EP...")
            print(f"[Thread-{thread_id}] Max dynamic batch: {self.max_dynamic_batch}")

        # Set environment variables for MIGraphX
        os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH'] = self.cache_dir
        os.environ['ORT_MIGRAPHX_MAX_DYNAMIC_BATCH'] = str(self.max_dynamic_batch)

        # Configure session options
        session_options = ort.SessionOptions()
        
        # Set logging verbosity
        if self.verbose:
            session_options.log_severity_level = 0  # Verbose (0 = Verbose, 1 = Info, 2 = Warning, 3 = Error, 4 = Fatal)
            session_options.log_verbosity_level = 0  # Detailed logs
            if self.show_io:
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
        session_options.graph_optimization_level = opt_level_map.get(self.optimization_level, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
        if self.show_io:
            opt_level_names = {
                0: "ORT_DISABLE_ALL",
                1: "ORT_ENABLE_BASIC",
                2: "ORT_ENABLE_EXTENDED",
                99: "ORT_ENABLE_ALL"
            }
            print(f"[Thread-{thread_id}] Graph optimization: {opt_level_names.get(self.optimization_level, 'ORT_DISABLE_ALL')}")

        # Configure MIGraphX execution provider
        migraphx_provider_options = {
            'device_id': 0,
            'migraphx_fp16_enable': 1 if self.fp16_enable else 0,
            'migraphx_bf16_enable': 1 if self.bf16_enable else 0,
            'migraphx_fp8_enable': 0,
            'migraphx_int8_enable': 0,
            'migraphx_exhaustive_tune': 0,
            #'migraphx_max_dynamic_batch': self.max_dynamic_batch,
        }

        if self.show_io:
            print(f"[Thread-{thread_id}] MIGraphX Provider Options:")
            print(f"[Thread-{thread_id}]   FP16 Enable: {'ON' if self.fp16_enable else 'OFF'}")
            print(f"[Thread-{thread_id}]   BF16 Enable: {'ON' if self.bf16_enable else 'OFF'}")
            print(f"[Thread-{thread_id}]   FP16 Data: {'ON' if self.fp16_data else 'OFF'}")
            print(f"[Thread-{thread_id}]   BF16 Data: {'ON' if self.bf16_data else 'OFF'}")

        # Create session with MIGraphX EP
        providers = [
            ('MIGraphXExecutionProvider', migraphx_provider_options),
            'CPUExecutionProvider'
        ]

        try:
            session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )

            if self.show_io:
                print(f"[Thread-{thread_id}] Session created successfully")
                print(f"[Thread-{thread_id}] Active providers: {session.get_providers()}")

            # Print input/output info (only if show_io enabled)
            if self.show_io:
                print(f"[Thread-{thread_id}] Model inputs:")
                for inp in session.get_inputs():
                    print(f"  - {inp.name}: {inp.shape} ({inp.type})")

                print(f"[Thread-{thread_id}] Model outputs:")
                for out in session.get_outputs():
                    print(f"  - {out.name}: {out.shape} ({out.type})")

            return session

        except Exception as e:
            print(f"[Thread-{thread_id}] ERROR creating session: {e}")
            raise

    def generate_input_data(self, session, batch_size):
        """Generate random input data for ALL model inputs"""
        inputs = {}

        for inp in session.get_inputs():
            shape = list(inp.shape)

            # Replace dynamic batch dimension (first dimension) with actual batch size
            if len(shape) > 0 and (isinstance(shape[0], str) or shape[0] < 0):
                shape[0] = batch_size

            # Replace any other dynamic dimensions with reasonable defaults
            for i in range(len(shape)):
                if isinstance(shape[i], str) or shape[i] < 0:
                    shape[i] = 1  # Default to 1 for other dynamic dims

            # Generate random data based on input type
            if inp.type == 'tensor(float)' or inp.type == 'tensor(float32)':
                data = np.random.randn(*shape).astype(np.float32)
                # Convert to fp16 or bf16 if requested
                if self.fp16_data:
                    data = data.astype(np.float16)
                elif self.bf16_data:
                    # bfloat16 support - convert via float32
                    # Note: numpy doesn't have native bfloat16, but ONNX Runtime can handle it
                    # We'll use float32 and let the EP handle conversion
                    # If ML dtypes is available, we could use ml_dtypes.bfloat16
                    try:
                        import ml_dtypes
                        data = data.astype(ml_dtypes.bfloat16)
                    except ImportError:
                        print(f"  WARNING: ml_dtypes not available for bfloat16, using float32")
                        data = data.astype(np.float32)
            elif inp.type == 'tensor(float16)':
                # Model expects float16 inputs - generate float16 data directly
                data = np.random.randn(*shape).astype(np.float16)
            elif inp.type == 'tensor(bfloat16)':
                # Model expects bfloat16 inputs - generate bfloat16 data
                try:
                    import ml_dtypes
                    data = np.random.randn(*shape).astype(np.float32).astype(ml_dtypes.bfloat16)
                except ImportError:
                    print(f"  WARNING: ml_dtypes not available for bfloat16, using float16")
                    data = np.random.randn(*shape).astype(np.float16)
            elif inp.type == 'tensor(double)' or inp.type == 'tensor(float64)':
                data = np.random.randn(*shape).astype(np.float64)
            elif inp.type == 'tensor(int64)':
                data = np.random.randint(0, 100, size=shape).astype(np.int64)
            elif inp.type == 'tensor(int32)':
                data = np.random.randint(0, 100, size=shape).astype(np.int32)
            elif inp.type == 'tensor(int16)':
                data = np.random.randint(0, 100, size=shape).astype(np.int16)
            elif inp.type == 'tensor(int8)':
                data = np.random.randint(0, 100, size=shape).astype(np.int8)
            elif inp.type == 'tensor(uint64)':
                data = np.random.randint(0, 100, size=shape).astype(np.uint64)
            elif inp.type == 'tensor(uint32)':
                data = np.random.randint(0, 100, size=shape).astype(np.uint32)
            elif inp.type == 'tensor(uint16)':
                data = np.random.randint(0, 100, size=shape).astype(np.uint16)
            elif inp.type == 'tensor(uint8)':
                data = np.random.randint(0, 100, size=shape).astype(np.uint8)
            elif inp.type == 'tensor(bool)':
                data = np.random.randint(0, 2, size=shape).astype(np.bool_)
            else:
                # Default to float32 for unknown types
                print(f"  WARNING: Unknown input type '{inp.type}' for '{inp.name}', using float32")
                data = np.random.randn(*shape).astype(np.float32)

            inputs[inp.name] = data

        return inputs

    def run_inference(self, thread_id, batch_size, iterations):
        """Run inference with specified batch size"""
        try:
            # Create session (each thread gets its own session)
            session = self.create_session(thread_id)

            if self.show_io:
                print(f"\n[Thread-{thread_id}] Starting {iterations} iterations with batch_size={batch_size}")
                print(f"[Thread-{thread_id}] " + "="*60)

            # Get output info for IO binding
            output_infos = session.get_outputs()

            # WARMUP: Run warmup iterations to exclude compilation time from measurements
            if self.warmup > 0:
                if self.show_io:
                    print(f"[Thread-{thread_id}] Running {self.warmup} warmup iterations (not measured)...")
                for w in range(self.warmup):
                    warmup_inputs = self.generate_input_data(session, batch_size)
                    warmup_start = time.perf_counter()
                    _ = session.run(None, warmup_inputs)
                    warmup_time = (time.perf_counter() - warmup_start) * 1000
                    if self.show_io:
                        print(f"[Thread-{thread_id}] Warmup {w+1}/{self.warmup}: {warmup_time:.2f}ms")
                if self.show_io:
                    print(f"[Thread-{thread_id}] Warmup complete, starting measured iterations...")

            # Run iterations with progress bar (when show_io is off)
            if self.show_io:
                iter_range = range(iterations)
            else:
                iter_range = tqdm(range(iterations), 
                                 desc=f"Thread-{thread_id} batch={batch_size}", 
                                 position=thread_id-1, 
                                 leave=True,
                                 ncols=80)
            
            for i in iter_range:
                iter_start = time.perf_counter()

                # Generate input data
                inputs = self.generate_input_data(session, batch_size)

                # Log input shapes (only if show_io enabled)
                if self.show_io:
                    input_shapes = {name: data.shape for name, data in inputs.items()}
                    print(f"\n[Thread-{thread_id}] Iteration {i+1}/{iterations}")
                    print(f"[Thread-{thread_id}] Input shapes: {input_shapes}")
                    print(f"[Thread-{thread_id}] Batch size requested: {batch_size}")

                # Use IO Binding for accurate GPU-only timing
                io_binding = session.io_binding()
                
                # Bind inputs to device (GPU)
                for name, data in inputs.items():
                    io_binding.bind_cpu_input(name, data)
                
                # Bind outputs to device (GPU)
                for output_info in output_infos:
                    io_binding.bind_output(output_info.name, 'cuda')

                # Run inference with IO binding - time only GPU execution
                inference_start = time.perf_counter()
                session.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()  # Wait for GPU to complete
                inference_time = (time.perf_counter() - inference_start) * 1000  # ms
                
                # Get outputs from GPU (after timing)
                outputs = io_binding.copy_outputs_to_cpu()

                output_shapes = {}
                
                # Verify batch size in ALL outputs
                batch_match = True
                batch_mismatch_details = []
                
                for output_idx, (output_info, output_data) in enumerate(zip(output_infos, outputs)):
                    output_name = output_info.name
                    output_shape = output_data.shape
                    output_type = output_info.type
                    output_shapes[output_name] = output_shape
                    
                    # Check if output has batch dimension
                    if len(output_shape) > 0:
                        output_batch = output_shape[0]
                        
                        if output_batch != batch_size:
                            batch_match = False
                            batch_mismatch_details.append(
                                f"  ❌ Output {output_idx} '{output_name}': "
                                f"shape={output_shape}, type={output_type}, "
                                f"batch={output_batch} (expected {batch_size})"
                            )

                # Clear IO binding for next iteration
                io_binding.clear_binding_inputs()
                io_binding.clear_binding_outputs()

                # Update progress bar with latest latency (when not show_io)
                match_symbol = "✅" if batch_match else "❌"
                if not self.show_io and hasattr(iter_range, 'set_postfix'):
                    iter_range.set_postfix({'ms': f'{inference_time:.1f}', 'ok': match_symbol})

                # Log output shapes with detailed info (only if show_io enabled)
                if self.show_io:
                    print(f"[Thread-{thread_id}] Verifying {len(outputs)} outputs:")
                    for output_idx, (output_info, output_data) in enumerate(zip(output_infos, outputs)):
                        output_name = output_info.name
                        output_shape = output_data.shape
                        if len(output_shape) > 0:
                            output_batch = output_shape[0]
                            if output_batch != batch_size:
                                print(f"[Thread-{thread_id}]   ❌ {output_name}: {output_shape} "
                                      f"(batch={output_batch}, expected={batch_size})")
                            else:
                                print(f"[Thread-{thread_id}]   ✅ {output_name}: {output_shape} "
                                      f"(batch={output_batch})")
                        else:
                            print(f"[Thread-{thread_id}]   ⚠️  {output_name}: {output_shape} "
                                  f"(scalar, no batch dim)")

                    # Summary verification
                    if batch_match:
                        print(f"[Thread-{thread_id}] ✅ All {len(outputs)} outputs have correct batch size: {batch_size}")
                    else:
                        print(f"[Thread-{thread_id}] ❌ BATCH SIZE MISMATCH DETECTED!")
                        for detail in batch_mismatch_details:
                            print(f"[Thread-{thread_id}] {detail}")

                iter_time = (time.perf_counter() - iter_start) * 1000  # ms
                if self.show_io:
                    print(f"[Thread-{thread_id}] Inference time: {inference_time:.2f}ms, "
                          f"Total time: {iter_time:.2f}ms")

                # Store results
                with self.lock:
                    self.results[thread_id].append({
                        'iteration': i + 1,
                        'batch_size': batch_size,
                        'inference_time_ms': inference_time,
                        'total_time_ms': iter_time,
                        'batch_match': batch_match,
                        'output_shapes': output_shapes
                    })

                # Small delay between iterations
                if i < iterations - 1:
                    time.sleep(0.1)

            if self.show_io:
                print(f"\n[Thread-{thread_id}] Completed all {iterations} iterations")
                print(f"[Thread-{thread_id}] " + "="*60)

        except Exception as e:
            print(f"\n[Thread-{thread_id}] ERROR during inference: {e}")
            import traceback
            traceback.print_exc()

    def print_summary(self):
        """Print summary of all results"""
        print("\n" + "="*80)
        print("PERFORMANCE METRICS AND SUMMARY")
        print("="*80)

        all_latencies = []
        total_inferences = 0
        total_samples = 0
        total_batch_mismatches = 0

        # First pass: collect totals
        for thread_id, results in sorted(self.results.items()):
            if not results:
                continue
            batch_size = results[0]['batch_size']
            inference_times = [r['inference_time_ms'] for r in results]
            all_latencies.extend(inference_times)
            iterations = len(results)
            total_inferences += iterations
            total_samples += iterations * batch_size
            mismatches = sum(1 for r in results if not r['batch_match'])
            total_batch_mismatches += mismatches

        # Per-thread statistics (only if thread_details is enabled)
        if self.thread_details:
            print("\n--- Per-Thread Statistics ---")
            for thread_id, results in sorted(self.results.items()):
                if not results:
                    print(f"\nThread-{thread_id}: No results recorded")
                    continue

                batch_size = results[0]['batch_size']
                inference_times = [r['inference_time_ms'] for r in results]

                iterations = len(results)
                mismatches = sum(1 for r in results if not r['batch_match'])

                # Calculate percentiles
                p50 = np.percentile(inference_times, 50)
                p90 = np.percentile(inference_times, 90)
                p95 = np.percentile(inference_times, 95)
                p99 = np.percentile(inference_times, 99)
                avg = np.mean(inference_times)
                min_lat = np.min(inference_times)
                max_lat = np.max(inference_times)
                std = np.std(inference_times)

                print(f"\nThread-{thread_id} (Batch Size: {batch_size}):")
                print(f"  Iterations: {iterations}")
                print(f"  Samples Processed: {iterations * batch_size}")
                if mismatches > 0:
                    print(f"  ❌ Batch Mismatches: {mismatches}")
                print(f"  Inference Latency (ms):")
                print(f"    Min:    {min_lat:8.2f}")
                print(f"    Mean:   {avg:8.2f}")
                print(f"    Median: {p50:8.2f}")
                print(f"    P90:    {p90:8.2f}")
                print(f"    P95:    {p95:8.2f}")
                print(f"    P99:    {p99:8.2f}")
                print(f"    Max:    {max_lat:8.2f}")
                print(f"    StdDev: {std:8.2f}")

                # Single-thread throughput
                total_time_sec = sum(inference_times) / 1000
                single_thread_throughput = (iterations * batch_size) / total_time_sec if total_time_sec > 0 else 0
                print(f"  Single-Thread Throughput: {single_thread_throughput:.2f} inferences/sec")

        # Overall combined statistics
        if all_latencies:
            print("\n--- Overall Statistics (All Threads Combined) ---")
            p50_all = np.percentile(all_latencies, 50)
            p90_all = np.percentile(all_latencies, 90)
            p95_all = np.percentile(all_latencies, 95)
            p99_all = np.percentile(all_latencies, 99)
            avg_all = np.mean(all_latencies)
            min_all = np.min(all_latencies)
            max_all = np.max(all_latencies)
            std_all = np.std(all_latencies)

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
            print(f"Total Samples Processed: {total_samples}")

            # Validation summary
            print(f"\n--- Validation Summary ---")
            print(f"Total Threads: {len(self.results)}")
            if total_batch_mismatches > 0:
                print(f"❌ Batch Mismatches: {total_batch_mismatches}/{total_inferences} ({100*total_batch_mismatches/total_inferences:.1f}%)")
            else:
                print(f"✅ All {total_inferences} inferences passed batch validation")

        print("\n" + "="*80)


def main():
    """Main test execution"""
    parser = argparse.ArgumentParser(description='Concurrent MIGraphX Execution Provider Test')
    parser.add_argument('--model', type=str, default="/models/Genrec_model.onnx/model.onnx",
                       help='Path to ONNX model (default: /models/Genrec_model.onnx/model.onnx)')
    parser.add_argument('--cache', type=str, default="/models/",
                       help='Cache directory (default: /models/)')
    parser.add_argument('--max-dynamic-batch', type=int, default=0,
                       help='Max dynamic batch size for pre-compilation (0=disabled)')
    parser.add_argument('--threads', type=int, default=2,
                       help='Number of concurrent threads (default: 2)')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Iterations per thread (default: 5)')
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 2, 4, 8],
                       help='Batch sizes to test (default: 1 2 4 8)')
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
    parser.add_argument('--random-batches', action='store_true', default=False,
                       help='Use random batch sizes per iteration (default: cycle through --batches)')
    parser.add_argument('--fp16-enable', action='store_true', default=False,
                       help='Enable FP16 precision in MIGraphX execution provider (default: OFF)')
    parser.add_argument('--bf16-enable', action='store_true', default=False,
                       help='Enable BF16 precision in MIGraphX execution provider (default: OFF)')
    parser.add_argument('--fp16-data', action='store_true', default=False,
                       help='Use FP16 input data instead of FP32 (default: OFF)')
    parser.add_argument('--bf16-data', action='store_true', default=False,
                       help='Use BF16 input data instead of FP32 (default: OFF)')

    args = parser.parse_args()
    
    print("="*80)
    print("MIGraphX Concurrent Inference Test")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Cache dir: {args.cache}")
    print(f"Max dynamic batch: {args.max_dynamic_batch} {'(disabled)' if args.max_dynamic_batch == 0 else ''}")
    print(f"Concurrent threads: {args.threads}")
    print(f"Iterations per thread: {args.iterations}")
    print(f"Batch sizes: {args.batches}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Show I/O details: {'ON' if args.show_io else 'OFF (only mismatches)'}")
    print(f"Random batches: {'YES' if args.random_batches else 'NO (cycle through list)'}")
    print(f"Verbose logging: {'ON' if args.verbose else 'OFF'}")
    opt_level_names = {0: "DISABLE_ALL", 1: "ENABLE_BASIC", 2: "ENABLE_EXTENDED", 99: "ENABLE_ALL"}
    print(f"Graph optimization: {opt_level_names.get(args.optimization_level, 'DISABLE_ALL')}")
    print(f"FP16 enable: {'ON' if args.fp16_enable else 'OFF'}")
    print(f"BF16 enable: {'ON' if args.bf16_enable else 'OFF'}")
    print(f"FP16 data: {'ON' if args.fp16_data else 'OFF'}")
    print(f"BF16 data: {'ON' if args.bf16_data else 'OFF'}")
    print("="*80)

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        print("Please update --model argument")
        return 1

    # Create test harness
    test = ConcurrentInferenceTest(args.model, args.cache, args.max_dynamic_batch, 
                                   args.verbose, args.optimization_level, args.warmup, args.show_io,
                                   args.thread_details, args.fp16_enable, args.bf16_enable,
                                   args.fp16_data, args.bf16_data)

    # Define test scenarios based on args.threads
    test_scenarios = []
    for i in range(args.threads):
        thread_id = i + 1
        # Cycle through batch sizes
        batch_size = args.batches[i % len(args.batches)]
        test_scenarios.append((thread_id, batch_size))

    print(f"\nTest scenarios ({args.threads} threads):")
    for tid, bs in test_scenarios:
        print(f"  Thread-{tid}: batch_size={bs}, iterations={args.iterations}")

    print("\nStarting concurrent inference...\n")
    start_time = time.time()

    # Create and start threads
    threads = []
    for thread_id, batch_size in test_scenarios:
        thread = threading.Thread(
            target=test.run_inference,
            args=(thread_id, batch_size, args.iterations),
            name=f"InferenceThread-{thread_id}"
        )
        threads.append(thread)
        thread.start()

        # Small stagger to avoid simultaneous session creation
        time.sleep(0.5)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    total_time = time.time() - start_time

    # Print summary
    test.print_summary()

    # Print throughput metrics
    print_throughput_metrics(test, total_time)

    print(f"\nTotal Wall-Clock Time: {total_time:.2f}s")
    print("\nTest completed!")

    return 0


def print_throughput_metrics(test, wall_clock_time):
    """Calculate and print throughput metrics"""
    print("\n" + "="*80)
    print("THROUGHPUT ANALYSIS")
    print("="*80)

    total_inferences = 0
    total_samples = 0
    all_latencies = []

    for thread_id, results in test.results.items():
        if not results:
            continue
        batch_size = results[0]['batch_size']
        iterations = len(results)
        total_inferences += iterations
        total_samples += iterations * batch_size
        all_latencies.extend([r['inference_time_ms'] for r in results])

    if total_inferences == 0:
        print("No inferences completed")
        return

    # Concurrent throughput (samples / wall clock time)
    concurrent_throughput = total_samples / wall_clock_time
    print(f"\n🚀 Concurrent Throughput: {concurrent_throughput:.2f} inferences/sec")
    print(f"   (Based on {total_samples} samples in {wall_clock_time:.2f}s wall-clock time)")

    # Aggregate throughput (total inferences / sum of all latencies)
    total_latency_seconds = sum(all_latencies) / 1000
    aggregate_throughput = total_inferences / total_latency_seconds if total_latency_seconds > 0 else 0
    print(f"\n📊 Aggregate Throughput: {aggregate_throughput:.2f} inferences/sec")
    print(f"   (Based on {total_inferences} inferences with {total_latency_seconds:.2f}s total latency)")

    # Speedup from concurrency
    num_threads = len(test.results)
    if num_threads > 1:
        speedup = concurrent_throughput / (aggregate_throughput / num_threads) if aggregate_throughput > 0 else 0
        efficiency = (speedup / num_threads) * 100 if num_threads > 0 else 0
        print(f"\n⚡ Concurrency Speedup: {speedup:.2f}x")
        print(f"   (Speedup from using {num_threads} threads)")
        print(f"\n📈 Parallel Efficiency: {efficiency:.1f}%")
        print(f"   (Ideal: 100%, Actual: {efficiency:.1f}%)")

    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
