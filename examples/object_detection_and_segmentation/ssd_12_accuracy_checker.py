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
#####################################################################################
"""
Download SSD-12 and its test data from the ONNX Model Zoo, then compare
MIGraphX results with ONNX Runtime and the packaged reference outputs.
"""

import argparse
import sys

import onnx

from model_zoo import (ModelZooFixture, default_cache_dir, get_fixture)
from ssd_accuracy_common import (compare_outputs, load_tensor_set,
                                 model_io_names, print_performance,
                                 run_migraphx, run_ort, validate_ssd_contract,
                                 validate_ssd_outputs)

SSD_FIXTURE = ModelZooFixture(
    directory="ssd",
    archive="ssd-12.tar.gz",
    model="ssd-12.onnx",
    sha256="43e57501da34abe0c01ea30902303bdb14f590f59c18fbb15ea3473810b0e86b",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MIGraphX SSD-12 outputs with ONNX Runtime.")
    parser.add_argument(
        "--cache-dir",
        default=default_cache_dir(),
        help="Directory used for downloaded Model Zoo data",
    )
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="ONNX Runtime execution provider",
    )
    parser.add_argument("--target",
                        default="gpu",
                        help="MIGraphX compilation target")
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for floating-point outputs",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for floating-point outputs",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of untimed inference warmup runs",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of timed inference runs",
    )
    parser.add_argument(
        "--ort-run",
        action="store_true",
        help="Only run ONNX Runtime and compare it with the packaged outputs",
    )
    parser.add_argument(
        "--ort-logging",
        action="store_true",
        help="Enable verbose ONNX Runtime logging",
    )
    parser.add_argument(
        "--ort-disable-affinity",
        dest="ort_intra_op_num_threads",
        type=int,
        default=None,
        help="Set ONNX Runtime intra-op threads and disable automatic affinity",
    )
    parser.add_argument(
        "--disable-offload-copy",
        dest="offload_copy",
        action="store_false",
        default=True,
        help="Disable MIGraphX offload copies",
    )
    parser.add_argument(
        "--disable-fast-math",
        dest="fast_math",
        action="store_false",
        default=True,
        help="Disable MIGraphX fast-math transformations",
    )
    parser.add_argument(
        "--exhaustive-tune",
        action="store_true",
        help="Enable exhaustive MIGraphX tuning",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed mismatch information",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    return args


def main():
    args = parse_args()
    model_path, data_dir = get_fixture(SSD_FIXTURE, args.cache_dir)

    model_proto = onnx.load(str(model_path), load_external_data=False)
    input_names, output_names = model_io_names(model_proto)
    inputs = load_tensor_set(data_dir, "input", input_names)
    golden_outputs = load_tensor_set(data_dir, "output", output_names)
    validate_ssd_contract(input_names, output_names, inputs, golden_outputs)

    print("Running ONNX Runtime reference...")
    ort_outputs, ort_durations = run_ort(model_path, inputs, output_names, args)
    ort_box_count = validate_ssd_outputs(ort_outputs)
    print_performance("ONNX Runtime", ort_durations)
    print("Comparing ONNX Runtime with packaged outputs:")
    golden_passed = compare_outputs(
        golden_outputs,
        ort_outputs,
        output_names,
        args.rtol,
        args.atol,
        args.verbose,
    )
    if args.ort_run:
        return 0 if golden_passed else 1

    print(f"Compiling and running MIGraphX target {args.target!r}...")
    migraphx_outputs, migraphx_durations = run_migraphx(
        model_path, inputs, output_names, args)
    migraphx_box_count = validate_ssd_outputs(migraphx_outputs)
    print_performance(f"MIGraphX {args.target}", migraphx_durations)
    box_counts_match = ort_box_count == migraphx_box_count
    box_count_status = "PASSED" if box_counts_match else "FAILED"
    print(f"{box_count_status}: output box count: "
          f"ORT={ort_box_count}, MIGraphX={migraphx_box_count}")
    print("Comparing MIGraphX with ONNX Runtime:")
    migraphx_passed = compare_outputs(
        ort_outputs,
        migraphx_outputs,
        output_names,
        args.rtol,
        args.atol,
        args.verbose,
    )

    if golden_passed and box_counts_match and migraphx_passed:
        print("PASSED: SSD-12 MIGraphX outputs match ONNX Runtime")
        return 0
    print("FAILED: SSD-12 accuracy check")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
