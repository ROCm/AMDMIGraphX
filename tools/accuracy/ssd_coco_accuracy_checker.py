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
"""Compare MIGraphX and ONNX Runtime on ten prepared COCO SSD-10 inputs."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx

import migraphx

from ssd_accuracy_common import benchmark
from ssd_accuracy_common import compare_outputs
from ssd_accuracy_common import load_tensor_set
from ssd_accuracy_common import make_ort_session
from ssd_accuracy_common import model_io_names
from ssd_accuracy_common import print_performance
from ssd_accuracy_common import SSD_INPUT_NAME
from ssd_accuracy_common import SSD_OUTPUT_NAMES
from ssd_accuracy_common import validate_ssd_input
from ssd_accuracy_common import validate_ssd_outputs

IMAGE_COUNT = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Compare MIGraphX SSD-10 outputs with ONNX Runtime for "
                     "ten prepared COCO images."))
    parser.add_argument(
        "test_dir",
        type=Path,
        help="Extracted SSD-10 directory containing model.onnx",
    )
    parser.add_argument(
        "fixtures_dir",
        type=Path,
        help=("Prepared directory containing manifest.json and "
              "test_data_set_0 through test_data_set_9"),
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
        default=0,
        help="Number of untimed inference warmup runs per image",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of timed inference runs per image",
    )
    parser.add_argument(
        "--ort-run",
        action="store_true",
        help="Only run ONNX Runtime and validate its output contracts",
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


def load_cases(fixtures_dir, input_names):
    manifest_path = fixtures_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing fixture manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    cases = manifest.get("cases")
    if manifest.get("image_count") != IMAGE_COUNT or not isinstance(
            cases, list) or len(cases) != IMAGE_COUNT:
        raise ValueError(f"Expected manifest with exactly {IMAGE_COUNT} cases")
    if manifest.get("input_name") != SSD_INPUT_NAME:
        raise ValueError(f"Expected manifest input {SSD_INPUT_NAME!r}, "
                         f"found {manifest.get('input_name')!r}")

    result = []
    for index, case in enumerate(cases):
        expected_name = f"test_data_set_{index}"
        if case.get("case") != expected_name:
            raise ValueError(
                f"Expected manifest case {expected_name!r}, found "
                f"{case.get('case')!r}")
        case_dir = fixtures_dir / expected_name
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Missing prepared case: {case_dir}")
        inputs = load_tensor_set(case_dir, "input", input_names)
        validate_ssd_input(inputs)
        manifest_shape = tuple(case.get("tensor_shape", ()))
        if manifest_shape != inputs[SSD_INPUT_NAME].shape:
            raise ValueError(
                f"Manifest shape {manifest_shape} does not match "
                f"{expected_name} input shape {inputs[SSD_INPUT_NAME].shape}")
        result.append((case, inputs))
    return result


class OrtRunner:
    def __init__(self, model_path, output_names, args):
        self.output_names = output_names
        self.session = make_ort_session(model_path, args)
        self.input_names = [value.name for value in self.session.get_inputs()]
        self.args = args

    def run(self, inputs):
        missing = set(self.input_names) - set(inputs)
        if missing:
            raise ValueError(
                f"No prepared data for ONNX Runtime inputs: {sorted(missing)}")
        ort_inputs = {name: inputs[name] for name in self.input_names}

        def run_once():
            values = self.session.run(self.output_names, ort_inputs)
            return dict(zip(self.output_names, values))

        return benchmark(run_once, self.args.warmup, self.args.iterations)


class MigraphxRunner:
    def __init__(self, model_path, output_names, inputs, args):
        self.output_names = output_names
        self.args = args
        self.model = migraphx.parse_onnx(str(model_path))
        parameter_shapes = self.model.get_parameter_shapes()
        if set(parameter_shapes) != set(inputs):
            raise ValueError(
                "MIGraphX parameters do not match prepared inputs: "
                f"parameters={sorted(parameter_shapes)}, inputs={sorted(inputs)}"
            )
        for name, shape in parameter_shapes.items():
            if tuple(shape.lens()) != inputs[name].shape:
                raise ValueError(
                    f"MIGraphX parameter {name!r} expects {tuple(shape.lens())}, "
                    f"but fixture has {inputs[name].shape}")
        self.model.compile(
            migraphx.get_target(args.target),
            offload_copy=args.offload_copy,
            fast_math=args.fast_math,
            exhaustive_tune=args.exhaustive_tune,
        )

    def run(self, inputs):
        parameters = {}
        for name, data in inputs.items():
            argument = migraphx.argument(data)
            if not self.args.offload_copy:
                argument = migraphx.to_gpu(argument)
            parameters[name] = argument

        def run_once():
            raw_outputs = self.model.run(parameters)
            if len(raw_outputs) != len(self.output_names):
                raise ValueError(
                    f"MIGraphX returned {len(raw_outputs)} outputs, "
                    f"expected {len(self.output_names)}")
            outputs = {}
            for name, output in zip(self.output_names, raw_outputs):
                if not self.args.offload_copy:
                    output = migraphx.from_gpu(output)
                outputs[name] = np.array(output)
            return outputs

        return benchmark(run_once, self.args.warmup, self.args.iterations)


def align_near_tied_outputs(reference, actual, rtol, atol):
    """Align equivalent detections only within contiguous reference score ties."""
    reference_boxes = reference["bboxes"][0]
    reference_scores = reference["scores"][0]
    actual_boxes = actual["bboxes"][0]
    actual_scores = actual["scores"][0]
    count = reference_scores.size
    permutation = list(range(count))
    reordered = 0

    group_starts = [0]
    for index in range(1, count):
        if not np.isclose(reference_scores[index - 1],
                          reference_scores[index],
                          rtol=rtol,
                          atol=atol):
            group_starts.append(index)
    group_starts.append(count)

    for start, stop in zip(group_starts, group_starts[1:]):
        if stop - start == 1:
            continue
        group = range(start, stop)
        candidates = {}
        for reference_index in group:
            preferred_order = sorted(group,
                                     key=lambda actual_index:
                                     (actual_index != reference_index,
                                      abs(actual_index - reference_index)))
            candidates[reference_index] = [
                actual_index for actual_index in preferred_order
                if np.isclose(reference_scores[reference_index],
                              actual_scores[actual_index],
                              rtol=rtol,
                              atol=atol)
                and np.allclose(reference_boxes[reference_index],
                                actual_boxes[actual_index],
                                rtol=rtol,
                                atol=atol)
            ]

        actual_to_reference = {}
        reference_to_actual = {}

        def match(reference_index, visited):
            for actual_index in candidates[reference_index]:
                if actual_index in visited:
                    continue
                visited.add(actual_index)
                previous = actual_to_reference.get(actual_index)
                if previous is None or match(previous, visited):
                    actual_to_reference[actual_index] = reference_index
                    reference_to_actual[reference_index] = actual_index
                    return True
            return False

        if not all(match(reference_index, set()) for reference_index in group):
            continue
        for reference_index in group:
            actual_index = reference_to_actual[reference_index]
            permutation[reference_index] = actual_index
            reordered += reference_index != actual_index

    if reordered == 0:
        return actual, reordered
    return {
        "bboxes": actual["bboxes"][:, permutation, :],
        "labels": actual["labels"][:, permutation],
        "scores": actual["scores"][:, permutation],
    }, reordered


def print_case_header(index, case):
    print(f"\n=== [{index + 1}/{IMAGE_COUNT}] {case['image_id']} ===")


def main():
    args = parse_args()
    model_path = args.test_dir / "model.onnx"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing SSD model: {model_path}")
    if not args.fixtures_dir.is_dir():
        raise FileNotFoundError(
            f"Missing prepared fixture directory: {args.fixtures_dir}")

    model_proto = onnx.load(str(model_path), load_external_data=False)
    input_names, output_names = model_io_names(model_proto)
    if input_names != [SSD_INPUT_NAME]:
        raise ValueError(
            f"Expected SSD input {SSD_INPUT_NAME!r}, found {input_names}")
    if tuple(output_names) != SSD_OUTPUT_NAMES:
        raise ValueError(f"Expected SSD outputs {SSD_OUTPUT_NAMES}, "
                         f"found {tuple(output_names)}")

    cases = load_cases(args.fixtures_dir, input_names)
    ort_runner = OrtRunner(model_path, output_names, args)
    migraphx_runner = None
    if not args.ort_run:
        print(
            "Compiling MIGraphX once for fixed input shape "
            f"{cases[0][1][SSD_INPUT_NAME].shape}...",
            flush=True)
        migraphx_runner = MigraphxRunner(model_path, output_names, cases[0][1],
                                         args)

    results = []
    for index, (case, inputs) in enumerate(cases):
        image_id = case["image_id"]
        print_case_header(index, case)
        ort_outputs, ort_durations = ort_runner.run(inputs)
        ort_count = validate_ssd_outputs(ort_outputs)
        print_performance("ONNX Runtime", ort_durations)

        if args.ort_run:
            print(f"PASSED: {image_id}: ONNX Runtime produced "
                  f"{ort_count} detections")
            results.append((image_id, ort_count, True))
            continue

        migraphx_outputs, migraphx_durations = migraphx_runner.run(inputs)
        migraphx_count = validate_ssd_outputs(migraphx_outputs)
        print_performance(f"MIGraphX {args.target}", migraphx_durations)
        counts_match = ort_count == migraphx_count
        status = "PASSED" if counts_match else "FAILED"
        print(f"{status}: detection count: ORT={ort_count}, "
              f"MIGraphX={migraphx_count}")
        aligned_outputs = migraphx_outputs
        reordered = 0
        if counts_match:
            aligned_outputs, reordered = align_near_tied_outputs(
                ort_outputs, migraphx_outputs, args.rtol, args.atol)
        if reordered:
            print(f"Aligned {reordered} near-tied detections for comparison")
        outputs_match = compare_outputs(
            ort_outputs,
            aligned_outputs,
            output_names,
            args.rtol,
            args.atol,
            args.verbose,
        )
        passed = counts_match and outputs_match
        results.append((image_id, ort_count, passed))
        print(f"{'PASSED' if passed else 'FAILED'}: image {image_id}")

    print("\nSummary:")
    for image_id, detection_count, passed in results:
        print(f"  {'PASSED' if passed else 'FAILED'} {image_id} "
              f"detections={detection_count}")
    passed_count = sum(passed for _, _, passed in results)
    print(f"{passed_count}/{IMAGE_COUNT} images PASSED")
    return 0 if passed_count == IMAGE_COUNT else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
