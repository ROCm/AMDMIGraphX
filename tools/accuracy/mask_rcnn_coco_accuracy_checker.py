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
"""Compare MIGraphX and ONNX Runtime on ten prepared COCO Mask R-CNN inputs."""

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import onnx

import migraphx

from mask_rcnn_accuracy_checker import benchmark
from mask_rcnn_accuracy_checker import compare_outputs
from mask_rcnn_accuracy_checker import find_model_path
from mask_rcnn_accuracy_checker import load_tensor_set
from mask_rcnn_accuracy_checker import make_ort_session
from mask_rcnn_accuracy_checker import model_io_names
from mask_rcnn_accuracy_checker import print_performance
from mask_rcnn_accuracy_checker import validate_mask_rcnn_input
from mask_rcnn_accuracy_checker import validate_mask_rcnn_outputs

IMAGE_COUNT = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Compare MIGraphX Mask R-CNN outputs with ONNX Runtime "
                     "for ten prepared COCO images."))
    parser.add_argument(
        "test_dir",
        type=Path,
        help="Directory containing exactly one Mask R-CNN ONNX model",
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


def load_cases(fixtures_dir, input_names, output_names):
    manifest_path = fixtures_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing fixture manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    cases = manifest.get("cases")
    if manifest.get("image_count") != IMAGE_COUNT or not isinstance(
            cases, list) or len(cases) != IMAGE_COUNT:
        raise ValueError(f"Expected manifest with exactly {IMAGE_COUNT} cases")

    result = []
    for index, case in enumerate(cases):
        expected_name = f"test_data_set_{index}"
        if case.get("case") != expected_name:
            raise ValueError(
                f"Expected manifest case {expected_name!r}, found {case.get('case')!r}"
            )
        case_dir = fixtures_dir / expected_name
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Missing prepared case: {case_dir}")
        inputs = load_tensor_set(case_dir, "input", input_names)
        validate_mask_rcnn_input(input_names, output_names, inputs)
        expected_shape = tuple(case.get("tensor_shape", ()))
        if expected_shape != inputs[input_names[0]].shape:
            raise ValueError(
                f"Manifest shape {expected_shape} does not match "
                f"{expected_name} input shape {inputs[input_names[0]].shape}")
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
    def __init__(self, model_path, output_names, args):
        self.model_path = model_path
        self.output_names = output_names
        self.args = args

    def compile(self, inputs):
        input_dims = {name: list(data.shape) for name, data in inputs.items()}
        model = migraphx.parse_onnx(str(self.model_path),
                                    map_input_dims=input_dims)
        parameter_shapes = model.get_parameter_shapes()
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
        model.compile(
            migraphx.get_target(self.args.target),
            offload_copy=self.args.offload_copy,
            fast_math=self.args.fast_math,
            exhaustive_tune=self.args.exhaustive_tune,
        )
        return model

    def run(self, model, inputs):
        parameters = {}
        for name, data in inputs.items():
            argument = migraphx.argument(data)
            if not self.args.offload_copy:
                argument = migraphx.to_gpu(argument)
            parameters[name] = argument

        def run_once():
            raw_outputs = model.run(parameters)
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

        outputs, durations = benchmark(run_once, self.args.warmup,
                                       self.args.iterations)
        return outputs, durations


def group_cases_by_shape(cases, input_name):
    groups = {}
    for index, (case, inputs) in enumerate(cases):
        shape = tuple(inputs[input_name].shape)
        groups.setdefault(shape, []).append((index, case, inputs))
    return groups


def print_case_header(index, case, shape):
    print(f"\n=== [{index + 1}/{IMAGE_COUNT}] {case['image_id']} "
          f"shape={shape} ===")


def main():
    args = parse_args()
    model_path = find_model_path(args.test_dir)
    if not args.fixtures_dir.is_dir():
        raise FileNotFoundError(
            f"Missing prepared fixture directory: {args.fixtures_dir}")

    model_proto = onnx.load(str(model_path), load_external_data=False)
    input_names, output_names = model_io_names(model_proto)
    cases = load_cases(args.fixtures_dir, input_names, output_names)
    ort_runner = OrtRunner(model_path, output_names, args)
    migraphx_runner = None if args.ort_run else MigraphxRunner(
        model_path, output_names, args)

    results = [None] * IMAGE_COUNT
    if args.ort_run:
        for index, (case, inputs) in enumerate(cases):
            image_id = case["image_id"]
            shape = inputs[input_names[0]].shape
            print_case_header(index, case, shape)
            ort_outputs, ort_durations = ort_runner.run(inputs)
            ort_count = validate_mask_rcnn_outputs(ort_outputs, output_names,
                                                   "ONNX Runtime")
            print_performance("ONNX Runtime", ort_durations)
            print(f"PASSED: {image_id}: ONNX Runtime produced "
                  f"{ort_count} detections")
            results[index] = (image_id, shape, True)
    else:
        groups = group_cases_by_shape(cases, input_names[0])
        print(f"Processing {IMAGE_COUNT} images in {len(groups)} shape groups",
              flush=True)
        for group_index, (shape, group) in enumerate(groups.items()):
            print(
                f"\nCompiling MIGraphX shape group "
                f"{group_index + 1}/{len(groups)}: {shape}",
                flush=True)
            model = migraphx_runner.compile(group[0][2])
            try:
                for index, case, inputs in group:
                    image_id = case["image_id"]
                    print_case_header(index, case, shape)
                    ort_outputs, ort_durations = ort_runner.run(inputs)
                    ort_count = validate_mask_rcnn_outputs(
                        ort_outputs, output_names, "ONNX Runtime")
                    print_performance("ONNX Runtime", ort_durations)

                    migraphx_outputs, migraphx_durations = migraphx_runner.run(
                        model, inputs)
                    print_performance(f"MIGraphX {args.target}",
                                      migraphx_durations)
                    migraphx_count = validate_mask_rcnn_outputs(
                        migraphx_outputs, output_names, "MIGraphX")
                    counts_match = ort_count == migraphx_count
                    status = "PASSED" if counts_match else "FAILED"
                    print(f"{status}: detection count: ORT={ort_count}, "
                          f"MIGraphX={migraphx_count}")
                    outputs_match = compare_outputs(
                        ort_outputs,
                        migraphx_outputs,
                        output_names,
                        args.rtol,
                        args.atol,
                        args.verbose,
                    )
                    passed = counts_match and outputs_match
                    results[index] = (image_id, shape, passed)
                    print(
                        f"{'PASSED' if passed else 'FAILED'}: image {image_id}"
                    )
                    del ort_outputs
                    del migraphx_outputs
            finally:
                del model
                gc.collect()
                print(f"Released MIGraphX model for shape {shape}", flush=True)

    print("\nSummary:")
    for image_id, shape, passed in results:
        print(f"  {'PASSED' if passed else 'FAILED'} {image_id} {shape}")
    passed_count = sum(passed for _, _, passed in results)
    print(f"{passed_count}/{IMAGE_COUNT} images PASSED")
    return 0 if passed_count == IMAGE_COUNT else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
