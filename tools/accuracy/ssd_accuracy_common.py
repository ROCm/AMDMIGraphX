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
"""Shared helpers for SSD Model Zoo accuracy checkers."""

import argparse
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

import migraphx

SSD_INPUT_NAME = "image"
SSD_INPUT_SHAPE = (1, 3, 1200, 1200)
SSD_OUTPUT_NAMES = ("bboxes", "labels", "scores")
SSD_OUTPUT_DTYPES = {
    "bboxes": np.float32,
    "labels": np.int64,
    "scores": np.float32,
}


def parse_args(model_name):
    parser = argparse.ArgumentParser(
        description=f"Compare MIGraphX {model_name} outputs with ONNX Runtime."
    )
    parser.add_argument(
        "test_dir",
        type=Path,
        help=(f"Extracted {model_name} directory containing the ONNX model "
              "and test_data_set_0"),
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


def load_tensor(path):
    tensor = onnx.TensorProto()
    tensor.ParseFromString(path.read_bytes())
    return tensor.name, np.ascontiguousarray(numpy_helper.to_array(tensor))


def load_tensor_set(data_dir, prefix, names):
    files = sorted(
        data_dir.glob(f"{prefix}_*.pb"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    if len(files) != len(names):
        raise ValueError(
            f"Expected {len(names)} {prefix} tensors in {data_dir}, found {len(files)}"
        )

    result = {}
    for expected_name, path in zip(names, files):
        tensor_name, data = load_tensor(path)
        name = tensor_name or expected_name
        if name not in names:
            raise ValueError(f"Unexpected tensor name {name!r} in {path}")
        if name in result:
            raise ValueError(f"Duplicate tensor name {name!r} in {data_dir}")
        result[name] = data

    missing = set(names) - set(result)
    if missing:
        raise ValueError(f"Missing {prefix} tensors: {sorted(missing)}")
    return result


def model_io_names(model):
    initializer_names = {
        initializer.name
        for initializer in model.graph.initializer
    }
    input_names = [
        value.name for value in model.graph.input
        if value.name not in initializer_names
    ]
    output_names = [value.name for value in model.graph.output]
    return input_names, output_names


def validate_ssd_input(inputs):
    if set(inputs) != {SSD_INPUT_NAME}:
        raise ValueError(
            f"Expected SSD input {SSD_INPUT_NAME!r}, found {sorted(inputs)}")

    image = inputs[SSD_INPUT_NAME]
    if image.shape != SSD_INPUT_SHAPE or image.dtype != np.float32:
        raise ValueError(
            f"Expected {SSD_INPUT_NAME} float32 shape {SSD_INPUT_SHAPE}, "
            f"found {image.dtype} shape {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError(
            f"SSD input {SSD_INPUT_NAME!r} contains non-finite values")


def validate_ssd_outputs(outputs):
    if set(outputs) != set(SSD_OUTPUT_NAMES):
        raise ValueError(
            f"Expected SSD outputs {SSD_OUTPUT_NAMES}, found {tuple(outputs)}")

    for name, expected_dtype in SSD_OUTPUT_DTYPES.items():
        value = outputs[name]
        if value.dtype != expected_dtype:
            raise ValueError(
                f"Expected SSD output {name!r} dtype {np.dtype(expected_dtype)}, "
                f"found {value.dtype}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"SSD output {name!r} contains non-finite values")

    bboxes = outputs["bboxes"]
    labels = outputs["labels"]
    scores = outputs["scores"]
    if bboxes.ndim != 3 or bboxes.shape[0] != 1 or bboxes.shape[2] != 4:
        raise ValueError(
            f"Expected bboxes shape (1, N, 4), found {bboxes.shape}")
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError(f"Expected labels shape (1, N), found {labels.shape}")
    if scores.ndim != 2 or scores.shape[0] != 1:
        raise ValueError(f"Expected scores shape (1, N), found {scores.shape}")

    detection_count = bboxes.shape[1]
    if labels.shape[1] != detection_count or scores.shape[1] != detection_count:
        raise ValueError(
            "SSD output detection counts are inconsistent: "
            f"bboxes={detection_count}, labels={labels.shape[1]}, "
            f"scores={scores.shape[1]}")
    return detection_count


def validate_ssd_contract(input_names, output_names, inputs, golden_outputs):
    if input_names != [SSD_INPUT_NAME]:
        raise ValueError(
            f"Expected SSD input {SSD_INPUT_NAME!r}, found {input_names}")
    if tuple(output_names) != SSD_OUTPUT_NAMES:
        raise ValueError(
            f"Expected SSD outputs {SSD_OUTPUT_NAMES}, found {tuple(output_names)}"
        )
    validate_ssd_input(inputs)
    validate_ssd_outputs(golden_outputs)


def make_ort_session(model_path, args):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if args.ort_intra_op_num_threads is not None:
        options.intra_op_num_threads = args.ort_intra_op_num_threads
        options.inter_op_num_threads = 1
    if args.ort_logging:
        options.log_verbosity_level = 0
        options.log_severity_level = 0
    else:
        options.log_severity_level = 3
    return ort.InferenceSession(str(model_path),
                                sess_options=options,
                                providers=[args.provider])


def benchmark(run, warmup, iterations):
    for _ in range(warmup):
        run()

    durations = []
    result = None
    for _ in range(iterations):
        start = time.perf_counter()
        result = run()
        durations.append(time.perf_counter() - start)
    return result, np.asarray(durations)


def print_performance(name, durations):
    milliseconds = durations * 1000
    print(f"{name} inference over {len(milliseconds)} iterations: "
          f"mean={np.mean(milliseconds):.3f} ms, "
          f"median={np.median(milliseconds):.3f} ms, "
          f"min={np.min(milliseconds):.3f} ms, "
          f"max={np.max(milliseconds):.3f} ms")


def run_ort(model_path, inputs, output_names, args):
    session = make_ort_session(model_path, args)
    session_inputs = [value.name for value in session.get_inputs()]
    missing = set(session_inputs) - set(inputs)
    if missing:
        raise ValueError(
            f"No packaged data for ONNX Runtime inputs: {sorted(missing)}")
    ort_inputs = {name: inputs[name] for name in session_inputs}

    def run():
        return dict(zip(output_names, session.run(output_names, ort_inputs)))

    return benchmark(run, args.warmup, args.iterations)


def run_migraphx(model_path, inputs, output_names, args):
    model = migraphx.parse_onnx(str(model_path))
    parameter_shapes = model.get_parameter_shapes()
    if set(parameter_shapes) != set(inputs):
        raise ValueError(
            "MIGraphX parameters do not match packaged inputs: "
            f"parameters={sorted(parameter_shapes)}, inputs={sorted(inputs)}")

    for name, shape in parameter_shapes.items():
        if tuple(shape.lens()) != inputs[name].shape:
            raise ValueError(
                f"MIGraphX parameter {name!r} expects {tuple(shape.lens())}, "
                f"but fixture has {inputs[name].shape}")

    model.compile(
        migraphx.get_target(args.target),
        offload_copy=args.offload_copy,
        fast_math=args.fast_math,
        exhaustive_tune=args.exhaustive_tune,
    )

    parameters = {}
    for name, data in inputs.items():
        argument = migraphx.argument(data)
        if not args.offload_copy:
            argument = migraphx.to_gpu(argument)
        parameters[name] = argument

    def run():
        raw_outputs = model.run(parameters)
        if len(raw_outputs) != len(output_names):
            raise ValueError(f"MIGraphX returned {len(raw_outputs)} outputs, "
                             f"expected {len(output_names)}")

        outputs = {}
        for name, output in zip(output_names, raw_outputs):
            if not args.offload_copy:
                output = migraphx.from_gpu(output)
            outputs[name] = np.array(output)
        return outputs

    # Compilation and parameter setup happen above; benchmark only inference.
    return benchmark(run, args.warmup, args.iterations)


def compare_outputs(reference, actual, output_names, rtol, atol, verbose):
    passed = True
    for name in output_names:
        expected = reference[name]
        result = actual[name]
        if expected.shape != result.shape:
            print(f"FAILED: {name} shape mismatch: "
                  f"expected {expected.shape}, actual {result.shape}")
            passed = False
            continue

        if np.issubdtype(expected.dtype, np.integer) or np.issubdtype(
                expected.dtype, np.bool_):
            matches = np.array_equal(expected, result)
            mismatch_count = int(np.count_nonzero(expected != result))
            detail = f"{mismatch_count} mismatched values"
        else:
            matches = np.allclose(expected,
                                  result,
                                  rtol=rtol,
                                  atol=atol,
                                  equal_nan=True)
            difference = np.abs(
                expected.astype(np.float64) - result.astype(np.float64))
            max_difference = float(
                np.max(difference)) if difference.size else 0.0
            detail = f"max abs difference {max_difference:.8g}"

        status = "PASSED" if matches else "FAILED"
        print(f"{status}: {name}: shape {result.shape}, {detail}")
        if not matches:
            passed = False
            if verbose:
                with np.printoptions(threshold=32, edgeitems=4):
                    print(f"Expected {name}:\n{expected}")
                    print(f"Actual {name}:\n{result}")
    return passed


def run_model_zoo_accuracy(model_name, model_filename):
    args = parse_args(model_name)
    model_path = args.test_dir / model_filename
    data_dir = args.test_dir / "test_data_set_0"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing {model_name} model: {model_path}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing {model_name} test data: {data_dir}")

    model_proto = onnx.load(str(model_path), load_external_data=False)
    input_names, output_names = model_io_names(model_proto)
    inputs = load_tensor_set(data_dir, "input", input_names)
    golden_outputs = load_tensor_set(data_dir, "output", output_names)
    validate_ssd_contract(input_names, output_names, inputs, golden_outputs)

    print("Running ONNX Runtime reference...")
    ort_outputs, ort_durations = run_ort(model_path, inputs, output_names,
                                         args)
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
        print(f"PASSED: {model_name} MIGraphX outputs match ONNX Runtime")
        return 0
    print(f"FAILED: {model_name} accuracy check")
    return 1
