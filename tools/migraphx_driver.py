#!/usr/bin/env python3
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
"""Python version of the MIGraphX driver CLI.

Provides the same subcommands as the C++ migraphx-driver binary:
read, params, compile, run, time, perf, verify.

Requires: pip install migraphx numpy
"""

import argparse
import hashlib
import json
import os
import sys
import time as time_mod

try:
    import numpy as np
except ImportError:
    sys.exit("error: numpy is required. Install with: pip install numpy")

try:
    import migraphx
except ImportError:
    sys.exit("error: migraphx module not found. "
             "Ensure MIGraphX is installed and PYTHONPATH includes "
             "/opt/rocm/lib or your build directory.")


def parse_at_args(raw_list):
    """Parse '@name value ...' argument lists into {name: [values]}.

    Used for --input-dim, --dyn-input-dim, --dim-param, --output-names, --fill0/1.
    """
    result = {}
    name = None
    for item in raw_list:
        if item.startswith("@"):
            name = item[1:]
            if name not in result:
                result[name] = []
        elif name is not None:
            result[name].append(item)
    return result


def parse_param_dims(raw_list):
    """Parse --input-dim '@name d1 d2 ...' into {name: [d1,d2,...]}."""
    parsed = parse_at_args(raw_list)
    return {k: [int(x) for x in v] for k, v in parsed.items()}


def parse_dyn_dims_json(dd_json):
    """Parse a dynamic dimension JSON string into a list of dynamic_dimension.

    Format: "[{min:1,max:64,optimals:[1,2,4,8]},3,224,224]"
    """
    normalized = dd_json.replace("min:", '"min":').replace(
        "max:", '"max":').replace("optimals:", '"optimals":')
    items = json.loads(normalized)
    dyn_dims = []
    for item in items:
        if isinstance(item, dict):
            optimals = set(item.get("optimals", []))
            dyn_dims.append(
                migraphx.shape.dynamic_dimension(item["min"], item["max"],
                                                 optimals))
        else:
            dyn_dims.append(migraphx.shape.dynamic_dimension(item, item))
    return dyn_dims


def parse_dyn_param_dims(raw_list):
    """Parse --dyn-input-dim '@name json_str' pairs."""
    parsed = parse_at_args(raw_list)
    result = {}
    for name, values in parsed.items():
        for val in values:
            result[name] = parse_dyn_dims_json(val)
    return result


def parse_dim_params(raw_list):
    """Parse --dim-param '@name value_or_json' pairs."""
    parsed = parse_at_args(raw_list)
    result = {}
    for name, values in parsed.items():
        for val in values:
            if val.isdigit():
                d = int(val)
                result[name] = migraphx.shape.dynamic_dimension(d, d)
            else:
                dims = parse_dyn_dims_json("[" + val + "]")
                if len(dims) != 1:
                    raise ValueError(
                        f"dim_param '{name}' must specify one dimension")
                result[name] = dims[0]
    return result


def parse_output_names(raw_list):
    """Parse --output-names 'name1 name2 ...'."""
    return list(raw_list)


def get_file_type(filepath):
    """Infer file type from extension."""
    if filepath.endswith(".onnx"):
        return "onnx"
    elif filepath.endswith(".pb"):
        return "tf"
    elif filepath.endswith(".json"):
        return "json"
    else:
        return "migraphx"


def test_gemm():
    """Create a simple GEMM program for testing."""
    p = migraphx.program()
    mm = p.get_main_module()
    a = mm.add_parameter("a",
                         migraphx.shape(type="float", lens=[4, 5]))
    b = mm.add_parameter("b",
                         migraphx.shape(type="float", lens=[5, 3]))
    mm.add_instruction(migraphx.op("dot"), [a, b])
    return p


def load_model(args):
    """Load a model based on CLI arguments. Returns a migraphx.program."""
    if getattr(args, "test", False):
        return test_gemm()

    filepath = args.file
    if not os.path.exists(filepath):
        print(f"error: Path does not exist: {filepath}", file=sys.stderr)
        sys.exit(1)

    file_type = getattr(args, "file_type", None) or get_file_type(filepath)
    batch = getattr(args, "batch", 1)

    if file_type == "onnx":
        onnx_opts = {"default_dim_value": batch}
        if getattr(args, "skip_unknown_operators", False):
            onnx_opts["skip_unknown_operators"] = True
        if getattr(args, "debug_symbols", False):
            onnx_opts["use_debug_symbols"] = True
        if getattr(args, "input_dim", None):
            onnx_opts["map_input_dims"] = parse_param_dims(args.input_dim)
        if getattr(args, "dyn_input_dim", None):
            onnx_opts["map_dyn_input_dims"] = parse_dyn_param_dims(
                args.dyn_input_dim)
        if getattr(args, "dim_param", None):
            onnx_opts["dim_params"] = parse_dim_params(args.dim_param)
        if getattr(args, "default_dyn_dim", None):
            dims = parse_dyn_dims_json("[" + args.default_dyn_dim + "]")
            onnx_opts["default_dyn_dim_value"] = dims[0]
            onnx_opts.pop("default_dim_value", None)
        p = migraphx.parse_onnx(filepath, **onnx_opts)
    elif file_type == "tf":
        tf_opts = {
            "is_nhwc": getattr(args, "nhwc", True),
            "batch_size": batch,
        }
        if getattr(args, "input_dim", None):
            tf_opts["map_input_dims"] = parse_param_dims(args.input_dim)
        if getattr(args, "output_names", None):
            tf_opts["output_names"] = parse_output_names(args.output_names)
        p = migraphx.parse_tf(filepath, **tf_opts)
    elif file_type == "json":
        p = migraphx.load(filepath, format="json")
    elif file_type == "migraphx":
        p = migraphx.load(filepath)
    else:
        print(f"error: Unknown file type '{file_type}'", file=sys.stderr)
        sys.exit(1)

    return p


def get_target_name(args):
    """Get target name from CLI flags."""
    return getattr(args, "target_name", "gpu")


def get_target(args):
    """Create migraphx target from CLI flags."""
    return migraphx.get_target(get_target_name(args))


def name_hash(name):
    """Deterministic hash for parameter name, matching C++ std::hash<string>."""
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % (2**32)


def generate_params(prog, args):
    """Generate parameter map with random or filled data."""
    param_shapes = prog.get_parameter_shapes()
    batch = getattr(args, "batch", 1)
    fill0_names = set()
    fill1_names = set()
    if getattr(args, "fill0", None):
        fill0_names = set(args.fill0)
    if getattr(args, "fill1", None):
        fill1_names = set(args.fill1)

    params = {}
    for name, shape in param_shapes.items():
        if name in fill0_names:
            params[name] = migraphx.fill_argument(shape, 0)
        elif name in fill1_names:
            params[name] = migraphx.fill_argument(shape, 1)
        else:
            params[name] = migraphx.generate_argument(shape, name_hash(name))
    return params


def quantize_program(prog, args, target):
    """Apply quantization passes if requested."""
    if getattr(args, "fp16", False):
        migraphx.quantize_fp16(prog)
    if getattr(args, "bf16", False):
        migraphx.quantize_bf16(prog)
    if getattr(args, "int8", False):
        migraphx.quantize_int8(prog, target)
    if getattr(args, "fp8", False):
        migraphx.quantize_fp8(prog, target)


def compile_program(args):
    """Load, quantize, and compile a program. Returns (program, target)."""
    prog = load_model(args)
    target = get_target(args)
    offload_copy = getattr(args, "enable_offload_copy", False)
    fast_math = not getattr(args, "disable_fast_math", False)
    exhaustive_tune = getattr(args, "exhaustive_tune", False)

    if prog.is_compiled():
        print("The program is already compiled, skipping compilation ...",
              file=sys.stderr)
    else:
        quantize_program(prog, args, target)
        t0 = time_mod.perf_counter()
        prog.compile(target,
                     offload_copy=offload_copy,
                     fast_math=fast_math,
                     exhaustive_tune=exhaustive_tune)
        elapsed_ms = (time_mod.perf_counter() - t0) * 1000
        print(f"Compilation time: {elapsed_ms:.0f}ms", file=sys.stderr)

    return prog, target


def save_program(prog, args):
    """Save or print the program based on output flags."""
    output_type = getattr(args, "output_type", None) or ""
    output_file = getattr(args, "output", None) or ""

    if not output_type:
        output_type = "binary" if output_file else "text"

    if output_type == "text":
        if output_file:
            with open(output_file, "w") as f:
                f.write(repr(prog))
                f.write("\n")
        else:
            print(repr(prog))
    elif output_type == "py":
        py_str = prog.to_py()
        if output_file:
            with open(output_file, "w") as f:
                f.write(py_str)
        else:
            print(py_str)
    elif output_type == "json":
        migraphx.save(prog, output_file if output_file else "/dev/stdout",
                      format="json")
    elif output_type == "binary":
        if not output_file:
            print(
                "error: --binary requires --output/-o to specify output file",
                file=sys.stderr)
            sys.exit(1)
        migraphx.save(prog, output_file)
    else:
        print(f"warning: output type '{output_type}' not supported in Python "
              f"driver, falling back to text",
              file=sys.stderr)
        print(repr(prog))


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_read(args):
    """Load and display a model."""
    prog = load_model(args)
    save_program(prog, args)


def cmd_params(args):
    """Load a model and print parameter shapes."""
    prog = load_model(args)
    for name, shape in prog.get_parameter_shapes().items():
        print(f"{name}: {shape}")


def cmd_compile(args):
    """Load, quantize, compile, and optionally save a model."""
    prog, _ = compile_program(args)
    save_program(prog, args)


def cmd_run(args):
    """Compile a model, generate inputs, and run inference."""
    prog, _ = compile_program(args)
    print("Allocating params ...", file=sys.stderr)
    params = generate_params(prog, args)
    prog.run(params)
    print(repr(prog))


def cmd_time(args):
    """Compile a model and benchmark inference time."""
    prog, _ = compile_program(args)
    n = getattr(args, "iterations", 100)
    print("Allocating params ...", file=sys.stderr)
    params = generate_params(prog, args)

    # Warmup
    prog.run(params)

    print("Running ...", file=sys.stderr)
    t0 = time_mod.perf_counter()
    for _ in range(n):
        prog.run(params)
    total_ms = (time_mod.perf_counter() - t0) * 1000
    avg_ms = total_ms / n
    print(f"Total time: {avg_ms:.4f}ms")


def cmd_perf(args):
    """Compile a model and run a performance report.

    Note: The C++ driver's detailed per-instruction perf_report is not
    available via the Python API. This command provides aggregate timing.
    """
    prog, _ = compile_program(args)
    n = getattr(args, "iterations", 100)
    print("Allocating params ...", file=sys.stderr)
    params = generate_params(prog, args)

    # Warmup
    prog.run(params)

    print("Running performance report ...", file=sys.stderr)
    times = []
    for _ in range(n):
        t0 = time_mod.perf_counter()
        prog.run(params)
        times.append((time_mod.perf_counter() - t0) * 1000)

    times_arr = np.array(times)
    print(f"Summary (n={n}):")
    print(f"  Mean:   {np.mean(times_arr):.4f}ms")
    print(f"  Median: {np.median(times_arr):.4f}ms")
    print(f"  Std:    {np.std(times_arr):.4f}ms")
    print(f"  Min:    {np.min(times_arr):.4f}ms")
    print(f"  Max:    {np.max(times_arr):.4f}ms")
    print(f"  Total:  {np.sum(times_arr):.4f}ms")


def cmd_verify(args):
    """Compile on target and ref, compare outputs."""
    prog = load_model(args)
    target_name = get_target_name(args)
    offload_copy = getattr(args, "enable_offload_copy", False)
    fast_math = not getattr(args, "disable_fast_math", False)
    exhaustive_tune = getattr(args, "exhaustive_tune", False)

    if target_name == "ref":
        print("error: verify requires a non-ref target (--gpu or --cpu)",
              file=sys.stderr)
        sys.exit(1)

    print(repr(prog))

    # Generate inputs on host
    param_shapes = prog.get_parameter_shapes()
    inputs = {}
    for name, shape in param_shapes.items():
        inputs[name] = migraphx.generate_argument(shape, name_hash(name))

    # Run on ref
    ref_prog = migraphx.parse_onnx(
        args.file) if get_file_type(args.file) == "onnx" else load_model(args)
    ref_target = migraphx.get_target("ref")
    ref_prog.compile(ref_target, offload_copy=True, fast_math=fast_math)
    ref_outputs = ref_prog.run(inputs)

    # Quantize and compile on target
    target = migraphx.get_target(target_name)
    quantize_program(prog, args, target)
    prog.compile(target,
                 offload_copy=offload_copy,
                 fast_math=fast_math,
                 exhaustive_tune=exhaustive_tune)

    if offload_copy:
        target_inputs = inputs
    else:
        target_inputs = {}
        for name, arg in inputs.items():
            target_inputs[name] = migraphx.generate_argument(
                prog.get_parameter_shapes()[name], name_hash(name))
    target_outputs = prog.run(target_inputs)

    # Compare
    rms_tol = getattr(args, "rms_tol", None) or 0.001
    atol = getattr(args, "atol", None) or 0.001
    rtol = getattr(args, "rtol", None) or 0.001

    passed = True
    for i, (ref_out, tgt_out) in enumerate(zip(ref_outputs, target_outputs)):
        ref_arr = np.array(ref_out)
        tgt_arr = np.array(tgt_out)

        if ref_arr.shape != tgt_arr.shape:
            print(
                f"FAILED output {i}: shape mismatch {ref_arr.shape} != {tgt_arr.shape}"
            )
            passed = False
            continue

        diff = np.abs(ref_arr.astype(np.float64) - tgt_arr.astype(np.float64))
        rms = np.sqrt(np.mean(diff**2))
        max_abs_diff = np.max(diff)
        ref_abs = np.abs(ref_arr.astype(np.float64))
        ref_abs_safe = np.where(ref_abs > 0, ref_abs, 1.0)
        max_rel_diff = np.max(diff / ref_abs_safe)

        output_passed = rms <= rms_tol and max_abs_diff <= atol and max_rel_diff <= rtol
        status = "PASSED" if output_passed else "FAILED"
        print(f"Output {i}: {status}  "
              f"rms={rms:.6e}  max_abs={max_abs_diff:.6e}  "
              f"max_rel={max_rel_diff:.6e}")
        if not output_passed:
            passed = False

    if passed:
        print("MIGraphX verification passed successfully.")
    else:
        print("MIGraphX verification FAILED.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------


def add_loader_args(parser):
    """Add model loading arguments shared across subcommands."""
    parser.add_argument("file",
                        nargs="?",
                        default=None,
                        metavar="<input file>",
                        help="Path to the model file")
    parser.add_argument("--test",
                        action="store_true",
                        help="Run a single GEMM to test MIGraphX")
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--onnx",
                     dest="file_type",
                     action="store_const",
                     const="onnx",
                     help="Load as ONNX")
    fmt.add_argument("--tf",
                     dest="file_type",
                     action="store_const",
                     const="tf",
                     help="Load as TensorFlow")
    fmt.add_argument("--migraphx",
                     dest="file_type",
                     action="store_const",
                     const="migraphx",
                     help="Load as MIGraphX")
    fmt.add_argument("--migraphx-json",
                     dest="file_type",
                     action="store_const",
                     const="json",
                     help="Load as MIGraphX JSON")
    parser.add_argument("--batch",
                        type=int,
                        default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--nhwc",
                        dest="nhwc",
                        action="store_true",
                        default=True,
                        help="Treat TF format as NHWC (default)")
    parser.add_argument("--nchw",
                        dest="nhwc",
                        action="store_false",
                        help="Treat TF format as NCHW")
    parser.add_argument("--skip-unknown-operators",
                        action="store_true",
                        help="Skip unknown operators when parsing")
    parser.add_argument(
        "--debug-symbols",
        action="store_true",
        help="Parse ONNX node names into instructions as debug symbols")
    parser.add_argument("--input-dim",
                        nargs="+",
                        action="append",
                        default=[],
                        metavar="ARG",
                        help="Dim of a parameter: @name d1 d2 dn")
    parser.add_argument("--dyn-input-dim",
                        nargs="+",
                        action="append",
                        default=[],
                        metavar="ARG",
                        help="Dynamic dims of a parameter: @name json_str")
    parser.add_argument("--dim-param",
                        nargs="+",
                        action="append",
                        default=[],
                        metavar="ARG",
                        help="Symbolic dim param: @name value_or_json")
    parser.add_argument("--default-dyn-dim",
                        default=None,
                        help="Default dynamic dimension JSON")
    parser.add_argument("--output-names",
                        nargs="+",
                        default=[],
                        help="Names of output nodes")


def add_output_args(parser):
    """Add output format arguments."""
    out_fmt = parser.add_mutually_exclusive_group()
    out_fmt.add_argument("--graphviz",
                         "-g",
                         dest="output_type",
                         action="store_const",
                         const="graphviz",
                         help="Print as graphviz (not supported in Python)")
    out_fmt.add_argument("--cpp",
                         dest="output_type",
                         action="store_const",
                         const="cpp",
                         help="Print as C++ (not supported in Python)")
    out_fmt.add_argument("--python",
                         "--py",
                         dest="output_type",
                         action="store_const",
                         const="py",
                         help="Print as Python program")
    out_fmt.add_argument("--json",
                         dest="output_type",
                         action="store_const",
                         const="json",
                         help="Print as JSON")
    out_fmt.add_argument("--text",
                         dest="output_type",
                         action="store_const",
                         const="text",
                         help="Print in text format")
    out_fmt.add_argument("--binary",
                         dest="output_type",
                         action="store_const",
                         const="binary",
                         help="Save in binary format")
    parser.add_argument("--output",
                        "-o",
                        default="",
                        help="Output to file")


def add_compiler_args(parser):
    """Add compilation arguments."""
    tgt = parser.add_mutually_exclusive_group()
    tgt.add_argument("--gpu",
                     dest="target_name",
                     action="store_const",
                     const="gpu",
                     help="Compile on the GPU (default)")
    tgt.add_argument("--cpu",
                     dest="target_name",
                     action="store_const",
                     const="cpu",
                     help="Compile on the CPU")
    tgt.add_argument("--ref",
                     dest="target_name",
                     action="store_const",
                     const="ref",
                     help="Compile on the reference implementation")
    parser.set_defaults(target_name="gpu")

    parser.add_argument("--enable-offload-copy",
                        action="store_true",
                        help="Enable implicit offload copying")
    parser.add_argument("--disable-fast-math",
                        action="store_true",
                        help="Disable fast math optimization")
    parser.add_argument("--exhaustive-tune",
                        action="store_true",
                        help="Exhaustively search for best tuning parameters")
    parser.add_argument("--fp16",
                        action="store_true",
                        help="Quantize for fp16")
    parser.add_argument("--bf16",
                        action="store_true",
                        help="Quantize for bf16")
    parser.add_argument("--int8",
                        action="store_true",
                        help="Quantize for int8")
    parser.add_argument("--fp8",
                        action="store_true",
                        help="Quantize for fp8")


def add_param_args(parser):
    """Add parameter generation arguments."""
    parser.add_argument("--fill0",
                        nargs="+",
                        default=[],
                        help="Fill parameter(s) with 0s")
    parser.add_argument("--fill1",
                        nargs="+",
                        default=[],
                        help="Fill parameter(s) with 1s")


def flatten_nested_lists(lst):
    """Flatten nested lists from argparse append+nargs combination."""
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(
        prog="migraphx_driver",
        description="MIGraphX Driver (Python)")
    parser.add_argument("-v",
                        "--version",
                        action="version",
                        version=f"MIGraphX Version: {migraphx.__version__}")

    subparsers = parser.add_subparsers(dest="command",
                                       title="commands",
                                       metavar="<command>")

    # read
    p_read = subparsers.add_parser("read", help="Read and display a model")
    add_loader_args(p_read)
    add_output_args(p_read)
    p_read.set_defaults(func=cmd_read)

    # params
    p_params = subparsers.add_parser("params",
                                     help="Show model parameter shapes")
    add_loader_args(p_params)
    p_params.set_defaults(func=cmd_params)

    # compile
    p_compile = subparsers.add_parser("compile", help="Compile a model")
    add_loader_args(p_compile)
    add_output_args(p_compile)
    add_compiler_args(p_compile)
    p_compile.set_defaults(func=cmd_compile)

    # run
    p_run = subparsers.add_parser("run", help="Compile and run a model")
    add_loader_args(p_run)
    add_compiler_args(p_run)
    add_param_args(p_run)
    p_run.set_defaults(func=cmd_run)

    # time
    p_time = subparsers.add_parser("time", help="Benchmark model inference")
    add_loader_args(p_time)
    add_compiler_args(p_time)
    add_param_args(p_time)
    p_time.add_argument("--iterations",
                        "-n",
                        type=int,
                        default=100,
                        help="Number of iterations (default: 100)")
    p_time.set_defaults(func=cmd_time)

    # perf
    p_perf = subparsers.add_parser("perf", help="Performance report")
    add_loader_args(p_perf)
    add_compiler_args(p_perf)
    add_param_args(p_perf)
    p_perf.add_argument("--iterations",
                        "-n",
                        type=int,
                        default=100,
                        help="Number of iterations (default: 100)")
    p_perf.set_defaults(func=cmd_perf)

    # verify
    p_verify = subparsers.add_parser(
        "verify", help="Verify target output against ref")
    add_loader_args(p_verify)
    add_compiler_args(p_verify)
    p_verify.add_argument("--rms-tol",
                          type=float,
                          default=None,
                          help="Tolerance for RMS error")
    p_verify.add_argument("--atol",
                          type=float,
                          default=None,
                          help="Absolute tolerance")
    p_verify.add_argument("--rtol",
                          type=float,
                          default=None,
                          help="Relative tolerance")
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()

    # Flatten nested lists from nargs+append
    for attr in ("input_dim", "dyn_input_dim", "dim_param"):
        if hasattr(args, attr):
            setattr(args, attr, flatten_nested_lists(getattr(args, attr)))

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if not getattr(args, "test", False) and not getattr(args, "file", None):
        print("error: an input file is required (or use --test)",
              file=sys.stderr)
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
