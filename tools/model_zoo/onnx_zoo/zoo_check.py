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
"""Check a zoo model against its golden data and optionally emit matching
driver arguments."""
import argparse
import os
import sys

import onnx

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
import test_runner as tr  # noqa: E402
import migraphx  # noqa: E402
import compare  # noqa: E402
import golden  # noqa: E402

# Quantized operators that the name patterns in is_quantized() do not cover.
QUANTIZED_OPS = ('MatMulNBits', 'MatMulBnb4')


def parse_args():
    p = argparse.ArgumentParser(description="Model zoo accuracy check")
    p.add_argument('test_dir', metavar='test_loc')
    p.add_argument('--name', help='test name to print (default: folder name)')
    p.add_argument('--target', default='gpu')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--compare',
                   choices=['allclose', 'rms'],
                   default='allclose')
    p.add_argument('--atol', type=float, default=1e-3)
    p.add_argument('--rtol', type=float, default=1e-3)
    p.add_argument('--atol-frac',
                   type=float,
                   default=0.0,
                   help='atol as a fraction of max(abs(expected))')
    p.add_argument('--rms-tol',
                   type=float,
                   default=1e-3,
                   help='largest RMS error of an output that passes')
    p.add_argument('--gold', choices=['archive', 'ref'], default='archive')
    p.add_argument('--emit-driver-args')
    p.add_argument('--precisions',
                   action='store_true',
                   help='print the precisions to run the model in and exit')

    return p.parse_args()


def is_quantized(model_path):
    """True when the graph already computes in integers, so that quantizing
    it to fp16 is not a separate check."""
    model = onnx.load(model_path, load_external_data=False)
    ops = {node.op_type for node in model.graph.node}
    return any('Quantize' in op or op.startswith('QLinear')
               or op.endswith('Integer') or op in QUANTIZED_OPS for op in ops)


def build(model_path, shapes, args, target=None):
    model = migraphx.parse_onnx(model_path, map_input_dims=shapes)
    if args.fp16:
        migraphx.quantize_fp16(model)
    model.compile(migraphx.get_target(target or args.target))

    return model


def write_driver_args(path, shapes, args, dim_params=None):
    """Write driver arguments matching the graded target, shapes, and precision."""
    # One argument per line so the shell can restore the exact argv with
    # mapfile. Dimensions must remain separate arguments for the driver parser.
    lines = ['--' + args.target]
    for name, dims in shapes.items():
        lines += ['--input-dim', '@{}'.format(name)]
        lines += [str(int(d)) for d in dims]
    for name, size in (dim_params or {}).items():
        lines += ['--dim-param', '@{}'.format(name), str(size)]
    if args.fp16:
        lines.append('--fp16')
    with open(path, 'w') as pfile:
        pfile.write(''.join(line + '\n' for line in lines))


def main():
    args = parse_args()
    test_loc = args.test_dir
    if args.precisions:
        model_path = os.path.join(test_loc, tr.get_model_name(test_loc))
        print('int8' if is_quantized(model_path) else 'fp32 fp16')
        return

    name = args.name or os.path.basename(os.path.normpath(test_loc))
    rms = args.compare == 'rms'
    if rms:
        tolerance = 'rms_tol {:g}'.format(args.rms_tol)
    else:
        tolerance = 'atol {:g}, rtol {:g}, atol_frac {:g}'.format(
            args.atol, args.rtol, args.atol_frac)

    print("Running test \"{}\" on target \"{}\" ...".format(name, args.target))
    print("Grading: {}, gold {}, {}\n".format(
        'fp16' if args.fp16 else 'as-parsed', args.gold, tolerance))

    model_name = tr.get_model_name(test_loc)
    if not model_name:
        print("No .onnx model found in {}".format(test_loc))
        sys.exit(1)
    model_path = os.path.join(test_loc, model_name)
    params = tr.model_parameter_names(model_path)
    outs = tr.model_output_names(model_path)

    shapes, cases = golden.load_cases(test_loc, params, outs)
    if not cases:
        # Nothing to grade. Perf can still run with the sizes that
        # <model>.onnx.json gives to the model's symbolic dimensions.
        if args.emit_driver_args:
            write_driver_args(args.emit_driver_args, {}, args,
                              golden.read_dim_params(model_path))
        print("SKIPPED: no golden data (test_data_set_* or test_data_*.npz) "
              "in {}".format(test_loc))
        return
    for pname, dims in shapes.items():
        print("Input: {}, shape: {}".format(pname, tuple(dims)))
    print()

    if args.emit_driver_args:
        write_driver_args(args.emit_driver_args, shapes, args)

    model = build(model_path, shapes, args)
    # Apply the same optional FP16 transform to reference gold so only the target differs.
    ref = build(model_path, shapes, args,
                'ref') if args.gold == 'ref' else None

    correct = 0
    worst_dev = worst_frac = 0.0
    for label, inputs, gold in cases:
        tuned = tr.tune_input_shape(model, inputs)
        if tuned:
            model = build(model_path, tuned, args)
            if ref is not None:
                ref = build(model_path, tuned, args, 'ref')
        if ref is not None:
            gold = tr.run_one_case(ref, inputs)

        outputs = tr.run_one_case(model, inputs)
        if rms:
            ok, dev, frac = compare.rms(gold, outputs, args.rms_tol, outs)
        else:
            ok, dev, frac = compare.allclose(gold, outputs, args.atol,
                                             args.rtol, args.atol_frac)
        correct += ok
        worst_dev, worst_frac = max(worst_dev, dev), max(worst_frac, frac)
        print("\tCase {}: {}".format(label, "PASSED" if ok else "FAILED"))

    print("\nTest \"{}\" has {} cases:".format(name, len(cases)))
    print("\t Passed: {}".format(correct))
    print("\t Failed: {}".format(len(cases) - correct))
    print("\t Worst {}: {:.6g} ({:.6g} of tolerance)".format(
        'RMS error' if rms else 'deviation', worst_dev, worst_frac))
    if correct < len(cases):
        print("{} cases failed!".format(len(cases) - correct))
        sys.exit(1)


if __name__ == "__main__":
    main()
