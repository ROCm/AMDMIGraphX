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
"""Zoo accuracy check. Imports tools/test_runner.py for archive I/O (that script
is shared with other pipelines) and adds a range-relative tolerance, an optional
ref reference, and driver flags for the perf run."""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
import test_runner as tr  # noqa: E402
import migraphx  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="ONNX zoo accuracy check")
    p.add_argument('test_dir', metavar='test_loc')
    p.add_argument('--target', default='gpu')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--atol', type=float, default=1e-3)
    p.add_argument('--rtol', type=float, default=1e-3)
    p.add_argument('--atol-frac',
                   type=float,
                   default=0.0,
                   help='atol as a fraction of max(abs(expected))')
    p.add_argument('--gold', choices=['archive', 'ref'], default='archive')
    p.add_argument('--emit-driver-args')
    p.add_argument('--emit-metrics',
                   help='write the accuracy columns of the results row here')

    return p.parse_args()


def deviation(gold, actual, atol, rtol):
    """(max abs deviation, deviation as a fraction of the tolerance it was
    graded against). The second value is <= 1 exactly when np.allclose passes,
    so it stays comparable across models whose ranges differ by orders of
    magnitude, and shows how much headroom a passing model has left."""
    if gold.size == 0:
        return 0.0, 0.0

    dev = np.abs(gold.astype(np.float64) - actual.astype(np.float64))
    # A NaN is never within tolerance; rank it as an outright miss rather than
    # letting it propagate into a number that reads as a pass.
    dev = np.where(np.isnan(dev), np.inf, dev)
    tol = atol + rtol * np.abs(actual.astype(np.float64))
    frac = np.divide(dev,
                     tol,
                     out=np.where(dev > 0, np.inf, 0.0),
                     where=tol > 0)

    return float(dev.max()), float(frac.max())


def check(gold_outputs, outputs, args):
    """(all outputs within tolerance, worst abs deviation, worst tolerance
    fraction) over every output of one case."""
    if len(gold_outputs) != len(outputs):
        print("Expected {} outputs, got {}".format(len(gold_outputs),
                                                   len(outputs)))
        return False, float('inf'), float('inf')

    ok = True
    worst_dev = worst_frac = 0.0
    for i, (gold, actual) in enumerate(zip(gold_outputs, outputs)):
        gold, actual = np.asarray(gold), np.asarray(actual)
        if gold.shape != actual.shape:
            print("\nOutput {} shape mismatch: expected {}, got {}".format(
                i, gold.shape, actual.shape))
            ok, worst_dev, worst_frac = False, float('inf'), float('inf')
            continue

        rng = float(np.abs(gold).max()) if gold.size else 0.0
        atol = max(args.atol, args.atol_frac * rng)
        dev, frac = deviation(gold, actual, atol, args.rtol)
        worst_dev, worst_frac = max(worst_dev, dev), max(worst_frac, frac)
        if not np.allclose(gold, actual, args.rtol, atol):
            print(
                "\nOutput {} is incorrect ... max abs diff {:.6g} vs atol {:.6g}, "
                "rtol {:.6g}, expected range {:.6g}".format(
                    i, dev, atol, args.rtol, rng))
            print("Expected value: \n{}\nActual value: \n{}\n".format(
                gold, actual))
            ok = False

    return ok, worst_dev, worst_frac


def build(model_path, shapes, args, target=None):
    model = migraphx.parse_onnx(model_path, map_input_dims=shapes)
    if args.fp16:
        migraphx.quantize_fp16(model)
    model.compile(migraphx.get_target(target or args.target))

    return model


def write_driver_args(path, shapes, args):
    """Driver flags reproducing what we just graded, so the timings in the
    results row belong to the same program as the verdict beside them."""
    # One argument per line so the shell can restore the exact argv with
    # mapfile. Dimensions must remain separate arguments for the driver parser.
    lines = ['--' + args.target]
    for name, dims in shapes.items():
        lines += ['--input-dim', '@{}'.format(name)]
        lines += [str(int(d)) for d in dims]
    if args.fp16:
        lines.append('--fp16')
    with open(path, 'w') as pfile:
        pfile.write(''.join(line + '\n' for line in lines))


def write_metrics(path,
                  status,
                  cases=0,
                  passed=0,
                  max_diff=None,
                  tol_frac=None):
    """The accuracy fields of one results.csv row, in the order test_models.sh
    declares them. Blank rather than zero for the metrics when nothing was
    graded, so aggregating the column cannot mistake a skip for a clean run."""
    fields = [status, cases, passed] + [
        '' if x is None else '{:.6g}'.format(x) for x in (max_diff, tol_frac)
    ]
    with open(path, 'w') as mfile:
        mfile.write(','.join(str(f) for f in fields) + '\n')


def load_cases(test_loc, params, outs):
    """(input_shapes, [(label, inputs, gold)]) from test_data_set_*/ folders or
    the legacy caffe2-era test_data_*.npz."""
    folders = tr.get_test_cases(test_loc)
    if folders:
        cases = [(f, tr.wrapup_inputs(os.path.join(test_loc, f), params),
                  tr.read_outputs(os.path.join(test_loc, f), outs))
                 for f in folders]
        return tr.get_input_shapes(os.path.join(test_loc, folders[0]),
                                   params), cases

    cases = []
    for i, path in enumerate(
            sorted(glob.glob(os.path.join(test_loc, 'test_data_*.npz')))):
        inputs, gold = tr.load_npz_case(path)
        # Positional mapping: the i-th stored tensor feeds the i-th model input.
        cases.append(('test_data_{}'.format(i),
                      {params[j]: inputs[j]
                       for j in range(len(params))}, gold))
    if not cases:
        return None, []

    return {name: cases[0][1][name].shape for name in params}, cases


def main():
    args = parse_args()
    test_loc = args.test_dir
    name = os.path.basename(os.path.normpath(test_loc))

    print("Running test \"{}\" on target \"{}\" ...".format(name, args.target))
    print(
        "Grading: {}, gold {}, atol {:g}, rtol {:g}, atol_frac {:g}\n".format(
            'fp16' if args.fp16 else 'as-parsed', args.gold, args.atol,
            args.rtol, args.atol_frac))

    model_name = tr.get_model_name(test_loc)
    if not model_name:
        print("No .onnx model found in {}".format(test_loc))
        sys.exit(1)
    model_path = os.path.join(test_loc, model_name)
    params = tr.model_parameter_names(model_path)

    shapes, cases = load_cases(test_loc, params,
                               tr.model_output_names(model_path))
    if not cases:
        print("No test_data_set_* or test_data_*.npz found in {}".format(
            test_loc))
        # Nothing to grade against is a gap in the archive, not a model defect.
        if args.emit_metrics:
            write_metrics(args.emit_metrics, 'skipped')
        sys.exit(1)
    for pname, dims in shapes.items():
        print("Input: {}, shape: {}".format(pname, tuple(dims)))
    print()

    if args.emit_driver_args:
        write_driver_args(args.emit_driver_args, shapes, args)

    model = build(model_path, shapes, args)
    # Quantized identically on both sides, so what remains is the target's doing
    # rather than quantization error.
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

        ok, dev, frac = check(gold, tr.run_one_case(model, inputs), args)
        correct += ok
        worst_dev, worst_frac = max(worst_dev, dev), max(worst_frac, frac)
        print("\tCase {}: {}".format(label, "PASSED" if ok else "FAILED"))

    print("\nTest \"{}\" has {} cases:".format(name, len(cases)))
    print("\t Passed: {}".format(correct))
    print("\t Failed: {}".format(len(cases) - correct))
    print("\t Worst deviation: {:.6g} ({:.6g} of tolerance)".format(
        worst_dev, worst_frac))
    if args.emit_metrics:
        write_metrics(args.emit_metrics,
                      'pass' if correct == len(cases) else 'fail', len(cases),
                      correct, worst_dev, worst_frac)
    if correct < len(cases):
        print("{} cases failed!".format(len(cases) - correct))
        sys.exit(1)


if __name__ == "__main__":
    main()
