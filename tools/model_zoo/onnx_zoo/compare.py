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
"""Compare the outputs of one test case against its golden outputs.

allclose() and rms() take the golden and the actual outputs in model output
order and return (passed, worst error, worst error as a fraction of the
tolerance). Both need only numpy.
"""
import numpy as np


def deviation(gold, actual, atol, rtol):
    """Return maximum absolute deviation and error-to-tolerance ratio."""
    if gold.size == 0:
        return 0.0, 0.0

    dev = np.abs(gold.astype(np.float64) - actual.astype(np.float64))
    # Convert undefined deviations to infinity before computing maxima.
    dev = np.where(np.isnan(dev), np.inf, dev)
    tol = atol + rtol * np.abs(gold.astype(np.float64))
    frac = np.divide(dev,
                     tol,
                     out=np.where(dev > 0, np.inf, 0.0),
                     where=tol > 0)

    return float(dev.max()), float(frac.max())


def rms_error(gold, actual):
    """RMS of the difference divided by the largest magnitude in either
    tensor: the metric of verify::rms_range, used by migraphx-driver verify."""
    gold = np.asarray(gold, dtype=np.float64).ravel()
    actual = np.asarray(actual, dtype=np.float64).ravel()
    if gold.size != actual.size:
        return float('inf')
    if gold.size == 0:
        return 0.0

    with np.errstate(all='ignore'):
        diff = np.sqrt(np.square(gold - actual).sum())
        mag = max(np.abs(gold).max(), np.abs(actual).max())
        mag = max(mag, np.finfo(np.float64).tiny)
        error = diff / (np.sqrt(gold.size) * mag)

    return float('inf') if np.isnan(error) else float(error)


def allclose(gold_outputs, outputs, atol=1e-3, rtol=1e-3, atol_frac=0.0):
    """Pass when every element of every output is within
    atol + rtol * abs(gold). atol_frac raises atol to that fraction of
    max(abs(gold)) of the output."""
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
        tol = max(atol, atol_frac * rng)
        dev, frac = deviation(gold, actual, tol, rtol)
        worst_dev, worst_frac = max(worst_dev, dev), max(worst_frac, frac)
        if not np.allclose(actual, gold, rtol, tol):
            print(
                "\nOutput {} is incorrect ... max abs diff {:.6g} vs atol {:.6g}, "
                "rtol {:.6g}, expected range {:.6g}".format(
                    i, dev, tol, rtol, rng))
            print("Expected value: \n{}\nActual value: \n{}\n".format(
                gold, actual))
            ok = False

    return ok, worst_dev, worst_frac


def rms(gold_outputs, outputs, tol=1e-3, names=None):
    """Pass when the RMS error of every output is at most tol. Prints the RMS
    error and PASSED or FAILED for each output."""
    if len(gold_outputs) != len(outputs):
        print("Expected {} outputs, got {}".format(len(gold_outputs),
                                                   len(outputs)))
        return False, float('inf'), float('inf')

    ok = True
    worst = 0.0
    for i, (gold, actual) in enumerate(zip(gold_outputs, outputs)):
        gold, actual = np.asarray(gold), np.asarray(actual)
        label = 'Output {}{}'.format(i,
                                     ' ({})'.format(names[i]) if names else '')
        if gold.shape != actual.shape:
            print("{} shape mismatch: expected {}, got {}: FAILED".format(
                label, gold.shape, actual.shape))
            ok, worst = False, float('inf')
            continue

        error = rms_error(gold, actual)
        worst = max(worst, error)
        passed = error <= tol
        ok = ok and passed
        print("{}: RMS error {:.6g}, tolerance {:.6g}: {}".format(
            label, error, tol, "PASSED" if passed else "FAILED"))

    if tol > 0:
        frac = worst / tol
    else:
        frac = float('inf') if worst > 0 else 0.0

    return ok, worst, frac
