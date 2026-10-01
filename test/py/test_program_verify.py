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
import migraphx


def make_program():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(type="float", lens=[2, 2])
    x = mm.add_parameter("x", s)
    out = mm.add_instruction(migraphx.op("relu"), [x])
    mm.add_return([out])
    return p, s


def test_program_verify_outputs():
    p, _ = make_program()
    result = p.verify(migraphx.get_target("ref"))
    assert result.passed()
    assert result.mode == migraphx.program_verify_mode.outputs
    assert result.failure_step is None
    assert len(result.results) == 1
    assert result.failures() == []
    layer = result.results[0]
    assert layer.name == ""
    assert layer.operator == "@return"
    assert layer.message == ""
    assert layer.index == 0
    assert layer.rms_error == 0
    assert layer.passed
    assert not layer.exception


def test_program_verify_modes_and_options():
    p, s = make_program()
    target = migraphx.get_target("ref")
    params = {"x": migraphx.generate_argument(s)}
    options = migraphx.program_verify_options()
    options.rms_tol = 1e-3
    options.atol = 1e-3
    options.rtol = 1e-3
    options.precision = migraphx.program_verify_precision.fp32
    options.ref_use_double = False
    options.name = "python"
    options.offload_copy = False
    options.fast_math = True
    options.exhaustive_tune = False
    options.compile_mode = migraphx.compile_modes.balanced

    output = p.verify(target, migraphx.program_verify_mode.outputs, params,
                      options)
    assert output.passed()
    assert output.results[0].name == "python"
    assert p.verify(target, migraphx.program_verify_mode.instructions).passed()
    assert p.verify(target, migraphx.program_verify_mode.reduce,
                    params).passed()
    assert p.verify(target, migraphx.program_verify_mode.bisect,
                    params).passed()
    assert p.verify(target, migraphx.program_verify_mode.layerwise,
                    params).passed()
    options.set_backend_option("verify_test", True)


test_program_verify_outputs()
test_program_verify_modes_and_options()
