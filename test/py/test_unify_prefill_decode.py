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


def _options():
    return {
        "use_symbolic_shapes": True,
        "unify_prefill_decode": True,
        "dim_params": {
            "sequence_length": migraphx.shape.dynamic_dimension(1, 4)
        },
    }


def _is_unified(program):
    return any(ins.name() == "select_module"
               for ins in program.get_main_module())


def test_parse_onnx_unify_prefill_decode():
    program = migraphx.parse_onnx("unify_prefill_decode_test.onnx",
                                  **_options())
    assert _is_unified(program)
    assert program.get_parameter_shapes()["x"].dyn_dims()[1].is_symbolic()


def test_parse_onnx_buffer_unify_prefill_decode():
    with open("unify_prefill_decode_test.onnx", "rb") as model:
        program = migraphx.parse_onnx_buffer(model.read(), **_options())
    assert _is_unified(program)


if __name__ == "__main__":
    test_parse_onnx_unify_prefill_decode()
    test_parse_onnx_buffer_unify_prefill_decode()
