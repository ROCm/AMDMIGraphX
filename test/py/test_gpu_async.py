#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
import ctypes

libname = "libamdhip64.so"
hip = ctypes.cdll.LoadLibrary(libname)

def test_conv_relu():
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")
    #print(p)
    #print("Compiling ...")
    p.compile(migraphx.get_target("gpu"))
    #print(p)
    params = {}

    for key, value in p.get_parameter_shapes().items():
        params[key] = migraphx.generate_argument(value)

    stream = ctypes.c_void_p()
    err = ctypes.c_int(hip.hipStreamCreate(ctypes.addressof(stream)))

    err = ctypes.c_int(hip.hipStreamDestroy(stream))
    print(err)

test_conv_relu()
