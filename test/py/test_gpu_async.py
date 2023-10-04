#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
import os
import glob

def test_conv_relu():

    library = "libamdhip64.so"

    rocm_path_env_var = "rocm_path"

    rocm_path_var = os.getenv(rocm_path_env_var, default="/opt/rocm")

    library_file = os.path.join(rocm_path_var, "lib", library)

    # Check if the library file exists at the specified path
    if os.path.exists(library_file):
        library = library_file
    else:
        # Pattern match path for rocm paths: /opt/rocm-*
        rocm_path_pattern = "/opt/rocm-*/lib/libamdhip64.so"
        matching_libraries = glob.glob(rocm_path_pattern)

        if matching_libraries:
            library = matching_libraries[0]

    hip = ctypes.cdll.LoadLibrary(library)

    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")
    print(p)
    print("Compiling ...")
    # Need to have offload_copy = False to avoid syncs() back to the host device
    p.compile(migraphx.get_target("gpu"), offload_copy=False)
    print(p)
    params = {}

    # Using default value in api for hipSuccess which is always 0
    hipSuccess = ctypes.c_long(0)

    # Alloc a stream
    stream = ctypes.c_void_p()

    err = ctypes.c_long(
        hip.hipStreamCreateWithFlags(ctypes.byref(stream), ctypes.c_uint(0)))

    if err.value != hipSuccess.value:
        print("FAILED hipStreamCreate")
        return err

    # Use to_gpu to push generated argument to the GPU before we perform a run
    for key, value in p.get_parameter_shapes().items():
        params[key] = migraphx.to_gpu(migraphx.generate_argument(value))

    result = migraphx.from_gpu(
        p.run_async(params, stream.value, "ihipStream_t")[-1])

    # Wait for all commands in stream to complete
    err = ctypes.c_long(hip.hipStreamSynchronize(stream))
    if err.value != hipSuccess.value:
        print("FAILED: hipStreamSyncronize")
        return err

    # Cleanup Stream
    err = ctypes.c_long(hip.hipStreamDestroy(stream))
    if err.value != hipSuccess.value:
        print("FAILED: hipStreamDestroy")
        return err

    print(result)


test_conv_relu()
