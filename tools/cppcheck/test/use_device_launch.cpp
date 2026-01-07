/*
* The MIT License (MIT)
*
* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
*/
// Test for UseDeviceLaunch check

#include <cstddef>
#include <cstdio>
#include <cstdlib>

// Mock functions and types for compilation
typedef int dim3;
typedef int hipMemcpyKind;
const hipMemcpyKind hipMemcpyDeviceToHost = 0;

void kernel() {}
void another_kernel() {}
int hipMalloc(void**, size_t) { return 0; }
int hipMemcpy(void*, const void*, size_t, hipMemcpyKind) { return 0; }
// TODO: migraphx-UseDeviceLaunch false positive - function definition triggers the check
// cppcheck-suppress migraphx-UseDeviceLaunch
int hipLaunchKernelGGL(int, int, int, int, int, ...) { return 0; }
void myLaunchKernel(int) {}

void test_positive_cases()
{
    // Should trigger: hipLaunchKernelGGL usage
    int gridSize  = 1;
    int blockSize = 256;
    int args      = 0;

    // cppcheck-suppress migraphx-UseDeviceLaunch
    hipLaunchKernelGGL(0, gridSize, blockSize, 0, 0, args);

    // Should trigger: another hipLaunchKernelGGL call
    // cppcheck-suppress migraphx-UseDeviceLaunch
    hipLaunchKernelGGL(1, dim3(1), dim3(256), 0, 0);
}

void test_negative_cases()
{
    // Should not trigger: using device::launch instead
    // device::launch(kernel, gridSize, blockSize, args);

    // Should not trigger: other HIP functions
    hipMalloc(nullptr, 1024);
    hipMemcpy(nullptr, nullptr, 1024, hipMemcpyDeviceToHost);

    // Should not trigger: regular function calls
    printf("Hello world\n");

    // Should not trigger: custom functions with similar names
    int args = 0;
    myLaunchKernel(args);
}
