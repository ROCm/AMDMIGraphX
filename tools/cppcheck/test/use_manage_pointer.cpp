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
// Test for UseManagePointer check
// Note: This test file defines mock functions to avoid actual memory operations.
// The mock functions are defined before use to satisfy cppcheck's analysis.

#include <cstdio>
#include <cstdlib>

// Forward declarations and mocks for HIP functions
// TODO: migraphx-UseManagePointer false positive - forward declarations trigger the check
// cppcheck-suppress migraphx-UseManagePointer
int hipFree(void*);
// cppcheck-suppress migraphx-UseManagePointer
int hipHostFree(void*);
// cppcheck-suppress migraphx-UseManagePointer
int hipStreamDestroy(void*);
// cppcheck-suppress migraphx-UseManagePointer
int hipEventDestroy(void*);
int hipMalloc(void**, unsigned long);

void test_fclose(FILE* file)
{
    // cppcheck-suppress migraphx-UseManagePointer
    fclose(file);
}

void test_free(void* ptr)
{
    // cppcheck-suppress migraphx-UseManagePointer
    free(ptr);
}

void test_hip_functions(void* gpu_ptr)
{
    // cppcheck-suppress migraphx-UseManagePointer
    hipFree(gpu_ptr);

    // cppcheck-suppress migraphx-UseManagePointer
    hipHostFree(gpu_ptr);

    // cppcheck-suppress migraphx-UseManagePointer
    hipStreamDestroy(gpu_ptr);

    // cppcheck-suppress migraphx-UseManagePointer
    hipEventDestroy(gpu_ptr);
}

void test_negative_cases()
{
    // Should not trigger: other functions
    int x    = 5;
    int* ptr = &x;
    (void)x; // Use variables to avoid warnings
    (void)ptr;

    // Should not trigger: allocation functions (not deallocation)
    hipMalloc(nullptr, 100);
}
