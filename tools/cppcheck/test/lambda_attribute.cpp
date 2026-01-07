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
// Test for LambdaAttribute check

#ifndef CPPCHECK
// cppcheck-suppress defineUpperCase
// cppcheck-suppress definePrefix
#define __device__
// cppcheck-suppress defineUpperCase
// cppcheck-suppress definePrefix
#define __host__
#endif

void test_device_attribute_before_params()
{
    int x = 5;
    // cppcheck-suppress migraphx-LambdaAttribute
    auto lambda1 = [] __device__(int a) { return a * 2; };
    (void)x;
    (void)lambda1;
}

void test_host_attribute_before_params()
{
    // cppcheck-suppress migraphx-LambdaAttribute
    auto lambda2 = [] __host__(int a) { return a + 1; };
    (void)lambda2;
}

void test_device_attribute_before_brace()
{
    // cppcheck-suppress migraphx-LambdaAttribute
    auto lambda3 = [] __device__ { return 42; };
    (void)lambda3;
}

void test_attribute_after_params()
{
    // cppcheck-suppress legacyUninitvar
    auto lambda1 = [](int a) __device__ { return a * 2; };
    (void)lambda1;
}

void test_no_attributes()
{
    auto lambda2 = [](int a) { return a + 1; };
    (void)lambda2;
}

void test_capture_list_only()
{
    int x        = 5;
    auto lambda3 = [x](int a) { return a + x; };
    (void)lambda3;
}

void test_attribute_in_correct_position()
{
    // cppcheck-suppress legacyUninitvar
    auto lambda4 = [](int a) __host__ __device__ { return a; };
    (void)lambda4;
}
