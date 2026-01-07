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
// Test for RedundantCast check

void test_redundant_auto_cast_1()
{
    // cppcheck-suppress migraphx-RedundantCast
    auto x = static_cast<int>(5);
    (void)x;
}

void test_redundant_auto_cast_2()
{
    int y = 10;
    // cppcheck-suppress migraphx-RedundantCast
    auto z = static_cast<int>(y);
    (void)y;
    (void)z;
}

void test_redundant_auto_cast_3()
{
    // cppcheck-suppress migraphx-RedundantCast
    auto w = static_cast<double>(3.14);
    (void)w;
}

void test_explicit_type_cast()
{
    // cppcheck-suppress migraphx-RedundantCast
    int x = static_cast<int>(5.5);
    (void)x;
}

void test_cast_to_different_type()
{
    int y = 10;
    // cppcheck-suppress migraphx-RedundantCast
    double z = static_cast<double>(y);
    (void)z;
}

void test_cast_with_const()
{
    const int a = 5;
    // cppcheck-suppress migraphx-RedundantCast
    auto b = static_cast<const int&>(a);
    (void)b;
}

void test_no_cast()
{
    auto c = 42;
    int d  = 5;
    (void)c;
    (void)d;
}
