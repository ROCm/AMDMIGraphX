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
// Test for EmptySwitchStatement check

void test_empty_switch()
{
    int x = 5;
    // cppcheck-suppress migraphx-EmptySwitchStatement
    switch(x)
    {
    }
}

void test_empty_switch_with_different_variable()
{
    int y = 10;
    // cppcheck-suppress migraphx-EmptySwitchStatement
    switch(y)
    {
    }
}

void test_switch_with_cases()
{
    int x = 5;
    switch(x)
    {
    case 1: x = 10; break;
    case 2: x = 20; break;
    default: x = 0; break;
    }
    (void)x; // Use variable to avoid warning
}

void test_switch_with_single_case()
{
    int x = 5;
    switch(x)
    {
    case 1: x = 100; break;
    }
    (void)x;
}
