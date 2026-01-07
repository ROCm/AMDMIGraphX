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
// Test for EmptyIfStatement check

void test_empty_if_1(int x)
{
    // cppcheck-suppress migraphx-EmptyIfStatement
    if(x > 0) {}
}

void test_empty_if_2(int x)
{
    // cppcheck-suppress migraphx-EmptyIfStatement
    if(x == 5) {}
}

void test_if_with_statement(int x)
{
    if(x > 0)
    {
        x = 10;
    }
    (void)x;
}

void test_if_with_else(int x)
{
    // Empty if body still triggers even with else clause
    // cppcheck-suppress migraphx-EmptyIfStatement
    if(x > 0) {}
    else
    {
        x = 0;
    }
    (void)x;
}

void test_if_with_multiple_statements(int x)
{
    if(x > 0)
    {
        int y = x;
        y     = y * 2;
        (void)y;
    }
}
