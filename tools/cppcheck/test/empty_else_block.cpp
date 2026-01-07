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
// Test for EmptyElseBlock check

void test_empty_else_1(int x)
{
    if(x > 0)
    {
        x = 10;
    }
    // cppcheck-suppress migraphx-EmptyElseBlock
    else {}
    (void)x;
}

void test_empty_else_2(int x)
{
    if(x < 0)
    {
        x = -x;
    }
    // cppcheck-suppress migraphx-EmptyElseBlock
    else {}
    (void)x;
}

void test_else_with_statement(int x)
{
    if(x > 0)
    {
        x = 10;
    }
    else
    {
        x = 0;
    }
    (void)x;
}

void test_no_else_block(int x)
{
    if(x > 0)
    {
        x = 10;
    }
    (void)x;
}

void test_else_if_chain(int x)
{
    if(x > 0)
    {
        x = 10;
    }
    else if(x < 0)
    {
        x = 0;
    }
    (void)x;
}
