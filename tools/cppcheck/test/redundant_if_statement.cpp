/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
// Test for RedundantIfStatement check

bool test_redundant_if_return_boolean_1()
{
    bool condition = true;
    // cppcheck-suppress migraphx-RedundantIfStatement
    if(condition)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool test_redundant_if_return_boolean_2()
{
    bool condition = false;
    // cppcheck-suppress migraphx-RedundantIfStatement
    if(condition)
    {
        return false;
    }
    else
    {
        return true;
    }
}

bool test_same_return_values(int x)
{
    // TODO: migraphx-RedundantIfStatement false positive - same value in both branches
    // cppcheck-suppress migraphx-RedundantIfStatement
    if(x > 0)
    {
        return true;
    }
    else
    {
        return true;
    }
}

bool test_complex_boolean_returns(int x)
{
    if(x > 0)
    {
        return x > 5;
    }
    else
    {
        return x < 0;
    }
}

int test_non_boolean_returns(int x)
{
    if(x > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
