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
// Test for MultipleUnaryOperator check

void test_double_negative()
{
    int x = 5;
    // cppcheck-suppress migraphx-MultipleUnaryOperator
    int y = -(-x);
    (void)y; // Use variable to avoid warning
}

void test_double_positive()
{
    int x = 5;
    // TODO: migraphx-MultipleUnaryOperator false negative - double positive not detected
    int z = +(+x);
    (void)z; // Use variable to avoid warning
}

void test_double_not()
{
    bool b = true;
    // cppcheck-suppress migraphx-MultipleUnaryOperator
    bool result = not(not b);
    (void)result; // Use variable to avoid warning
}

void test_multiple_bitwise_not()
{
    unsigned int u = 0xFF;
    // cppcheck-suppress migraphx-MultipleUnaryOperator
    unsigned int inverted = ~~u;
    (void)inverted; // Use variable to avoid warning
}

void test_single_unary_operators()
{
    int x = 5;
    int y = -x;
    int z = +x;
    (void)y; // Use variables to avoid warnings
    (void)z;
}

void test_single_logical_not()
{
    bool b      = true;
    bool result = not b;
    (void)result; // Use variable to avoid warning
}

void test_binary_operators()
{
    int x  = 5;
    int y  = 10;
    int z  = 15;
    int a  = x + y;
    int b2 = x - z;
    (void)a; // Use variables to avoid warnings
    (void)b2;
}

void test_increment_decrement()
{
    int x = 5;
    x++;
    ++x;
    x--;
    --x;
    (void)x;
}
