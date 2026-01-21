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
// Test for RedundantConditionalOperator check

void test_redundant_ternary_true_false(bool condition)
{
    // cppcheck-suppress migraphx-RedundantConditionalOperator
    bool result1 = condition ? true : false;
    (void)result1; // Use variable to avoid warning
}

void test_redundant_ternary_false_true(bool condition)
{
    // cppcheck-suppress migraphx-RedundantConditionalOperator
    bool result2 = condition ? false : true;
    (void)result2; // Use variable to avoid warning
}

void test_redundant_ternary_both_true(bool condition)
{
    // cppcheck-suppress migraphx-RedundantConditionalOperator
    // cppcheck-suppress duplicateValueTernary
    bool result3 = condition ? true : true;
    (void)result3; // Use variable to avoid warning
}

void test_redundant_ternary_both_false(bool condition)
{
    // cppcheck-suppress migraphx-RedundantConditionalOperator
    // cppcheck-suppress duplicateValueTernary
    bool result4 = condition ? false : false;
    (void)result4; // Use variable to avoid warning
}

void test_different_values(bool condition, int x, int y)
{
    int result1 = condition ? x : y;
    (void)result1; // Use variable to avoid warning
}

void test_non_boolean_values(bool condition)
{
    int result2 = condition ? 1 : 0;
    (void)result2; // Use variable to avoid warning
}

void test_expressions(bool condition, int x, int y)
{
    int result3 = condition ? x + 1 : y - 1;
    (void)result3; // Use variable to avoid warning
}

void test_function_calls(bool condition, int x)
{
    int result4 = condition ? x : 42;
    (void)result4; // Use variable to avoid warning
}
