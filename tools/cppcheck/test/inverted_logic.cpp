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
// Test for InvertedLogic rule from rules.xml

void test_inverted_logic_with_if_else_1(int x, int y)
{
    // cppcheck-suppress InvertedLogic
    if(x != y)
    {
        x = 0;
    }
    else
    {
        x = 1;
    }
    (void)x; // Use variable to avoid warning
}

void test_inverted_logic_with_negation(bool flag)
{
    int x = 5;
    // cppcheck-suppress InvertedLogic
    if(not flag)
    {
        x = 2;
    }
    else
    {
        x = 3;
    }
    (void)x; // Use variable to avoid warning
}

void test_inverted_logic_ternary_1(int x, int y)
{
    // cppcheck-suppress InvertedLogic
    int result1 = (x != y) ? 0 : 1;
    (void)result1; // Use variable to avoid warning
}

void test_inverted_logic_ternary_2(bool flag)
{
    // cppcheck-suppress InvertedLogic
    int result2 = (not flag) ? 0 : 1;
    (void)result2; // Use variable to avoid warning
}

void test_positive_logic_equality(int x, int y)
{
    if(x == y)
    {
        x = 0;
    }
    else
    {
        x = 1;
    }
    (void)x; // Use variable to avoid warning
}

void test_positive_logic_boolean(bool flag)
{
    int x = 5;
    if(flag)
    {
        x = 2;
    }
    else
    {
        x = 3;
    }
    (void)x; // Use variable to avoid warning
}

void test_other_comparisons(int x, int y)
{
    if(x > y)
    {
        x = 4;
    }
    else
    {
        x = 5;
    }
    (void)x; // Use variable to avoid warning
}

void test_positive_ternary(int x, int y, bool flag)
{
    int result1 = (x == y) ? 0 : 1;
    int result2 = flag ? 0 : 1;
    (void)result1; // Use variables to avoid warnings
    (void)result2;
}
