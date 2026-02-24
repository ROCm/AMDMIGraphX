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
// Test for defineUpperCase and definePrefix rules from rules.xml

// cppcheck-suppress defineUpperCase
// cppcheck-suppress definePrefix
#define myMacro 42

// cppcheck-suppress defineUpperCase
// cppcheck-suppress definePrefix
#define TestMacro 100

// cppcheck-suppress definePrefix
#define MY_CONSTANT 5

// cppcheck-suppress definePrefix
#define OTHER_PREFIX_VALUE 10

// cppcheck-suppress defineUpperCase
// cppcheck-suppress definePrefix
#define badMacro 15

#define MIGRAPHX_CORRECT_MACRO 20
#define MIGRAPHX_ANOTHER_MACRO 30

void test_macro_not_uppercase_1()
{
    int x = myMacro;
    (void)x;
}

void test_macro_not_uppercase_2()
{
    int y = TestMacro;
    (void)y;
}

void test_macro_missing_prefix_1()
{
    int z = MY_CONSTANT;
    (void)z;
}

void test_macro_missing_prefix_2()
{
    int w = OTHER_PREFIX_VALUE;
    (void)w;
}

void test_macro_both_issues()
{
    int v = badMacro;
    (void)v;
}

void test_correct_macros()
{
    int correct = MIGRAPHX_CORRECT_MACRO;
    int another = MIGRAPHX_ANOTHER_MACRO;
    (void)correct;
    (void)another;
}

void test_standard_library_macros()
{
#ifndef NULL
// cppcheck-suppress definePrefix
#define NULL nullptr
#endif
}
