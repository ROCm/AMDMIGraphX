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
// Test for UnusedDeref rule from rules.xml

void test_redundant_deref_with_increment()
{
    int x = 5;
    // cppcheck-suppress UnusedDeref
    int* ptr = &x;
    // cppcheck-suppress clarifyStatement
    *ptr++;
}

void test_redundant_deref_with_decrement()
{
    int x = 5;
    // cppcheck-suppress UnusedDeref
    int* ptr = &x;
    // cppcheck-suppress clarifyStatement
    *ptr--;
}

void test_redundant_deref_with_increment_variant()
{
    int x = 5;
    // cppcheck-suppress UnusedDeref
    int* ptr = &x;
    // cppcheck-suppress clarifyStatement
    *ptr++;
}

void test_proper_dereference_should_not_trigger()
{
    // Should not trigger: proper dereference for reading value
    int x    = 5;
    int* ptr = &x;
    int y    = *ptr;
    (void)y; // Suppress unused variable warning
}

void test_assignment_through_dereference_should_not_trigger()
{
    // Should not trigger: assignment through dereference
    int x    = 5;
    int* ptr = &x;
    *ptr     = 10;
}

void test_increment_without_dereference_should_not_trigger()
{
    // Should not trigger: increment without dereference
    int x    = 5;
    int* ptr = &x;
    ptr++;
    (void)ptr;
}

void test_dereference_in_expression_should_not_trigger()
{
    // Should not trigger: dereference used in expression
    int x    = 5;
    int* ptr = &x;
    int z    = (*ptr) + 1;
    (void)z; // Suppress unused variable warning
}
