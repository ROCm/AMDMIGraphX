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
// Test for RedundantStaticCastDecl check

#include <cstddef>
#include <cstdint>

int src();
long lsrc();
void sink_u(unsigned);
void sink_l(long);
void sink_s(std::size_t);
void sink_ul(unsigned long);
void observe(int*);

// Positive cases: every read of the variable is a static_cast to the same type
// that differs from its declared type, so it should just be declared as that
// type.

void test_two_reads()
{
    // cppcheck-suppress migraphx-RedundantStaticCastDecl
    int x = src();
    sink_u(static_cast<unsigned>(x));
    sink_u(static_cast<unsigned>(x));
}

unsigned test_single_read()
{
    // cppcheck-suppress migraphx-RedundantStaticCastDecl
    int a = src();
    return static_cast<unsigned>(a);
}

void test_widen_to_size_t()
{
    // cppcheck-suppress migraphx-RedundantStaticCastDecl
    long n = lsrc();
    sink_s(static_cast<std::size_t>(n));
    sink_s(static_cast<std::size_t>(n));
}

long test_widen_signed()
{
    // cppcheck-suppress migraphx-RedundantStaticCastDecl
    int w = src();
    return static_cast<long>(w);
}

// Negative cases: nothing should be reported below.

// The variable also has a plain (uncast) read.
void test_plain_use()
{
    int y = src();
    sink_u(static_cast<unsigned>(y));
    sink_l(y + 1);
}

// The variable is cast to two different types.
void test_different_types()
{
    int w = src();
    sink_u(static_cast<unsigned>(w));
    sink_ul(static_cast<unsigned long>(w));
}

// The cast is to the variable's own type, so it is a same-type cast rather than
// a declaration change.
int test_same_type()
{
    int b = src();
    return static_cast<int>(b);
}

// The address of the variable is taken, so the type cannot simply change.
void test_address_taken()
{
    int c = src();
    observe(&c);
    sink_u(static_cast<unsigned>(c));
}

// A compound assignment reads the variable in its declared type.
unsigned test_compound_assign()
{
    int g = src();
    g += 1;
    return static_cast<unsigned>(g);
}

// A function parameter is not a local declaration.
unsigned test_parameter(int p) { return static_cast<unsigned>(p); }

// A reference cannot be redeclared as a value type.
unsigned test_reference(const int& base)
{
    const int& r = base;
    return static_cast<unsigned>(r);
}
