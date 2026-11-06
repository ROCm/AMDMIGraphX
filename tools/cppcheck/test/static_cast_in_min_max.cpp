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
// Test for StaticCastInMinMax check

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

// Positive cases: a static_cast on an argument of std::min/std::max should be
// replaced with an explicit template argument on the call itself.

std::uint64_t test_min_second_arg(std::uint64_t x, int y)
{
    // cppcheck-suppress migraphx-StaticCastInMinMax
    return std::min(x, static_cast<std::uint64_t>(y));
}

std::uint64_t test_max_first_arg(int y, std::uint64_t x)
{
    // cppcheck-suppress migraphx-StaticCastInMinMax
    return std::max(static_cast<std::uint64_t>(y), x);
}

std::size_t test_min_size_t(std::size_t n, int i)
{
    // cppcheck-suppress migraphx-StaticCastInMinMax
    return std::min(n, static_cast<std::size_t>(i));
}

unsigned int test_max_fundamental_type(unsigned int a, int b)
{
    // cppcheck-suppress migraphx-StaticCastInMinMax
    return std::max(a, static_cast<unsigned int>(b));
}

std::uint64_t test_min_both_args_cast(int a, int b)
{
    // cppcheck-suppress migraphx-StaticCastInMinMax
    return std::min(static_cast<std::uint64_t>(a), static_cast<std::uint64_t>(b));
}

// Negative cases: nothing should be reported below.

// Already uses an explicit template argument, which is what we recommend.
std::uint64_t test_already_templated(std::uint64_t x, int y)
{
    return std::min<std::uint64_t>(x, y);
}

// No cast at all.
std::uint64_t test_no_cast(std::uint64_t x, std::uint64_t y) { return std::max(x, y); }

// The cast is nested in a larger expression, not a direct argument, so the
// template-argument form would not be equivalent.
std::uint64_t test_nested_cast(std::uint64_t x, int y)
{
    return std::min(x, static_cast<std::uint64_t>(y) + 1);
}

// A member function named max, not std::max.
struct widget
{
    int max(int v) const;
};

int test_member_function(const widget& w, int b) { return w.max(static_cast<int>(b)); }

// std::numeric_limits<T>::max() takes no arguments.
int test_numeric_limits() { return std::numeric_limits<int>::max(); }

// A free function that merely happens to be named min.
int my_min(int a, int b);

int test_unqualified_min(int a, int b) { return my_min(a, static_cast<int>(b)); }
