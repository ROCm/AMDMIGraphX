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
#include <rocm/utility/move.hpp>
#include <rocm/type_traits.hpp>
#include <migraphx/kernels/test.hpp>

// ---- helpers to inspect value category ----

template <class T>
constexpr bool is_lvalue(T&)
{
    return true;
}

template <class T>
constexpr bool is_lvalue(T&&)
{
    return false;
}

struct movable
{
    int value;
};

// ---- basic move converts lvalue to rvalue ----

TEST_CASE(move_lvalue)
{
    int x = 42;
    EXPECT(not is_lvalue(rocm::move(x)));
}

TEST_CASE(move_preserves_value)
{
    int x = 7;
    EXPECT(rocm::move(x) == 7);
}

// ---- return type is rvalue reference ----

TEST_CASE(move_return_type_from_lvalue)
{
    int x = 1;
    EXPECT(rocm::is_rvalue_reference<decltype(rocm::move(x))>{});
    EXPECT(not rocm::is_lvalue_reference<decltype(rocm::move(x))>{});
}

TEST_CASE(move_return_type_from_rvalue)
{
    EXPECT(rocm::is_rvalue_reference<decltype(rocm::move(42))>{});
}

TEST_CASE(move_return_type_from_lvalue_ref)
{
    int x   = 1;
    int& rx = x;
    EXPECT(rocm::is_rvalue_reference<decltype(rocm::move(rx))>{});
}

// ---- exact type: move(T) yields remove_reference_t<T>&& ----

TEST_CASE(move_exact_type_int)
{
    int x = 0;
    EXPECT(rocm::is_same<decltype(rocm::move(x)), int&&>{});
}

TEST_CASE(move_exact_type_int_ref)
{
    int x   = 0;
    int& rx = x;
    EXPECT(rocm::is_same<decltype(rocm::move(rx)), int&&>{});
}

TEST_CASE(move_exact_type_int_rvalue) { EXPECT(rocm::is_same<decltype(rocm::move(0)), int&&>{}); }

// ---- const lvalue: move yields const T&& ----

TEST_CASE(move_const_lvalue)
{
    const int x = 10;
    EXPECT(not is_lvalue(rocm::move(x)));
    EXPECT(rocm::is_rvalue_reference<decltype(rocm::move(x))>{});
    EXPECT(rocm::is_same<decltype(rocm::move(x)), const int&&>{});
}

// ---- user-defined type ----

TEST_CASE(move_udt)
{
    movable m{5};
    EXPECT(not is_lvalue(rocm::move(m)));
    EXPECT(rocm::move(m).value == 5);
    EXPECT(rocm::is_same<decltype(rocm::move(m)), movable&&>{});
}

TEST_CASE(move_const_udt)
{
    const movable m{9};
    EXPECT(not is_lvalue(rocm::move(m)));
    EXPECT(rocm::move(m).value == 9);
    EXPECT(rocm::is_same<decltype(rocm::move(m)), const movable&&>{});
}

// ---- noexcept ----

TEST_CASE(move_is_noexcept)
{
    int x = 0;
    EXPECT(noexcept(rocm::move(x)));
    EXPECT(noexcept(rocm::move(0)));
    const int cx = 0;
    EXPECT(noexcept(rocm::move(cx)));
}

// ---- move in expression context ----

TEST_CASE(move_in_addition)
{
    int a = 3;
    int b = 4;
    EXPECT(rocm::move(a) + rocm::move(b) == 7);
}

TEST_CASE(move_chained)
{
    int x = 42;
    EXPECT(rocm::move(rocm::move(x)) == 42);
    EXPECT(rocm::is_rvalue_reference<decltype(rocm::move(rocm::move(x)))>{});
}
