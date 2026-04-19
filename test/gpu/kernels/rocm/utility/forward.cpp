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
#include <rocm/utility/forward.hpp>
#include <rocm/type_traits.hpp>
#include <migraphx/kernels/test.hpp>

// ---- helpers to inspect value category of a forwarded expression ----

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

// ---- forward lvalue ref (first overload: remove_reference_t<T>& -> T&&) ----

TEST_CASE(forward_lvalue_as_lvalue)
{
    int x = 42;
    // T = int&, so T&& collapses to int& (lvalue)
    EXPECT(is_lvalue(rocm::forward<int&>(x)));
}

TEST_CASE(forward_lvalue_as_rvalue)
{
    int x = 42;
    // T = int, so T&& is int&& (rvalue)
    EXPECT(not is_lvalue(rocm::forward<int>(x)));
}

TEST_CASE(forward_lvalue_preserves_value)
{
    int x = 7;
    EXPECT(rocm::forward<int&>(x) == 7);
    EXPECT(rocm::forward<int>(x) == 7);
}

// ---- forward rvalue ref (second overload: remove_reference_t<T>&& -> T&&) ----

TEST_CASE(forward_rvalue_as_rvalue)
{
    // T = int, forwarding a prvalue through the rvalue overload
    EXPECT(not is_lvalue(rocm::forward<int>(42)));
}

TEST_CASE(forward_rvalue_preserves_value) { EXPECT(rocm::forward<int>(99) == 99); }

// ---- return type checks ----

TEST_CASE(forward_return_type_lvalue_ref)
{
    // forward<int&>(...) should return int&
    int x = 1;
    EXPECT(rocm::is_lvalue_reference<decltype(rocm::forward<int&>(x))>{});
    EXPECT(not rocm::is_rvalue_reference<decltype(rocm::forward<int&>(x))>{});
}

TEST_CASE(forward_return_type_rvalue_ref)
{
    // forward<int>(...) should return int&&
    int x = 1;
    EXPECT(rocm::is_rvalue_reference<decltype(rocm::forward<int>(x))>{});
    EXPECT(not rocm::is_lvalue_reference<decltype(rocm::forward<int>(x))>{});
}

TEST_CASE(forward_return_type_from_rvalue)
{
    // forward<int>(rvalue) should return int&&
    EXPECT(rocm::is_rvalue_reference<decltype(rocm::forward<int>(42))>{});
}

// ---- const qualifiers ----

TEST_CASE(forward_const_lvalue)
{
    const int x = 10;
    EXPECT(rocm::forward<const int&>(x) == 10);
    EXPECT(rocm::is_lvalue_reference<decltype(rocm::forward<const int&>(x))>{});
}

TEST_CASE(forward_const_rvalue)
{
    const int x = 10;
    EXPECT(rocm::is_rvalue_reference<decltype(rocm::forward<const int>(x))>{});
}

// ---- user-defined type ----

TEST_CASE(forward_udt_lvalue)
{
    movable m{5};
    EXPECT(is_lvalue(rocm::forward<movable&>(m)));
    EXPECT(rocm::forward<movable&>(m).value == 5);
}

TEST_CASE(forward_udt_rvalue)
{
    movable m{5};
    EXPECT(not is_lvalue(rocm::forward<movable>(m)));
    EXPECT(rocm::forward<movable>(m).value == 5);
}

// ---- perfect forwarding in a generic context ----

template <class T>
constexpr int perfect_forward(T&& x)
{
    return rocm::forward<T>(x);
}

TEST_CASE(perfect_forwarding_lvalue)
{
    int x = 3;
    EXPECT(perfect_forward(x) == 3);
}

TEST_CASE(perfect_forwarding_rvalue) { EXPECT(perfect_forward(8) == 8); }

// ---- noexcept ----

TEST_CASE(forward_is_noexcept)
{
    int x = 0;
    EXPECT(noexcept(rocm::forward<int&>(x)));
    EXPECT(noexcept(rocm::forward<int>(x)));
    EXPECT(noexcept(rocm::forward<int>(0)));
}
