/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <type_traits_test.hpp>

template <class T>
struct check : rocm::bool_constant<false>
{
};

template <>
struct check<long> : rocm::bool_constant<true>
{
};

struct construct
{
    template <class T>
    constexpr construct(T, typename rocm::enable_if<check<T>{}>::type* = nullptr) : v(true)
    {
    }
    template <class T>
    constexpr construct(T, typename rocm::enable_if<not check<T>{}>::type* = nullptr) : v(false)
    {
    }
    constexpr bool value() const { return v; }

    private:
    bool v;
};

template <class T, class E = void>
struct specialize;

template <class T>
struct specialize<T, typename rocm::enable_if<check<T>{}>::type> : rocm::bool_constant<true>
{
};

template <class T>
struct specialize<T, typename rocm::enable_if<not check<T>{}>::type> : rocm::bool_constant<false>
{
};

template <class T>
constexpr typename rocm::enable_if<check<T>{}, bool>::type returns(T)
{
    return true;
}

template <class T>
constexpr typename rocm::enable_if<not check<T>{}, bool>::type returns(T)
{
    return false;
}

template <class T>
constexpr rocm::enable_if_t<check<T>{}, bool> alias(T)
{
    return true;
}

template <class T>
constexpr rocm::enable_if_t<not check<T>{}, bool> alias(T)
{
    return false;
}

ROCM_DUAL_TEST_CASE()
{
    static_assert(not construct(1).value());
    static_assert(construct(1L).value());
    static_assert(not specialize<int>{});
    static_assert(specialize<long>{});
    static_assert(not returns(1));
    static_assert(returns(1L));
    static_assert(not alias(1));
    static_assert(alias(1L));
}
