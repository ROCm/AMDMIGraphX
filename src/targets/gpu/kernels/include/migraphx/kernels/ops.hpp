/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
 */
#ifndef MIGRAPHX_GUARD_KERNELS_OPS_HPP
#define MIGRAPHX_GUARD_KERNELS_OPS_HPP

#include <migraphx/kernels/math.hpp>

namespace migraphx {
namespace op {

struct sum
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return x + y;
    }
};

struct product
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return x * y;
    }
};

struct id
{
    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x) const
    {
        return x;
    }
};

struct mean
{
    index_int item_num = 1;
    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x) const
    {
        return x / static_cast<T>(item_num);
    }
};

struct max
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return migraphx::max(x, y);
    }
};

struct min
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return migraphx::min(x, y);
    }
};
} // namespace op

struct lowest
{
    template <class T>
    constexpr operator T() const
    {
        return numeric_lowest<vec_type<T>>();
    }
};

struct highest
{
    template <class T>
    constexpr operator T() const
    {
        return numeric_max<vec_type<T>>();
    }
};
} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_OPS_HPP
