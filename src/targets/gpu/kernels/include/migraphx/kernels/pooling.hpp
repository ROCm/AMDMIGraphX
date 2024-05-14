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
#ifndef MIGRAPHX_GUARD_KERNELS_POOLING_HPP
#define MIGRAPHX_GUARD_KERNELS_POOLING_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/array.hpp>

namespace migraphx {

struct max_pool
{
    MIGRAPHX_DEVICE_CONSTEXPR auto init() const { return lowest{}; }

    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return max(x, y);
    }

    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T final(T x, index_int) const
    {
        return x;
    }
};

struct average_pool
{
    MIGRAPHX_DEVICE_CONSTEXPR auto init() const { return 0.0; }

    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return x + y;
    }

    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T final(T x, index_int y) const
    {
        return (y == 0) ? T{0.0} : T{x / y};
    }
};

struct lpnorm_pool_base
{
};

template <index_int P>
struct lpnorm_pool : lpnorm_pool_base
{
    MIGRAPHX_DEVICE_CONSTEXPR auto init() const { return 0.0; }

    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T apply(T x) const
    {
        if constexpr(P == 0)
            return 1;
        else if constexpr(P == 1)
            return migraphx::abs(x);
        else if constexpr(P == 2)
            return x * x;
        else
            return migraphx::pow(migraphx::abs(x), T(P));
    }

    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return x + apply(y);
    }

    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T final(T x, index_int) const
    {
        if constexpr(P == 0)
            return 1;
        else if constexpr(P == 1)
            return x;
        else if constexpr(P == 2)
            return migraphx::sqrt(x);
        else
            return migraphx::pow(x, 1. / P);
    }
};

template <class Window, class Stride, class Padding>
struct window
{
    Window win      = {};
    Stride stride   = {};
    Padding padding = {};

    using rank = decltype(Window{}.size());

    constexpr auto size() const
    {
        return return_c([] { return Window{}.product(); });
    }

    constexpr auto has_padding() const
    {
        return return_c([] { return Padding{} == 0; });
    }

    template <class Index, class F>
    constexpr void visit(Index i, F f) const
    {
        auto win_start = generate_array<diff_int>(rank{}, [&](auto j) {
            diff_int w   = win[j];
            diff_int dim = i[j];
            MIGRAPHX_ASSERT(w >= 1);
            if(w == 1)
                return dim;
            diff_int s = stride[j];
            diff_int p = padding[j];
            return (dim * s) - p;
        });
        repeat(size(), [&](auto j) { f(win_start + win.multi(j)); });
    }
};

template <class Window, class Stride, class Padding>
constexpr window<Window, Stride, Padding> make_window(Window w, Stride s, Padding p)
{
    return {w, s, p};
}

template <bool IncludePad, class Op, class Window, class Output, class Input>
__device__ void pooling(Op op, Window w, Output output, Input input)
{
    auto idx   = make_index();
    using type = typename Output::type;
    idx.global_stride(output.get_shape().elements(), [&](auto i) {
        auto out_idx        = output.get_shape().multi(i);
        index_int pool_size = w.size();
        type x              = op.init();
        w.visit(out_idx, [&](auto j) {
            if(j < input.get_shape().lens)
            {
                x = op(x, input[j]);
            }
            else
            {
                if constexpr(is_base_of<Op, lpnorm_pool_base>{})
                {
                    x = op(x, op.init());
                }
                if constexpr(not IncludePad and is_same<Op, average_pool>{})
                {
                    pool_size--;
                }
            }
        });
        output[out_idx] = op.final(x, pool_size);
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_POOLING_HPP
