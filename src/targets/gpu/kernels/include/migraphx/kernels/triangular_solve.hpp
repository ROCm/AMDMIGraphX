/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef MIGRAPHX_GUARD_KERNELS_TRIANGULAR_SOLVE_HPP
#define MIGRAPHX_GUARD_KERNELS_TRIANGULAR_SOLVE_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/uninitialized_buffer.hpp>

namespace migraphx {

// The shape of x with the row axis collapsed: one lane per (batch, column).
// The lens are type-encoded, so the axis is dropped at the type level.
template <class Shape>
constexpr auto solve_lanes(Shape s)
{
    auto lens = transform_i(
        s.lens, [](auto x, auto i) { return i == Shape{}.lens.size() - 2 ? 1 : x; });
    return make_shape(lens, s.strides);
}

// Substitution over the trailing (n, n) x (n, k) matrices, one workgroup per
// batch problem: A is staged in LDS once and each lane walks the rows of a
// column with the column kept in registers. The loops are fully unrolled so
// the column stays in registers, which bounds n.
template <bool Lower, bool UnitDiagonal, class A, class B, class X>
__device__ void triangular_solve_registers(A a, B b, X x)
{
    using type           = typename X::type;
    constexpr auto xs    = get_shape_c<X>{};
    constexpr auto ndim  = xs.lens.size();
    constexpr auto n     = xs.lens[ndim - 2];
    constexpr auto k     = xs.lens[ndim - 1];
    constexpr auto lanes = solve_lanes(xs);
    constexpr auto mat   = make_shape(index_ints<n, n>{});
    __shared__ uninitialized_buffer<type, mat.elements()> lds;
    auto idx  = make_index();
    auto base = lanes.multi(idx.group * k); // (batch..., 0, 0) of this problem
    idx.local_stride(mat.elements(), [&](auto i) {
        auto rc      = mat.multi(i);
        auto ai      = base;
        ai[ndim - 2] = rc[0];
        ai[ndim - 1] = rc[1];
        lds[i]       = a[ai];
    });
    __syncthreads();
    idx.local_stride(k, [&](auto col) {
        auto xi      = base;
        xi[ndim - 1] = col;
        array<type, n> column;
        repeat_c<n>([&](auto step) {
            constexpr index_int t = Lower ? step : n - 1 - step;
            xi[ndim - 2]          = t;
            auto acc              = b[xi];
            repeat(step, [&](auto j) {
                constexpr index_int s = Lower ? j : n - 1 - j;
                acc -= lds[mat.index({t, s})] * column[s];
            });
            if constexpr(not UnitDiagonal)
                acc /= lds[mat.index({t, t})];
            column[t] = acc;
            x[xi]     = acc;
        });
    });
}

// Large n: one lane per (batch, column) walks the rows in solve order and
// reads the rows it has already written back from x
template <bool Lower, bool UnitDiagonal, class A, class B, class X>
__device__ void triangular_solve_global(A a, B b, X x)
{
    constexpr auto xs    = get_shape_c<X>{};
    constexpr auto ndim  = xs.lens.size();
    constexpr auto n     = xs.lens[ndim - 2];
    constexpr auto lanes = solve_lanes(xs);
    auto idx             = make_index();
    idx.global_stride(lanes.elements(), [&](auto i) {
        auto xi = lanes.multi(i); // (batch..., 0, col)
        auto ai = xi;             // (batch..., t, s)
        for(index_int step = 0; step < n; step++)
        {
            const index_int t = Lower ? step : n - 1 - step;
            xi[ndim - 2]      = t;
            ai[ndim - 2]      = t;
            auto acc          = b[xi];
            for(index_int j = 0; j < step; j++)
            {
                const index_int s = Lower ? j : n - 1 - j;
                ai[ndim - 1]      = s;
                auto xr           = xi;
                xr[ndim - 2]      = s;
                acc -= a[ai] * x[xr];
            }
            if constexpr(not UnitDiagonal)
            {
                ai[ndim - 1] = t;
                acc /= a[ai];
            }
            x[xi] = acc;
        }
    });
}

template <bool Lower, bool UnitDiagonal, bool Registers, class A, class B, class X>
__device__ void triangular_solve(A a, B b, X x)
{
    if constexpr(Registers)
        triangular_solve_registers<Lower, UnitDiagonal>(a, b, x);
    else
        triangular_solve_global<Lower, UnitDiagonal>(a, b, x);
}

} // namespace migraphx
#endif
