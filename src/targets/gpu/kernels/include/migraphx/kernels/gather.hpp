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
#ifndef MIGRAPHX_GUARD_KERNELS_GATHERND_HPP
#define MIGRAPHX_GUARD_KERNELS_GATHERND_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>

namespace migraphx {

template <class T>
struct gather_settings
{
    T axes{};
};

template <class... Ts>
constexpr gather_settings<Ts...> make_gather_settings(Ts... xs)
{
    return {xs...};
}

template <class T, class U, class V, class Settings>
__device__ void gather(const T& data_t, const U& indices_t, const V& output_t, Settings s)
{
    auto ind           = make_index();
    auto axis          = s.axis;
    auto output_shape  = output_t.get_shape();
    auto indices_shape = indices_t.get_shape();
    auto data_shape    = data_t.get_shape();

    auto axis_dim_size = data_shape.lens().at(axis);

    auto indices_shape_lens = indices_shape.lens;
    auto data_shape_lens    = data_shape.lens;

    const auto* indices_ptr = indices_t.data();
    auto* output_ptr        = output_t.data();

    ind.global_stride(output_shape.elements(), [&](auto i) {

    });
}

} // namespace migraphx
#endif
