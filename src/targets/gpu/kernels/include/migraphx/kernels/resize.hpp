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
#ifndef MIGRAPHX_GUARD_KERNELS_RESIZE_HPP
#define MIGRAPHX_GUARD_KERNELS_RESIZE_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/vec.hpp>

namespace migraphx {

struct resize_interp_params
{
    index_int i0{};
    index_int i1{};
    float weight{};
};

MIGRAPHX_DEVICE_CONSTEXPR resize_interp_params
compute_interp_params_asymmetric(index_int in_len, index_int out_idx, float scale)
{
    if(in_len <= 1 or scale == 0.0f)
        return {0, 0, 0.0f};

    float coord = static_cast<float>(out_idx) / scale;
    float max_c = in_len > 0 ? static_cast<float>(in_len - 1) : 0.0f;
    coord        = migraphx::max(0.0f, migraphx::min(max_c, coord));
    auto base_f  = migraphx::floor(coord);
    auto base    = static_cast<index_int>(base_f);
    auto next    = (base + 1 < in_len) ? (base + 1) : (in_len - 1);
    float frac   = coord - static_cast<float>(base);
    return {base, next, frac};
}

template <class Input, class Output, class Scales>
__device__ void resize_linear_asymmetric(const Input& input,
                                         Output& output,
                                         const Scales& scales)
{
    auto idx        = make_index();
    auto in_shape  = input.get_shape();
    auto out_shape = output.get_shape();
    auto in_lens   = in_shape.lens.template to<index_int>();

    idx.global_stride(out_shape.elements(), [&](auto gid) {
        auto out_idx = out_shape.multi(gid).template to<index_int>();
        constexpr index_int ndim = decltype(out_idx.size()){};
        array<resize_interp_params, ndim> params{};
        for(index_int d = 0; d < ndim; ++d)
        {
            params[d] = compute_interp_params_asymmetric(in_lens[d], out_idx[d], scales[d]);
        }

        float acc              = 0.0f;
        const index_int corners = (ndim == 0) ? 1 : (1u << ndim);
        array<index_int, ndim> in_idx{};

        for(index_int mask = 0; mask < corners; ++mask)
        {
            float w = 1.0f;
            for(index_int d = 0; d < ndim; ++d)
            {
                const bool use_high = ((mask >> d) & 1u) != 0u;
                w *= use_high ? params[d].weight : (1.0f - params[d].weight);
                in_idx[d] = use_high ? params[d].i1 : params[d].i0;
            }
            if(w == 0.0f)
                continue;
            acc += w * migraphx::convert<float>(input[in_idx]);
        }

        output[out_idx] = implicit_conversion(acc);
    });
}

} // namespace migraphx

#endif
