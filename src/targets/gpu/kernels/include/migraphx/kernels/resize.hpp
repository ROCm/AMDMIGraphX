/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/kernels/dfor.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {

// Coordinate transformation mode functors
// Use double precision to match ref implementation and avoid float precision issues
struct coord_transform_half_pixel
{
    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR double operator()(index_int, index_int, index_int idx, T scale) const
    {
        return (static_cast<double>(idx) + 0.5) / static_cast<double>(scale) - 0.5;
    }
};

struct coord_transform_pytorch_half_pixel
{
    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR float
    operator()(index_int, index_int l_out, index_int idx, T scale) const
    {
        return l_out > 1 ? (static_cast<float>(idx) + 0.5) / static_cast<float>(scale) - 0.5 : 0.0;
    }
};

struct coord_transform_align_corners
{
    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR float
    operator()(index_int l_in, index_int l_out, index_int idx, T) const
    {
        return (l_out == 1) ? 0.0
                            : (1.0 * static_cast<float>(idx) * static_cast<float>(l_in - 1) /
                               static_cast<float>(l_out - 1));
    }
};

struct coord_transform_asymmetric
{
    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR float operator()(index_int, index_int, index_int idx, T scale) const
    {
        return static_cast<float>(idx) / static_cast<float>(scale);
    }
};

struct coord_transform_tf_half_pixel_for_nn
{
    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR float operator()(index_int, index_int, index_int idx, T scale) const
    {
        return (static_cast<float>(idx) + 0.5) / static_cast<float>(scale);
    }
};

// Nearest mode functors
struct nearest_floor
{
    MIGRAPHX_DEVICE_CONSTEXPR index_int operator()(index_int d_in, float val) const
    {
        val = max(0.0f, min(static_cast<float>(d_in - 1), val));
        return static_cast<index_int>(floor(val));
    }
};

struct nearest_ceil
{
    MIGRAPHX_DEVICE_CONSTEXPR index_int operator()(index_int d_in, float val) const
    {
        val = max(0.0f, min(static_cast<float>(d_in - 1), val));
        return static_cast<index_int>(ceil(val));
    }
};

struct nearest_round_prefer_floor
{
    MIGRAPHX_DEVICE_CONSTEXPR index_int operator()(index_int d_in, float val) const
    {
        val = max(0.0f, min(static_cast<float>(d_in - 1), val));
        return static_cast<index_int>(ceil(val - 0.5));
    }
};

struct nearest_round_prefer_ceil
{
    MIGRAPHX_DEVICE_CONSTEXPR index_int operator()(index_int d_in, float val) const
    {
        val = max(0.0f, min(static_cast<float>(d_in - 1), val));
        return static_cast<index_int>(round(val));
    }
};

// Settings for resize operation
template <class T1, class T2, class T3>
struct resize_settings
{
    T1 coord_transform{};
    T2 nearest_op{};
    T3 scales{};
};

template <class... Ts>
MIGRAPHX_DEVICE_CONSTEXPR resize_settings<Ts...> make_resize_settings(Ts... xs)
{
    return {xs...};
}

// Compute input indices for nearest neighbor mode
template <class InShape, class OutShape, class NearestOp, class CoordOp, class Scales>
MIGRAPHX_DEVICE_CONSTEXPR auto compute_nearest_idx(InShape in_shape,
                                                   OutShape out_shape,
                                                   index_int out_idx,
                                                   const NearestOp& nearest_op,
                                                   const CoordOp& coord_op,
                                                   const Scales& scales)
{
    auto out_multi      = out_shape.multi(out_idx);
    constexpr auto ndim = InShape{}.lens.size();
    array<index_int, ndim> in_multi{};

    for(index_int i = 0; i < ndim; ++i)
    {
        auto coord  = coord_op(in_shape.lens[i], out_shape.lens[i], out_multi[i], scales[i]);
        in_multi[i] = nearest_op(in_shape.lens[i], coord);
    }

    return in_shape.index(in_multi);
}

// Compute interpolation parameters for linear mode
template <class T>
struct interp_params
{
    index_int i0; // lower index
    index_int i1; // upper index
    T weight;     // interpolation weight (0.0 to 1.0)
};

template <class T, class CoordOp>
MIGRAPHX_DEVICE_CONSTEXPR interp_params<T> compute_interp_params_1d(
    index_int in_len, index_int out_len, index_int out_idx, T scale, const CoordOp& coord_op)
{
    // Handle degenerate dimension (length 1) to avoid NaNs
    if(in_len <= 1)
    {
        return {0, 0, T{0.0}};
    }

    // Compute the original floating-point coordinate
    T coord = coord_op(in_len, out_len, out_idx, scale);

    // Clamp to valid input range [0, in_len-1]
    T max_c = in_len > 0 ? static_cast<T>(in_len - 1) : T{0.0};
    coord   = max(T{0.0}, min(max_c, coord));

    index_int base = static_cast<index_int>(floor(coord));
    index_int next = min(base + 1, in_len > 0 ? in_len - 1 : 0);
    T frac         = coord - static_cast<T>(base);

    return {base, next, frac};
}

// Resize nearest kernel
template <class Input, class Output, class Settings>
__device__ void resize_nearest(Input input, Output output, Settings settings)
{
    auto idx           = make_index();
    auto in_shape      = input.get_shape();
    auto out_shape     = output.get_shape();
    const auto& scales = settings.scales;

    idx.global_stride(out_shape.elements(), [&](auto out_idx) {
        auto in_idx = compute_nearest_idx(
            in_shape, out_shape, out_idx, settings.nearest_op, settings.coord_transform, scales);
        output[out_idx] = input[in_idx];
    });
}

// Resize linear kernel
template <class Input, class Output, class Settings>
__device__ void resize_linear(Input input, Output output, Settings settings)
{
    auto idx            = make_index();
    auto in_shape       = input.get_shape();
    auto out_shape      = output.get_shape();
    constexpr auto ndim = get_shape_c<Input>{}.lens.size();
    const auto& scales  = settings.scales;

    idx.global_stride(out_shape.elements(), [&](auto out_idx) {
        auto out_multi = out_shape.multi(out_idx);

        // Precompute interpolation parameters for each dimension
        array<interp_params<double>, ndim> params{};
        for(index_int d = 0; d < ndim; ++d)
        {
            params[d] = compute_interp_params_1d<double>(in_shape.lens[d],
                                                         out_shape.lens[d],
                                                         out_multi[d],
                                                         scales[d],
                                                         settings.coord_transform);
        }

        // Accumulate over 2^ndim corners
        float acc               = 0.0;
        const index_int corners = (ndim == 0) ? 1 : (1 << ndim);
        array<index_int, ndim> in_multi{};

        for(index_int mask = 0; mask < corners; ++mask)
        {
            float w = 1.0;
            for(index_int d = 0; d < ndim; ++d)
            {
                const bool use_high = ((mask >> d) & 1U) != 0U;
                w *= use_high ? params[d].weight : (1.0 - params[d].weight);
                in_multi[d] = use_high ? params[d].i1 : params[d].i0;
            }

            if(w == 0.0)
                continue;

            acc += w * migraphx::convert<float>(input[in_multi]);
        }

        output[out_idx] = implicit_conversion(acc);
    });
}

} // namespace migraphx

#endif
