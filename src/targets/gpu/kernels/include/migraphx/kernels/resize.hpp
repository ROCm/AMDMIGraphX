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
#include <migraphx/kernels/dfor.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/bit.hpp>
#include <migraphx/kernels/algorithm.hpp>

namespace migraphx {

// Coordinate transformation mode functors
struct coord_transform_half_pixel
{
    MIGRAPHX_DEVICE_CONSTEXPR float operator()(index_int, index_int, float idx, float scale) const
    {
        MIGRAPHX_ASSERT(scale > 0 or scale < 0);
        return (idx + 0.5f) / scale - 0.5f;
    }
};

struct coord_transform_pytorch_half_pixel
{
    MIGRAPHX_DEVICE_CONSTEXPR float
    operator()(index_int, index_int l_out, float idx, float scale) const
    {
        MIGRAPHX_ASSERT(scale > 0 or scale < 0);
        return l_out > 1 ? (idx + 0.5f) / scale - 0.5f : 0.0f;
    }
};

struct coord_transform_align_corners
{
    MIGRAPHX_DEVICE_CONSTEXPR float
    operator()(index_int l_in, index_int l_out, float idx, float) const
    {
        return (l_out == 1) ? 0.0f : (idx * (l_in - 1.0f) / (l_out - 1.0f));
    }
};

struct coord_transform_asymmetric
{
    MIGRAPHX_DEVICE_CONSTEXPR float operator()(index_int, index_int, float idx, float scale) const
    {
        MIGRAPHX_ASSERT(scale > 0 or scale < 0);
        return idx / scale;
    }
};

struct coord_transform_tf_half_pixel_for_nn
{
    MIGRAPHX_DEVICE_CONSTEXPR float operator()(index_int, index_int, float idx, float scale) const
    {
        MIGRAPHX_ASSERT(scale > 0 or scale < 0);
        return (idx + 0.5f) / scale;
    }
};

// Nearest mode functors
struct nearest_floor
{
    MIGRAPHX_DEVICE_CONSTEXPR index_int operator()(index_int d_in, float val) const
    {
        return migraphx::floor(max(0.0f, min(static_cast<float>(d_in - 1), val)));
    }
};

struct nearest_ceil
{
    MIGRAPHX_DEVICE_CONSTEXPR index_int operator()(index_int d_in, float val) const
    {
        return migraphx::ceil(max(0.0f, min(static_cast<float>(d_in - 1), val)));
    }
};

struct nearest_round_prefer_floor
{
    MIGRAPHX_DEVICE_CONSTEXPR index_int operator()(index_int d_in, float val) const
    {
        return migraphx::ceil(max(0.0f, min(static_cast<float>(d_in - 1), val)) - 0.5f);
    }
};

struct nearest_round_prefer_ceil
{
    MIGRAPHX_DEVICE_CONSTEXPR index_int operator()(index_int d_in, float val) const
    {
        return migraphx::round(max(0.0f, min(static_cast<float>(d_in - 1), val)));
    }
};

// Compute input indices for nearest neighbor mode
template <class CoordOp, class NearestOp, class InShape, class OutShape, class Scales>
MIGRAPHX_DEVICE_CONSTEXPR auto
compute_nearest_idx(InShape in_shape, OutShape out_shape, index_int out_idx, const Scales& scales)
{
    auto out_multi      = out_shape.multi(out_idx);
    constexpr auto ndim = InShape{}.lens.size();
    array<index_int, ndim> in_multi{};

    for(index_int i = 0; i < ndim; ++i)
    {
        auto coord  = CoordOp{}(in_shape.lens[i], out_shape.lens[i], out_multi[i], scales[i]);
        in_multi[i] = NearestOp{}(in_shape.lens[i], coord);
    }

    return in_multi;
}

// Compute interpolation parameters for linear mode
struct interp_params
{
    index_int i0; // lower index
    index_int i1; // upper index
    float weight; // interpolation weight (0.0 to 1.0)
};

// Cubic interpolation parameters for one dimension (4 neighbors)
struct cubic_params
{
    array<index_int, 4> indices;
    array<float, 4> weights;
};

// Cubic kernel function (Keys bicubic)
MIGRAPHX_DEVICE_CONSTEXPR float cubic_kernel(float s, float a)
{
    float abs_s = migraphx::abs(s);
    if(abs_s < 1.0f)
        return (a + 2.0f) * abs_s * abs_s * abs_s - (a + 3.0f) * abs_s * abs_s + 1.0f;
    if(abs_s < 2.0f)
        return a * abs_s * abs_s * abs_s - 5.0f * a * abs_s * abs_s + 8.0f * a * abs_s - 4.0f * a;
    return 0.0f;
}

template <class CoordOp>
MIGRAPHX_DEVICE_CONSTEXPR interp_params
compute_interp_params_1d(index_int in_len, index_int out_len, index_int out_idx, float scale)
{
    // Handle degenerate dimension (length 1) to avoid NaNs
    if(in_len <= 1)
    {
        return {0, 0, 0.0f};
    }

    // Compute the original floating-point coordinate
    float coord = CoordOp{}(in_len, out_len, out_idx, scale);

    // Clamp to valid input range [0, in_len-1]
    float max_c         = in_len > 0 ? float(in_len - 1) : 0.0f;
    float clamped_coord = max(0.0f, min(max_c, coord));

    index_int base = migraphx::floor(clamped_coord);
    index_int next = min(base + 1, in_len > 0 ? in_len - 1 : 0);
    float frac     = clamped_coord - float(base);

    return {base, next, frac};
}

// Compute cubic interpolation parameters for a single dimension
template <class CoordOp>
MIGRAPHX_DEVICE_CONSTEXPR cubic_params compute_cubic_params_1d(
    index_int in_len, index_int out_len, index_int out_idx, float scale, float cubic_a)
{
    cubic_params result{};

    if(in_len == 0)
    {
        result.indices = {0, 0, 0, 0};
        result.weights = {0.0f, 0.0f, 0.0f, 0.0f};
        return result;
    }

    float coord = CoordOp{}(in_len, out_len, out_idx, scale);
    // Use signed arithmetic to avoid underflow when base is 0 and we compute base-1
    diff_int base_i = migraphx::floor(coord);

    for(diff_int i = 0; i < 4; ++i)
    {
        diff_int pos      = base_i - 1 + i;
        float t           = coord - float(pos);
        result.weights[i] = cubic_kernel(t, cubic_a);
        // Clamp to valid range [0, in_len-1]
        result.indices[i] =
            max(diff_int{0}, min(pos, static_cast<diff_int>(in_len - 1)));
    }

    return result;
}

// Resize nearest kernel
template <class CoordOp, class NearestOp, class Input, class Output, class Scales>
__device__ void resize_nearest(Input input, Output output, Scales scales)
{
    auto idx       = make_index();
    auto in_shape  = input.get_shape();
    auto out_shape = output.get_shape();

    idx.global_stride(out_shape.elements(), [&](auto out_idx) {
        auto in_idx = compute_nearest_idx<CoordOp, NearestOp>(in_shape, out_shape, out_idx, scales);
        output[out_idx] = input[in_idx];
    });
}

// Resize linear kernel
// Optimized to only iterate over 2^k corners where k is the number of dimensions
// that actually need interpolation (i.e., where i0 != i1)
template <class CoordOp, class NearestOp, class Input, class Output, class Scales>
__device__ void resize_linear(Input input, Output output, Scales scales)
{
    auto idx            = make_index();
    auto in_shape       = input.get_shape();
    auto out_shape      = output.get_shape();
    constexpr auto ndim = get_shape_c<Input>{}.lens.size();

    idx.global_stride(out_shape.elements(), [&](auto out_idx) {
        auto out_multi = out_shape.multi(out_idx);

        // Precompute interpolation parameters for each dimension
        auto params = array_transform(in_shape.lens, out_shape.lens, out_multi, scales)(
            [](auto... xs) { return compute_interp_params_1d<CoordOp>(xs...); });

        index_int active_count =
            count_if(scales.begin(), scales.end(), [](auto scale) { return scale != 1.0f; });
        MIGRAPHX_ASSERT(active_count < 32);

        // Initialize in_multi with non-interpolated dimensions (where i0 == i1)
        auto in_multi = array_transform(params)([](const interp_params& p) { return p.i0; });

        // Accumulate over 2^active_count corners instead of 2^ndim
        const index_int corners = (1u << active_count);
        float acc               = 0.0f;

        for(index_int subset = 0; subset < corners; ++subset)
        {
            float w              = 1.0f;
            index_int active_bit = 0;

            for(index_int d = 0; d < ndim; ++d)
            {
                if(scales[d] == 1.0f)
                    continue;
                // This dimension needs interpolation
                const bool use_high = get_bit(subset, active_bit);
                w *= use_high ? params[d].weight : (1.0f - params[d].weight);
                in_multi[d] = use_high ? params[d].i1 : params[d].i0;
                ++active_bit;
            }

            acc += w * migraphx::convert<float>(input[in_multi]);
        }

        output[out_idx] = implicit_conversion(acc);
    });
}

// Resize cubic kernel
// Uses separable bicubic interpolation with 4 neighbors per dimension
template <class CoordOp, class NearestOp, class Input, class Output, class Scales>
__device__ void resize_cubic(Input input, Output output, Scales scales, float cubic_coeff)
{
    auto idx            = make_index();
    auto in_shape       = input.get_shape();
    auto out_shape      = output.get_shape();
    constexpr auto ndim = get_shape_c<Input>{}.lens.size();

    idx.global_stride(out_shape.elements(), [&](auto out_idx) {
        auto out_multi = out_shape.multi(out_idx);

        // Precompute cubic interpolation parameters for each dimension
        array<cubic_params, ndim> params{};
        for(index_int d = 0; d < ndim; ++d)
        {
            params[d] = compute_cubic_params_1d<CoordOp>(
                in_shape.lens[d], out_shape.lens[d], out_multi[d], scales[d], cubic_coeff);
        }

        // Count dimensions that need interpolation (scale != 1.0)
        auto active_count = count_if(scales.begin(), scales.end(), 
            [&](auto scale) { return scale != 1.0f; });

        array<index_int, ndim> active_dims{};
        auto r = range(ndim);
        copy_if(r.begin(), r.end(), active_dims.begin(), [&](auto d) { 
            return scales[d] != 1.0f; 
        });

        // Initialize in_multi: for non-interpolated dimensions, use output index directly
        // (since input and output sizes are the same for those dimensions)
        array<index_int, ndim> in_multi{};
        for(index_int d = 0; d < ndim; ++d)
        {
            if(scales[d] == 1.0f)
            {
                in_multi[d] = out_multi[d];
            }
            else
            {
                in_multi[d] = params[d].indices[0];
            }
        }

        // Number of combinations: 4^active_count
        // For efficiency, limit to reasonable number of active dimensions
        index_int total_combos = 1;
        for(index_int i = 0; i < active_count; ++i)
        {
            total_combos <<= 2; // multiply by 4
        }

        float acc = 0.0f;

        for(index_int combo = 0; combo < total_combos; ++combo)
        {
            float w      = 1.0f;
            index_int tc = combo;

            for(index_int i = 0; i < active_count; ++i)
            {
                index_int d            = active_dims[i];
                index_int neighbor_idx = tc % 4;
                tc >>= 2; // divide by 4
                w *= params[d].weights[neighbor_idx];
                in_multi[d] = params[d].indices[neighbor_idx];
            }

            if(migraphx::abs(w) > 1e-10f)
            {
                acc += w * migraphx::convert<float>(input[in_multi]);
            }
        }

        output[out_idx] = implicit_conversion(acc);
    });
}

} // namespace migraphx

#endif
