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
 */
#ifndef MIGRAPHX_GUARD_KERNELS_ROIALIGN_HPP
#define MIGRAPHX_GUARD_KERNELS_ROIALIGN_HPP

// #include <migraphx/kernels/debug.hpp>
// #include <migraphx/kernels/print.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/dfor.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/array.hpp>

namespace migraphx {

struct max_pool
{
    MIGRAPHX_DEVICE_CONSTEXPR auto init() { return lowest{}; }

    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T operator()(T x, T y)
    {
        return max(x, y);
    }

    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T final(T x, index_int)
    {
        return (x);
    }
};

struct avg_pool
{
    MIGRAPHX_DEVICE_CONSTEXPR auto init() { return 0.0; }

    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T operator()(T x, T y)
    {
        return x + y;
    }

    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T final(T x, index_int y)
    {
        return (y == 0) ? T{0.0} : T{x / y};
    }
};

template <class Iterator, class Op>
MIGRAPHX_DEVICE_CONSTEXPR typename Iterator::value_type bilinear_interpolate(
    const Iterator data, const array<index_int, 2>& dims, array<float, 2> xy, Op pooling)
{
    // TODO:  The values calculated here are geometric, depending only
    // on the ROI/batch layer pairs, and are the same
    // locations for each channel c.  They could be precalculated and reused as they are in the ref.
    // implementation.
    array<int, 2> low{};
    array<int, 2> high{};
    for(index_int ii = 0; ii < xy.size(); ++ii)
    {
        if(xy[ii] < -1.0f or xy[ii] > dims[ii])
        {
            return implicit_conversion(0);
        }

        xy[ii]   = migraphx::max(xy[ii], 0.0f);
        low[ii]  = xy[ii];
        high[ii] = low[ii] + 1;
        if(low[ii] >= dims[ii] - 1)
        {
            xy[ii] = high[ii] = low[ii] = dims[ii] - 1;
        }
    }
    array<index_int, 4> locs = {low[1] * dims[0] + low[0],
                                low[1] * dims[0] + high[0],
                                high[1] * dims[0] + low[0],
                                high[1] * dims[0] + high[0]};

    // lx, ly, hx, hy are the distances from (x, y) to the upper and lower edges of the
    // containing pixel; aka the fractional part of a floating-point number
    float lx = xy[0] - low[0];
    float ly = xy[1] - low[1];

    float hy = 1.0f - ly;
    float hx = 1.0f - lx;

    // weighting values, which in bilinear interpolation are based on distance
    // of the point from the index points.
    // do calculations in floating point and convert final result to required type
    array<float, 4> ws = {hy * hx, hy * lx, ly * hx, ly * lx};

    auto v01 = pooling(data[locs[0]] * ws[0], data[locs[1]] * ws[1]);
    auto v23 = pooling(data[locs[2]] * ws[2], data[locs[3]] * ws[3]);
    return implicit_conversion(pooling(v01, v23));
}

// Calculate a single pooled output value
template <class Iterator, class Op>
MIGRAPHX_DEVICE_CONSTEXPR auto calc_pooling(const Iterator& data,
                                            const array<float, 2>& roi_starts,
                                            const array<float, 2>& bin_size,
                                            const array<int, 2>& idx,
                                            const array<index_int, 2>& bin_grid_size,
                                            const array<index_int, 2>& dims,
                                            Op op)
{
    // for one idx (output height and width coordinates) we iterate through all bin_grid values
    using in_dtype      = typename Iterator::value_type;
    in_dtype output_val = in_dtype{op.init()};
    const int64_t count = bin_grid_size[0] * bin_grid_size[1];
    dfor(bin_grid_size[0], bin_grid_size[1])([&](auto iy, auto ix) {
        array<index_int, 2> id = {iy, ix};
        array<float, 2> locs = roi_starts + idx * bin_size + bin_size * (id + 0.5f) / bin_grid_size;
        auto val   = bilinear_interpolate(data, dims, locs, op);
        output_val = op(output_val, val);
    });
    return op.final(output_val, count);
}

template <class T1, class T2, class T3, class T4>
struct roalign_settings
{
    T1 roi_offset{};
    T2 is_avg_pooling{};
    T3 sampling_ratio{};
    T4 spatial_scale{};
};

template <class... Ts>
constexpr roalign_settings<Ts...> make_roalign_settings(Ts... xs)
{
    return {xs...};
}

template <class T, class U, class V, class W, class Settings>
__device__ void roialign(const T& x_t, const U& rois_t, const V& ind_t, W& y_t, Settings s)
{
    auto index      = make_index();
    const auto x    = x_t.begin();
    const auto rois = rois_t.begin();
    const auto ind  = ind_t.begin();
    // input shape
    auto x_lens      = x_t.get_shape().lens;
    auto channel_num = x_lens[1];
    // input dims of height and width, in all 2-dim arrays, the first dim
    // is for height and second dim is for width
    array<index_int, 2> in_dims = {x_lens[3], x_lens[2]};

    const auto stride   = index.nglobal();
    auto out_s          = y_t.get_shape();
    auto roi_column_num = rois_t.get_shape().lens[1];

    // output dims of height and width, in all 2-dim arrays, the first dim
    // is for height and second dim is for width
    const auto& out_lens = out_s.lens;

    array<index_int, 2> out_dims = {out_lens[3], out_lens[2]};

    // Compute lens and strides vectors for use in reindexing output.
    array<index_int, 4> visit_lens({out_lens[0], out_lens[1], out_lens[3], out_lens[2]});
    // Todo: look for a less indirect way to reconcile the ordering of iteration
    // between this op. and the reference.
    array<size_t, 4> m_lens{out_lens[0], out_lens[1], out_lens[2], out_lens[3]};
    array<size_t, 4> m_strides;
    m_strides[3] = 1;
    for(int k = 2; k >= 0; k--)
    {
        m_strides[k] = m_strides[k + 1] * m_lens[k + 1];
    }
    for(index_int i = index.global; i < out_s.elements(); i += stride)
    {
        auto idx = visit_lens.multi(i);
        int n    = idx[0];
        int c    = idx[1];
        int ph   = idx[2];
        int pw   = idx[3];

        const auto offset_rois = rois + (n * roi_column_num);
        const int batch_ind    = ind[n];

        // Note that roi_offset in src/targets/gpu/jit/roialign.cpp uses a negative value, so we add
        // rather than subtract it here
        array<float, 2> roi_starts = {
            static_cast<float>(offset_rois[0]) * static_cast<float>(s.spatial_scale) + s.roi_offset,
            static_cast<float>(offset_rois[1]) * static_cast<float>(s.spatial_scale) +
                s.roi_offset};

        array<float, 2> roi_ends = {
            static_cast<float>(offset_rois[2]) * static_cast<float>(s.spatial_scale) + s.roi_offset,
            static_cast<float>(offset_rois[3]) * static_cast<float>(s.spatial_scale) +
                s.roi_offset};

        array<float, 2> roi_size{};
        array<float, 2> bin_size{};
        array<index_int, 2> bin_grid_size{};

        for(index_int ii = 0; ii < roi_size.size(); ++ii)
        {
            roi_size[ii] = roi_ends[ii] - roi_starts[ii];
            if(s.roi_offset == 0.f)
                roi_size[ii] = migraphx::max(roi_size[ii], 1.0f);

            bin_size[ii]      = roi_size[ii] / out_dims[ii];
            bin_grid_size[ii] = (s.sampling_ratio > 0)
                                    ? s.sampling_ratio
                                    : migraphx::ceil(roi_size[ii] / out_dims[ii]);
        }
        const auto offset_x = x + ((batch_ind * channel_num + c) * in_dims[0] * in_dims[1]);

        //
        //  Reindexing.  Calculations to this point did not iterate in the same order as
        // in the reference op; we now calculate the output index corresponding to i
        //
        size_t pp = i;
        size_t jj = (pp / m_strides[0]) * m_strides[0];
        pp        = pp % m_strides[0];
        jj += (pp / m_strides[1]) * m_strides[1];
        pp %= m_strides[1];
        pp = pp / m_lens[2] + (pp % m_lens[2]) * m_strides[2];
        jj += pp;

        if constexpr(s.is_avg_pooling)
        {
            y_t[jj] = calc_pooling(
                offset_x, roi_starts, bin_size, {ph, pw}, bin_grid_size, in_dims, avg_pool{});
        }
        else
        {
            y_t[jj] = calc_pooling(
                offset_x, roi_starts, bin_size, {ph, pw}, bin_grid_size, in_dims, max_pool{});
        }
    }
}

} // namespace migraphx
#endif
