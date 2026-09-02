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
#ifndef MIGRAPHX_GUARD_KERNELS_GRIDSAMPLE_HPP
#define MIGRAPHX_GUARD_KERNELS_GRIDSAMPLE_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/ops.hpp>

namespace migraphx {

enum gridsample_padding : int
{
    gridsample_zeros      = 0,
    gridsample_border     = 1,
    gridsample_reflection = 2
};

enum gridsample_mode : int
{
    gridsample_mode_nearest = 0,
    gridsample_mode_linear  = 1,
    gridsample_mode_cubic   = 2
};

MIGRAPHX_DEVICE_CONSTEXPR float gridsample_cubic_weight_1(float t)
{
    constexpr float a = -0.75f;
    return ((a + 2.0f) * t - (a + 3.0f)) * t * t + 1.0f;
}

MIGRAPHX_DEVICE_CONSTEXPR float gridsample_cubic_weight_2(float t)
{
    constexpr float a = -0.75f;
    return ((a * t - 5.0f * a) * t + 8.0f * a) * t - 4.0f * a;
}

template <bool AlignCorners>
MIGRAPHX_DEVICE_CONSTEXPR float gridsample_unnormalize(float c, float size)
{
    if constexpr(AlignCorners)
        return (c + 1.0f) * ((size - 1.0f) / 2.0f);
    else
        return (c + 1.0f) * (size / 2.0f) - 0.5f;
}

MIGRAPHX_DEVICE_CONSTEXPR float gridsample_reflect(float c, float size, float corner_start)
{
    float idx        = migraphx::abs(corner_start - c);
    float size_times = migraphx::floor(migraphx::floor(idx) / size);
    float extra      = idx - size_times * size;
    bool even        = migraphx::fmod(size_times, 2.0f) == 0.0f;
    return even ? extra + corner_start : (size - extra) + corner_start;
}

template <bool AlignCorners, int PaddingMode>
MIGRAPHX_DEVICE_CONSTEXPR float gridsample_pad(float c, float size)
{
    if constexpr(PaddingMode == gridsample_reflection)
    {
        c = gridsample_reflect(c, AlignCorners ? size - 1.0f : size, AlignCorners ? 0.0f : -0.5f);
    }
    if constexpr(PaddingMode != gridsample_zeros)
    {
        c = migraphx::min(migraphx::max(c, 0.0f), size - 1.0f);
    }
    return c;
}

// One thread per output element. Taps are computed inline from the grid
// coordinate; no index tensors are materialized.
template <bool AlignCorners, int PaddingMode, int Mode, class T, class G, class U>
__device__ void gridsample(const T& x_t, const G& grid_t, U& y_t)
{
    auto index       = make_index();
    const auto out_s = y_t.get_shape();
    const auto x_s   = x_t.get_shape();

    const float in_h  = x_s.lens[2];
    const float in_w  = x_s.lens[3];
    const float h_max = in_h - 1.0f;
    const float w_max = in_w - 1.0f;

    index.global_stride(out_s.elements(), [&](auto i) {
        const auto idx = out_s.multi(i);
        const auto n   = idx[0];
        const auto c   = idx[1];
        const auto h   = idx[2];
        const auto w   = idx[3];

        const float gx = grid_t[array<index_int, 4>{n, h, w, 0}];
        const float gy = grid_t[array<index_int, 4>{n, h, w, 1}];

        const float px = gridsample_pad<AlignCorners, PaddingMode>(
            gridsample_unnormalize<AlignCorners>(gx, in_w), in_w);
        const float py = gridsample_pad<AlignCorners, PaddingMode>(
            gridsample_unnormalize<AlignCorners>(gy, in_h), in_h);

        if constexpr(Mode == gridsample_mode_nearest)
        {
            // Bounds-checked on the rounded float value, matching
            // nearest_sampler in the ONNX parser (round -> clip ->
            // compare-equal). See op::gridsample::compute() for why the
            // check has to happen before any cast to an unsigned index type.
            const float rx   = migraphx::nearbyint<float>(px);
            const float ry   = migraphx::nearbyint<float>(py);
            const bool valid = rx >= 0.0f and rx <= w_max and ry >= 0.0f and ry <= h_max;

            const index_int xi = valid ? rx : 0.0f;
            const index_int yi = valid ? ry : 0.0f;

            y_t[idx] = valid ? implicit_conversion(x_t[migraphx::array<index_int, 4>{n, c, yi, xi}])
                             : implicit_conversion(0.0f);
        }
        else if constexpr(Mode == gridsample_mode_cubic)
        {
            // 4x4 tap cubic convolution, mirrors bicubic_sampler in the ONNX
            // parser and op::gridsample::compute(): gridsample_pad() is
            // applied once above for px/py, and again per corner below,
            // since border/reflection padding must reflect corners that
            // fall outside the image independently of the base coordinate.
            const float floor_x = migraphx::floor<float>(px);
            const float floor_y = migraphx::floor<float>(py);
            const float fx      = px - floor_x;
            const float fy      = py - floor_y;

            const migraphx::array<float, 4> x_weight = {gridsample_cubic_weight_2(fx + 1.0f),
                                                       gridsample_cubic_weight_1(fx),
                                                       gridsample_cubic_weight_1(1.0f - fx),
                                                       gridsample_cubic_weight_2(2.0f - fx)};
            const migraphx::array<float, 4> y_weight = {gridsample_cubic_weight_2(fy + 1.0f),
                                                       gridsample_cubic_weight_1(fy),
                                              gridsample_cubic_weight_1(1.0f - fy),
                                              gridsample_cubic_weight_2(2.0f - fy)};

            migraphx::array<index_int, 4> x_idx;
            migraphx::array<index_int, 4> y_idx;
            migraphx::array<bool, 4> x_valid;
            migraphx::array<bool, 4> y_valid;
            repeat_c<4>([&](auto k) {
                const float cx =
                    gridsample_pad<AlignCorners, PaddingMode>(floor_x - 1.0f + k, in_w);
                const float cy =
                    gridsample_pad<AlignCorners, PaddingMode>(floor_y - 1.0f + k, in_h);
                x_valid[k] = cx >= 0.0f and cx <= w_max;
                y_valid[k] = cy >= 0.0f and cy <= h_max;
                x_idx[k]   = x_valid[k] ? cx : 0.0f;
                y_idx[k]   = y_valid[k] ? cy : 0.0f;
            });

            float acc = 0.0f;
            repeat_c<4>([&](auto j) {
                float row = 0.0f;
                repeat_c<4>([&](auto xk) {
                    if(x_valid[xk] and y_valid[j])
                        row += x_t[array<index_int, 4>{n, c, y_idx[j], x_idx[xk]}] * x_weight[xk];
                });
                acc += row * y_weight[j];
            });

            y_t[idx] = implicit_conversion(acc);
        }
        else
        {
            const float fx0 = migraphx::floor<float>(px);
            const float fy0 = migraphx::floor<float>(py);
            const float fx  = px - fx0;
            const float fy  = py - fy0;

            // In-range test on the sample coordinate, matching the clip-then-compare
            // validation in the parser decomposition.
            const bool x0_ok = fx0 >= 0.0f and fx0 <= w_max;
            const bool x1_ok = (fx0 + 1.0f) >= 0.0f and (fx0 + 1.0f) <= w_max;
            const bool y0_ok = fy0 >= 0.0f and fy0 <= h_max;
            const bool y1_ok = (fy0 + 1.0f) >= 0.0f and (fy0 + 1.0f) <= h_max;

            const index_int x0 = migraphx::min(migraphx::max(fx0, 0.0f), w_max);
            const index_int y0 = migraphx::min(migraphx::max(fy0, 0.0f), h_max);
            const index_int x1 = migraphx::min(migraphx::max(fx0 + 1.0f, 0.0f), w_max);
            const index_int y1 = migraphx::min(migraphx::max(fy0 + 1.0f, 0.0f), h_max);

            // Accumulated in the same order as the parser: (x0,y0), (x1,y0), (x0,y1), (x1,y1)
            float acc = 0.0f;
            if(x0_ok and y0_ok)
                acc += x_t[migraphx::array<index_int, 4>{n, c, y0, x0}] * ((1.0f - fy) * (1.0f - fx));
            if(x1_ok and y0_ok)
                acc += x_t[migraphx::array<index_int, 4>{n, c, y0, x1}] * ((1.0f - fy) * fx);
            if(x0_ok and y1_ok)
                acc += x_t[migraphx::array<index_int, 4>{n, c, y1, x0}] * (fy * (1.0f - fx));
            if(x1_ok and y1_ok)
                acc += x_t[migraphx::array<index_int, 4>{n, c, y1, x1}] * (fy * fx);

            y_t[idx] = implicit_conversion(acc);
        }
    });
}

} // namespace migraphx
#endif
