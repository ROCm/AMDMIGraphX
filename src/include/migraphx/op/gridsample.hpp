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
#ifndef MIGRAPHX_GUARD_OPERATORS_GRIDSAMPLE_HPP
#define MIGRAPHX_GUARD_OPERATORS_GRIDSAMPLE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/value.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * GridSample (ONNX GridSample) for 4D inputs, mode in {"nearest", "linear",
 * "cubic"}.
 *
 * Inputs:  x    {N, C, H_in,  W_in}
 *          grid {N, H_out, W_out, 2}   -- normalized (x, y) in [-1, 1]
 * Output:       {N, C, H_out, W_out}
 *
 * The semantics here are deliberately identical to the ONNX-parser
 * decomposition in src/onnx/parse_gridsample.cpp (struct linear_sampler /
 * nearest_sampler)
 */
struct gridsample
{
    std::string mode = "linear";

    std::string padding_mode = "zeros";
    bool align_corners       = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.mode, "mode"),
                    f(self.padding_mode, "padding_mode"),
                    f(self.align_corners, "align_corners"));
    }

    std::string name() const { return "gridsample"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).same_type();
        bool supported_modes = mode == "nearest" or mode == "linear" or mode == "bilinear" or
                               mode == "cubic" or mode == "bicubic";
        if(not supported_modes)
            MIGRAPHX_THROW("GRIDSAMPLE: only modes \"nearest\", \"linear\" and \"cubic\" are "
                           "supported or its legacy variants, got \"" +
                           mode + "\"");
        if(padding_mode != "zeros" and padding_mode != "border" and padding_mode != "reflection")
            MIGRAPHX_THROW("GRIDSAMPLE: unknown padding_mode \"" + padding_mode + "\"");

        const auto& x_s = inputs.at(0);
        const auto& g_s = inputs.at(1);
        if(x_s.lens().size() != 4)
            MIGRAPHX_THROW("GRIDSAMPLE: input must be 4 dimensions");
        if(g_s.lens().size() != 4 or g_s.lens().at(3) != 2)
            MIGRAPHX_THROW("GRIDSAMPLE: grid must be {N, H_out, W_out, 2}");
        if(x_s.lens().at(0) != g_s.lens().at(0))
            MIGRAPHX_THROW("GRIDSAMPLE: input and grid must have the same batch size");

        return {x_s.type(),
                {x_s.lens().at(0), x_s.lens().at(1), g_s.lens().at(1), g_s.lens().at(2)}};
    }

    float unnormalize(float c, float size) const
    {
        return align_corners ? (c + 1.0f) * ((size - 1.0f) / 2.0f)
                             : (c + 1.0f) * (size / 2.0f) - 0.5f;
    }

    static float reflect_coord(float c, float size, float corner_start)
    {
        float idx        = std::abs(corner_start - c);
        float size_times = std::floor(std::floor(idx) / size);
        float extra      = idx - size_times * size;
        bool even        = float_equal(std::fmod(size_times, 2.0f), 0.0f);
        return even ? extra + corner_start : (size - extra) + corner_start;
    }

    float pad_coord(float c, float size) const
    {
        if(padding_mode == "reflection")
        {
            c = reflect_coord(c, align_corners ? size - 1.0f : size, align_corners ? 0.0f : -0.5f);
        }
        if(padding_mode != "zeros")
        {
            c = std::min(std::max(c, 0.0f), size - 1.0f);
        }
        return c;
    }

    static float cubic_weight_1(float t)
    {
        constexpr float a = -0.75f;
        return ((a + 2.0f) * t - (a + 3.0f)) * t * t + 1.0f;
    }

    static float cubic_weight_2(float t)
    {
        constexpr float a = -0.75f;
        return ((a * t - 5.0f * a) * t + 8.0f * a) * t - 4.0f * a;
    }

    // The single output pixel a mode function is responsible for, together with
    // the already-padded sample coordinate and the input extents. Bundled so
    // each mode takes one argument instead of eight.
    struct sample_site
    {
        std::size_t n;
        std::size_t h;
        std::size_t w;
        std::size_t n_chan;
        float px;
        float py;
        float in_w;
        float in_h;
    };

    template <class Output, class X>
    static void sample_linear(Output& output, const X& x, const sample_site& s)
    {
        const float w_max = s.in_w - 1.0f;
        const float h_max = s.in_h - 1.0f;
        const float fx0   = std::floor(s.px);
        const float fy0   = std::floor(s.py);
        const float fx    = s.px - fx0;
        const float fy    = s.py - fy0;

        // In-range test on the *sample* coordinate, matching the
        // clip-then-compare validation in the parser decomposition.
        const bool x0_ok = fx0 >= 0.0f and fx0 <= w_max;
        const bool x1_ok = (fx0 + 1.0f) >= 0.0f and (fx0 + 1.0f) <= w_max;
        const bool y0_ok = fy0 >= 0.0f and fy0 <= h_max;
        const bool y1_ok = (fy0 + 1.0f) >= 0.0f and (fy0 + 1.0f) <= h_max;

        const std::size_t x0 = std::min(std::max(fx0, 0.0f), w_max);
        const std::size_t x1 = std::min(std::max(fx0 + 1.0f, 0.0f), w_max);
        const std::size_t y0 = std::min(std::max(fy0, 0.0f), h_max);
        const std::size_t y1 = std::min(std::max(fy0 + 1.0f, 0.0f), h_max);

        for(std::size_t c = 0; c < s.n_chan; ++c)
        {
            float acc = (x0_ok and y0_ok) ? x(s.n, c, y0, x0) * ((1.0f - fy) * (1.0f - fx)) : 0.0f;
            if(x1_ok and y0_ok)
                acc += x(s.n, c, y0, x1) * ((1.0f - fy) * fx);
            if(x0_ok and y1_ok)
                acc += x(s.n, c, y1, x0) * (fy * (1.0f - fx));
            if(x1_ok and y1_ok)
                acc += x(s.n, c, y1, x1) * (fy * fx);

            output(s.n, c, s.h, s.w) = acc;
        }
    }

    template <class Output, class X>
    static void sample_nearest(Output& output, const X& x, const sample_site& s)
    {
        const float w_max = s.in_w - 1.0f;
        const float h_max = s.in_h - 1.0f;
        const float rx    = std::nearbyint(s.px);
        const float ry    = std::nearbyint(s.py);
        const bool valid  = rx >= 0.0f and rx <= w_max and ry >= 0.0f and ry <= h_max;

        const std::size_t x_nearest = valid ? rx : 0.0f;
        const std::size_t y_nearest = valid ? ry : 0.0f;

        for(std::size_t c = 0; c < s.n_chan; ++c)
        {
            if(valid)
                output(s.n, c, s.h, s.w) = x(s.n, c, y_nearest, x_nearest);
            else
                output(s.n, c, s.h, s.w) = 0;
        }
    }

    template <class Output, class X>
    void sample_cubic(Output& output, const X& x, const sample_site& s) const
    {
        const float w_max = s.in_w - 1.0f;
        const float h_max = s.in_h - 1.0f;
        const float fx0   = std::floor(s.px);
        const float fy0   = std::floor(s.py);
        const float fx    = s.px - fx0;
        const float fy    = s.py - fy0;

        const std::array<float, 4> x_weight = {cubic_weight_2(fx + 1.0f),
                                               cubic_weight_1(fx),
                                               cubic_weight_1(1.0f - fx),
                                               cubic_weight_2(2.0f - fx)};
        const std::array<float, 4> y_weight = {cubic_weight_2(fy + 1.0f),
                                               cubic_weight_1(fy),
                                               cubic_weight_1(1.0f - fy),
                                               cubic_weight_2(2.0f - fy)};

        std::array<std::size_t, 4> x_idx{};
        std::array<std::size_t, 4> y_idx{};
        std::array<bool, 4> x_valid{};
        std::array<bool, 4> y_valid{};
        for(std::size_t k = 0; k < 4; ++k)
        {
            // pad_coord is applied once by the caller for px/py and again per
            // tap here, since border/reflection padding must fold each tap that
            // falls outside the image independently of the base coordinate.
            const float cx = pad_coord(fx0 - 1.0f + k, s.in_w);
            const float cy = pad_coord(fy0 - 1.0f + k, s.in_h);
            x_valid[k]     = cx >= 0.0f and cx <= w_max;
            y_valid[k]     = cy >= 0.0f and cy <= h_max;
            x_idx[k]       = x_valid[k] ? cx : 0.0f;
            y_idx[k]       = y_valid[k] ? cy : 0.0f;
        }

        for(std::size_t c = 0; c < s.n_chan; ++c)
        {
            float acc = 0.0f;
            for(std::size_t j = 0; j < 4; ++j)
            {
                float row = 0.0f;
                for(std::size_t xk = 0; xk < 4; ++xk)
                {
                    if(x_valid[xk] and y_valid[j])
                        row += x(s.n, c, y_idx[j], x_idx[xk]) * x_weight[xk];
                }
                acc += row * y_weight[j];
            }
            output(s.n, c, s.h, s.w) = acc;
        }
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        const std::vector<std::size_t>& out_lens = output_shape.lens();
        const std::size_t n_batch                = out_lens[0];
        const std::size_t n_chan                 = out_lens[1];
        const std::size_t out_h                  = out_lens[2];
        const std::size_t out_w                  = out_lens[3];

        const std::vector<std::size_t>& in_lens = args.at(0).get_shape().lens();
        const float in_h                        = in_lens[2];
        const float in_w                        = in_lens[3];

        visit_all(result, args.at(0), args.at(1))([&](auto output, auto x, auto grid) {
            par_for(n_batch * out_h * out_w, [&](auto i) {
                const std::size_t w = i % out_w;
                const std::size_t h = (i / out_w) % out_h;
                const std::size_t n = i / (out_w * out_h);

                const sample_site site{n,
                                       h,
                                       w,
                                       n_chan,
                                       pad_coord(unnormalize(grid(n, h, w, 0), in_w), in_w),
                                       pad_coord(unnormalize(grid(n, h, w, 1), in_h), in_h),
                                       in_w,
                                       in_h};

                if(contains(mode, "linear"))
                {
                    sample_linear(output, x, site);
                }
                else if(contains(mode, "nearest"))
                {
                    sample_nearest(output, x, site);
                }
                else if(contains(mode, "cubic"))
                {
                    sample_cubic(output, x, site);
                }
                else
                {
                    // compute_shape rejects every other spelling, so this is
                    // unreachable unless the two validations drift apart.
                    MIGRAPHX_THROW("GRIDSAMPLE: only modes \"nearest\", \"linear\" and \"cubic\" "
                                   "are supported or its legacy variants, got \"" +
                                   mode + "\"");
                }
            });
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
