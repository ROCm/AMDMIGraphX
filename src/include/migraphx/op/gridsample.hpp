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
#include <cmath>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Bilinear GridSample (ONNX GridSample, mode="linear"/"bilinear") for 4D inputs.
 *
 * Inputs:  x    {N, C, H_in,  W_in}
 *          grid {N, H_out, W_out, 2}   -- normalized (x, y) in [-1, 1]
 * Output:       {N, C, H_out, W_out}
 *
 * The semantics here are deliberately identical to the ONNX-parser
 * decomposition in src/onnx/parse_gridsample.cpp (struct linear_sampler): same
 * unnormalization, same padding handling, same corner weights, same
 * accumulation order.  The difference is purely that this computes the four
 * taps inline instead of materializing gathernd index tensors.
 */
struct gridsample
{
    std::string mode         = "linear";
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
        check_shapes{inputs, *this}.has(2).standard();
        if(mode != "linear")
            MIGRAPHX_THROW("GRIDSAMPLE: only mode=\"linear\" is supported, got \"" + mode + "\"");
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
        if(x_s.type() != g_s.type())
            MIGRAPHX_THROW("GRIDSAMPLE: input and grid must have the same type");

        return {x_s.type(),
                {x_s.lens().at(0), x_s.lens().at(1), g_s.lens().at(1), g_s.lens().at(2)}};
    }

    // (c + 1) * (size - 1) / 2   with align_corners
    // (c + 1) * size / 2 - 0.5   otherwise
    float unnormalize(float c, float size) const
    {
        return align_corners ? (c + 1.0f) * ((size - 1.0f) / 2.0f)
                             : (c + 1.0f) * (size / 2.0f) - 0.5f;
    }

    // Mirrors grid_sampler::reflect_coordinates() in the ONNX parser, including
    // the floor() applied before the division.
    static float reflect_coord(float c, float size, float corner_start)
    {
        float idx        = std::abs(corner_start - c);
        float size_times = std::floor(std::floor(idx) / size);
        float extra      = idx - size_times * size;
        bool even        = float_equal(std::fmod(size_times, 2.0f), 0.0f);
        return even ? extra + corner_start : (size - extra) + corner_start;
    }

    // Applies padding_mode to an unnormalized coordinate.
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

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        const auto& out_lens = output_shape.lens();
        const auto n_batch   = out_lens[0];
        const auto n_chan    = out_lens[1];
        const auto out_h     = out_lens[2];
        const auto out_w     = out_lens[3];

        const auto x_s    = args.at(0).get_shape();
        const auto g_s    = args.at(1).get_shape();
        const auto in_h   = x_s.lens()[2];
        const auto in_w   = x_s.lens()[3];
        const float h_max = in_h - 1.0f;
        const float w_max = in_w - 1.0f;

        visit_all(result, args.at(0), args.at(1))([&](auto output, auto x, auto grid) {
            par_for(n_batch * out_h * out_w, [&](auto i) {
                const auto w = i % out_w;
                const auto h = (i / out_w) % out_h;
                const auto n = i / (out_w * out_h);

                float px = pad_coord(unnormalize(grid[g_s.index({n, h, w, 0})], in_w), in_w);
                float py = pad_coord(unnormalize(grid[g_s.index({n, h, w, 1})], in_h), in_h);

                const float fx0 = std::floor(px);
                const float fy0 = std::floor(py);
                const float fx  = px - fx0;
                const float fy  = py - fy0;

                // In-range test on the *sample* coordinate, matching the
                // clip-then-compare validation in the parser decomposition.
                const bool x0_ok = fx0 >= 0.0f and fx0 <= w_max;
                const bool x1_ok = (fx0 + 1.0f) >= 0.0f and (fx0 + 1.0f) <= w_max;
                const bool y0_ok = fy0 >= 0.0f and fy0 <= h_max;
                const bool y1_ok = (fy0 + 1.0f) >= 0.0f and (fy0 + 1.0f) <= h_max;

                const auto x0 = static_cast<std::size_t>(std::min(std::max(fx0, 0.0f), w_max));
                const auto x1 =
                    static_cast<std::size_t>(std::min(std::max(fx0 + 1.0f, 0.0f), w_max));
                const auto y0 = static_cast<std::size_t>(std::min(std::max(fy0, 0.0f), h_max));
                const auto y1 =
                    static_cast<std::size_t>(std::min(std::max(fy0 + 1.0f, 0.0f), h_max));

                for(std::size_t c = 0; c < n_chan; ++c)
                {
                    // Accumulated in the same order as the parser: (x0,y0), (x1,y0), (x0,y1),
                    // (x1,y1)
                    float acc = (x0_ok and y0_ok)
                                    ? static_cast<float>(x[x_s.index({n, c, y0, x0})]) *
                                          ((1.0f - fy) * (1.0f - fx))
                                    : 0.0f;
                    if(x1_ok and y0_ok)
                        acc +=
                            static_cast<float>(x[x_s.index({n, c, y0, x1})]) * ((1.0f - fy) * fx);
                    if(x0_ok and y1_ok)
                        acc +=
                            static_cast<float>(x[x_s.index({n, c, y1, x0})]) * (fy * (1.0f - fx));
                    if(x1_ok and y1_ok)
                        acc += static_cast<float>(x[x_s.index({n, c, y1, x1})]) * (fy * fx);

                    output[output_shape.index({n, c, h, w})] = acc;
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
