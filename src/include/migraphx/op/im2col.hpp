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
#ifndef MIGRAPHX_GUARD_OPERATORS_IM2COL_HPP
#define MIGRAPHX_GUARD_OPERATORS_IM2COL_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct im2col
{
    std::vector<std::size_t> padding{0, 0};
    std::vector<std::size_t> stride{1, 1};
    std::vector<std::size_t> dilation{1, 1};

    padding_mode_t padding_mode = default_;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.padding, "padding"),
                    f(self.stride, "stride"),
                    f(self.dilation, "dilation"),
                    f(self.padding_mode, "padding_mode"));
    }

    std::string name() const { return "im2col"; }

    value attributes() const { return {{"normalize_padding", "padding"}}; }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        const auto& input   = inputs[0];
        const auto& weights = inputs[1];
        auto batch_size     = input.lens()[0];
        auto input_channels = weights.lens()[1];
        auto kernel_height  = weights.lens()[2];
        auto kernel_width   = weights.lens()[3];
        check_shapes{inputs, *this}.has(2);
        if(batch_size != 1)
            MIGRAPHX_THROW("im2col only support batch_size 1");

        auto padding_h = 2 * padding[0];
        auto padding_w = 2 * padding[1];
        if(padding.size() == 2 * stride.size())
        {
            padding_h = padding[0] + padding[2];
            padding_w = padding[1] + padding[3];
        }
        auto output_height = std::size_t(std::max<std::ptrdiff_t>(
            1,
            (input.lens()[2] - (1 + dilation[0] * (kernel_height - 1)) + padding_h) / stride[0] +
                1));
        auto output_width  = std::size_t(std::max<std::ptrdiff_t>(
            1,
            (input.lens()[3] - (1 + dilation[1] * (kernel_width - 1)) + padding_w) / stride[1] +
                1));

        auto channels_col = kernel_height * kernel_width * input_channels;
        return {input.type(), {output_height * output_width, channels_col}};
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        argument result{dyn_out.computed_shape};
        auto input_shape   = args[0].get_shape();
        auto weights_shape = args[1].get_shape();
        visit_all(result, args[0])([&](auto col, auto input) {
            const std::size_t& height   = input_shape.lens()[2];
            const std::size_t& width    = input_shape.lens()[3];
            const std::size_t& channels = weights_shape.lens()[1];
            const std::size_t& kernel_h = weights_shape.lens()[2];
            const std::size_t& kernel_w = weights_shape.lens()[3];
            const std::size_t& pad_h    = padding[0];
            const std::size_t& pad_w    = padding[1];
            const std::size_t& stride_h = stride[0];
            const std::size_t& stride_w = stride[1];

            long kdiv2_h = long(kernel_h) / 2;
            long kdiv2_w = long(kernel_w) / 2;
            const std::size_t col_height = (height - kernel_h + 2 * pad_h) / stride_h + 1;
            const std::size_t col_width  = (width - kernel_w + 2 * pad_w) / stride_w + 1;
            long iinput = kdiv2_h - long(pad_h);
            for(std::size_t ioutput = 0; ioutput < col_height; ioutput++, iinput += stride_h)
            {
                long jinput = kdiv2_w - long(pad_w);
                for(std::size_t joutput = 0; joutput < col_width; joutput++, jinput += stride_w)
                {
                    std::size_t ldx = ioutput * col_width + joutput;
                    std::size_t p   = 0;
                    dfor(channels,
                         kernel_h,
                         kernel_w)([&](std::size_t c, std::size_t koffset, std::size_t loffset) {
                        auto idx    = iinput + long(koffset) - kdiv2_h;
                        auto jdx    = jinput + long(loffset) - kdiv2_w;
                        col(ldx, p) =
                            ((idx >= 0) and (idx < height) and (jdx >= 0) and (jdx < width))
                                ? input(0, c, idx, jdx)
                                : 0;
                        p++;
                    });
                }
            }
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
