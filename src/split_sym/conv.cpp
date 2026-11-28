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
#include <migraphx/split_sym/analyzer.hpp>
#include <migraphx/op/common.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

axis_desc classify_convolution_axis(std::size_t axis,
                                    bool default_padding,
                                    const std::vector<std::size_t>& padding,
                                    std::size_t spatial_dimensions)
{
    if(not default_padding)
        return axis_desc{};
    return padded_axis(fill_kind::zero, windowed_zero_pad(padding, spatial_dimensions, axis));
}

struct analyze_conv : analyzer<analyze_conv>
{
    bool matches(const operation& op) const
    {
        return op.name() == "convolution" or
               op.attributes().get("general_data_type", std::string{}) == "convolution";
    }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        const auto& op       = ins->get_operator();
        bool default_padding = false;
        std::size_t group    = 0;
        std::vector<std::size_t> padding;
        std::size_t spatial_dimensions = 0;
        if(op.name() == "convolution" or op.name() == "quant_convolution")
        {
            auto attributes    = op.to_value();
            default_padding    = attributes.at("padding_mode").to<op::padding_mode_t>() ==
                                 op::padding_mode_t::default_;
            group              = attributes.at("group").to<std::size_t>();
            padding            = attributes.at("padding").to_vector<std::size_t>();
            spatial_dimensions = attributes.at("stride").to_vector<std::size_t>().size();
        }
        return analyze_axes(
            ins,
            [&](std::size_t axis) {
                return axis < 2 or
                       classify_convolution_axis(axis, default_padding, padding, spatial_dimensions)
                               .handling == axis_handling::pad;
            },
            [&](std::size_t input, std::size_t axis) {
                if(input == 0)
                {
                    if(axis == 0)
                        return parallel_axis();
                    if(axis == 1)
                    {
                        if(group != 1)
                            return axis_desc{};
                        return contracted_axis(fill_kind::zero);
                    }
                    return classify_convolution_axis(
                        axis, default_padding, padding, spatial_dimensions);
                }
                if(input != 1 or group != 1)
                    return axis_desc{};
                if(axis == 0)
                    return parallel_axis();
                if(axis == 1)
                    return contracted_axis(fill_kind::zero);
                return axis_desc{};
            });
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
