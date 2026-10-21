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
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

struct pooling_axes
{
    bool default_padding = false;
    fill_kind fill       = fill_kind::dont_care;
    bool ceil_mode       = false;
    bool dyn_global      = false;
    std::vector<std::size_t> padding;
    std::size_t spatial_dimensions = 0;

    axis_desc classify(std::size_t axis) const
    {
        if(axis < 2)
            return parallel_axis();
        if(not default_padding or fill == fill_kind::dont_care)
            return axis_desc{};
        if(dyn_global)
            return contracted_axis(fill);
        return padded_axis(fill,
                           not ceil_mode and windowed_zero_pad(padding, spatial_dimensions, axis));
    }
};

struct analyze_pooling : analyzer<analyze_pooling>
{
    bool matches(const operation& op) const { return op.name() == "pooling"; }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        auto attributes = ins->get_operator().to_value();
        auto mode       = attributes.at("mode").to<op::pooling_mode>();
        auto ceil_mode  = attributes.at("ceil_mode").to<bool>();
        fill_kind fill  = fill_kind::zero;
        if(mode == op::pooling_mode::max)
            fill = fill_kind::lowest;
        else if(mode == op::pooling_mode::average and
                (not attributes.at("count_include_pad").to<bool>() or ceil_mode))
            fill = fill_kind::dont_care;
        auto dyn_global = attributes.at("dyn_global").to<bool>();
        if(dyn_global and mode == op::pooling_mode::average)
            fill = fill_kind::dont_care;
        auto default_padding =
            attributes.at("padding_mode").to<op::padding_mode_t>() == op::padding_mode_t::default_;
        auto padding            = attributes.at("padding").to_vector<std::size_t>();
        auto spatial_dimensions = attributes.at("stride").to_vector<std::size_t>().size();
        pooling_axes axes{
            default_padding, fill, ceil_mode, dyn_global, std::move(padding), spatial_dimensions};
        return analyze_axes(
            ins,
            [&](std::size_t axis) { return axes.classify(axis).handling == axis_handling::pad; },
            [&](std::size_t, std::size_t axis) { return axes.classify(axis); });
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
