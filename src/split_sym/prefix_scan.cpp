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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

struct analyze_prefix_scan : analyzer<analyze_prefix_scan>
{
    bool matches(const operation& op) const { return op.name() == "prefix_scan_sum"; }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        const auto& input = ins->inputs().front()->get_shape();
        auto attributes   = ins->get_operator().to_value();
        auto axis         = normalize_axis(attributes.at("axis").to<int64_t>(), input.ndim());
        if(not axis.has_value())
            return symbolic_op_info{ins};
        auto reverse   = attributes.at("reverse").to<bool>();
        auto exclusive = attributes.at("exclusive").to<bool>();
        auto result = analyze_axes(ins, [axis = *axis, reverse](std::size_t, std::size_t current) {
            return current == axis and reverse ? contracted_axis(fill_kind::zero) : parallel_axis();
        });
        if(not reverse and not exclusive)
            result.propagate_padding = propagate_padding;
        return result;
    }

    static std::optional<std::vector<input_padding_requirement>>
    propagate_padding(instruction_ref ins, const padding_requirement& requirement)
    {
        auto axis = normalize_axis(ins->get_operator().to_value().at("axis").to<int64_t>(),
                                   ins->get_shape().ndim());
        if(not axis.has_value() or requirement.axis != *axis)
            return std::nullopt;
        if(requirement.semantics == padding_semantics::max_neutral)
            return propagate_padding_to_input(
                ins, 0, requirement, {fill_kind::zero}, padding_semantics::exact);
        if(requirement.semantics == padding_semantics::min_neutral)
        {
            auto fill =
                shape::is_integral(ins->get_shape().type()) ? fill_kind::zero : fill_kind::one;
            return propagate_padding_to_input(
                ins, 0, requirement, {fill}, padding_semantics::exact);
        }
        return std::nullopt;
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
