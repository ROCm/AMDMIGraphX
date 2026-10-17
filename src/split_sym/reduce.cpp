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
#include <migraphx/make_op.hpp>
#include <algorithm>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

std::optional<fill_kind> reduce_identity(const std::string& name)
{
    if(name == "reduce_max")
        return fill_kind::lowest;
    if(name == "reduce_min")
        return fill_kind::highest;
    if(name == "reduce_prod" or name == "reduce_all")
        return fill_kind::one;
    if(name == "reduce_sum" or name == "reduce_any")
        return fill_kind::zero;
    return std::nullopt;
}

std::vector<int64_t> reduce_axes(const operation& op, std::size_t ndim)
{
    auto attributes = op.to_value();
    std::vector<int64_t> axes;
    if(attributes.contains("axes"))
        axes = attributes.at("axes").to_vector<int64_t>();
    int64_t rank = ndim;
    std::transform(axes.begin(), axes.end(), axes.begin(), [rank](auto axis) {
        return axis < 0 ? axis + rank : axis;
    });
    return axes;
}

struct analyze_reduce : analyzer<analyze_reduce>
{
    bool matches(const operation& op) const { return op.attributes().contains("reduce"); }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        const auto& op = ins->get_operator();
        auto inputs    = to_shapes(ins->inputs());
        std::vector<std::vector<int64_t>> reduction_axes(inputs.size());
        std::transform(inputs.begin(),
                       inputs.end(),
                       reduction_axes.begin(),
                       [&](const shape& input) { return reduce_axes(op, input.ndim()); });
        auto identity = reduce_identity(op.name());
        auto result   = analyze_axes(ins, [&](std::size_t input, std::size_t axis) {
            const auto& axes = reduction_axes.at(input);
            if(axes.empty())
                return axis_desc{};
            if(not contains(axes, axis))
                return parallel_axis();
            return identity.has_value() ? contracted_axis(*identity) : axis_desc{};
        });
        if(identity.has_value() and result.supported)
            result.freezer = freeze;
        return result;
    }

    static instruction_ref freeze(module& m,
                                  instruction_ref source,
                                  const std::vector<instruction_ref>& args,
                                  const std::unordered_map<sym::expr, std::size_t>&)
    {
        auto identity = reduce_identity(source->name());
        assert(identity.has_value());
        auto padded_args = args;
        for(auto& arg : padded_args)
            if(arg->get_shape().dynamic())
                arg = m.add_instruction(make_op("fixed_pad", {{"value", fill_value(*identity)}}),
                                        arg);
        return m.add_instruction(source->get_operator(), padded_args);
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
