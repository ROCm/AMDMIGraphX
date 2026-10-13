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

struct analyze_topk : analyzer<analyze_topk>
{
    bool matches(const operation& op) const { return op.name() == "topk"; }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        auto inputs        = to_shapes(ins->inputs());
        const auto& output = ins->get_shape();
        if(inputs.empty() or inputs.size() > 2 or not inputs.front().symbolic() or
           output.type() != shape::tuple_type or output.sub_shapes().size() != 2)
            return symbolic_op_info{ins};

        auto attributes = ins->get_operator().to_value();
        auto axis = normalize_axis(attributes.at("axis").to<int64_t>(), inputs.front().ndim());
        if(not axis.has_value() or not is_variable_axis(inputs.front().dyn_dims().at(*axis)) or
           any_of(output.sub_shapes(), [](const auto& s) { return not s.symbolic(); }))
            return symbolic_op_info{ins};

        auto topk_fill =
            attributes.at("largest").to<bool>() ? fill_kind::lowest : fill_kind::highest;
        return analyze_axes(
            ins,
            output.sub_shapes().front(),
            [axis = *axis](std::size_t) { return true; },
            [axis = *axis, topk_fill](std::size_t input, std::size_t current_axis) {
                if(current_axis != axis)
                    return parallel_axis();
                return input == 0 ? contracted_axis(topk_fill)
                                  : contracted_axis(fill_kind::dont_care);
            });
    }
};

struct analyze_topk_get_tuple_elem : analyzer<analyze_topk_get_tuple_elem>
{
    bool matches(const operation& op) const { return op.name() == "get_tuple_elem"; }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        const auto& inputs = ins->inputs();
        if(inputs.size() != 1 or inputs.front()->name() != "topk")
            return symbolic_op_info{ins};
        auto result    = analyze_axes(ins);
        result.freezer = freeze;
        return result;
    }

    static instruction_ref freeze(module& m,
                                  instruction_ref source,
                                  const std::vector<instruction_ref>& args,
                                  const std::unordered_map<sym::expr, std::size_t>&)
    {
        return m.add_instruction(source->get_operator(), args);
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
