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
#include <migraphx/value.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

bool is_prefix_stable_dyn_slice(instruction_ref ins)
{
    const auto& inputs = ins->inputs();
    if(inputs.size() != 3)
        return false;
    auto attributes = ins->get_operator().to_value();
    if(not attributes.contains("axes"))
        return false;
    auto starts_value = inputs.at(1)->sym_eval();
    auto ends_value   = inputs.at(2)->sym_eval();
    if(starts_value.empty() or ends_value.empty())
        return false;
    auto axes   = attributes.at("axes").to_vector<int64_t>();
    auto starts = starts_value.get().to_vector();
    auto ends   = ends_value.get().to_vector();
    if(axes.size() != starts.size() or axes.size() != ends.size())
        return false;

    const auto& input = inputs.front()->get_shape();
    auto input_dims   = input.to_symbolic().dyn_dims();
    for(std::size_t i = 0; i < axes.size(); ++i)
    {
        auto axis = normalize_axis(axes.at(i), input.ndim());
        if(not axis.has_value() or not sym::find_variables(starts.at(i)).empty())
            return false;
        const auto& end = ends.at(i);
        if(not sym::find_variables(end).empty() and not(end == input_dims.at(*axis).sym_expr))
            return false;
    }
    return true;
}

std::vector<int64_t> evaluate_slice_input(const std::vector<instruction_ref>& args,
                                          std::size_t index,
                                          const std::unordered_map<sym::expr, std::size_t>& values)
{
    auto symbolic_value = args.at(index)->sym_eval();
    assert(not symbolic_value.empty());
    auto expressions = symbolic_value.get();
    std::vector<int64_t> result(expressions.size());
    std::transform(expressions.begin(), expressions.end(), result.begin(), [&](const auto& e) {
        return e.eval_uint(values);
    });
    return result;
}

struct analyze_slice : analyzer<analyze_slice>
{
    bool matches(const operation& op) const
    {
        return op.name() == "slice" or op.name() == "dyn_slice";
    }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        const auto& op = ins->get_operator();
        auto inputs    = to_shapes(ins->inputs());
        if(op.name() == "slice")
        {
            if(inputs.size() != 1)
                return symbolic_op_info{ins};
            auto axes = op.to_value().at("axes").to_vector<int64_t>();
            for(auto& axis : axes)
            {
                auto normalized_axis = normalize_axis(axis, inputs.front().ndim());
                if(not normalized_axis.has_value() or
                   is_variable_axis(inputs.front().dyn_dims().at(*normalized_axis)))
                    return symbolic_op_info{ins};
                axis = *normalized_axis;
            }
            return analyze_axes(ins);
        }

        if(inputs.size() != 3 or not is_prefix_stable_dyn_slice(ins))
            return symbolic_op_info{ins};
        auto result    = analyze_axes(ins);
        result.freezer = freeze;
        return result;
    }

    static instruction_ref freeze(module& m,
                                  instruction_ref source,
                                  const std::vector<instruction_ref>& args,
                                  const std::unordered_map<sym::expr, std::size_t>& values)
    {
        assert(args.size() == 3);
        auto attributes = source->get_operator().to_value();
        return m.add_instruction(make_op("slice",
                                         {{"axes", attributes.at("axes")},
                                          {"starts", evaluate_slice_input(args, 1, values)},
                                          {"ends", evaluate_slice_input(args, 2, values)}}),
                                 args.front());
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
