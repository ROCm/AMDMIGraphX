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
#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

std::optional<std::vector<sym::expr>>
find_slice_expressions(instruction_ref ins, std::size_t input_index, const std::string& name)
{
    auto attributes = ins->get_operator().to_value();
    if(attributes.contains(name))
        return from_value<std::vector<sym::expr>>(attributes.at(name));
    auto value = ins->inputs().at(input_index)->sym_eval();
    if(value.empty())
        return std::nullopt;
    return value.get().to_vector();
}

bool is_prefix_stable_dyn_slice(instruction_ref ins)
{
    const auto& inputs = ins->inputs();
    if(inputs.size() != 3)
        return false;
    auto attributes = ins->get_operator().to_value();
    if(not attributes.contains("axes"))
        return false;
    auto starts = find_slice_expressions(ins, 1, "starts");
    auto ends   = find_slice_expressions(ins, 2, "ends");
    if(not starts.has_value() or not ends.has_value())
        return false;
    auto axes = attributes.at("axes").to_vector<int64_t>();
    if(axes.size() != starts->size() or axes.size() != ends->size())
        return false;

    const auto& input = inputs.front()->get_shape();
    auto input_dims   = input.to_symbolic().dyn_dims();
    auto zero         = sym::lit(int64_t{0});
    for(std::size_t i = 0; i < axes.size(); ++i)
    {
        auto axis         = normalize_axis(axes.at(i), input.ndim());
        const auto& start = starts->at(i);
        if(not axis.has_value() or not sym::find_variables(start).empty() or
           sym::strict_less(start, zero).value_or(false))
            return false;
        const auto& end = ends->at(i);
        if(sym::strict_less(end, zero).value_or(false) or
           (not sym::find_variables(end).empty() and not(end == input_dims.at(*axis).sym_expr)))
            return false;
    }
    return true;
}

std::vector<int64_t>
evaluate_slice_expressions(const std::vector<sym::expr>& expressions,
                           const std::unordered_map<sym::expr, std::size_t>& values)
{
    std::vector<int64_t> result(expressions.size());
    std::transform(expressions.begin(), expressions.end(), result.begin(), [&](const auto& e) {
        return e.eval_uint(values);
    });
    return result;
}

std::vector<int64_t> evaluate_slice_bound(instruction_ref source,
                                          std::size_t input_index,
                                          const std::string& name,
                                          const std::unordered_map<sym::expr, std::size_t>& values)
{
    auto expressions = find_slice_expressions(source, input_index, name);
    assert(expressions.has_value());
    return evaluate_slice_expressions(*expressions, values);
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
        auto attributes = op.to_value();
        auto result     = attributes.contains("starts") and attributes.contains("ends")
                              ? analyze_axes(ins, {1, 2})
                              : analyze_axes(ins);
        result.freezer  = freeze;
        return result;
    }

    static instruction_ref freeze(module& m,
                                  instruction_ref source,
                                  const std::vector<instruction_ref>& args,
                                  const freezer_context& context)
    {
        assert(args.size() == 1 or args.size() == 3);
        auto attributes = source->get_operator().to_value();
        return m.add_instruction(
            make_op("slice",
                    {{"axes", attributes.at("axes")},
                     {"starts", evaluate_slice_bound(source, 1, "starts", context.values)},
                     {"ends", evaluate_slice_bound(source, 2, "ends", context.values)}}),
            args.front());
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
