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
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

struct dyn_slice_bounds
{
    std::vector<sym::expr> starts;
    std::vector<sym::expr> ends;
};

// A dyn_slice keeps a prefix on every axis when each start is a non-negative constant and each
// end is a non-negative constant or the full input extent.
std::optional<dyn_slice_bounds> prefix_stable_dyn_slice_bounds(instruction_ref ins)
{
    const auto& inputs = ins->inputs();
    if(inputs.size() != 3)
        return std::nullopt;
    auto attributes = ins->get_operator().to_value();
    if(not attributes.contains("axes"))
        return std::nullopt;
    auto starts_value = inputs.at(1)->sym_eval();
    auto ends_value   = inputs.at(2)->sym_eval();
    if(starts_value.empty() or ends_value.empty())
        return std::nullopt;
    auto axes   = attributes.at("axes").to_vector<int64_t>();
    auto starts = starts_value.get().to_vector();
    auto ends   = ends_value.get().to_vector();
    if(axes.size() != starts.size() or axes.size() != ends.size())
        return std::nullopt;

    const auto& input = inputs.front()->get_shape();
    auto input_dims   = input.to_symbolic().dyn_dims();
    auto zero         = sym::lit(int64_t{0});
    for(std::size_t i = 0; i < axes.size(); ++i)
    {
        auto axis         = normalize_axis(axes.at(i), input.ndim());
        const auto& start = starts.at(i);
        if(not axis.has_value() or not sym::find_variables(start).empty() or
           sym::strict_less(start, zero).value_or(false))
            return std::nullopt;
        const auto& end = ends.at(i);
        if(sym::strict_less(end, zero).value_or(false) or
           (not sym::find_variables(end).empty() and not(end == input_dims.at(*axis).sym_expr)))
            return std::nullopt;
    }
    return dyn_slice_bounds{std::move(starts), std::move(ends)};
}

std::vector<int64_t> evaluate_slice_bounds(const std::vector<sym::expr>& expressions,
                                           const std::unordered_map<sym::expr, std::size_t>& values)
{
    std::vector<int64_t> result(expressions.size());
    std::transform(expressions.begin(), expressions.end(), result.begin(), [&](const auto& e) {
        return static_cast<int64_t>(e.eval_uint(values));
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
            const auto& input = inputs.front();
            auto axes         = op.to_value().at("axes").to_vector<int64_t>();
            if(any_of(axes, [&](auto axis) {
                   auto normalized = normalize_axis(axis, input.ndim());
                   return not normalized.has_value() or
                          is_variable_axis(input.dyn_dims().at(*normalized));
               }))
                return symbolic_op_info{ins};
            return analyze_axes(ins);
        }

        auto bounds = prefix_stable_dyn_slice_bounds(ins);
        if(not bounds.has_value())
            return symbolic_op_info{ins};
        auto result    = analyze_axes(ins);
        result.freezer = [bounds = std::move(*bounds)](
                             module& m,
                             instruction_ref source,
                             const std::vector<instruction_ref>& args,
                             const std::unordered_map<sym::expr, std::size_t>& values) {
            assert(args.size() == 3);
            auto attributes = source->get_operator().to_value();
            return m.add_instruction(
                make_op("slice",
                        {{"axes", attributes.at("axes")},
                         {"starts", evaluate_slice_bounds(bounds.starts, values)},
                         {"ends", evaluate_slice_bounds(bounds.ends, values)}}),
                args.front());
        };
        return result;
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
