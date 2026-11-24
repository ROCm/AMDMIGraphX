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
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

std::optional<double> scalar_constant(instruction_ref ins)
{
    if(contains({"broadcast", "broadcast_with_dims", "contiguous", "convert", "multibroadcast"},
                ins->name()) and
       not ins->inputs().empty())
        return scalar_constant(ins->inputs().front());
    auto value = ins->eval();
    if(value.empty() or value.get_shape().elements() != 1)
        return std::nullopt;
    std::optional<double> result;
    value.visit([&](auto values) { result = static_cast<double>(values.front()); });
    return result;
}

bool requires_exact_fill(const padding_requirement& requirement, fill_kind fill)
{
    return requirement.semantics == padding_semantics::exact and
           requirement.fills == std::set<fill_kind>{fill};
}

struct analyze_pointwise : analyzer<analyze_pointwise>
{
    bool matches(const operation& op) const { return op.attributes().contains("pointwise"); }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        auto result              = analyze_axes(ins);
        result.propagate_padding = propagate_padding;
        return result;
    }

    static std::optional<std::vector<input_padding_requirement>>
    propagate_padding(instruction_ref ins, const padding_requirement& requirement)
    {
        const auto& inputs = ins->inputs();
        if(ins->name() == "where" and inputs.size() == 3)
        {
            auto condition = propagate_padding_to_input(
                ins, 0, requirement, {fill_kind::zero}, padding_semantics::exact);
            auto false_value = propagate_padding_to_input(ins,
                                                          2,
                                                          requirement,
                                                          requirement.fills,
                                                          requirement.semantics,
                                                          requirement.materializable);
            if(not condition.has_value() or not false_value.has_value())
                return std::nullopt;
            condition->insert(condition->end(), false_value->begin(), false_value->end());
            return condition;
        }
        if(ins->name() == "equal" and inputs.size() == 2 and
           requires_exact_fill(requirement, fill_kind::zero))
        {
            auto input_indices = range(inputs.size());
            auto input =
                std::find_if(input_indices.begin(), input_indices.end(), [&](auto input_index) {
                    auto other = scalar_constant(inputs.at(1 - input_index));
                    return other.has_value() and *other == 0.0;
                });
            if(input == input_indices.end())
                return std::nullopt;
            std::set<fill_kind> nonzero_fills = {fill_kind::one};
            auto input_type                   = inputs.at(*input)->get_shape().type();
            if(supports_fill(input_type, fill_kind::neg_one))
                nonzero_fills.insert(fill_kind::neg_one);
            return propagate_padding_to_input(
                ins, *input, requirement, nonzero_fills, padding_semantics::exact);
        }
        if(ins->name() == "sub" and inputs.size() == 2 and
           contains({padding_semantics::max_neutral, padding_semantics::min_neutral},
                    requirement.semantics))
        {
            auto scalar = scalar_constant(inputs.back());
            if(scalar.has_value())
            {
                const bool materializable =
                    requirement.materializable and
                    (not shape::is_integral(ins->get_shape().type()) or *scalar == 0.0);
                return propagate_padding_to_input(
                    ins, 0, requirement, requirement.fills, requirement.semantics, materializable);
            }
        }
        if(inputs.size() != 1)
            return std::nullopt;
        if(ins->name() == "exp")
        {
            if(requirement.semantics == padding_semantics::max_neutral or
               requires_exact_fill(requirement, fill_kind::zero))
                return propagate_padding_to_input(
                    ins, 0, requirement, {fill_kind::neg_inf}, padding_semantics::exact);
            if(requires_exact_fill(requirement, fill_kind::one))
                return propagate_padding_to_input(
                    ins, 0, requirement, {fill_kind::zero}, padding_semantics::exact);
        }
        if(ins->name() == "identity" or
           (ins->name() == "convert" and
            inputs.front()->get_shape().type() == ins->get_shape().type()))
            return preserve_padding(ins, requirement);
        if(ins->name() == "convert" and requirement.semantics == padding_semantics::exact)
        {
            std::set<fill_kind> preserved;
            std::copy_if(
                requirement.fills.begin(),
                requirement.fills.end(),
                std::inserter(preserved, preserved.end()),
                [](auto fill) { return contains({fill_kind::zero, fill_kind::one}, fill); });
            if(not preserved.empty())
                return propagate_padding_to_input(ins,
                                                  0,
                                                  requirement,
                                                  preserved,
                                                  padding_semantics::exact,
                                                  requirement.materializable);
        }
        return std::nullopt;
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
