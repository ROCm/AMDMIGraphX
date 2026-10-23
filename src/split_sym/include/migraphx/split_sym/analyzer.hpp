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
#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_SPLIT_SYM_ANALYZER_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_SPLIT_SYM_ANALYZER_HPP

#include <migraphx/config.hpp>
#include <migraphx/auto_register.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/zip_view.hpp>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {

enum class fill_kind
{
    dont_care,
    zero,
    lowest,
    neg_inf,
    highest,
    one
};

enum class mask_role
{
    // A softmax-style axis masked with -inf so the padded region contributes zeros.
    normalized,
    // A reduction axis masked with the reduction identity.
    contracted
};

enum class axis_handling
{
    unsupported,
    pad,
    mask
};

struct axis_desc
{
    axis_handling handling = axis_handling::unsupported;
    fill_kind fill         = fill_kind::dont_care;
    // Padded values on this axis cannot leak into the result, so a cross-block input need not be
    // sliced back to its runtime extent.
    bool coalesce_safe = false;
    mask_role role     = mask_role::contracted;
};

axis_desc padded_axis(fill_kind fill, bool coalesce_safe);

axis_desc parallel_axis();

axis_desc contracted_axis(fill_kind fill);

axis_desc masked_axis(mask_role role, fill_kind fill);

struct axis_mask
{
    std::size_t axis;
    sym::expr extent;
    fill_kind fill;
    mask_role role;

    bool zeros_contracted_region(const axis_mask& consumer) const
    {
        return role == mask_role::normalized and fill == fill_kind::neg_inf and
               consumer.role == mask_role::contracted and consumer.fill == fill_kind::zero and
               axis == consumer.axis and sym::same_symbol(extent, consumer.extent);
    }
};

struct operand_plan
{
    std::optional<float> pad_value;
    // Axes where producer padding would change the result; cross-block inputs are sliced back to
    // their runtime extent on these axes.
    std::vector<std::size_t> retained_slice_axes;
    // Runtime masks applied inside the clone instead of relying on the pad fill.
    std::vector<axis_mask> masks;

    bool zeros_contracted_region(const axis_mask& consumer) const
    {
        return any_of(masks,
                      [&](const auto& mask) { return mask.zeros_contracted_region(consumer); });
    }
};

// Re-emits the operation in static form for the given fixed symbol values.
using op_freezer =
    std::function<instruction_ref(module&,
                                  instruction_ref,
                                  const std::vector<instruction_ref>&,
                                  const std::unordered_map<sym::expr, std::size_t>&)>;

struct symbolic_op_info
{
    explicit symbolic_op_info(instruction_ref ref) : ins(ref) {}

    instruction_ref ins;
    std::vector<std::size_t> output_symbolic_axes;
    std::vector<operand_plan> operands;
    op_freezer freezer;
    // Inputs that only carry shape metadata, such as allocate dims; they are not operands.
    std::vector<std::size_t> shape_input_indices;
    bool supported = false;

    const shape& get_output_shape() const { return ins->get_shape(); }

    std::vector<shape> get_input_shapes() const { return to_shapes(ins->inputs()); }

    bool can_specialize() const
    {
        const bool needs_padding =
            any_of(operands, [](const auto& operand) { return operand.pad_value.has_value(); });
        const bool needs_specialization =
            needs_padding or (freezer and not output_symbolic_axes.empty());
        return supported and ins->module_inputs().empty() and needs_specialization;
    }

    bool starts_specialization_block() const
    {
        return not output_symbolic_axes.empty() and can_specialize();
    }

    bool needs_deferred_specialization_block() const
    {
        return output_symbolic_axes.empty() and can_specialize();
    }

    bool zeros_contracted_region(const axis_mask& consumer) const
    {
        return any_of(operands, [&](const auto& operand) {
            return operand.zeros_contracted_region(consumer);
        });
    }
};

bool is_variable_axis(const shape::dynamic_dimension& d);

bool supports_mask(shape::type_t type, fill_kind fill);

float fill_value(fill_kind fill);

std::optional<std::size_t> normalize_axis(int64_t axis, std::size_t rank);

bool windowed_zero_pad(const std::vector<std::size_t>& padding,
                       std::size_t spatial_dimensions,
                       std::size_t axis);

template <class OutputRule, class InputRule>
symbolic_op_info analyze_axes(instruction_ref ins,
                              std::vector<std::size_t> shape_input_indices,
                              const OutputRule& output_rule,
                              const InputRule& input_rule)
{
    symbolic_op_info info{ins};
    info.shape_input_indices = std::move(shape_input_indices);
    const auto& output_shape = info.get_output_shape();
    auto output_axes         = range(output_shape.ndim());
    auto output_dims         = output_shape.to_symbolic().dyn_dims();

    std::copy_if(output_axes.begin(),
                 output_axes.end(),
                 std::back_inserter(info.output_symbolic_axes),
                 [&](auto axis) { return is_variable_axis(output_dims.at(axis)); });

    info.supported = all_of(info.output_symbolic_axes, output_rule);

    auto input_shapes = info.get_input_shapes();
    assert(
        all_of(info.shape_input_indices, [&](auto index) { return index < input_shapes.size(); }));

    info.operands.resize(input_shapes.size());
    auto input_indices = range(input_shapes.size());
    for(auto&& [index, input, operand] : views::zip(input_indices, input_shapes, info.operands))
    {
        if(contains(info.shape_input_indices, index))
            continue;
        if(not input.symbolic())
            continue;
        const auto& input_dims = input.dyn_dims();
        auto variable_axes     = find_all(
            range(input.ndim()), [&](auto axis) { return is_variable_axis(input_dims.at(axis)); });
        if(variable_axes.empty())
            continue;
        fill_kind fill = fill_kind::dont_care;
        for(std::size_t axis : variable_axes)
        {
            const auto& dimension = input_dims.at(axis);
            auto desc             = input_rule(index, axis);
            if(desc.handling == axis_handling::unsupported)
            {
                info.supported = false;
                operand.retained_slice_axes.push_back(axis);
                continue;
            }
            if(desc.handling == axis_handling::mask)
            {
                if(not supports_mask(input.type(), desc.fill))
                    info.supported = false;
                operand.masks.push_back({axis, dimension.sym_expr, desc.fill, desc.role});
                continue;
            }
            if(not desc.coalesce_safe)
                operand.retained_slice_axes.push_back(axis);
            if(desc.fill == fill_kind::dont_care)
                continue;
            if(fill != fill_kind::dont_care and fill != desc.fill)
                MIGRAPHX_THROW("SPLIT_SYM_DIM: conflicting padding fills on one operand");
            fill = desc.fill;
        }
        operand.pad_value = fill_value(fill);
    }
    return info;
}

template <class OutputRule, class InputRule>
symbolic_op_info
analyze_axes(instruction_ref ins, const OutputRule& output_rule, const InputRule& input_rule)
{
    return analyze_axes(ins, {}, output_rule, input_rule);
}

template <class InputRule>
symbolic_op_info analyze_axes(instruction_ref ins, const InputRule& input_rule)
{
    return analyze_axes(ins, [](std::size_t) { return true; }, input_rule);
}

symbolic_op_info analyze_axes(instruction_ref ins,
                              std::vector<std::size_t> shape_input_indices = {});

struct symbolic_analyzer
{
    std::function<bool(const operation&)> matches;
    std::function<symbolic_op_info(instruction_ref)> analyze;
};

void register_analyzer(symbolic_analyzer analyzer);

symbolic_op_info analyze_instruction(instruction_ref ins);

template <class T>
void register_analyzer()
{
    T analyzer;
    register_analyzer({[=](const operation& op) { return analyzer.matches(op); },
                       [=](instruction_ref ins) { return analyzer.analyze(ins); }});
}

struct register_analyzer_action
{
    template <class T>
    static void apply()
    {
        register_analyzer<T>();
    }
};

template <class T>
using analyzer = auto_register<register_analyzer_action, T>;

} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
