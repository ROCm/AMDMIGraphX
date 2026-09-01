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

#include <migraphx/split_sym_dim.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape_transform_descriptor.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/value.hpp>
#include <migraphx/zip_view.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

enum class fill_kind
{
    dont_care,
    zero,
    lowest,
    neg_inf,
    highest,
    one
};

struct io_ref
{
    bool is_output    = false;
    std::size_t index = 0;
};

bool is_variable_axis(const shape::dynamic_dimension& d)
{
    return d.is_symbolic() and not d.is_fixed();
}

bool has_variable_axis(const shape& s)
{
    return s.symbolic() and any_of(s.dyn_dims(), is_variable_axis);
}

bool is_pointwise(const operation& op) { return op.attributes().contains("pointwise"); }
bool is_reduce(const operation& op) { return op.attributes().contains("reduce"); }
bool is_dot(const operation& op)
{
    return op.name() == "dot" or op.attributes().get("general_data_type", std::string{}) == "dot";
}
bool is_conv(const operation& op)
{
    return op.name() == "convolution" or
           op.attributes().get("general_data_type", std::string{}) == "convolution";
}
bool is_pooling(const operation& op) { return op.name() == "pooling"; }
bool is_softmax(const operation& op) { return op.name() == "softmax"; }
bool is_shape_transform(const operation& op)
{
    return contains({"contiguous", "flatten", "reshape", "transpose"}, op.name());
}

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
    for(auto& a : axes)
        if(a < 0)
            a += rank;
    return axes;
}

bool windowed_zero_pad(const std::vector<std::size_t>& padding,
                       std::size_t spatial_dimensions,
                       std::size_t axis)
{
    if(axis < 2)
        return false;
    std::size_t spatial_dimension = axis - 2;
    if(spatial_dimension >= spatial_dimensions)
        return false;
    if(padding.size() == spatial_dimensions)
        return padding[spatial_dimension] == 0;
    if(padding.size() == 2 * spatial_dimensions)
        return padding[spatial_dimension] == 0 and
               padding[spatial_dimensions + spatial_dimension] == 0;
    return false;
}

enum class mask_role
{
    normalized,
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
    bool coalesce_safe     = false;
    mask_role role         = mask_role::contracted;
};

axis_desc padded_axis(fill_kind fill, bool coalesce_safe)
{
    return {axis_handling::pad, fill, coalesce_safe};
}

axis_desc parallel_axis() { return padded_axis(fill_kind::dont_care, true); }

axis_desc contracted_axis(fill_kind fill) { return padded_axis(fill, false); }

axis_desc masked_axis(mask_role role, fill_kind fill)
{
    return {axis_handling::mask, fill, true, role};
}

struct axis_mask
{
    std::size_t axis;
    sym::expr extent;
    fill_kind fill;
    mask_role role;
};

struct operand_plan
{
    std::optional<float> pad_value;
    std::vector<std::size_t> retained_slice_axes;
    std::vector<axis_mask> masks;
};

using optimal_map = std::unordered_map<sym::expr, sym::expr>;
using freeze_map  = std::unordered_map<sym::expr, std::size_t>;
using op_freezer  = std::function<instruction_ref(
    module&, instruction_ref, const std::vector<instruction_ref>&, const freeze_map&)>;

struct symbolic_op_info
{
    instruction_ref ins;
    shape output_shape;
    std::vector<shape> input_shapes;
    std::vector<std::size_t> output_symbolic_axes;
    std::vector<operand_plan> operands;
    op_freezer freezer;
    std::vector<std::size_t> shape_input_indices;
    bool supported = false;
};

std::vector<instruction_ref> select_data_inputs(const std::vector<instruction_ref>& inputs,
                                                const std::vector<std::size_t>& shape_input_indices)
{
    assert(all_of(shape_input_indices, [&](auto index) { return index < inputs.size(); }));
    std::vector<instruction_ref> result;
    for(std::size_t index = 0; index < inputs.size(); ++index)
        if(not contains(shape_input_indices, index))
            result.push_back(inputs.at(index));
    return result;
}

bool supports_mask(shape::type_t type, fill_kind fill)
{
    if(fill != fill_kind::neg_inf)
        return true;
    return contains({shape::half_type, shape::float_type, shape::double_type, shape::bf16_type},
                    type);
}

float fill_value(fill_kind fill)
{
    switch(fill)
    {
    case fill_kind::dont_care:
    case fill_kind::zero: return 0.0f;
    case fill_kind::lowest: return std::numeric_limits<float>::lowest();
    case fill_kind::neg_inf: return -std::numeric_limits<float>::infinity();
    case fill_kind::highest: return std::numeric_limits<float>::max();
    case fill_kind::one: return 1.0f;
    }
    MIGRAPHX_THROW("SPLIT_SYM_DIM: unsupported fill kind");
}

template <class Rule>
void analyze_axes(symbolic_op_info& info, const Rule& rule)
{
    info.supported = true;
    for(std::size_t axis = 0; axis < info.output_shape.ndim(); ++axis)
    {
        const auto& dimension = info.output_shape.dyn_dims().at(axis);
        if(not is_variable_axis(dimension))
            continue;
        info.output_symbolic_axes.push_back(axis);
        if(rule(io_ref{true, 0}, axis).handling != axis_handling::pad)
            info.supported = false;
    }

    assert(all_of(info.shape_input_indices,
                  [&](auto index) { return index < info.input_shapes.size(); }));
    info.operands.resize(info.input_shapes.size());
    for(std::size_t index = 0; index < info.input_shapes.size(); ++index)
    {
        if(contains(info.shape_input_indices, index))
            continue;
        const auto& input = info.input_shapes.at(index);
        if(not has_variable_axis(input))
            continue;
        auto& operand  = info.operands.at(index);
        fill_kind fill = fill_kind::dont_care;
        for(std::size_t axis = 0; axis < input.ndim(); ++axis)
        {
            const auto& dimension = input.dyn_dims().at(axis);
            if(not is_variable_axis(dimension))
                continue;
            operand.pad_value = fill_value(fill);
            auto desc         = rule(io_ref{false, index}, axis);
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
        if(operand.pad_value.has_value())
            operand.pad_value = fill_value(fill);
    }
}

void analyze_pointwise(symbolic_op_info& info)
{
    analyze_axes(info, [](io_ref, std::size_t) { return parallel_axis(); });
}

void analyze_reduce(symbolic_op_info& info)
{
    const auto& op     = info.ins->get_operator();
    const auto& inputs = info.input_shapes;
    std::vector<std::vector<int64_t>> reduction_axes(inputs.size());
    std::transform(inputs.begin(), inputs.end(), reduction_axes.begin(), [&](const shape& input) {
        return reduce_axes(op, input.ndim());
    });
    auto identity = reduce_identity(op.name());
    analyze_axes(info, [&](io_ref io, std::size_t axis) {
        if(io.is_output)
            return parallel_axis();
        const auto& axes = reduction_axes.at(io.index);
        if(axes.empty())
            return axis_desc{};
        if(not contains(axes, axis))
            return parallel_axis();
        return identity.has_value() ? contracted_axis(*identity) : axis_desc{};
    });
}

void analyze_dot(symbolic_op_info& info)
{
    analyze_axes(info, [&](io_ref io, std::size_t axis) {
        if(io.is_output)
            return parallel_axis();
        std::size_t rank = info.input_shapes.at(io.index).ndim();
        assert(rank >= 2);
        std::size_t contraction_axis = (io.index == 0) ? rank - 1 : rank - 2;
        return axis == contraction_axis ? masked_axis(mask_role::contracted, fill_kind::zero)
                                        : parallel_axis();
    });
}

std::vector<shape::dynamic_dimension> symbolic_broadcast_dims(const operation& op)
{
    if(not contains({"broadcast", "multibroadcast", "broadcast_with_dims"}, op.name()))
        return {};
    return from_value<std::vector<shape::dynamic_dimension>>(op.to_value().at("out_dyn_dims"));
}

instruction_ref freeze_broadcast(module& m,
                                 instruction_ref source,
                                 const std::vector<instruction_ref>& args,
                                 const freeze_map& freeze)
{
    const auto& op   = source->get_operator();
    auto output_dims = symbolic_broadcast_dims(op);
    std::vector<std::size_t> lens(output_dims.size());
    std::transform(output_dims.begin(), output_dims.end(), lens.begin(), [&](const auto& d) {
        return d.sym_expr.eval_uint(freeze);
    });
    if(op.name() == "broadcast")
    {
        auto axis = op.to_value().at("axis").to<std::size_t>();
        return m.add_instruction(make_op("broadcast", {{"axis", axis}, {"out_lens", lens}}), args);
    }
    if(not contains({"multibroadcast", "broadcast_with_dims"}, op.name()))
        MIGRAPHX_THROW("SPLIT_SYM_DIM: unsupported symbolic broadcast " + op.name());
    return m.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}), args);
}

std::optional<shape> symbolic_allocate_shape(const operation& op)
{
    if(op.name() != "allocate")
        return std::nullopt;
    auto attributes    = op.to_value();
    const auto& target = attributes.at("shape");
    if(target.is_null())
        return std::nullopt;
    auto s = from_value<shape>(target);
    if(not s.symbolic())
        return std::nullopt;
    return s;
}

instruction_ref freeze_allocate(module& m,
                                instruction_ref source_ins,
                                const std::vector<instruction_ref>& args,
                                const freeze_map& freeze)
{
    auto source = symbolic_allocate_shape(source_ins->get_operator());
    assert(source.has_value());
    std::vector<std::size_t> lens(source->ndim());
    std::transform(source->dyn_dims().begin(),
                   source->dyn_dims().end(),
                   lens.begin(),
                   [&](const auto& d) { return d.sym_expr.eval_uint(freeze); });
    std::vector<std::size_t> strides(source->ndim());
    std::transform(source->dyn_strides().begin(),
                   source->dyn_strides().end(),
                   strides.begin(),
                   [&](const auto& stride) { return stride.eval_uint(freeze); });
    shape target{source->type(), lens, strides};
    return m.add_instruction(make_op("allocate", {{"shape", to_value(target)}}), args);
}

operation reshape_from_shape(const shape& target, const optimal_map& substitutions)
{
    std::vector<dim_like> dims(target.ndim());
    std::transform(
        target.dyn_dims().begin(), target.dyn_dims().end(), dims.begin(), [&](const auto& d) {
            return shape::dynamic_dimension{d.sym_expr.subs(substitutions)};
        });
    return make_op("reshape", {{"dims", to_value(dims)}});
}

instruction_ref freeze_reshape(module& m,
                               instruction_ref source,
                               const std::vector<instruction_ref>& args,
                               const freeze_map& freeze)
{
    assert(source->inputs().size() == 2);
    assert(args.size() == 1);
    const auto& target = source->inputs().back()->get_shape();
    std::vector<int64_t> dims(target.ndim());
    std::transform(
        target.dyn_dims().begin(), target.dyn_dims().end(), dims.begin(), [&](const auto& d) {
            return static_cast<int64_t>(d.sym_expr.eval_uint(freeze));
        });
    return m.add_instruction(make_op("reshape", {{"dims", dims}}), args);
}

bool is_symbolic_broadcast(const operation& op, std::size_t ninputs)
{
    if(symbolic_broadcast_dims(op).empty())
        return false;
    if(op.name() == "multibroadcast")
        return ninputs >= 2;
    return ninputs == 2;
}

bool matches_broadcast(const operation& op)
{
    return contains({"broadcast", "multibroadcast", "broadcast_with_dims"}, op.name());
}

void analyze_broadcast(symbolic_op_info& info)
{
    if(not is_symbolic_broadcast(info.ins->get_operator(), info.input_shapes.size()))
        return;
    info.freezer = freeze_broadcast;
    info.shape_input_indices.resize(info.input_shapes.size() - 1);
    std::iota(info.shape_input_indices.begin(), info.shape_input_indices.end(), std::size_t{1});
    analyze_axes(info, [](io_ref, std::size_t) { return parallel_axis(); });
}

bool matches_allocate(const operation& op) { return op.name() == "allocate"; }

void analyze_allocate(symbolic_op_info& info)
{
    if(info.input_shapes.size() != 1 or
       not symbolic_allocate_shape(info.ins->get_operator()).has_value())
        return;
    info.freezer             = freeze_allocate;
    info.shape_input_indices = {0};
    analyze_axes(info, [](io_ref, std::size_t) { return parallel_axis(); });
}

void analyze_shape_transform(symbolic_op_info& info)
{
    const auto& op     = info.ins->get_operator();
    const auto& inputs = info.input_shapes;
    auto descriptor_op = op;
    const auto& output = info.output_shape;
    if(op.name() == "reshape" and inputs.size() == 1)
    {
        auto non_unit_dims = [](const shape& s) {
            std::vector<sym::expr> result;
            std::transform(s.dyn_dims().begin(),
                           s.dyn_dims().end(),
                           std::back_inserter(result),
                           [](const auto& d) { return d.sym_expr; });
            result.erase(std::remove(result.begin(), result.end(), sym::lit(1)), result.end());
            return result;
        };
        if(non_unit_dims(inputs.front()) == non_unit_dims(output))
        {
            analyze_axes(info, [](io_ref, std::size_t) { return parallel_axis(); });
            return;
        }
    }
    if(op.name() == "reshape" and inputs.size() == 2 and inputs.back().symbolic())
    {
        auto target_reshape  = reshape_from_shape(inputs.back(), {});
        descriptor_op        = make_op("reshape", {{"dims", to_value(inputs.back().max_lens())}});
        auto input_elements  = inputs.front().sym_elements();
        auto output_elements = output.sym_elements();
        if(sym::strict_less(input_elements, output_elements).value_or(false) or
           sym::strict_less(output_elements, input_elements).value_or(false))
            return;
        if(target_reshape.compute_shape({inputs.front()}) != output)
            return;
        info.freezer             = freeze_reshape;
        info.shape_input_indices = {1};
    }
    else if(inputs.size() != 1)
        return;
    auto desc = shape_transform_descriptor::create(inputs.front().max_lens(), {descriptor_op});
    if(desc.empty())
        return;
    auto source_dims = inputs.front().to_symbolic().dyn_dims();
    auto output_dims = output.to_symbolic().dyn_dims();
    analyze_axes(info, [&](io_ref io, std::size_t axis) {
        if(not io.is_output)
        {
            auto dst_axes = desc.get_dst_axes_from_src(axis);
            if(dst_axes.size() != 1)
                return axis_desc{};
            auto dst_axis = dst_axes.front();
            return source_dims.at(axis).sym_expr == output_dims.at(dst_axis).sym_expr
                       ? parallel_axis()
                       : axis_desc{};
        }

        auto source_axes = range(source_dims.size());
        auto count = std::count_if(source_axes.begin(), source_axes.end(), [&](auto source_axis) {
            auto dst_axes = desc.get_dst_axes_from_src(source_axis);
            return dst_axes.size() == 1 and dst_axes.front() == axis and
                   source_dims.at(source_axis).sym_expr == output_dims.at(axis).sym_expr;
        });
        return count == 1 ? parallel_axis() : axis_desc{};
    });
}

std::optional<std::size_t> normalize_axis(int64_t axis, std::size_t rank)
{
    if(axis < 0)
        axis += static_cast<int64_t>(rank);
    if(axis < 0 or axis >= static_cast<int64_t>(rank))
        return std::nullopt;
    return static_cast<std::size_t>(axis);
}

bool matches_gather(const operation& op) { return op.name() == "gather"; }

void analyze_gather(symbolic_op_info& info)
{
    const auto& inputs = info.input_shapes;
    if(inputs.size() != 2)
        return;
    auto axis = normalize_axis(info.ins->get_operator().to_value().at("axis").to<int64_t>(),
                               inputs.front().ndim());
    if(not axis.has_value())
        return;
    analyze_axes(info, [axis = *axis](io_ref io, std::size_t current_axis) {
        if(io.is_output or io.index == 1)
            return parallel_axis();
        return current_axis == axis ? axis_desc{} : parallel_axis();
    });
}

bool matches_concat(const operation& op) { return op.name() == "concat"; }

void analyze_concat(symbolic_op_info& info)
{
    const auto& inputs = info.input_shapes;
    if(inputs.empty())
        return;
    auto axis = normalize_axis(info.ins->get_operator().to_value().at("axis").to<int64_t>(),
                               inputs.front().ndim());
    if(not axis.has_value())
        return;
    if(any_of(inputs, [&](const auto& input) {
           return input.ndim() != inputs.front().ndim() or
                  is_variable_axis(input.dyn_dims().at(*axis));
       }))
        return;
    analyze_axes(info, [](io_ref, std::size_t) { return parallel_axis(); });
}

bool matches_unit_axis_transform(const operation& op)
{
    return op.name() == "squeeze" or op.name() == "unsqueeze";
}

void analyze_unit_axis_transform(symbolic_op_info& info)
{
    if(info.input_shapes.size() != 1)
        return;
    analyze_axes(info, [](io_ref, std::size_t) { return parallel_axis(); });
}

bool matches_fill(const operation& op) { return op.name() == "fill"; }

void analyze_fill(symbolic_op_info& info)
{
    if(info.input_shapes.size() != 2)
        return;
    analyze_axes(info, [](io_ref, std::size_t) { return parallel_axis(); });
}

std::optional<shape::dynamic_dimension> symbolic_range_dim(const operation& op)
{
    if(op.name() != "dynamic_range")
        return std::nullopt;
    auto attributes = op.to_value();
    if(not attributes.contains("output_dim") or attributes.at("output_dim").is_null())
        return std::nullopt;
    auto output_dim = from_value<shape::dynamic_dimension>(attributes.at("output_dim"));
    if(not output_dim.is_symbolic())
        return std::nullopt;
    return output_dim;
}

instruction_ref freeze_dynamic_range(module& m,
                                     instruction_ref source,
                                     const std::vector<instruction_ref>& args,
                                     const freeze_map& freeze)
{
    assert(args.size() == 3);
    auto output_dim = symbolic_range_dim(source->get_operator());
    assert(output_dim.has_value());
    auto length = output_dim->sym_expr.eval_uint(freeze);
    std::vector<int64_t> indices(length);
    std::iota(indices.begin(), indices.end(), int64_t{0});
    auto index = m.add_literal(literal{shape{shape::int64_type, {length}}, indices});
    auto start =
        m.add_instruction(make_op("multibroadcast", {{"out_lens", {length}}}), args.front());
    auto delta =
        m.add_instruction(make_op("multibroadcast", {{"out_lens", {length}}}), args.back());
    auto scaled = m.add_instruction(make_op("mul"), index, delta);
    return m.add_instruction(make_op("add"), start, scaled);
}

bool matches_dynamic_range(const operation& op) { return op.name() == "dynamic_range"; }

void analyze_dynamic_range(symbolic_op_info& info)
{
    const auto& op     = info.ins->get_operator();
    const auto& inputs = info.input_shapes;
    if(inputs.size() != 3 or inputs.front().type() != shape::int64_type or
       not symbolic_range_dim(op).has_value())
        return;
    info.freezer = freeze_dynamic_range;
    analyze_axes(info, [](io_ref, std::size_t) { return parallel_axis(); });
}

instruction_ref freeze_dyn_slice(module& m,
                                 instruction_ref source,
                                 const std::vector<instruction_ref>& args,
                                 const freeze_map& freeze)
{
    assert(args.size() == 3);
    const auto& op            = source->get_operator();
    auto attributes           = op.to_value();
    auto evaluate_expressions = [&](const std::string& key) {
        auto expressions = from_value<std::vector<sym::expr>>(attributes.at(key));
        std::vector<int64_t> result(expressions.size());
        std::transform(expressions.begin(), expressions.end(), result.begin(), [&](const auto& e) {
            return static_cast<int64_t>(e.eval_uint(freeze));
        });
        return result;
    };
    return m.add_instruction(make_op("slice",
                                     {{"axes", attributes.at("axes")},
                                      {"starts", evaluate_expressions("starts")},
                                      {"ends", evaluate_expressions("ends")}}),
                             args.front());
}

bool is_prefix_stable_dyn_slice(const operation& op, const shape& input)
{
    auto attributes = op.to_value();
    if(not attributes.contains("axes") or not attributes.contains("starts") or
       not attributes.contains("ends"))
        return false;
    auto axes   = attributes.at("axes").to_vector<int64_t>();
    auto starts = from_value<std::vector<sym::expr>>(attributes.at("starts"));
    auto ends   = from_value<std::vector<sym::expr>>(attributes.at("ends"));
    if(axes.size() != starts.size() or axes.size() != ends.size())
        return false;

    auto input_dims = input.to_symbolic().dyn_dims();
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

bool matches_slice(const operation& op) { return op.name() == "slice" or op.name() == "dyn_slice"; }

void analyze_slice(symbolic_op_info& info)
{
    const auto& op     = info.ins->get_operator();
    const auto& inputs = info.input_shapes;
    if(op.name() == "slice")
    {
        if(inputs.size() != 1)
            return;
        auto axes = op.to_value().at("axes").to_vector<int64_t>();
        for(auto& axis : axes)
        {
            auto normalized_axis = normalize_axis(axis, inputs.front().ndim());
            if(not normalized_axis.has_value() or
               is_variable_axis(inputs.front().dyn_dims().at(*normalized_axis)))
                return;
            axis = static_cast<int64_t>(*normalized_axis);
        }
        analyze_axes(info, [](io_ref, std::size_t) { return parallel_axis(); });
        return;
    }

    if(inputs.size() != 3 or not is_prefix_stable_dyn_slice(op, inputs.front()))
        return;
    info.freezer = freeze_dyn_slice;
    analyze_axes(info, [](io_ref, std::size_t) { return parallel_axis(); });
}

using scatter_prefix_axis = std::pair<std::size_t, sym::expr>;

instruction_ref make_scatter_prefix_mask(module& m,
                                         const std::vector<std::size_t>& prefix_lens,
                                         const std::vector<scatter_prefix_axis>& prefix_axes)
{
    assert(not prefix_axes.empty());
    std::optional<instruction_ref> valid;
    auto sources = m.get_parameters();
    for(const auto& [axis, extent_expr] : prefix_axes)
    {
        std::vector<int64_t> positions(prefix_lens.at(axis));
        std::iota(positions.begin(), positions.end(), int64_t{0});
        auto index =
            m.add_literal(literal{shape{shape::int64_type, {positions.size()}}, positions});
        index = m.add_instruction(make_op("broadcast", {{"axis", axis}, {"out_lens", prefix_lens}}),
                                  index);
        auto extent = m.add_instruction(
            make_op("eval_expr_from_shape",
                    {{"expressions", to_value(std::vector<sym::expr>{extent_expr})}}),
            sources);
        extent = m.add_instruction(make_op("multibroadcast", {{"out_lens", prefix_lens}}), extent);
        auto current = m.add_instruction(make_op("less"), index, extent);
        valid = valid.has_value() ? m.add_instruction(make_op("logical_and"), *valid, current)
                                  : current;
    }
    return *valid;
}

instruction_ref freeze_scatternd(module& m,
                                 instruction_ref source,
                                 const std::vector<instruction_ref>& args,
                                 const freeze_map&)
{
    assert(args.size() == 3);
    const auto& index_shape = source->inputs().at(1)->get_shape();
    auto index_rank         = index_shape.ndim();
    assert(index_rank > 0);
    std::vector<scatter_prefix_axis> prefix_axes;
    const auto& index_dims = index_shape.dyn_dims();
    for(std::size_t axis = 0; axis + 1 < index_rank; ++axis)
        if(is_variable_axis(index_dims.at(axis)))
            prefix_axes.emplace_back(axis, index_dims.at(axis).sym_expr);
    assert(not prefix_axes.empty());

    const auto& op  = source->get_operator();
    auto data       = args.front();
    auto indices    = args.at(1);
    auto updates    = args.back();
    auto index_lens = indices->get_shape().lens();
    assert(not index_lens.empty());
    auto index_depth = index_lens.back();
    index_lens.pop_back();

    auto valid                 = make_scatter_prefix_mask(m, index_lens, prefix_axes);
    auto effective_index_depth = index_depth;
    if(index_depth == 0)
    {
        data                  = m.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), data);
        effective_index_depth = 1;
    }

    auto data_lens = data->get_shape().lens();
    assert(effective_index_depth <= data_lens.size());
    std::vector<int64_t> pads(data_lens.size() * 2, 0);
    std::fill(pads.begin() + data_lens.size(),
              pads.begin() + data_lens.size() + effective_index_depth,
              int64_t{1});
    auto padded_data = m.add_instruction(make_op("pad", {{"pads", pads}}), data);

    auto rewritten_index_lens = index_lens;
    rewritten_index_lens.push_back(effective_index_depth);
    auto condition =
        m.add_instruction(make_op("unsqueeze", {{"axes", {index_lens.size()}}}), valid);
    condition = m.add_instruction(make_op("multibroadcast", {{"out_lens", rewritten_index_lens}}),
                                  condition);

    std::vector<int64_t> sink_values(effective_index_depth);
    std::transform(data_lens.begin(),
                   data_lens.begin() + effective_index_depth,
                   sink_values.begin(),
                   [](auto x) { return static_cast<int64_t>(x); });
    auto sink =
        m.add_literal(literal{shape{shape::int64_type, {effective_index_depth}}, sink_values});
    sink = m.add_instruction(make_op("multibroadcast", {{"out_lens", rewritten_index_lens}}), sink);

    if(index_depth == 0)
    {
        std::vector<int64_t> zeros(effective_index_depth, 0);
        indices = m.add_literal(literal{shape{shape::int64_type, {effective_index_depth}}, zeros});
        indices = m.add_instruction(make_op("multibroadcast", {{"out_lens", rewritten_index_lens}}),
                                    indices);
    }
    indices = m.add_instruction(make_op("where"), condition, indices, sink);

    auto scattered = m.add_instruction(op, padded_data, indices, updates);
    std::vector<int64_t> axes(effective_index_depth);
    std::iota(axes.begin(), axes.end(), int64_t{0});
    std::vector<int64_t> starts(effective_index_depth, 0);
    std::vector<int64_t> ends(effective_index_depth);
    std::transform(data_lens.begin(),
                   data_lens.begin() + effective_index_depth,
                   ends.begin(),
                   [](auto x) { return static_cast<int64_t>(x); });
    auto result = m.add_instruction(
        make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}), scattered);
    if(index_depth == 0)
        result = m.add_instruction(make_op("squeeze", {{"axes", {0}}}), result);
    return m.add_instruction(make_op("contiguous"), result);
}

bool matches_scatternd(const operation& op) { return op.name() == "scatternd_none"; }

void analyze_scatternd(symbolic_op_info& info)
{
    const auto& inputs = info.input_shapes;
    if(inputs.size() != 3)
        return;
    const auto& indices = inputs.at(1);
    auto index_rank     = indices.ndim();
    assert(index_rank > 0);
    const auto& index_dims = indices.dyn_dims();
    bool has_prefix_axis   = false;
    for(std::size_t axis = 0; axis + 1 < index_rank; ++axis)
        has_prefix_axis = has_prefix_axis or is_variable_axis(index_dims.at(axis));
    if(has_prefix_axis and (indices.type() != shape::int64_type or
                            any_of(inputs, [](const auto& input) { return not input.standard(); })))
        return;
    if(has_prefix_axis)
        info.freezer = freeze_scatternd;
    analyze_axes(info, [](io_ref, std::size_t) { return parallel_axis(); });
}

void analyze_softmax(symbolic_op_info& info)
{
    const auto& inputs = info.input_shapes;
    if(inputs.size() != 1)
        return;
    auto axis = normalize_axis(info.ins->get_operator().to_value().at("axis").to<int64_t>(),
                               inputs.front().ndim());
    if(not axis.has_value())
        return;
    analyze_axes(info, [axis = *axis](io_ref io, std::size_t current_axis) {
        if(io.is_output)
            return parallel_axis();
        return current_axis == axis ? masked_axis(mask_role::normalized, fill_kind::neg_inf)
                                    : parallel_axis();
    });
}

void analyze_conv(symbolic_op_info& info)
{
    const auto& op       = info.ins->get_operator();
    bool default_padding = false;
    std::size_t group    = 0;
    std::vector<std::size_t> padding;
    std::size_t spatial_dimensions = 0;
    if(op.name() == "convolution" or op.name() == "quant_convolution")
    {
        auto attributes = op.to_value();
        default_padding =
            attributes.at("padding_mode").to<op::padding_mode_t>() == op::padding_mode_t::default_;
        group              = attributes.at("group").to<std::size_t>();
        padding            = attributes.at("padding").to_vector<std::size_t>();
        spatial_dimensions = attributes.at("stride").to_vector<std::size_t>().size();
    }
    analyze_axes(info, [&](io_ref io, std::size_t axis) {
        auto spatial = [&](std::size_t spatial_axis) {
            if(not default_padding)
                return axis_desc{};
            return padded_axis(fill_kind::zero,
                               windowed_zero_pad(padding, spatial_dimensions, spatial_axis));
        };
        if(io.is_output)
            return axis < 2 ? parallel_axis() : spatial(axis);
        if(io.index == 0)
        {
            if(axis == 0)
                return parallel_axis();
            if(axis == 1)
            {
                if(group != 1)
                    return axis_desc{};
                return contracted_axis(fill_kind::zero);
            }
            return spatial(axis);
        }
        if(io.index != 1 or group != 1)
            return axis_desc{};
        if(axis == 0)
            return parallel_axis();
        if(axis == 1)
            return contracted_axis(fill_kind::zero);
        return axis_desc{};
    });
}

void analyze_pooling(symbolic_op_info& info)
{
    auto attributes = info.ins->get_operator().to_value();
    auto mode       = attributes.at("mode").to<op::pooling_mode>();
    auto ceil_mode  = attributes.at("ceil_mode").to<bool>();
    fill_kind fill  = fill_kind::zero;
    if(mode == op::pooling_mode::max)
        fill = fill_kind::lowest;
    else if(mode == op::pooling_mode::average and
            (not attributes.at("count_include_pad").to<bool>() or ceil_mode))
        fill = fill_kind::dont_care;
    if(attributes.at("dyn_global").to<bool>() and mode == op::pooling_mode::average)
        fill = fill_kind::dont_care;
    auto default_padding =
        attributes.at("padding_mode").to<op::padding_mode_t>() == op::padding_mode_t::default_;
    auto padding            = attributes.at("padding").to_vector<std::size_t>();
    auto spatial_dimensions = attributes.at("stride").to_vector<std::size_t>().size();
    analyze_axes(info, [&](io_ref, std::size_t axis) {
        if(axis < 2)
            return parallel_axis();
        if(not default_padding or fill == fill_kind::dont_care)
            return axis_desc{};
        return padded_axis(fill,
                           not ceil_mode and windowed_zero_pad(padding, spatial_dimensions, axis));
    });
}

struct op_family
{
    bool (*matches)(const operation&);
    void (*analyze)(symbolic_op_info&);
};

void analyze_op(symbolic_op_info& info)
{
    static const std::array<op_family, 16> families = {{
        {matches_gather, analyze_gather},
        {matches_concat, analyze_concat},
        {matches_slice, analyze_slice},
        {matches_unit_axis_transform, analyze_unit_axis_transform},
        {matches_fill, analyze_fill},
        {matches_dynamic_range, analyze_dynamic_range},
        {matches_scatternd, analyze_scatternd},
        {is_pointwise, analyze_pointwise},
        {is_reduce, analyze_reduce},
        {is_dot, analyze_dot},
        {matches_broadcast, analyze_broadcast},
        {matches_allocate, analyze_allocate},
        {is_shape_transform, analyze_shape_transform},
        {is_softmax, analyze_softmax},
        {is_conv, analyze_conv},
        {is_pooling, analyze_pooling},
    }};
    const auto& op                                  = info.ins->get_operator();
    for(const auto& family : families)
    {
        if(family.matches(op))
        {
            family.analyze(info);
            return;
        }
    }
}

bool has_symbolic_param(const module& m)
{
    auto param_shapes = m.get_parameter_shapes();
    return any_of(param_shapes, [](const auto& p) { return p.second.symbolic(); });
}

std::unordered_map<sym::expr, instruction_ref> find_root_sources(const module& m)
{
    std::unordered_map<sym::expr, instruction_ref> result;
    for(const auto& name : m.get_parameter_names())
    {
        auto parameter = m.get_parameter(name);
        const auto& s  = parameter->get_shape();
        if(not s.symbolic())
            continue;
        for(const auto& d : s.dyn_dims())
            if(d.is_symbolic() and d.sym_expr.name() == "variable")
                result.emplace(sym::as_symbol(d.sym_expr), parameter);
    }
    return result;
}

struct resolve_symbolic_dimensions_of_match : match::supports_dynamic_shapes
{
    std::unordered_map<sym::expr, instruction_ref> root_sources;
    std::vector<instruction_ref> sources;

    auto matcher() const { return match::name("dimensions_of")(match::nargs(1)); }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins                = mr.result;
        const auto& input_shape = ins->inputs().front()->get_shape();
        if(not input_shape.symbolic())
            return;
        const auto symbolic_value = ins->sym_eval();
        if(symbolic_value.empty())
            return;
        const auto expressions = symbolic_value.get().to_vector();
        if(any_of(expressions, [&](const auto& expression) {
               auto variables = sym::find_variables(expression);
               return any_of(variables, [&](const auto& variable) {
                   return not contains(root_sources, variable);
               });
           }))
            return;
        m.replace_instruction(
            ins,
            make_op("eval_expr_from_shape", {{"expressions", to_value(expressions)}}),
            sources);
    }
};

std::vector<instruction_ref> find_symbolic_instructions(const module& m)
{
    return find_all(iterator_for(m),
                    [](instruction_ref ins) { return ins->get_shape().symbolic(); });
}

symbolic_op_info analyze_instruction(instruction_ref ins)
{
    auto input_shapes = to_shapes(ins->inputs());
    symbolic_op_info info;
    info.ins          = ins;
    info.output_shape = ins->get_shape();
    info.input_shapes = std::move(input_shapes);
    analyze_op(info);
    return info;
}

std::vector<symbolic_op_info>
analyze_symbolic_instructions(const std::vector<instruction_ref>& instructions)
{
    std::vector<symbolic_op_info> infos;
    for(auto ins : instructions)
    {
        if(starts_with(ins->name(), "@"))
            continue;
        infos.push_back(analyze_instruction(ins));
    }
    return infos;
}

void elide_masks_zeroed_by_softmax(std::vector<symbolic_op_info>& infos)
{
    // Softmax -inf masking zeroes the padded tail consumed by matching contractions.
    auto zeros_contracted_region = [](const axis_mask& source, const axis_mask& consumer) {
        return source.role == mask_role::normalized and source.fill == fill_kind::neg_inf and
               consumer.role == mask_role::contracted and consumer.fill == fill_kind::zero and
               source.axis == consumer.axis and sym::same_symbol(source.extent, consumer.extent);
    };
    std::unordered_map<instruction_ref, const symbolic_op_info*> info_map;
    for(const auto& info : infos)
        info_map.emplace(info.ins, &info);
    for(auto& info : infos)
    {
        const auto& args = info.ins->inputs();
        std::vector<sym::expr> zeroed_contractions;
        for(auto&& [arg, operand] : views::zip(args, info.operands))
        {
            if(not contains(info_map, arg))
                continue;
            const auto* source = info_map.at(arg);
            if(not is_softmax(source->ins->get_operator()) or source->operands.empty())
                continue;
            const auto& source_masks = source->operands.front().masks;
            auto& masks              = operand.masks;
            masks.erase(
                std::remove_if(masks.begin(),
                               masks.end(),
                               [&](const auto& mask) {
                                   if(not any_of(source_masks, [&](const auto& source_mask) {
                                          return zeros_contracted_region(source_mask, mask);
                                      }))
                                       return false;
                                   zeroed_contractions.push_back(mask.extent);
                                   return true;
                               }),
                masks.end());
        }
        if(zeroed_contractions.empty() or not is_dot(info.ins->get_operator()))
            continue;
        for(auto& operand : info.operands)
        {
            auto& masks = operand.masks;
            masks.erase(
                std::remove_if(masks.begin(),
                               masks.end(),
                               [&](const auto& mask) {
                                   return mask.role == mask_role::contracted and
                                          mask.fill == fill_kind::zero and
                                          any_of(zeroed_contractions, [&](const auto& extent) {
                                              return sym::same_symbol(mask.extent, extent);
                                          });
                               }),
                masks.end());
        }
    }
}

struct root_spec
{
    sym::expr root;
    std::string name;
    shape::dynamic_dimension::interval interval;
    sym::expr optimal_symbol;
    std::vector<std::size_t> optimal_values;
    std::vector<shape::dynamic_dimension::interval> subranges;
};

std::optional<std::vector<root_spec>> collect_roots(const module& m)
{
    std::unordered_map<sym::expr, root_spec> root_specs;
    auto parameter_names = m.get_parameter_names();
    for(const auto& parameter_name : parameter_names)
    {
        const auto& s = m.get_parameter(parameter_name)->get_shape();
        if(s.dynamic() and not s.symbolic())
            return std::nullopt;
        if(not s.symbolic())
            continue;
        const auto& dims = s.dyn_dims();
        for(const auto& d : dims)
        {
            if(not d.is_symbolic())
                continue;
            if(d.sym_expr.name() != "variable")
            {
                if(is_variable_axis(d))
                    return std::nullopt;
                continue;
            }
            auto root        = sym::as_symbol(d.sym_expr);
            auto name        = root.to_string();
            auto interval    = d.get_interval();
            auto optimal_set = d.get_optimals();
            if(any_of(optimal_set, [&](auto x) { return x < interval.min or x > interval.max; }))
                return std::nullopt;
            optimal_set.insert(interval.min);
            optimal_set.insert(interval.max);
            std::vector<std::size_t> optimal_values(optimal_set.begin(), optimal_set.end());

            if(contains(root_specs, root))
            {
                const auto& existing = root_specs.at(root);
                if(existing.interval != interval or existing.optimal_values != optimal_values)
                    return std::nullopt;
                continue;
            }

            root_specs.emplace(
                root,
                root_spec{root, std::move(name), interval, {}, std::move(optimal_values), {}});
        }
    }
    if(root_specs.empty())
        return std::nullopt;

    std::vector<root_spec> roots;
    roots.reserve(root_specs.size());
    for(auto& entry : root_specs)
        roots.push_back(std::move(entry.second));
    std::sort(
        roots.begin(), roots.end(), [](const auto& x, const auto& y) { return x.name < y.name; });
    std::unordered_set<std::string> symbol_names;
    for(const auto& root : roots)
        symbol_names.insert(root.name);

    for(auto& root : roots)
    {
        std::string optimal_name = "#split_sym_dim_" + root.name + "_opt";
        while(contains(symbol_names, optimal_name))
            optimal_name += "_";
        symbol_names.insert(optimal_name);
        std::set<sym::scalar> optimal_scalars;
        std::transform(root.optimal_values.begin(),
                       root.optimal_values.end(),
                       std::inserter(optimal_scalars, optimal_scalars.end()),
                       [](auto x) { return sym::scalar{x}; });
        if(root.interval.min == root.interval.max)
            root.optimal_symbol = sym::lit(root.interval.min);
        else
            root.optimal_symbol = sym::var(std::move(optimal_name),
                                           {root.interval.min, root.interval.max},
                                           std::move(optimal_scalars));

        auto lower = root.interval.min;
        root.subranges.reserve(root.optimal_values.size());
        for(auto upper : root.optimal_values)
        {
            root.subranges.push_back({lower, upper});
            lower = upper + 1;
        }
    }
    return roots;
}

struct block_plan
{
    std::vector<const symbolic_op_info*> ops;
    std::vector<root_spec> roots;
    std::size_t clone_count = 0;
};

struct slice_spec
{
    std::vector<int64_t> axes;
    std::vector<sym::expr> starts;
    std::vector<sym::expr> ends;
};

struct clone_input
{
    instruction_ref source;
    std::vector<std::size_t> slice_axes;
};

bool operator==(const clone_input& x, const clone_input& y)
{
    return x.source == y.source and x.slice_axes == y.slice_axes;
}

struct clone_recipe
{
    std::vector<clone_input> inputs;
    std::vector<operand_plan> operands;
    op_freezer freezer;
    shape dispatch_output;
    slice_spec output_slice;
};

using clone_recipe_map = std::unordered_map<instruction_ref, clone_recipe>;

shape substitute_shape(const shape& s, const optimal_map& substitutions)
{
    if(not s.symbolic())
        return s;
    std::vector<shape::dynamic_dimension> dimensions(s.ndim());
    std::transform(
        s.dyn_dims().begin(), s.dyn_dims().end(), dimensions.begin(), [&](const auto& d) {
            return shape::dynamic_dimension{d.sym_expr.subs(substitutions)};
        });
    std::vector<sym::expr> strides(s.ndim());
    std::transform(s.dyn_strides().begin(),
                   s.dyn_strides().end(),
                   strides.begin(),
                   [&](const auto& stride) { return stride.subs(substitutions); });
    return {s.type(), dimensions, strides};
}

void gather_shape_roots(const shape& s, std::unordered_set<sym::expr>& result)
{
    if(not s.symbolic())
        return;
    for(const auto& d : s.dyn_dims())
    {
        auto variables = sym::find_variables(d.sym_expr);
        result.insert(variables.begin(), variables.end());
    }
    for(const auto& stride : s.dyn_strides())
    {
        auto variables = sym::find_variables(stride);
        result.insert(variables.begin(), variables.end());
    }
}

struct selected_block_roots
{
    std::vector<root_spec> roots;
    std::size_t clone_count = 1;
};

std::optional<selected_block_roots> select_block_roots(const block_plan& block,
                                                       const std::vector<root_spec>& roots,
                                                       std::size_t max_clones)
{
    std::unordered_set<sym::expr> required;
    for(const auto* op : block.ops)
    {
        const auto& info = *op;
        gather_shape_roots(info.output_shape, required);
        for(const auto& input : info.input_shapes)
            gather_shape_roots(input, required);
        for(const auto& operand : info.operands)
            for(const auto& mask : operand.masks)
            {
                auto variables = sym::find_variables(mask.extent);
                required.insert(variables.begin(), variables.end());
            }
    }

    selected_block_roots result;
    for(const auto& root : roots)
    {
        if(not contains(required, root.root))
            continue;
        required.erase(root.root);
        if(root.optimal_values.size() >
           std::numeric_limits<std::size_t>::max() / result.clone_count)
            return std::nullopt;
        result.clone_count *= root.optimal_values.size();
        if(max_clones != 0 and result.clone_count > max_clones)
            return std::nullopt;
        result.roots.push_back(root);
    }
    if(not required.empty() or result.roots.empty())
        return std::nullopt;
    return result;
}

bool can_specialize(const symbolic_op_info& info)
{
    return not info.output_symbolic_axes.empty() and info.supported and
           info.ins->module_inputs().empty() and
           (info.freezer or any_of(info.operands, [](const auto& operand) {
                return operand.pad_value.has_value();
            }));
}

bool absorbable_dependency(instruction_ref ins, const std::unordered_set<instruction_ref>& planned)
{
    if(contains(planned, ins))
        return true;
    if(starts_with(ins->name(), "@") or not ins->module_inputs().empty())
        return false;
    const auto& s = ins->get_shape();
    if(not s.dynamic())
        return true;
    if(not s.symbolic())
        return false;
    return all_of(s.dyn_strides(),
                  [](const auto& stride) { return sym::fixed_value(stride).has_value(); });
}

bool block_is_closed(const block_plan& block)
{
    std::unordered_set<instruction_ref> included;
    for(const auto* op : block.ops)
        included.insert(op->ins);

    for(const auto* op : block.ops)
    {
        const auto& info = *op;
        const auto& args = info.ins->inputs();
        assert(args.size() == info.operands.size());
        for(std::size_t index = 0; index < args.size(); ++index)
        {
            if(contains(info.shape_input_indices, index))
                continue;
            auto source         = args.at(index);
            const auto& operand = info.operands.at(index);
            if(contains(included, source))
            {
                if(operand.pad_value.has_value() and not operand.retained_slice_axes.empty())
                    return false;
                continue;
            }

            std::vector<instruction_ref> stack = {source};
            std::unordered_set<instruction_ref> visited;
            while(not stack.empty())
            {
                auto current = stack.back();
                stack.pop_back();
                if(not visited.insert(current).second)
                    continue;
                if(contains(included, current))
                    continue;
                if(absorbable_dependency(current, included))
                {
                    stack.insert(stack.end(), current->inputs().begin(), current->inputs().end());
                    continue;
                }

                std::vector<instruction_ref> boundary = {current};
                std::unordered_set<instruction_ref> boundary_visited;
                while(not boundary.empty())
                {
                    auto dependency = boundary.back();
                    boundary.pop_back();
                    if(not boundary_visited.insert(dependency).second)
                        continue;
                    if(contains(included, dependency))
                        return false;
                    boundary.insert(
                        boundary.end(), dependency->inputs().begin(), dependency->inputs().end());
                }
            }
        }
    }
    return true;
}

bool blocks_connected(const block_plan& x, const block_plan& y)
{
    std::unordered_set<instruction_ref> x_instructions;
    std::unordered_set<instruction_ref> y_instructions;
    for(const auto* op : x.ops)
        x_instructions.insert(op->ins);
    for(const auto* op : y.ops)
        y_instructions.insert(op->ins);

    auto has_edge = [&](const block_plan& source,
                        const std::unordered_set<instruction_ref>& targets) {
        return any_of(source.ops, [&](const auto* op) {
            return any_of(op->ins->inputs(),
                          [&](instruction_ref input) { return contains(targets, input); });
        });
    };
    return has_edge(x, y_instructions) or has_edge(y, x_instructions);
}

bool merge_block_into(block_plan& target,
                      const block_plan& source,
                      const std::vector<root_spec>& roots,
                      std::size_t max_clones)
{
    block_plan result;
    result.ops = target.ops;
    result.ops.insert(result.ops.end(), source.ops.begin(), source.ops.end());
    if(not block_is_closed(result))
        return false;
    auto selected = select_block_roots(result, roots, max_clones);
    if(not selected.has_value())
        return false;
    result.roots       = std::move(selected->roots);
    result.clone_count = selected->clone_count;
    target             = std::move(result);
    return true;
}

bool merge_one_block_pair(std::vector<block_plan>& blocks,
                          const std::vector<root_spec>& roots,
                          std::size_t max_clones,
                          bool require_edge)
{
    for(auto target = blocks.begin(); target != blocks.end(); ++target)
    {
        for(auto source = std::next(target); source != blocks.end(); ++source)
        {
            if(require_edge and not blocks_connected(*target, *source))
                continue;
            if(not merge_block_into(*target, *source, roots, max_clones))
                continue;
            blocks.erase(source);
            return true;
        }
    }
    return false;
}

std::vector<block_plan> discover_blocks(const std::vector<symbolic_op_info>& infos,
                                        const std::vector<root_spec>& roots,
                                        std::size_t max_clones)
{
    std::vector<block_plan> blocks;
    std::unordered_set<instruction_ref> specializable;
    for(const auto& info : infos)
    {
        if(can_specialize(info))
            specializable.insert(info.ins);
    }

    for(const auto& info : infos)
    {
        if(not contains(specializable, info.ins))
            continue;
        block_plan singleton{{&info}, {}, {}};
        auto selected = select_block_roots(singleton, roots, max_clones);
        if(selected.has_value())
        {
            singleton.roots       = std::move(selected->roots);
            singleton.clone_count = selected->clone_count;
            blocks.push_back(std::move(singleton));
        }
    }

    fix([&](auto self) {
        if(merge_one_block_pair(blocks, roots, max_clones, true) or
           merge_one_block_pair(blocks, roots, max_clones, false))
            self();
    })();
    return blocks;
}

using pad_cache = std::unordered_map<instruction_ref, std::vector<instruction_ref>>;

instruction_ref
add_or_reuse_pad(module& m, const operation& pad_op, instruction_ref input, pad_cache& cache)
{
    auto& candidates = cache[input];
    auto it = std::find_if(candidates.begin(), candidates.end(), [&](instruction_ref candidate) {
        return candidate->get_operator() == pad_op;
    });
    if(it != candidates.end())
        return *it;
    auto result = m.add_instruction(pad_op, input);
    candidates.push_back(result);
    return result;
}

using replacement_map       = std::unordered_map<instruction_ref, instruction_ref>;
using instruction_block_map = std::unordered_map<instruction_ref, std::size_t>;
using symbolic_info_map     = std::unordered_map<instruction_ref, const symbolic_op_info*>;

bool needs_fixed_retarget(const shape& s, const optimal_map& substitutions)
{
    if(not s.symbolic())
        return false;
    return any_of(s.dyn_dims(),
                  [&](const auto& d) {
                      return d.is_fixed() and d.sym_expr.subs(substitutions) != d.sym_expr;
                  }) or
           any_of(s.dyn_strides(),
                  [&](const auto& stride) { return stride.subs(substitutions) != stride; });
}

bool block_contains(const block_plan& block, instruction_ref ins)
{
    return any_of(block.ops, [&](const symbolic_op_info* info) { return info->ins == ins; });
}

slice_spec make_output_slice(const symbolic_op_info& info)
{
    slice_spec result;
    const auto& dimensions = info.output_shape.dyn_dims();
    for(auto axis : info.output_symbolic_axes)
    {
        assert(axis < dimensions.size());
        result.axes.push_back(axis);
        result.starts.push_back(sym::lit(int64_t{0}));
        result.ends.push_back(dimensions.at(axis).sym_expr);
    }
    return result;
}

clone_recipe make_clone_recipe(const symbolic_op_info& info,
                               const block_plan& block,
                               const instruction_block_map& block_for_instruction,
                               const symbolic_info_map& info_for_instruction,
                               const optimal_map& optimal_substitutions)
{
    const auto& args = info.ins->inputs();
    clone_recipe result;
    assert(info.input_shapes.size() == args.size());
    assert(info.operands.size() == args.size());
    for(std::size_t index = 0; index < args.size(); ++index)
    {
        if(contains(info.shape_input_indices, index))
        {
            if(not info.operands.at(index).masks.empty())
                MIGRAPHX_THROW(
                    "SPLIT_SYM_DIM: cannot remove an input that requires runtime masking");
            continue;
        }
        auto operand = info.operands.at(index);
        auto source  = args.at(index);
        clone_input input{source, {}};
        if(contains(block_for_instruction, source))
            input.slice_axes = info_for_instruction.at(source)->output_symbolic_axes;

        bool emit_pad = operand.pad_value.has_value() or
                        needs_fixed_retarget(info.input_shapes.at(index), optimal_substitutions);
        if(operand.pad_value.has_value() and contains(block_for_instruction, source) and
           (block_contains(block, source) or not operand.retained_slice_axes.empty()))
        {
            std::vector<std::size_t> kept_axes;
            std::copy_if(input.slice_axes.begin(),
                         input.slice_axes.end(),
                         std::back_inserter(kept_axes),
                         [&](auto axis) { return contains(operand.retained_slice_axes, axis); });
            if(kept_axes.empty())
            {
                input.slice_axes.clear();
                emit_pad = false;
            }
            else if(kept_axes.size() != input.slice_axes.size())
            {
                input.slice_axes = std::move(kept_axes);
            }
        }
        if(emit_pad)
            operand.pad_value = operand.pad_value.value_or(0.0f);
        else
            operand.pad_value.reset();
        result.inputs.push_back(std::move(input));
        result.operands.push_back(std::move(operand));
    }
    result.freezer         = info.freezer;
    result.dispatch_output = substitute_shape(info.output_shape, optimal_substitutions);
    result.output_slice    = make_output_slice(info);
    return result;
}

clone_recipe_map make_clone_recipes(const std::vector<block_plan>& blocks,
                                    const std::vector<root_spec>& roots)
{
    optimal_map optimal_substitutions;
    for(const auto& root : roots)
        optimal_substitutions.emplace(root.root, root.optimal_symbol);

    instruction_block_map block_for_instruction;
    symbolic_info_map info_for_instruction;
    for(std::size_t block_index = 0; block_index < blocks.size(); ++block_index)
        for(const auto* info : blocks.at(block_index).ops)
        {
            block_for_instruction.emplace(info->ins, block_index);
            info_for_instruction.emplace(info->ins, info);
        }

    clone_recipe_map result;
    for(const auto& block : blocks)
        for(const auto* info : block.ops)
            result.emplace(info->ins,
                           make_clone_recipe(*info,
                                             block,
                                             block_for_instruction,
                                             info_for_instruction,
                                             optimal_substitutions));
    return result;
}

shape clone_parameter_shape(
    const shape& s,
    const std::unordered_map<sym::expr, shape::dynamic_dimension::interval>& subranges,
    const freeze_map& freeze)
{
    if(not s.symbolic())
        return s;
    optimal_map substitutions;
    for(const auto& [root, interval] : subranges)
    {
        substitutions.emplace(root, sym::var(root.to_string(), {interval.min, interval.max}));
    }
    auto result = substitute_shape(s, substitutions);
    optimal_map frozen_symbols;
    for(const auto& [symbol, value] : freeze)
        if(not contains(subranges, symbol))
            frozen_symbols.emplace(symbol, sym::lit(value));
    auto dimensions = result.dyn_dims();
    std::transform(dimensions.begin(), dimensions.end(), dimensions.begin(), [&](const auto& d) {
        return shape::dynamic_dimension{d.sym_expr.subs(frozen_symbols)};
    });
    result = {result.type(), std::move(dimensions), result.dyn_strides()};
    if(s.is_fixed() and all_of(result.dyn_strides(), [](const auto& stride) {
           return sym::fixed_value(stride).has_value();
       }))
        return result.to_static();
    return result;
}

struct clone_output_case
{
    freeze_map freeze;
    std::vector<shape> outputs;
};

shape dispatch_shape_for_clones(const shape& planned,
                                const std::vector<clone_output_case>& clone_outputs,
                                std::size_t output_index)
{
    assert(planned.symbolic());
    assert(not clone_outputs.empty());
    auto matches = [&](const shape& candidate) {
        return all_of(clone_outputs, [&](const auto& clone_output) {
            const auto& output = clone_output.outputs.at(output_index);
            auto expected      = candidate.to_static(clone_output.freeze);
            if(expected.type() != output.type() or expected.lens() != output.lens())
                return false;
            if(expected.elements() == 0)
                return true;
            for(std::size_t axis = 0; axis < expected.ndim(); ++axis)
            {
                if(expected.lens()[axis] > 1 and expected.strides()[axis] != output.strides()[axis])
                    return false;
            }
            return true;
        });
    };
    if(matches(planned))
        return planned;

    for(const auto& clone_output : clone_outputs)
    {
        auto actual_layout =
            clone_output.outputs.at(output_index).with_lens(planned.type(), planned.dyn_dims());
        if(matches(actual_layout))
            return actual_layout;
    }

    std::vector<shape> outputs;
    std::transform(clone_outputs.begin(),
                   clone_outputs.end(),
                   std::back_inserter(outputs),
                   [&](const auto& clone_output) { return clone_output.outputs.at(output_index); });
    MIGRAPHX_THROW("SPLIT_SYM_DIM: planned dispatch shape " +
                   to_string_range(std::vector<shape>{planned}) +
                   " does not represent clone outputs " + to_string_range(outputs));
}

std::vector<clone_input> clone_inputs_for(const clone_recipe_map& plan, instruction_ref ins)
{
    auto found = plan.find(ins);
    if(found != plan.end())
        return found->second.inputs;
    std::vector<clone_input> result;
    std::transform(ins->inputs().begin(),
                   ins->inputs().end(),
                   std::back_inserter(result),
                   [](instruction_ref source) { return clone_input{source, {}}; });
    return result;
}

struct runtime_cache
{
    std::unordered_map<sym::expr, instruction_ref> extents;
    std::unordered_map<std::size_t, instruction_ref> indices;
    std::map<std::pair<shape::type_t, fill_kind>, instruction_ref> fills;
};

instruction_ref resolved_extent(module& m,
                                const sym::expr& expression,
                                const std::vector<instruction_ref>& sources,
                                runtime_cache& cache)
{
    auto cached = cache.extents.find(expression);
    if(cached != cache.extents.end())
        return cached->second;
    auto result =
        m.add_instruction(make_op("eval_expr_from_shape",
                                  {{"expressions", to_value(std::vector<sym::expr>{expression})}}),
                          sources);
    return cache.extents.emplace(expression, result).first->second;
}

instruction_ref index_literal(module& m, std::size_t n, runtime_cache& cache)
{
    auto cached = cache.indices.find(n);
    if(cached != cache.indices.end())
        return cached->second;
    std::vector<int64_t> indices(n);
    std::iota(indices.begin(), indices.end(), int64_t{0});
    auto result = m.add_literal(literal{shape{shape::int64_type, {n}}, indices});
    return cache.indices.emplace(n, result).first->second;
}

instruction_ref fill_literal(module& m, shape::type_t type, fill_kind fill, runtime_cache& cache)
{
    auto key    = std::make_pair(type, fill);
    auto cached = cache.fills.find(key);
    if(cached != cache.fills.end())
        return cached->second;
    auto result = m.add_literal(literal{shape{type, {1}}, std::vector<float>{fill_value(fill)}});
    return cache.fills.emplace(key, result).first->second;
}

instruction_ref add_runtime_mask(module& m,
                                 instruction_ref input,
                                 const axis_mask& mask,
                                 const std::vector<instruction_ref>& sources,
                                 const optimal_map& fixed_substitutions,
                                 runtime_cache& cache)
{
    const auto& s = input->get_shape();
    assert(not s.dynamic());
    assert(mask.axis < s.ndim());
    auto lens = s.lens();

    auto index  = m.add_instruction(make_op("broadcast", {{"axis", mask.axis}, {"out_lens", lens}}),
                                   index_literal(m, lens[mask.axis], cache));
    auto extent = m.add_instruction(
        make_op("multibroadcast", {{"out_lens", lens}}),
        resolved_extent(m, mask.extent.subs(fixed_substitutions), sources, cache));
    auto valid = m.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}),
                                   m.add_instruction(make_op("less"), index, extent));
    auto fill  = m.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}),
                                  fill_literal(m, s.type(), mask.fill, cache));
    return m.add_instruction(make_op("where"), valid, input, fill);
}

struct block_input
{
    clone_input logical;
    instruction_ref value;
};

struct block_frame
{
    std::vector<instruction_ref> body;
    std::vector<clone_input> outputs;
    std::vector<block_input> inputs;
    std::map<std::string, std::size_t> params;
    std::vector<std::size_t> literals;
    std::vector<std::size_t> extent_sources;
};

struct clone_body
{
    std::unordered_set<instruction_ref> instructions;
    std::vector<instruction_ref> ordered;
    std::vector<clone_input> outputs;
};

clone_input full_output_for(const clone_recipe_map& plan, instruction_ref source)
{
    clone_input result{source, {}};
    const auto& axes = plan.at(source).output_slice.axes;
    std::transform(axes.begin(), axes.end(), std::back_inserter(result.slice_axes), [](auto axis) {
        return static_cast<std::size_t>(axis);
    });
    return result;
}

std::optional<clone_body>
find_clone_body(const module& m, const block_plan& block, const clone_recipe_map& plan)
{
    std::vector<instruction_ref> stack;
    std::unordered_set<instruction_ref> planned;
    for(const auto* info : block.ops)
    {
        stack.push_back(info->ins);
        planned.insert(info->ins);
    }

    clone_body result;
    while(not stack.empty())
    {
        auto ins = stack.back();
        stack.pop_back();
        if(not result.instructions.insert(ins).second)
            continue;
        for(auto input : clone_inputs_for(plan, ins))
            if(absorbable_dependency(input.source, planned))
                stack.push_back(input.source);
    }
    if(result.instructions.empty())
        return std::nullopt;

    std::vector<clone_input> outputs;
    for(auto output : m.get_returns())
        if(contains(planned, output))
        {
            auto full_output = full_output_for(plan, output);
            if(not contains(outputs, full_output))
                outputs.push_back(std::move(full_output));
        }
    for(auto ins : iterator_for(m))
    {
        if(contains(result.instructions, ins))
        {
            result.ordered.push_back(ins);
            continue;
        }
        for(auto input : clone_inputs_for(plan, ins))
        {
            if(not contains(planned, input.source))
                continue;
            if(input.slice_axes.empty())
                input = full_output_for(plan, input.source);
            if(not contains(outputs, input))
                outputs.push_back(std::move(input));
        }
    }
    for(auto ins : iterator_for(m))
        for(const auto& output : outputs)
            if(output.source == ins)
                result.outputs.push_back(output);
    if(result.outputs.empty())
        return std::nullopt;
    return result;
}

std::vector<clone_input> collect_block_boundary(const clone_body& body,
                                                const clone_recipe_map& plan)
{
    std::vector<clone_input> result;
    for(auto ins : body.ordered)
        for(auto input : clone_inputs_for(plan, ins))
            if((not contains(body.instructions, input.source) or not input.slice_axes.empty()) and
               not contains(result, input))
                result.push_back(std::move(input));
    return result;
}

std::size_t add_block_input(std::vector<clone_input>& inputs, clone_input input)
{
    auto found = std::find(inputs.begin(), inputs.end(), input);
    if(found != inputs.end())
        return static_cast<std::size_t>(std::distance(inputs.begin(), found));
    inputs.push_back(std::move(input));
    return inputs.size() - 1;
}

std::optional<block_frame>
find_block_frame(const module& m,
                 const block_plan& block,
                 const clone_recipe_map& plan,
                 std::size_t block_number,
                 const std::unordered_map<instruction_ref, std::string>& parameter_names,
                 const std::unordered_map<sym::expr, instruction_ref>& root_sources)
{
    auto body = find_clone_body(m, block, plan);
    if(not body.has_value())
        return std::nullopt;

    auto boundary = collect_block_boundary(*body, plan);
    block_frame result;
    result.body    = std::move(body->ordered);
    result.outputs = std::move(body->outputs);

    for(const auto& root : block.roots)
    {
        if(not contains(root_sources, root.root))
            MIGRAPHX_THROW("SPLIT_SYM_DIM: no parameter resolves block root " + root.name);
        auto source      = root_sources.at(root.root);
        auto input_index = add_block_input(boundary, {source, {}});
        if(not contains(result.extent_sources, input_index))
            result.extent_sources.push_back(input_index);
    }
    std::transform(boundary.begin(),
                   boundary.end(),
                   std::back_inserter(result.inputs),
                   [](const clone_input& input) { return block_input{input, input.source}; });

    std::unordered_set<std::string> used_names;
    for(const auto& name : m.get_parameter_names())
        used_names.insert(name);
    const std::string input_prefix = "#split_sym_dim_input_";
    std::size_t generated_suffix   = 0;
    for(auto ins : iterator_for(m))
    {
        for(std::size_t input_index = 0; input_index < result.inputs.size(); ++input_index)
        {
            if(result.inputs.at(input_index).logical.source != ins)
                continue;
            if(contains(parameter_names, ins))
            {
                result.params.emplace(parameter_names.at(ins), input_index);
                continue;
            }
            if(ins->name() == "@literal")
            {
                result.literals.push_back(input_index);
                ++generated_suffix;
                continue;
            }
            auto name = input_prefix + std::to_string(block_number) + "_" +
                        std::to_string(generated_suffix++);
            while(not used_names.insert(name).second)
                name = input_prefix + std::to_string(block_number) + "_" +
                       std::to_string(generated_suffix++);
            result.params.emplace(std::move(name), input_index);
        }
    }
    if(result.params.size() + result.literals.size() != boundary.size())
        MIGRAPHX_THROW("SPLIT_SYM_DIM: failed to collect every block input");
    return result;
}

using cloned_input_values = std::vector<std::pair<clone_input, instruction_ref>>;

instruction_ref
find_cloned_input(const clone_input& input,
                  const std::unordered_map<instruction_ref, instruction_ref>& clone_map,
                  const cloned_input_values& input_values)
{
    if(input.slice_axes.empty())
        return clone_map.at(input.source);
    auto found = std::find_if(input_values.begin(), input_values.end(), [&](const auto& value) {
        return value.first == input;
    });
    assert(found != input_values.end());
    return found->second;
}

struct clone_context
{
    module& clone_module;
    const clone_recipe_map& plan;
    std::unordered_map<instruction_ref, instruction_ref>& clone_map;
    const cloned_input_values& input_values;
    const freeze_map& freeze;
    const std::vector<instruction_ref>& runtime_extent_sources;
    const optimal_map& fixed_substitutions;
    runtime_cache cache;
    pad_cache reusable_pads;

    clone_context(module& mod,
                  const clone_recipe_map& recipes,
                  std::unordered_map<instruction_ref, instruction_ref>& clones,
                  const cloned_input_values& inputs,
                  const freeze_map& frozen_values,
                  const std::vector<instruction_ref>& runtime_sources,
                  const optimal_map& substitutions)
        : clone_module(mod),
          plan(recipes),
          clone_map(clones),
          input_values(inputs),
          freeze(frozen_values),
          runtime_extent_sources(runtime_sources),
          fixed_substitutions(substitutions)
    {
    }

    instruction_ref emit(instruction_ref source)
    {
        auto found         = plan.find(source);
        auto source_inputs = clone_inputs_for(plan, source);
        std::vector<instruction_ref> args;
        std::transform(source_inputs.begin(),
                       source_inputs.end(),
                       std::back_inserter(args),
                       [&](const clone_input& input) {
                           return find_cloned_input(input, clone_map, input_values);
                       });
        if(found != plan.end())
        {
            const auto& operands = found->second.operands;
            assert(operands.size() == args.size());
            for(std::size_t index = 0; index < operands.size(); ++index)
                if(operands.at(index).pad_value.has_value())
                    args.at(index) = add_or_reuse_pad(
                        clone_module,
                        make_op("fixed_pad", {{"value", *operands.at(index).pad_value}}),
                        args.at(index),
                        reusable_pads);
            for(std::size_t index = 0; index < operands.size(); ++index)
                for(const auto& mask : operands.at(index).masks)
                    args.at(index) = add_runtime_mask(clone_module,
                                                      args.at(index),
                                                      mask,
                                                      runtime_extent_sources,
                                                      fixed_substitutions,
                                                      cache);
        }

        op_freezer freezer;
        std::vector<std::size_t> shape_input_indices;
        if(found != plan.end())
            freezer = found->second.freezer;
        else if(source->get_shape().dynamic())
        {
            auto info           = analyze_instruction(source);
            freezer             = info.freezer;
            shape_input_indices = std::move(info.shape_input_indices);
        }

        instruction_ref clone;
        if(freezer)
        {
            auto data_args = select_data_inputs(args, shape_input_indices);
            clone          = freezer(clone_module, source, data_args, freeze);
        }
        else
        {
            clone =
                clone_module.add_instruction(source->get_operator(), args, source->module_inputs());
        }

        if(clone->get_shape().dynamic())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: clone body is not fully static");
        clone_map[source] = clone;
        if(not source->get_debug_symbols().empty())
            clone_module.add_debug_symbols(clone, source->get_debug_symbols());
        return clone;
    }
};

struct clone_build
{
    module clone;
    clone_output_case output_case;
};

clone_build
build_clone(const std::string& name,
            const block_frame& frame,
            const clone_recipe_map& plan,
            const freeze_map& freeze,
            const std::unordered_map<sym::expr, shape::dynamic_dimension::interval>& subranges,
            const optimal_map& fixed_substitutions)
{
    module clone_module{name};
    std::unordered_map<instruction_ref, instruction_ref> clone_map;
    cloned_input_values input_values;
    for(const auto& [parameter_name, input_index] : frame.params)
    {
        const auto& input = frame.inputs.at(input_index);
        auto parameter    = clone_module.add_parameter(
            parameter_name, clone_parameter_shape(input.value->get_shape(), subranges, freeze));
        if(input.logical.slice_axes.empty())
            clone_map[input.logical.source] = parameter;
        else
            input_values.emplace_back(input.logical, parameter);
    }
    for(auto input_index : frame.literals)
    {
        auto source       = frame.inputs.at(input_index).logical.source;
        clone_map[source] = clone_module.add_literal(source->get_literal());
    }

    std::vector<instruction_ref> runtime_extent_sources;
    std::transform(frame.extent_sources.begin(),
                   frame.extent_sources.end(),
                   std::back_inserter(runtime_extent_sources),
                   [&](std::size_t input_index) {
                       return find_cloned_input(
                           frame.inputs.at(input_index).logical, clone_map, input_values);
                   });
    clone_context context{clone_module,
                          plan,
                          clone_map,
                          input_values,
                          freeze,
                          runtime_extent_sources,
                          fixed_substitutions};
    for(auto source : frame.body)
        context.emit(source);

    std::vector<instruction_ref> clone_outputs;
    std::transform(frame.outputs.begin(),
                   frame.outputs.end(),
                   std::back_inserter(clone_outputs),
                   [&](const clone_input& output) { return clone_map.at(output.source); });
    if(any_of(clone_outputs, [](instruction_ref output) { return output->get_shape().dynamic(); }))
        MIGRAPHX_THROW("SPLIT_SYM_DIM: clone output is not fully static");
    std::vector<shape> output_shapes;
    std::transform(clone_outputs.begin(),
                   clone_outputs.end(),
                   std::back_inserter(output_shapes),
                   [](instruction_ref output) { return output->get_shape(); });
    clone_module.add_return(clone_outputs);
    return {std::move(clone_module), {freeze, std::move(output_shapes)}};
}

instruction_ref resolve_replacement(module& m,
                                    instruction_ref source,
                                    replacement_map& replacements,
                                    const instruction_block_map& block_for_instruction)
{
    auto found = replacements.find(source);
    if(found != replacements.end())
        return found->second;
    if(contains(block_for_instruction, source))
        MIGRAPHX_THROW("SPLIT_SYM_DIM: block dependency was not specialized before use");

    auto args    = source->inputs();
    bool changed = false;
    for(auto& arg : args)
    {
        auto replacement = resolve_replacement(m, arg, replacements, block_for_instruction);
        if(replacement == arg)
            continue;
        arg     = replacement;
        changed = true;
    }
    if(not changed)
        return source;

    instruction_ref result;
    try
    {
        result = m.add_instruction(source->get_operator(), args, source->module_inputs());
    }
    catch(const std::exception& e)
    {
        std::vector<shape> input_shapes;
        std::transform(args.begin(), args.end(), std::back_inserter(input_shapes), [](auto arg) {
            return arg->get_shape();
        });
        MIGRAPHX_THROW("SPLIT_SYM_DIM: failed to rewire " + source->name() + " with inputs " +
                       to_string_range(input_shapes) + ": " + e.what());
    }
    if(not source->get_debug_symbols().empty())
        m.add_debug_symbols(result, source->get_debug_symbols());
    replacements.emplace(source, result);
    return result;
}

using output_value_map = std::vector<std::pair<clone_input, instruction_ref>>;

instruction_ref find_output_value(const clone_input& input, const output_value_map& output_values)
{
    auto found = std::find_if(output_values.begin(), output_values.end(), [&](const auto& value) {
        return value.first == input;
    });
    if(found == output_values.end())
        MIGRAPHX_THROW("SPLIT_SYM_DIM: block output was not specialized before use");
    return found->second;
}

void resolve_frame_inputs(module& m,
                          block_frame& frame,
                          replacement_map& replacements,
                          const instruction_block_map& block_for_instruction,
                          const output_value_map& output_values)
{
    for(std::size_t index = 0; index < frame.inputs.size(); ++index)
    {
        auto& input = frame.inputs.at(index);
        input.value =
            input.logical.slice_axes.empty()
                ? resolve_replacement(m, input.logical.source, replacements, block_for_instruction)
                : find_output_value(input.logical, output_values);
    }
}

slice_spec select_slice_axes(const slice_spec& slice, const std::vector<std::size_t>& axes)
{
    slice_spec result;
    for(std::size_t index = 0; index < slice.axes.size(); ++index)
    {
        auto axis = slice.axes.at(index);
        if(axis < 0 or not contains(axes, static_cast<std::size_t>(axis)))
            continue;
        result.axes.push_back(axis);
        result.starts.push_back(slice.starts.at(index));
        result.ends.push_back(slice.ends.at(index));
    }
    return result;
}

instruction_ref add_output_slice(module& m,
                                 const clone_input& output,
                                 instruction_ref selected_output,
                                 const clone_recipe& recipe)
{
    auto slice   = select_slice_axes(recipe.output_slice, output.slice_axes);
    auto sources = m.get_parameters();
    auto starts  = m.add_instruction(
        make_op("eval_expr_from_shape", {{"expressions", to_value(slice.starts)}}), sources);
    auto ends = m.add_instruction(
        make_op("eval_expr_from_shape", {{"expressions", to_value(slice.ends)}}), sources);
    auto result              = m.add_instruction(make_op("dyn_slice",
                                                         {{"axes", slice.axes},
                                                          {"starts", to_value(slice.starts)},
                                                          {"ends", to_value(slice.ends)}}),
                                    selected_output,
                                    starts,
                                    ends);
    const auto& source_shape = output.source->get_shape();
    auto dimensions          = source_shape.dyn_dims();
    for(auto axis : recipe.output_slice.axes)
        if(axis >= 0 and not contains(output.slice_axes, static_cast<std::size_t>(axis)))
            dimensions.at(axis) = recipe.dispatch_output.dyn_dims().at(axis);
    shape output_shape{source_shape.type(),
                       std::move(dimensions),
                       selected_output->get_shape().to_symbolic().dyn_strides()};
    instruction::replace(result, result->get_operator(), output_shape, result->inputs());
    result->set_normalized();
    if(not output.source->get_debug_symbols().empty())
        m.add_debug_symbols(result, output.source->get_debug_symbols());
    return result;
}

void wire_select_module(module_pass_manager& mpm,
                        const block_frame& frame,
                        const clone_recipe_map& plan,
                        std::vector<module> clones,
                        const std::vector<clone_output_case>& clone_outputs,
                        replacement_map& replacements,
                        output_value_map& output_values)
{
    module& m = mpm.get_module();
    std::vector<module_ref> submodules;
    submodules.reserve(clones.size());
    for(auto& clone : clones)
    {
        auto name = clone.name();
        submodules.push_back(mpm.create_module(name, std::move(clone)));
    }

    std::vector<instruction_ref> selection_inputs;
    std::transform(frame.params.begin(),
                   frame.params.end(),
                   std::back_inserter(selection_inputs),
                   [&](const auto& input) { return frame.inputs.at(input.second).value; });
    std::vector<shape> body_output_shapes;
    for(std::size_t output_index = 0; output_index < frame.outputs.size(); ++output_index)
    {
        auto source = frame.outputs.at(output_index).source;
        body_output_shapes.push_back(dispatch_shape_for_clones(
            plan.at(source).dispatch_output, clone_outputs, output_index));
    }
    auto selection = m.add_instruction(
        make_op("select_module", {{"output_dyn_shapes", to_value(shape{body_output_shapes})}}),
        selection_inputs,
        submodules);

    for(std::size_t output_index = 0; output_index < frame.outputs.size(); ++output_index)
    {
        const auto& output = frame.outputs.at(output_index);
        auto selected_output =
            m.add_instruction(make_op("get_tuple_elem", {{"index", output_index}}), selection);
        auto sliced = add_output_slice(m, output, selected_output, plan.at(output.source));
        output_values.emplace_back(output, sliced);
        if(output == full_output_for(plan, output.source))
            replacements.emplace(output.source, sliced);
    }
}

void specialize_blocks(module_pass_manager& mpm,
                       const std::vector<block_plan>& blocks,
                       const clone_recipe_map& plan)
{
    module& m = mpm.get_module();
    std::unordered_map<instruction_ref, std::string> parameter_names;
    for(const auto& name : m.get_parameter_names())
        parameter_names[m.get_parameter(name)] = name;
    auto root_sources = find_root_sources(m);
    instruction_block_map block_for_instruction;
    for(std::size_t block_index = 0; block_index < blocks.size(); ++block_index)
        for(const auto* info : blocks.at(block_index).ops)
            block_for_instruction.emplace(info->ins, block_index);
    replacement_map replacements;
    output_value_map output_values;
    auto original_outputs = m.get_returns();

    std::size_t block_number = 0;
    for(const auto& block : blocks)
    {
        auto frame = find_block_frame(m, block, plan, block_number, parameter_names, root_sources);
        if(not frame.has_value())
        {
            ++block_number;
            continue;
        }
        resolve_frame_inputs(m, *frame, replacements, block_for_instruction, output_values);

        optimal_map fixed_substitutions;
        for(const auto& root : block.roots)
            if(root.interval.min == root.interval.max)
                fixed_substitutions.emplace(root.root, root.optimal_symbol);
        assert(block.clone_count > 0);
        std::vector<module> clones;
        std::vector<clone_output_case> clone_outputs;
        clones.reserve(block.clone_count);
        clone_outputs.reserve(block.clone_count);
        for(std::size_t clone_index = 0; clone_index < block.clone_count; ++clone_index)
        {
            std::vector<std::size_t> choices;
            choices.reserve(block.roots.size());
            auto remaining = clone_index;
            for(const auto& root : block.roots)
            {
                choices.push_back(remaining % root.optimal_values.size());
                remaining /= root.optimal_values.size();
            }
            assert(remaining == 0);

            freeze_map freeze;
            std::unordered_map<sym::expr, shape::dynamic_dimension::interval> subranges;
            for(auto&& [root, choice] : views::zip(block.roots, choices))
            {
                auto target                 = root.optimal_values.at(choice);
                freeze[root.root]           = target;
                freeze[root.optimal_symbol] = target;
                subranges[root.root]        = root.subranges.at(choice);
            }
            auto name = m.name() + ":split_sym_dim_" + std::to_string(block_number) + "_" +
                        std::to_string(clone_index);
            auto built = build_clone(name, *frame, plan, freeze, subranges, fixed_substitutions);
            clones.push_back(std::move(built.clone));
            clone_outputs.push_back(std::move(built.output_case));
        }
        wire_select_module(
            mpm, *frame, plan, std::move(clones), clone_outputs, replacements, output_values);
        ++block_number;
    }
    std::vector<instruction_ref> outputs;
    std::transform(original_outputs.begin(),
                   original_outputs.end(),
                   std::back_inserter(outputs),
                   [&](instruction_ref output) {
                       return resolve_replacement(m, output, replacements, block_for_instruction);
                   });
    m.replace_return(outputs);
    m.sort();
}

} // namespace

void split_sym_dim::apply(module_pass_manager& mpm) const
{
    module& m = mpm.get_module();

    if(not has_symbolic_param(m))
        return;

    match::find_matches(m,
                        resolve_symbolic_dimensions_of_match{.root_sources = find_root_sources(m),
                                                             .sources      = m.get_parameters()});
    dead_code_elimination{}.apply(m);
    auto symbolic_instructions = find_symbolic_instructions(m);
    if(symbolic_instructions.empty())
        return;

    auto infos = analyze_symbolic_instructions(symbolic_instructions);
    elide_masks_zeroed_by_softmax(infos);

    auto roots = collect_roots(m);
    if(not roots.has_value())
        return;
    auto blocks = discover_blocks(infos, *roots, max_clones);
    if(blocks.empty())
        return;

    auto clone_recipes = make_clone_recipes(blocks, *roots);
    specialize_blocks(mpm, blocks, clone_recipes);
    run_passes(m, {dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
