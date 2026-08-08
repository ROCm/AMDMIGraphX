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
#include <migraphx/functional.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape_transform_descriptor.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/value.hpp>
#include <migraphx/zip_view.hpp>

#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

// How a symbolic axis is consumed by an operator.
enum class axis_kind
{
    parallel,   // elements independent (batch); padded rows sliced back off
    windowed,   // sliding window reads the axis (conv/pool spatial)
    normalized, // values interact along the axis (softmax)
    contracted, // axis reduced away (reduce, gemm K, conv channels)
    merged,     // reshape merge/split; hard coalescing boundary
    unknown     // no registered semantics -> forces a pass boundary
};

// Sentinel the padded region must hold so a sensitive op's result is unchanged.
enum class fill_kind
{
    none,    // op is insensitive on this axis (parallel); fill is a don't-care
    zero,    // sum/mean identity, gemm/conv contraction, conv zero-pad, lpnorm
    lowest,  // max identity (maxpool)
    neg_inf, // exact softmax mask
    highest, // min identity
    one      // product identity
};

// One operand (by index) or the output of an operator.
struct io_ref
{
    bool is_output    = false;
    std::size_t index = 0; // operand index when not is_output
};

// A symbolic axis that varies (an actual variable), not a constant sym::lit.
bool is_variable_axis(const shape::dynamic_dimension& d)
{
    return d.is_symbolic() and not d.is_fixed();
}

bool has_variable_axis(const shape& s)
{
    return s.symbolic() and any_of(s.dyn_dims(), is_variable_axis);
}

std::unordered_set<sym::expr> expression_variables(const sym::expr& e)
{
    std::unordered_set<sym::expr> result;
    std::unordered_set<sym::expr> visited;
    std::vector<sym::expr> stack = {e};
    while(not stack.empty())
    {
        auto current = std::move(stack.back());
        stack.pop_back();
        if(not visited.insert(current).second)
            continue;
        if(current.name() == "variable")
        {
            result.insert(sym::as_symbol(current));
            continue;
        }
        const auto& children = current.children();
        stack.insert(stack.end(), children.begin(), children.end());
    }
    return result;
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
    return contains({"contiguous", "flatten", "reshape", "squeeze", "transpose", "unsqueeze"},
                    op.name());
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

// Reduce axes attribute normalized against the operand rank.
std::vector<int64_t> reduce_axes(const operation& op, std::size_t ndim)
{
    auto v = op.to_value();
    std::vector<int64_t> axes;
    if(v.contains("axes"))
        axes = v.at("axes").to_vector<int64_t>();
    int64_t rank = ndim;
    for(auto& a : axes)
        if(a < 0)
            a += rank;
    return axes;
}

// Windowed axis is coalesce-safe only with zero padding (default mode, no ceil):
// the last window then never reaches the padded tail at any (symbolic) size.
bool windowed_zero_pad(const std::vector<std::size_t>& padding,
                       std::size_t nspatial,
                       std::size_t axis)
{
    if(axis < 2)
        return false;
    std::size_t sdim = axis - 2;
    if(sdim >= nspatial)
        return false;
    if(padding.size() == nspatial)
        return padding[sdim] == 0;
    if(padding.size() == 2 * nspatial)
        return padding[sdim] == 0 and padding[nspatial + sdim] == 0;
    return false;
}

struct axis_padding
{
    fill_kind fill     = fill_kind::none;
    bool coalesce_safe = false;
    bool supported     = false;
};

struct axis_masking
{
    fill_kind fill;
};

// Resolved padding and masking policies for one operator axis.
struct axis_desc
{
    axis_kind kind = axis_kind::unknown;
    axis_padding padding;
    std::optional<axis_masking> masking;
};

axis_desc parallel_axis() { return {axis_kind::parallel, {fill_kind::none, true, true}, {}}; }

axis_desc contracted_axis(fill_kind fill)
{
    return {axis_kind::contracted, {fill, false, true}, {}};
}

// Coalesce-safe only because specialize installs the mask the pad fill cannot express.
axis_desc masked_axis(axis_kind kind, fill_kind fill)
{
    return {kind, {fill_kind::none, true, true}, axis_masking{fill}};
}

using axis_policy = std::function<axis_desc(io_ref, std::size_t)>;

axis_policy unsupported_policy()
{
    return [](io_ref, std::size_t) { return axis_desc{}; };
}

using optimal_map      = std::unordered_map<sym::expr, sym::expr>;
using freeze_map       = std::unordered_map<sym::expr, std::size_t>;
using target_optimizer = std::function<operation(const operation&, const optimal_map&)>;
using target_freezer   = std::function<operation(const operation&, const freeze_map&)>;

struct symbolic_target_policy
{
    target_optimizer to_optimal;
    target_freezer to_static;
};

struct op_semantics
{
    axis_policy axes = unsupported_policy();
    std::optional<symbolic_target_policy> symbolic_target;
};

op_semantics axis_semantics(axis_policy axes) { return {std::move(axes), {}}; }

struct pointwise_family
{
    bool matches(const operation& op) const { return is_pointwise(op); }

    op_semantics describe(const operation&, const std::vector<shape>&) const
    {
        return axis_semantics([](io_ref, std::size_t) { return parallel_axis(); });
    }
};

struct reduce_family
{
    bool matches(const operation& op) const { return is_reduce(op); }

    op_semantics describe(const operation& op, const std::vector<shape>& inputs) const
    {
        std::vector<std::vector<int64_t>> raxes(inputs.size());
        std::transform(inputs.begin(), inputs.end(), raxes.begin(), [&](const shape& input) {
            return reduce_axes(op, input.ndim());
        });
        auto identity = reduce_identity(op.name());
        return axis_semantics([raxes = std::move(raxes), identity](io_ref io, std::size_t axis) {
            if(io.is_output)
                return parallel_axis();
            const auto& axes = raxes.at(io.index);
            if(axes.empty())
                return axis_desc{};
            if(not contains(axes, axis))
                return parallel_axis();
            return identity.has_value() ? contracted_axis(*identity) : axis_desc{};
        });
    }
};

struct dot_family
{
    bool matches(const operation& op) const { return is_dot(op); }

    op_semantics describe(const operation&, const std::vector<shape>& inputs) const
    {
        std::vector<std::size_t> ranks(inputs.size());
        std::transform(
            inputs.begin(), inputs.end(), ranks.begin(), [](const shape& s) { return s.ndim(); });
        // A[..., M, K] contracts its last axis; B[..., K, N] its second-to-last.
        return axis_semantics([ranks = std::move(ranks)](io_ref io, std::size_t axis) {
            if(io.is_output)
                return parallel_axis(); // M, N
            std::size_t nd = ranks.at(io.index);
            assert(nd >= 2);
            std::size_t k_axis = (io.index == 0) ? nd - 1 : nd - 2;
            return axis == k_axis ? masked_axis(axis_kind::contracted, fill_kind::zero)
                                  : parallel_axis();
        });
    }
};

std::vector<shape::dynamic_dimension> symbolic_broadcast_dims(const operation& op)
{
    if(not contains({"broadcast", "multibroadcast"}, op.name()))
        return {};
    return from_value<std::vector<shape::dynamic_dimension>>(op.to_value().at("out_dyn_dims"));
}

operation optimal_broadcast(const operation& op, const optimal_map& substitutions)
{
    auto output_dims = symbolic_broadcast_dims(op);
    std::vector<shape::dynamic_dimension> dims(output_dims.size());
    std::transform(output_dims.begin(), output_dims.end(), dims.begin(), [&](const auto& d) {
        return shape::dynamic_dimension{d.sym_expr.subs(substitutions)};
    });
    if(op.name() == "broadcast")
    {
        auto axis = op.to_value().at("axis").to<std::size_t>();
        return make_op("broadcast", {{"axis", axis}, {"out_dyn_dims", to_value(dims)}});
    }
    return make_op("multibroadcast", {{"out_dyn_dims", to_value(dims)}});
}

operation frozen_broadcast(const operation& op, const freeze_map& freeze)
{
    auto dims = symbolic_broadcast_dims(op);
    std::vector<std::size_t> lens(dims.size());
    std::transform(dims.begin(), dims.end(), lens.begin(), [&](const auto& d) {
        return d.sym_expr.eval_uint(freeze);
    });
    if(op.name() == "broadcast")
    {
        auto axis = op.to_value().at("axis").to<std::size_t>();
        return make_op("broadcast", {{"axis", axis}, {"out_lens", lens}});
    }
    return make_op("multibroadcast", {{"out_lens", lens}});
}

bool is_single_input_symbolic_broadcast(const operation& op, std::size_t ninputs)
{
    return ninputs == 1 and not symbolic_broadcast_dims(op).empty();
}

struct broadcast_family
{
    bool matches(const operation& op) const
    {
        return contains({"broadcast", "multibroadcast"}, op.name());
    }

    op_semantics describe(const operation& op, const std::vector<shape>& inputs) const
    {
        if(not is_single_input_symbolic_broadcast(op, inputs.size()))
            return {};
        return {[](io_ref, std::size_t) { return parallel_axis(); },
                symbolic_target_policy{optimal_broadcast, frozen_broadcast}};
    }
};

struct shape_transform_family
{
    bool matches(const operation& op) const { return is_shape_transform(op); }

    op_semantics describe(const operation& op, const std::vector<shape>& inputs) const
    {
        if(inputs.size() != 1)
            return {};
        auto desc = shape_transform_descriptor::create(inputs.front().max_lens(), {op});
        if(desc.empty())
            return {};
        auto input_rank = inputs.front().ndim();
        return axis_semantics([desc = std::move(desc), input_rank](io_ref io, std::size_t axis) {
            if(not io.is_output)
                return desc.get_dst_axes_from_src(axis).size() == 1 ? parallel_axis() : axis_desc{};

            auto source_axes = range(input_rank);
            auto count =
                std::count_if(source_axes.begin(), source_axes.end(), [&](auto source_axis) {
                    auto dst_axes = desc.get_dst_axes_from_src(source_axis);
                    return dst_axes.size() == 1 and dst_axes.front() == axis;
                });
            return count == 1 ? parallel_axis() : axis_desc{};
        });
    }
};

struct softmax_family
{
    bool matches(const operation& op) const { return is_softmax(op); }

    op_semantics describe(const operation& op, const std::vector<shape>& inputs) const
    {
        auto axis = op.to_value().at("axis").to<int64_t>();
        std::vector<int64_t> normalized_axes(inputs.size(), -1);
        std::transform(
            inputs.begin(), inputs.end(), normalized_axes.begin(), [&](const auto& input) {
                auto sm_axis = axis;
                int64_t ndim = input.ndim();
                if(sm_axis < 0)
                    sm_axis += ndim;
                return sm_axis >= 0 and sm_axis < ndim ? sm_axis : int64_t{-1};
            });
        return axis_semantics(
            [normalized_axes = std::move(normalized_axes)](io_ref io, std::size_t axis) {
                if(io.is_output)
                    return parallel_axis();
                auto sm_axis = normalized_axes.at(io.index);
                if(sm_axis < 0)
                    return axis_desc{};
                return axis == sm_axis ? masked_axis(axis_kind::normalized, fill_kind::neg_inf)
                                       : parallel_axis();
            });
    }
};

struct conv_family
{
    bool matches(const operation& op) const { return is_conv(op); }

    op_semantics describe(const operation& op, const std::vector<shape>&) const
    {
        bool default_padding = false;
        std::size_t group    = 0;
        std::vector<std::size_t> padding;
        std::size_t nspatial = 0;
        if(op.name() == "convolution" or op.name() == "quant_convolution")
        {
            auto attributes = op.to_value();
            default_padding = attributes.at("padding_mode").to<op::padding_mode_t>() ==
                              op::padding_mode_t::default_;
            group    = attributes.at("group").to<std::size_t>();
            padding  = attributes.at("padding").to_vector<std::size_t>();
            nspatial = attributes.at("stride").to_vector<std::size_t>().size();
        }
        // data [N, C, spatial...]; weights [K, C, kernel...]; out [N, K, spatial...].
        return axis_semantics([default_padding, group, padding = std::move(padding), nspatial](
                                  io_ref io, std::size_t axis) {
            auto spatial = [&](std::size_t spatial_axis) {
                if(not default_padding)
                    return axis_desc{};
                return axis_desc{
                    axis_kind::windowed,
                    {fill_kind::zero, windowed_zero_pad(padding, nspatial, spatial_axis), true},
                    {}};
            };
            if(io.is_output)
                return axis < 2 ? parallel_axis() : spatial(axis);
            if(io.index == 0)
            {
                if(axis == 0)
                    return parallel_axis(); // batch
                if(axis == 1)
                {
                    if(group != 1)
                        return axis_desc{};
                    return contracted_axis(fill_kind::zero); // input channels
                }
                return spatial(axis);
            }
            if(io.index != 1 or group != 1)
                return axis_desc{};
            if(axis == 0)
                return parallel_axis(); // batch
            if(axis == 1)
                return contracted_axis(fill_kind::zero); // input channels
            return axis_desc{};
        });
    }
};

struct pooling_family
{
    bool matches(const operation& op) const { return is_pooling(op); }

    op_semantics describe(const operation& op, const std::vector<shape>&) const
    {
        auto attributes = op.to_value();
        auto mode       = attributes.at("mode").to<op::pooling_mode>();
        fill_kind fill  = fill_kind::zero; // lpnorm, average count_include_pad
        if(mode == op::pooling_mode::max)
            fill = fill_kind::lowest;
        // avg-pool that excludes pad from its divisor can't be reproduced by
        // pad-to-optimal -> no fill -> not paddable.
        else if(mode == op::pooling_mode::average and
                not attributes.at("count_include_pad").to<bool>())
            fill = fill_kind::none;
        if(attributes.at("dyn_global").to<bool>() and mode == op::pooling_mode::average)
            fill = fill_kind::none;
        auto default_padding =
            attributes.at("padding_mode").to<op::padding_mode_t>() == op::padding_mode_t::default_;
        auto ceil_mode = attributes.at("ceil_mode").to<bool>();
        auto padding   = attributes.at("padding").to_vector<std::size_t>();
        auto nspatial  = attributes.at("stride").to_vector<std::size_t>().size();
        return axis_semantics(
            [fill, default_padding, ceil_mode, padding = std::move(padding), nspatial](
                io_ref, std::size_t axis) {
                if(axis < 2)
                    return parallel_axis(); // N, C
                if(not default_padding or fill == fill_kind::none)
                    return axis_desc{};
                return axis_desc{
                    axis_kind::windowed,
                    {fill, not ceil_mode and windowed_zero_pad(padding, nspatial, axis), true},
                    {}};
            });
    }
};

op_semantics describe_op(const operation& op, const std::vector<shape>& inputs)
{
    op_semantics semantics;
    bool matched  = false;
    auto classify = [&](auto family) {
        if(not matched and family.matches(op))
        {
            semantics = family.describe(op, inputs);
            matched   = true;
        }
    };
    each_args(classify,
              pointwise_family{},
              reduce_family{},
              dot_family{},
              broadcast_family{},
              shape_transform_family{},
              softmax_family{},
              conv_family{},
              pooling_family{});
    return semantics;
}

float fill_value(fill_kind f)
{
    switch(f)
    {
    case fill_kind::none:
    case fill_kind::zero: return 0.0f;
    case fill_kind::lowest: return std::numeric_limits<float>::lowest();
    case fill_kind::neg_inf: return -std::numeric_limits<float>::infinity();
    case fill_kind::highest: return std::numeric_limits<float>::max();
    case fill_kind::one: return 1.0f;
    }
    return 0.0f;
}

struct padding_plan
{
    bool required      = false;
    bool supported     = true;
    float fill         = 0.0f;
    bool coalesce_safe = true;
    std::vector<std::size_t> unsafe_axes;
};

struct mask_region
{
    axis_kind kind;
    std::size_t axis;
    fill_kind fill;
    sym::expr extent;
};

struct masking_plan
{
    bool required  = false;
    bool supported = true;
    std::vector<mask_region> regions;
};

struct operand_plan
{
    padding_plan padding;
    masking_plan masking;
};

bool supports_mask(shape::type_t type, fill_kind fill)
{
    if(fill != fill_kind::neg_inf)
        return true;
    return contains({shape::half_type, shape::float_type, shape::double_type, shape::bf16_type},
                    type);
}

operand_plan
describe_operand(const shape& input, std::size_t index, const axis_policy& describe_axis)
{
    operand_plan result;
    fill_kind fk     = fill_kind::none;
    const auto& dds  = input.dyn_dims();
    std::size_t axis = 0;
    for(const auto& d : dds)
    {
        auto current_axis = axis++;
        if(not is_variable_axis(d))
            continue;
        result.padding.required  = true;
        auto desc                = describe_axis(io_ref{false, index}, current_axis);
        result.padding.supported = result.padding.supported and desc.padding.supported;
        if(desc.masking.has_value())
        {
            result.masking.required = true;
            result.masking.supported =
                result.masking.supported and supports_mask(input.type(), desc.masking->fill);
            result.masking.regions.push_back(
                {desc.kind, current_axis, desc.masking->fill, d.sym_expr});
            continue;
        }
        result.padding.coalesce_safe = result.padding.coalesce_safe and desc.padding.coalesce_safe;
        if(not desc.padding.coalesce_safe)
            result.padding.unsafe_axes.push_back(current_axis);
        if(desc.padding.fill == fill_kind::none)
            continue;
        if(fk != fill_kind::none and fk != desc.padding.fill)
            MIGRAPHX_THROW("SPLIT_SYM_DIM: conflicting padding fills on one operand");
        fk = desc.padding.fill;
    }
    result.padding.fill = fill_value(fk);
    return result;
}

// Fully symbolic: every axis carries a sym::expr (constants as sym::lit).
bool is_symbolic(const shape& s) { return s.symbolic(); }

bool has_symbolic_param(const module& m)
{
    auto param_shapes = m.get_parameter_shapes();
    return any_of(param_shapes, [](const auto& p) { return is_symbolic(p.second); });
}

// Collect the instructions whose output shape is symbolic, in module order.
std::vector<instruction_ref> find_symbolic_instructions(module& m)
{
    return find_all(iterator_for(m),
                    [](instruction_ref ins) { return is_symbolic(ins->get_shape()); });
}

// An operator with its unmutated shapes and symbolic output axes.
struct symbolic_op_info
{
    instruction_ref ins;
    shape output;
    std::vector<shape> inputs;
    std::vector<std::size_t> sym_axes;
    std::vector<operand_plan> operands;
    std::optional<symbolic_target_policy> symbolic_target;
    bool paddable = true;
    bool maskable = true;
};

// Snapshot each symbolic operator's shapes and derive its output-axis and operand
// padding and masking requirements. Edge instructions are omitted.
std::vector<symbolic_op_info>
analyze_symbolic_instructions(const std::vector<instruction_ref>& instructions)
{
    std::vector<symbolic_op_info> infos;
    for(auto ins : instructions)
    {
        if(starts_with(ins->name(), "@"))
            continue;
        const auto& op = ins->get_operator();
        auto inputs    = to_shapes(ins->inputs());
        auto semantics = describe_op(op, inputs);
        symbolic_op_info info{ins,
                              ins->get_shape(),
                              std::move(inputs),
                              {},
                              {},
                              std::move(semantics.symbolic_target),
                              true,
                              true};
        auto describe_axis = std::move(semantics.axes);
        const auto& dds    = info.output.dyn_dims();
        std::size_t axis   = 0;
        for(const auto& d : dds)
        {
            auto current_axis = axis++;
            if(not is_variable_axis(d))
                continue;
            auto desc = describe_axis(io_ref{true, 0}, current_axis);
            info.sym_axes.push_back(current_axis);
            info.paddable = info.paddable and desc.padding.supported;
            info.maskable = info.maskable and not desc.masking.has_value();
        }
        info.operands.resize(info.inputs.size());
        std::size_t operand_index = 0;
        for(auto&& [input, operand] : views::zip(info.inputs, info.operands))
        {
            auto current_operand = operand_index++;
            if(not has_variable_axis(input))
                continue;
            operand       = describe_operand(input, current_operand, describe_axis);
            info.paddable = info.paddable and operand.padding.supported;
            info.maskable = info.maskable and operand.masking.supported;
        }
        infos.push_back(std::move(info));
    }
    return infos;
}

void elide_masks_zeroed_by_softmax(std::vector<symbolic_op_info>& infos)
{
    // A -inf mask before softmax leaves exact zeros in the padded region, so a consumer
    // contracting that region needs no zero mask. For dot, the zero factor also makes the
    // corresponding mask on the other operand redundant and preserves gemm-softmax-gemm fusion.
    auto zeroes_contracted_region = [](const mask_region& source, const mask_region& consumer) {
        return source.kind == axis_kind::normalized and source.fill == fill_kind::neg_inf and
               consumer.kind == axis_kind::contracted and consumer.fill == fill_kind::zero and
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
            const auto& source_masks = source->operands.front().masking.regions;
            auto& masking            = operand.masking;
            auto& masks              = masking.regions;
            masks.erase(
                std::remove_if(masks.begin(),
                               masks.end(),
                               [&](const auto& mask) {
                                   if(not any_of(source_masks, [&](const auto& source_mask) {
                                          return zeroes_contracted_region(source_mask, mask);
                                      }))
                                       return false;
                                   zeroed_contractions.push_back(mask.extent);
                                   return true;
                               }),
                masks.end());
            masking.required = not masks.empty();
        }
        if(zeroed_contractions.empty() or not is_dot(info.ins->get_operator()))
            continue;
        for(auto& operand : info.operands)
        {
            auto& masking = operand.masking;
            auto& masks   = masking.regions;
            masks.erase(
                std::remove_if(masks.begin(),
                               masks.end(),
                               [&](const auto& mask) {
                                   return mask.kind == axis_kind::contracted and
                                          mask.fill == fill_kind::zero and
                                          any_of(zeroed_contractions, [&](const auto& extent) {
                                              return sym::same_symbol(mask.extent, extent);
                                          });
                               }),
                masks.end());
            masking.required = not masks.empty();
        }
    }
}

struct root_spec
{
    sym::expr root;
    std::string name;
    shape::dynamic_dimension::interval interval;
    sym::expr opt_symbol;
    std::vector<std::size_t> optimals;
    std::vector<shape::dynamic_dimension::interval> subranges;
};

// Gather each distinct symbolic dimension used by the module parameters. For each dimension,
// validate its bounds and any supplied optimal sizes, create a unique internal symbol for the
// clone's target size, and divide the original range into subranges that select those targets at
// runtime. The interval bounds are always clone targets, even when no optimals were supplied.
// Clone limits are checked later for each discovered block, not for the whole module.
std::optional<std::vector<root_spec>> collect_roots(const module& m)
{
    std::unordered_map<sym::expr, root_spec> root_specs;
    auto param_names = m.get_parameter_names();
    for(const auto& param_name : param_names)
    {
        const auto& s = m.get_parameter(param_name)->get_shape();
        if(s.dynamic() and not s.symbolic())
            return std::nullopt;
        if(not s.symbolic())
            continue;
        const auto& dims = s.dyn_dims();
        for(const auto& d : dims)
        {
            if(not is_variable_axis(d))
                continue;
            if(d.sym_expr.name() != "variable")
                return std::nullopt;
            auto root     = sym::as_symbol(d.sym_expr);
            auto name     = root.to_string();
            auto interval = d.get_interval();
            auto opt_set  = d.get_optimals();
            if(any_of(opt_set, [&](auto x) { return x < interval.min or x > interval.max; }))
                return std::nullopt;
            opt_set.insert(interval.min);
            opt_set.insert(interval.max);
            std::vector<std::size_t> optimals(opt_set.begin(), opt_set.end());

            if(contains(root_specs, root))
            {
                const auto& existing = root_specs.at(root);
                if(existing.interval != interval or existing.optimals != optimals)
                    return std::nullopt;
                continue;
            }

            root_specs.emplace(
                root, root_spec{root, std::move(name), interval, {}, std::move(optimals), {}});
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
    std::set<std::string> symbol_names;
    for(const auto& root : roots)
        symbol_names.insert(root.name);

    for(auto& root : roots)
    {
        std::string opt_name = "#split_sym_dim_" + root.name + "_opt";
        while(contains(symbol_names, opt_name))
            opt_name += "_";
        symbol_names.insert(opt_name);
        std::set<sym::scalar> optimal_values;
        std::transform(root.optimals.begin(),
                       root.optimals.end(),
                       std::inserter(optimal_values, optimal_values.end()),
                       [](auto x) { return sym::scalar{x}; });
        root.opt_symbol = sym::var(
            std::move(opt_name), {root.interval.min, root.interval.max}, std::move(optimal_values));

        root.subranges.resize(root.optimals.size());
        auto lower = root.interval.min;
        std::transform(
            root.optimals.begin(), root.optimals.end(), root.subranges.begin(), [&](auto upper) {
                auto result = shape::dynamic_dimension::interval{lower, upper};
                lower       = upper + 1;
                return result;
            });
    }
    return roots;
}

// An interior slice->fixed_pad pair and the axes that must remain sliced.
struct coalesce_candidate
{
    instruction_ref pad;                  // fixed_pad to rewrite during coalescing
    instruction_ref slice;                // symbolic back-slice immediately before the pad
    instruction_ref producer;             // optimal-sized value before the slice
    std::vector<std::size_t> unsafe_axes; // axes that must remain sliced
};

struct consumer_mask
{
    std::size_t operand;
    std::size_t axis;
    fill_kind fill;   // sentinel for the invalid padded region
    sym::expr extent; // runtime extent of the valid region
};

struct block_plan
{
    std::vector<const symbolic_op_info*> ops; // non-owning references into the analysis
    std::vector<root_spec> roots;             // roots needed to route and freeze this block
};

struct materialize_result
{
    std::vector<coalesce_candidate> padding_candidates; // interior pairs for Phase 2
    // Masks are delayed until static clone construction.
    std::unordered_map<instruction_ref, std::vector<consumer_mask>> mask_plan;
    std::unordered_map<instruction_ref, target_freezer> target_freezers;
    // Materialized clone-body instructions owned by each planned block.
    std::unordered_map<const block_plan*, std::unordered_set<instruction_ref>> body_ops;
};

struct slice_spec
{
    std::vector<int64_t> axes;
    std::vector<sym::expr> starts;
    std::vector<sym::expr> ends;
};

void gather_shape_roots(const shape& s, std::unordered_set<sym::expr>& result)
{
    if(not s.symbolic())
        return;
    for(const auto& d : s.dyn_dims())
    {
        auto variables = expression_variables(d.sym_expr);
        result.insert(variables.begin(), variables.end());
    }
}

std::optional<std::vector<root_spec>> select_block_roots(const block_plan& block,
                                                         const std::vector<root_spec>& roots,
                                                         std::size_t max_clones)
{
    std::unordered_set<sym::expr> required;
    for(const auto* op : block.ops)
    {
        const auto& info = *op;
        gather_shape_roots(info.output, required);
        for(const auto& input : info.inputs)
            gather_shape_roots(input, required);
        for(const auto& operand : info.operands)
            for(const auto& mask : operand.masking.regions)
            {
                auto variables = expression_variables(mask.extent);
                required.insert(variables.begin(), variables.end());
            }
    }

    std::vector<root_spec> selected;
    std::size_t clone_count = 1;
    for(const auto& root : roots)
    {
        if(not contains(required, root.root))
            continue;
        required.erase(root.root);
        if(root.optimals.size() > std::numeric_limits<std::size_t>::max() / clone_count)
            return std::nullopt;
        clone_count *= root.optimals.size();
        if(max_clones != 0 and clone_count > max_clones)
            return std::nullopt;
        selected.push_back(root);
    }
    if(not required.empty() or selected.empty())
        return std::nullopt;
    return selected;
}

bool is_materializable(const symbolic_op_info& info)
{
    return not info.sym_axes.empty() and info.paddable and info.maskable and
           info.ins->module_inputs().empty() and
           (info.symbolic_target.has_value() or
            any_of(info.operands, [](const auto& operand) { return operand.padding.required; }));
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
        for(auto&& [source, operand] : views::zip(args, info.operands))
        {
            if(contains(included, source))
            {
                if(operand.padding.required and not operand.padding.coalesce_safe)
                    return false;
                continue;
            }

            // A block invocation cannot consume a value from an excluded operation
            // that itself depends on the block. That would leave and then re-enter
            // the block along one dependency path.
            std::vector<instruction_ref> stack = {source};
            std::unordered_set<instruction_ref> visited;
            while(not stack.empty())
            {
                auto current = stack.back();
                stack.pop_back();
                if(not visited.insert(current).second)
                    continue;
                if(contains(included, current))
                    return false;
                stack.insert(stack.end(), current->inputs().begin(), current->inputs().end());
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
    result.roots = std::move(*selected);
    target       = std::move(result);
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
    for(const auto& info : infos)
    {
        if(not is_materializable(info))
            continue;
        block_plan singleton{{&info}, {}};
        auto selected = select_block_roots(singleton, roots, max_clones);
        if(selected.has_value())
        {
            singleton.roots = std::move(*selected);
            blocks.push_back(std::move(singleton));
        }
    }

    // Prefer merging along dataflow edges. Once no connected pair can grow, also
    // merge independent regions when their union is valid; this makes the trivial
    // all-supported graph one block even when it contains independent branches.
    fix([&](auto self) {
        if(merge_one_block_pair(blocks, roots, max_clones, true) or
           merge_one_block_pair(blocks, roots, max_clones, false))
            self();
    })();
    return blocks;
}

instruction_ref
insert_dyn_slice(module& m, instruction_ref pos, instruction_ref input, const slice_spec& sl)
{
    auto sources = m.get_parameters();
    auto starts  = m.insert_instruction(
        pos, make_op("eval_expr_from_shape", {{"expressions", to_value(sl.starts)}}), sources);
    auto ends = m.insert_instruction(
        pos, make_op("eval_expr_from_shape", {{"expressions", to_value(sl.ends)}}), sources);
    return m.insert_instruction(
        pos,
        make_op("dyn_slice",
                {{"axes", sl.axes}, {"starts", to_value(sl.starts)}, {"ends", to_value(sl.ends)}}),
        input,
        starts,
        ends);
}

using pad_cache = std::unordered_map<instruction_ref, std::vector<instruction_ref>>;

instruction_ref insert_or_reuse_pad(module& m,
                                    instruction_ref pos,
                                    const operation& pad_op,
                                    instruction_ref input,
                                    pad_cache& cache)
{
    auto& candidates = cache[input];
    auto it = std::find_if(candidates.begin(), candidates.end(), [&](instruction_ref candidate) {
        return candidate->get_operator() == pad_op;
    });
    if(it != candidates.end())
        return *it;
    auto result = m.insert_instruction(pos, pad_op, input);
    candidates.push_back(result);
    return result;
}

// Phase 1: pad each symbolic input to its optimal, rerun the op, slice back to
// the real size. Returns the interior slice->fixed_pad pairs (with their decided
// coalesce-safety) for Phase 2.
materialize_result materialize(module& m,
                               const std::vector<symbolic_op_info>& infos,
                               const std::vector<block_plan>& blocks,
                               const std::vector<root_spec>& roots)
{
    materialize_result result;
    optimal_map optimal_substitutions;
    for(const auto& root : roots)
        optimal_substitutions.emplace(root.root, root.opt_symbol);

    std::unordered_map<instruction_ref, const block_plan*> block_for_instruction;
    for(const auto& block : blocks)
        for(const auto* op : block.ops)
            block_for_instruction.emplace(op->ins, &block);

    std::vector<instruction_ref> original_instructions;
    auto instruction_range = iterator_for(m);
    std::copy(instruction_range.begin(),
              instruction_range.end(),
              std::back_inserter(original_instructions));
    std::unordered_map<instruction_ref, instruction_ref> replacements;
    std::unordered_map<const block_plan*, pad_cache> reusable_pads;
    for(const auto& info : infos)
    {
        if(not contains(block_for_instruction, info.ins))
            continue;
        const auto* block     = block_for_instruction.at(info.ins);
        auto ins              = info.ins;
        const auto& op        = ins->get_operator();
        const auto& orig_dims = info.output.dyn_dims();
        const auto& inputs    = info.inputs;

        auto args = ins->inputs();
        assert(inputs.size() == args.size());
        bool padded_input = false;
        for(auto&& [arg, facts, input] : views::zip(args, info.operands, inputs))
        {
            auto source = arg;
            if(contains(replacements, source))
                arg = replacements.at(source);

            if(not facts.padding.required)
                continue;
            const auto& src_dds = input.dyn_dims();
            std::vector<sym::expr> target(src_dds.size());
            transform(src_dds, target.begin(), [&](const auto& d) {
                return d.sym_expr.subs(optimal_substitutions);
            });
            auto operand = arg;
            auto pad_op =
                make_op("fixed_pad", {{"dims", to_value(target)}, {"value", facts.padding.fill}});
            auto pad = operand->name() == "dyn_slice"
                           ? m.insert_instruction(ins, pad_op, operand)
                           : insert_or_reuse_pad(m, ins, pad_op, operand, reusable_pads[block]);
            result.body_ops[block].insert(pad);
            if(operand->name() == "dyn_slice" and contains(block_for_instruction, source) and
               (block_for_instruction.at(source) == block or not facts.padding.unsafe_axes.empty()))
            {
                assert(not operand->inputs().empty());
                result.padding_candidates.push_back(
                    {pad, operand, operand->inputs().front(), facts.padding.unsafe_axes});
            }
            arg          = pad;
            padded_input = true;
        }
        if(not padded_input and not info.symbolic_target.has_value())
            continue;

        auto materialized_op = op;
        if(info.symbolic_target.has_value())
            materialized_op = info.symbolic_target->to_optimal(op, optimal_substitutions);
        auto padded_op = m.insert_instruction(ins, materialized_op, args);
        if(info.symbolic_target.has_value())
            result.target_freezers.emplace(padded_op, info.symbolic_target->to_static);
        result.body_ops[block].insert(padded_op);
        std::size_t operand = 0;
        for(const auto& facts : info.operands)
        {
            auto current_operand = operand++;
            const auto& masking  = facts.masking;
            if(not masking.required)
                continue;
            assert(not masking.regions.empty());
            for(const auto& mask : masking.regions)
                result.mask_plan[padded_op].push_back(
                    {current_operand, mask.axis, mask.fill, mask.extent});
        }

        slice_spec sl;
        for(auto axis : info.sym_axes)
        {
            assert(axis < orig_dims.size());
            sl.axes.push_back(axis);
            sl.starts.push_back(sym::lit(int64_t{0}));
            sl.ends.push_back(orig_dims[axis].sym_expr);
        }
        auto sliced = insert_dyn_slice(m, ins, padded_op, sl);
        replacements.emplace(ins, sliced);
        if(not ins->get_debug_symbols().empty())
        {
            m.add_debug_symbols(padded_op, ins->get_debug_symbols());
            m.add_debug_symbols(sliced, ins->get_debug_symbols());
        }
    }

    // Values crossing out of a block use its real-size slice. Rewire every
    // original operation left in main, including unsupported symbolic operations
    // and static consumers, while leaving the replaced block bodies dead.
    for(auto ins : original_instructions)
    {
        if(starts_with(ins->name(), "@") or contains(block_for_instruction, ins))
            continue;
        auto args    = ins->inputs();
        bool changed = false;
        for(auto& arg : args)
        {
            if(not contains(replacements, arg))
                continue;
            arg     = replacements.at(arg);
            changed = true;
        }
        if(changed)
            m.replace_instruction(ins, ins->get_operator(), args, ins->module_inputs());
    }

    auto original_outputs = m.get_returns();
    std::vector<instruction_ref> outputs;
    std::transform(original_outputs.begin(),
                   original_outputs.end(),
                   std::back_inserter(outputs),
                   [&](instruction_ref out) {
                       return contains(replacements, out) ? replacements.at(out) : out;
                   });
    m.replace_return(outputs);
    return result;
}

slice_spec retain_slice_axes(const value& attributes, const std::vector<std::size_t>& axes)
{
    auto slice_axes = attributes.at("axes").to_vector<int64_t>();
    auto starts     = from_value<std::vector<sym::expr>>(attributes.at("starts"));
    auto ends       = from_value<std::vector<sym::expr>>(attributes.at("ends"));
    slice_spec result;
    for(auto&& [axis, start, end] : views::zip(slice_axes, starts, ends))
    {
        if(not contains(axes, axis))
            continue;
        result.axes.push_back(axis);
        result.starts.push_back(start);
        result.ends.push_back(end);
    }
    return result;
}

void coalesce(module& m, const std::vector<coalesce_candidate>& candidates)
{
    for(const auto& c : candidates)
    {
        auto attributes = c.slice->get_operator().to_value();
        auto sl         = retain_slice_axes(attributes, c.unsafe_axes);
        if(sl.axes.empty())
        {
            m.replace_instruction(c.pad, c.producer);
            continue;
        }
        if(sl.axes.size() == attributes.at("axes").size())
            continue;
        auto kept = insert_dyn_slice(m, c.pad, c.producer, sl);
        m.replace_instruction(c.pad, c.pad->get_operator(), {kept});
    }
}

// A back-slice materialize left at a module output: a `dyn_slice` still producing a
// symbolic (real-size) shape. Its input is the optimal-sized body output.
bool is_back_slice(instruction_ref ins)
{
    return ins->name() == "dyn_slice" and is_symbolic(ins->get_shape());
}

// A clone's parameter shape: each variable axis records its intended routing
// sub-range, constants keep their value, and a static parameter is copied unchanged.
shape clone_param_shape(
    const shape& s, const std::unordered_map<sym::expr, shape::dynamic_dimension::interval>& sub)
{
    if(not s.dynamic())
        return s;
    if(s.symbolic() and s.is_fixed())
        return s.to_static();
    std::vector<shape::dynamic_dimension> dds;
    for(const auto& d : s.dyn_dims())
    {
        if(is_variable_axis(d) and d.sym_expr.name() == "variable")
        {
            auto root = sym::as_symbol(d.sym_expr);
            if(contains(sub, root))
            {
                const auto& r = sub.at(root);
                dds.push_back(shape::dynamic_dimension{sym::var(root.to_string(), {r.min, r.max})});
                continue;
            }
        }
        dds.push_back(d);
    }
    return shape{s.type(), dds};
}

// Freeze a fixed_pad's target: substitute this clone's optimals for the optimal
// symbols so every `dims` entry is a concrete int64 and the padded output shape
// is static.
operation frozen_fixed_pad(const operation& op, const freeze_map& freeze)
{
    auto attributes  = op.to_value();
    auto source_dims = from_value<std::vector<sym::expr>>(attributes.at("dims"));
    std::vector<sym::expr> dims(source_dims.size());
    std::transform(source_dims.begin(), source_dims.end(), dims.begin(), [&](const auto& e) {
        return sym::lit(e.eval_uint(freeze));
    });
    return make_op("fixed_pad", {{"dims", to_value(dims)}, {"value", attributes.at("value")}});
}

struct runtime_cache
{
    std::unordered_map<sym::expr, instruction_ref> extents;
    std::map<std::size_t, instruction_ref> indices;
    std::map<std::pair<shape::type_t, fill_kind>, instruction_ref> fills;
};

instruction_ref resolved_extent(module& m,
                                const sym::expr& expression,
                                const std::vector<instruction_ref>& sources,
                                runtime_cache& cache)
{
    if(contains(cache.extents, expression))
        return cache.extents.at(expression);
    auto result =
        m.add_instruction(make_op("eval_expr_from_shape",
                                  {{"expressions", to_value(std::vector<sym::expr>{expression})}}),
                          sources);
    return cache.extents.emplace(expression, result).first->second;
}

instruction_ref index_literal(module& m, std::size_t n, runtime_cache& cache)
{
    if(contains(cache.indices, n))
        return cache.indices.at(n);
    std::vector<int64_t> indices(n);
    std::iota(indices.begin(), indices.end(), int64_t{0});
    auto result = m.add_literal(literal{shape{shape::int64_type, {n}}, indices});
    return cache.indices.emplace(n, result).first->second;
}

instruction_ref fill_literal(module& m, shape::type_t type, fill_kind fill, runtime_cache& cache)
{
    auto key = std::make_pair(type, fill);
    if(contains(cache.fills, key))
        return cache.fills.at(key);
    auto result = m.add_literal(literal{shape{type, {1}}, std::vector<float>{fill_value(fill)}});
    return cache.fills.emplace(key, result).first->second;
}

instruction_ref add_runtime_mask(module& m,
                                 instruction_ref input,
                                 const consumer_mask& mask,
                                 const std::vector<instruction_ref>& sources,
                                 runtime_cache& cache)
{
    const auto& s = input->get_shape();
    assert(not s.dynamic());
    assert(mask.axis < s.ndim());
    auto lens = s.lens();

    auto index  = m.add_instruction(make_op("broadcast", {{"axis", mask.axis}, {"out_lens", lens}}),
                                   index_literal(m, lens[mask.axis], cache));
    auto extent = m.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}),
                                    resolved_extent(m, mask.extent, sources, cache));
    auto valid  = m.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}),
                                   m.add_instruction(make_op("less"), index, extent));
    auto fill   = m.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}),
                                  fill_literal(m, s.type(), mask.fill, cache));
    return m.add_instruction(make_op("where"), valid, input, fill);
}

std::unordered_map<sym::expr, instruction_ref> find_root_sources(module& m)
{
    std::unordered_map<sym::expr, instruction_ref> result;
    for(const auto& name : m.get_parameter_names())
    {
        auto parameter = m.get_parameter(name);
        const auto& s  = parameter->get_shape();
        if(not s.symbolic())
            continue;
        for(const auto& d : s.dyn_dims())
            if(is_variable_axis(d) and d.sym_expr.name() == "variable")
                result.emplace(sym::as_symbol(d.sym_expr), parameter);
    }
    return result;
}

// Clone each maximal block for its own Cartesian product of root optimals. Values
// crossing a block boundary remain in main: select_module chooses an optimal-sized
// clone result, then the existing symbolic slice restores its runtime shape.
void specialize_blocks(module_pass_manager& mpm,
                       const std::vector<block_plan>& blocks,
                       const materialize_result& materialized)
{
    module& m = mpm.get_module();
    std::unordered_map<instruction_ref, std::string> param_name;
    for(const auto& name : m.get_parameter_names())
        param_name[m.get_parameter(name)] = name;
    auto root_sources = find_root_sources(m);
    std::unordered_set<instruction_ref> live;
    std::vector<instruction_ref> live_stack = m.get_returns();
    while(not live_stack.empty())
    {
        auto ins = live_stack.back();
        live_stack.pop_back();
        if(not live.insert(ins).second)
            continue;
        live_stack.insert(live_stack.end(), ins->inputs().begin(), ins->inputs().end());
    }

    std::size_t next_block_number = 0;
    for(const auto& block : blocks)
    {
        auto block_number        = next_block_number++;
        const auto& tracked_body = materialized.body_ops.at(&block);
        std::vector<instruction_ref> stack;
        for(auto ins : iterator_for(m))
            if(contains(live, ins) and is_back_slice(ins) and not ins->inputs().empty() and
               contains(tracked_body, ins->inputs().front()))
                stack.push_back(ins->inputs().front());

        std::unordered_set<instruction_ref> body;
        while(not stack.empty())
        {
            auto ins = stack.back();
            stack.pop_back();
            if(not body.insert(ins).second)
                continue;
            for(auto input : ins->inputs())
                if(contains(tracked_body, input))
                    stack.push_back(input);
        }
        if(body.empty())
            continue;

        std::vector<instruction_ref> body_instructions;
        std::vector<instruction_ref> outputs;
        std::unordered_set<instruction_ref> boundary;
        for(auto ins : iterator_for(m))
        {
            if(contains(body, ins))
            {
                body_instructions.push_back(ins);
                for(auto input : ins->inputs())
                    if(not contains(body, input))
                        boundary.insert(input);
            }
            if(contains(live, ins) and is_back_slice(ins) and not ins->inputs().empty() and
               contains(body, ins->inputs().front()))
                outputs.push_back(ins);
        }
        if(outputs.empty())
            continue;

        std::vector<instruction_ref> runtime_source_inputs;
        for(const auto& root : block.roots)
        {
            if(not contains(root_sources, root.root))
                MIGRAPHX_THROW("SPLIT_SYM_DIM: no parameter resolves block root " + root.name);
            auto source = root_sources.at(root.root);
            boundary.insert(source);
            if(not contains(runtime_source_inputs, source))
                runtime_source_inputs.push_back(source);
        }

        std::set<std::string> used_names;
        for(const auto& name : m.get_parameter_names())
            used_names.insert(name);
        std::map<std::string, instruction_ref> input_sources;
        std::size_t generated_name = 0;
        for(auto ins : iterator_for(m))
        {
            if(not contains(boundary, ins))
                continue;
            if(contains(param_name, ins))
            {
                input_sources.emplace(param_name.at(ins), ins);
                continue;
            }
            std::string name;
            std::string prefix =
                ins->name() == "@literal" ? "#split_sym_dim_literal_" : "#split_sym_dim_input_";
            do
            {
                name =
                    prefix + std::to_string(block_number) + "_" + std::to_string(generated_name++);
            } while(contains(used_names, name));
            used_names.insert(name);
            input_sources.emplace(std::move(name), ins);
        }
        if(input_sources.size() != boundary.size())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: failed to collect every block input");

        // Cartesian product of this block's root optimals.
        const auto& roots = block.roots;
        std::vector<std::size_t> idx(roots.size(), 0);
        std::vector<module> clones;
        std::size_t clone_index = 0;
        for(bool more = true; more;)
        {
            freeze_map freeze;
            std::unordered_map<sym::expr, shape::dynamic_dimension::interval> subrange;
            for(auto&& [root, choice] : views::zip(roots, idx))
            {
                freeze[root.opt_symbol] = root.optimals.at(choice);
                subrange[root.root]     = root.subranges.at(choice);
            }

            module sm{m.name() + ":split_sym_dim_" + std::to_string(block_number) + "_" +
                      std::to_string(clone_index++)};
            std::unordered_map<instruction_ref, instruction_ref> map;
            for(const auto& input : input_sources)
                map[input.second] = sm.add_parameter(
                    input.first, clone_param_shape(input.second->get_shape(), subrange));

            std::vector<instruction_ref> runtime_sources;
            std::transform(runtime_source_inputs.begin(),
                           runtime_source_inputs.end(),
                           std::back_inserter(runtime_sources),
                           [&](instruction_ref source) { return map.at(source); });
            runtime_cache cache;
            for(auto ins : body_instructions)
            {
                std::vector<instruction_ref> args;
                std::transform(ins->inputs().begin(),
                               ins->inputs().end(),
                               std::back_inserter(args),
                               [&](instruction_ref in) { return map.at(in); });
                if(contains(materialized.mask_plan, ins))
                    for(const auto& mask : materialized.mask_plan.at(ins))
                        args.at(mask.operand) = add_runtime_mask(
                            sm, args.at(mask.operand), mask, runtime_sources, cache);
                auto op = ins->get_operator();
                if(ins->name() == "fixed_pad")
                    op = frozen_fixed_pad(op, freeze);
                else if(contains(materialized.target_freezers, ins))
                    op = materialized.target_freezers.at(ins)(op, freeze);
                map[ins] = sm.add_instruction(op, args, ins->module_inputs());
                if(map[ins]->get_shape().dynamic())
                    MIGRAPHX_THROW("SPLIT_SYM_DIM: clone body is not fully static");
                if(not ins->get_debug_symbols().empty())
                    sm.add_debug_symbols(map.at(ins), ins->get_debug_symbols());
            }

            std::vector<instruction_ref> souts;
            std::transform(outputs.begin(),
                           outputs.end(),
                           std::back_inserter(souts),
                           [&](instruction_ref out) { return map.at(out->inputs().front()); });
            if(any_of(souts, [](instruction_ref out) { return out->get_shape().dynamic(); }))
                MIGRAPHX_THROW("SPLIT_SYM_DIM: clone output is not fully static");
            sm.add_return(souts);
            clones.push_back(std::move(sm));

            more = false;
            for(auto&& [root, choice] : views::zip(roots, idx))
            {
                if(++choice < root.optimals.size())
                {
                    more = true;
                    break;
                }
                choice = 0;
            }
        }

        std::vector<module_ref> submodules;
        submodules.reserve(clones.size());
        for(auto& clone : clones)
        {
            auto name = clone.name();
            submodules.push_back(mpm.create_module(name, std::move(clone)));
        }

        std::vector<instruction_ref> sm_inputs;
        std::transform(input_sources.begin(),
                       input_sources.end(),
                       std::back_inserter(sm_inputs),
                       [](const auto& input) { return input.second; });
        std::vector<shape> body_shapes;
        std::transform(outputs.begin(),
                       outputs.end(),
                       std::back_inserter(body_shapes),
                       [](instruction_ref out) { return out->inputs().front()->get_shape(); });
        auto sel = m.add_instruction(
            make_op("select_module", {{"output_dyn_shapes", to_value(shape{body_shapes})}}),
            sm_inputs,
            submodules);

        // Keep each boundary slice at its existing instruction_ref so downstream
        // blocks and unsupported consumers remain wired. Sorting after all blocks
        // moves the select/get_tuple_elem producers before these rewritten slices.
        std::size_t output_number = 0;
        for(auto output : outputs)
        {
            auto gte =
                m.add_instruction(make_op("get_tuple_elem", {{"index", output_number}}), sel);
            std::vector<instruction_ref> args = {gte};
            std::copy(std::next(output->inputs().begin()),
                      output->inputs().end(),
                      std::back_inserter(args));
            m.replace_instruction(output, output->get_operator(), args);
            ++output_number;
        }
    }
    m.sort();
}

} // namespace

void split_sym_dim::apply(module_pass_manager& mpm) const
{
    module& m = mpm.get_module();

    // Nothing to specialize unless some parameter is symbolic.
    if(not has_symbolic_param(m))
        return;

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

    auto materialized = materialize(m, infos, blocks, *roots);
    coalesce(m, materialized.padding_candidates);
    specialize_blocks(mpm, blocks, materialized);
    run_passes(m, {dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
