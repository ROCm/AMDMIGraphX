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
#include <variant>
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

// Windowed axis is coalesce-safe only with zero padding (default mode, no ceil):
// the last window then never reaches the padded tail at any (symbolic) size.
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

using optimal_map             = std::unordered_map<sym::expr, sym::expr>;
using freeze_map              = std::unordered_map<sym::expr, std::size_t>;
using optimal_op_rewriter     = std::function<operation(const operation&, const optimal_map&)>;
using static_op_freezer       = std::function<operation(const operation&, const freeze_map&)>;
using static_subgraph_emitter = std::function<instruction_ref(
    module&, const operation&, const std::vector<instruction_ref>&, const freeze_map&)>;

struct symbolic_target_policy
{
    optimal_op_rewriter retarget_to_optimal;
    static_op_freezer freeze_op;
    std::optional<std::vector<std::size_t>> selected_inputs;
    std::optional<static_subgraph_emitter> emit_static_subgraph;
};

bool retains_input(const std::optional<symbolic_target_policy>& target, std::size_t index)
{
    return not target.has_value() or not target->selected_inputs.has_value() or
           contains(*target->selected_inputs, index);
}

std::vector<instruction_ref> select_inputs(const std::vector<instruction_ref>& inputs,
                                           const std::optional<std::vector<std::size_t>>& selected)
{
    if(not selected.has_value())
        return inputs;
    std::vector<instruction_ref> result;
    std::transform(selected->begin(), selected->end(), std::back_inserter(result), [&](auto index) {
        return inputs.at(index);
    });
    return result;
}

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
            std::size_t rank = ranks.at(io.index);
            assert(rank >= 2);
            std::size_t contraction_axis = (io.index == 0) ? rank - 1 : rank - 2;
            return axis == contraction_axis ? masked_axis(axis_kind::contracted, fill_kind::zero)
                                            : parallel_axis();
        });
    }
};

std::vector<shape::dynamic_dimension> symbolic_broadcast_dims(const operation& op)
{
    if(not contains({"broadcast", "multibroadcast", "broadcast_with_dims"}, op.name()))
        return {};
    return from_value<std::vector<shape::dynamic_dimension>>(op.to_value().at("out_dyn_dims"));
}

operation retarget_broadcast_to_optimal(const operation& op,
                                        const std::vector<shape::dynamic_dimension>& output_dims,
                                        const optimal_map& substitutions)
{
    std::vector<shape::dynamic_dimension> dims(output_dims.size());
    std::transform(output_dims.begin(), output_dims.end(), dims.begin(), [&](const auto& d) {
        return shape::dynamic_dimension{d.sym_expr.subs(substitutions)};
    });
    if(op.name() == "broadcast")
    {
        auto axis = op.to_value().at("axis").to<std::size_t>();
        return make_op("broadcast", {{"axis", axis}, {"out_dyn_dims", to_value(dims)}});
    }
    if(not contains({"multibroadcast", "broadcast_with_dims"}, op.name()))
        MIGRAPHX_THROW("SPLIT_SYM_DIM: unsupported symbolic broadcast " + op.name());
    return make_op("multibroadcast", {{"out_dyn_dims", to_value(dims)}});
}

operation freeze_broadcast(const operation& op,
                           const std::vector<shape::dynamic_dimension>& output_dims,
                           const freeze_map& freeze)
{
    std::vector<std::size_t> lens(output_dims.size());
    std::transform(output_dims.begin(), output_dims.end(), lens.begin(), [&](const auto& d) {
        return d.sym_expr.eval_uint(freeze);
    });
    if(op.name() == "broadcast")
    {
        auto axis = op.to_value().at("axis").to<std::size_t>();
        return make_op("broadcast", {{"axis", axis}, {"out_lens", lens}});
    }
    if(not contains({"multibroadcast", "broadcast_with_dims"}, op.name()))
        MIGRAPHX_THROW("SPLIT_SYM_DIM: unsupported symbolic broadcast " + op.name());
    return make_op("multibroadcast", {{"out_lens", lens}});
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

operation retarget_allocate_to_optimal(const operation& op, const optimal_map& substitutions)
{
    auto source = symbolic_allocate_shape(op);
    assert(source.has_value());
    std::vector<shape::dynamic_dimension> dims(source->ndim());
    std::transform(
        source->dyn_dims().begin(), source->dyn_dims().end(), dims.begin(), [&](const auto& d) {
            return shape::dynamic_dimension{d.sym_expr.subs(substitutions)};
        });
    std::vector<sym::expr> strides(source->ndim());
    std::transform(source->dyn_strides().begin(),
                   source->dyn_strides().end(),
                   strides.begin(),
                   [&](const auto& stride) { return stride.subs(substitutions); });
    shape target{source->type(), dims, strides};
    return make_op("allocate", {{"shape", to_value(target)}});
}

operation freeze_allocate(const operation& op, const freeze_map& freeze)
{
    auto source = symbolic_allocate_shape(op);
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
    return make_op("allocate", {{"shape", to_value(target)}});
}

operation retarget_reshape_to_optimal(const shape& target, const optimal_map& substitutions)
{
    std::vector<dim_like> dims(target.ndim());
    std::transform(
        target.dyn_dims().begin(), target.dyn_dims().end(), dims.begin(), [&](const auto& d) {
            return shape::dynamic_dimension{d.sym_expr.subs(substitutions)};
        });
    return make_op("reshape", {{"dims", to_value(dims)}});
}

operation freeze_reshape(const operation& op, const freeze_map& freeze)
{
    auto source_dims = from_value<std::vector<dim_like>>(op.to_value().at("dims"));
    std::vector<int64_t> dims;
    dims.reserve(source_dims.size());
    std::transform(
        source_dims.begin(), source_dims.end(), std::back_inserter(dims), [&](const auto& d) {
            if(std::holds_alternative<int64_t>(d))
                return std::get<int64_t>(d);
            return static_cast<int64_t>(
                std::get<shape::dynamic_dimension>(d).sym_expr.eval_uint(freeze));
        });
    return make_op("reshape", {{"dims", dims}});
}

bool is_symbolic_broadcast(const operation& op, std::size_t ninputs)
{
    if(symbolic_broadcast_dims(op).empty())
        return false;
    if(op.name() == "multibroadcast")
        return ninputs >= 2;
    return ninputs == 2;
}

struct broadcast_family
{
    bool matches(const operation& op) const
    {
        return contains({"broadcast", "multibroadcast", "broadcast_with_dims"}, op.name());
    }

    op_semantics describe(const operation& op, const std::vector<shape>& inputs) const
    {
        if(not is_symbolic_broadcast(op, inputs.size()))
            return {};
        auto output_dims = symbolic_broadcast_dims(op);
        return {[](io_ref, std::size_t) { return parallel_axis(); },
                symbolic_target_policy{
                    [output_dims](const operation& source, const optimal_map& substitutions) {
                        return retarget_broadcast_to_optimal(source, output_dims, substitutions);
                    },
                    [output_dims](const operation& source, const freeze_map& freeze) {
                        return freeze_broadcast(source, output_dims, freeze);
                    },
                    std::vector<std::size_t>{0},
                    std::nullopt}};
    }
};

struct allocate_family
{
    bool matches(const operation& op) const { return op.name() == "allocate"; }

    op_semantics describe(const operation& op, const std::vector<shape>& inputs) const
    {
        if(inputs.size() != 1 or not symbolic_allocate_shape(op).has_value())
            return {};
        return {[](io_ref, std::size_t) { return parallel_axis(); },
                symbolic_target_policy{retarget_allocate_to_optimal,
                                       freeze_allocate,
                                       std::vector<std::size_t>{},
                                       std::nullopt}};
    }
};

struct shape_transform_family
{
    bool matches(const operation& op) const { return is_shape_transform(op); }

    op_semantics describe(const operation& op, const std::vector<shape>& inputs) const
    {
        auto optimal_op    = op;
        auto descriptor_op = op;
        std::optional<symbolic_target_policy> symbolic_target;
        auto output = op.compute_shape(inputs);
        if(op.name() == "reshape" and inputs.size() == 1)
        {
            auto nonunit_dims = [](const shape& s) {
                std::vector<sym::expr> result;
                std::transform(s.dyn_dims().begin(),
                               s.dyn_dims().end(),
                               std::back_inserter(result),
                               [](const auto& d) { return d.sym_expr; });
                result.erase(std::remove(result.begin(), result.end(), sym::lit(1)), result.end());
                return result;
            };
            if(nonunit_dims(inputs.front()) == nonunit_dims(output))
                return axis_semantics([](io_ref, std::size_t) { return parallel_axis(); });
        }
        if(op.name() == "reshape" and inputs.size() == 2 and inputs.back().symbolic())
        {
            optimal_op    = retarget_reshape_to_optimal(inputs.back(), {});
            descriptor_op = make_op("reshape", {{"dims", to_value(inputs.back().max_lens())}});
            auto input_elements  = inputs.front().sym_elements();
            auto output_elements = output.sym_elements();
            if(sym::strict_less(input_elements, output_elements).value_or(false) or
               sym::strict_less(output_elements, input_elements).value_or(false))
                return {};
            auto optimal_output = optimal_op.compute_shape({inputs.front()});
            if(optimal_output != output)
                return {};
            auto target     = inputs.back();
            symbolic_target = symbolic_target_policy{
                [target](const operation&, const optimal_map& substitutions) {
                    return retarget_reshape_to_optimal(target, substitutions);
                },
                freeze_reshape,
                std::vector<std::size_t>{0},
                std::nullopt};
        }
        else if(inputs.size() != 1)
            return {};
        auto desc = shape_transform_descriptor::create(inputs.front().max_lens(), {descriptor_op});
        if(desc.empty())
            return {};
        auto source_dims = inputs.front().to_symbolic().dyn_dims();
        auto output_dims = output.to_symbolic().dyn_dims();
        auto axes        = [desc        = std::move(desc),
                     source_dims = std::move(source_dims),
                     output_dims = std::move(output_dims)](io_ref io, std::size_t axis) {
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
            auto count =
                std::count_if(source_axes.begin(), source_axes.end(), [&](auto source_axis) {
                    auto dst_axes = desc.get_dst_axes_from_src(source_axis);
                    return dst_axes.size() == 1 and dst_axes.front() == axis and
                           source_dims.at(source_axis).sym_expr == output_dims.at(axis).sym_expr;
                });
            return count == 1 ? parallel_axis() : axis_desc{};
        };
        return {std::move(axes), std::move(symbolic_target)};
    }
};

int64_t normalize_axis(int64_t axis, std::size_t rank)
{
    if(axis < 0)
        axis += rank;
    return axis;
}

struct gather_family
{
    bool matches(const operation& op) const { return op.name() == "gather"; }

    op_semantics describe(const operation& op, const std::vector<shape>& inputs) const
    {
        if(inputs.size() != 2)
            return {};
        auto axis = normalize_axis(op.to_value().at("axis").to<int64_t>(), inputs.front().ndim());
        if(axis < 0 or axis >= inputs.front().ndim())
            return {};
        return axis_semantics([axis](io_ref io, std::size_t current_axis) {
            if(io.is_output or io.index == 1)
                return parallel_axis();
            return current_axis == axis ? axis_desc{} : parallel_axis();
        });
    }
};

struct concat_family
{
    bool matches(const operation& op) const { return op.name() == "concat"; }

    op_semantics describe(const operation& op, const std::vector<shape>& inputs) const
    {
        if(inputs.empty())
            return {};
        auto axis = normalize_axis(op.to_value().at("axis").to<int64_t>(), inputs.front().ndim());
        if(axis < 0 or axis >= inputs.front().ndim())
            return {};
        if(any_of(inputs, [&](const auto& input) {
               return input.ndim() != inputs.front().ndim() or
                      is_variable_axis(input.dyn_dims().at(axis));
           }))
            return {};
        return axis_semantics([](io_ref, std::size_t) { return parallel_axis(); });
    }
};

struct unit_axis_transform_family
{
    bool matches(const operation& op) const
    {
        return op.name() == "squeeze" or op.name() == "unsqueeze";
    }

    op_semantics describe(const operation&, const std::vector<shape>& inputs) const
    {
        if(inputs.size() != 1)
            return {};
        return axis_semantics([](io_ref, std::size_t) { return parallel_axis(); });
    }
};

struct fill_family
{
    bool matches(const operation& op) const { return op.name() == "fill"; }

    op_semantics describe(const operation&, const std::vector<shape>& inputs) const
    {
        if(inputs.size() != 2)
            return {};
        return axis_semantics([](io_ref, std::size_t) { return parallel_axis(); });
    }
};

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

operation retarget_dynamic_range_to_optimal(const operation& op, const optimal_map& substitutions)
{
    auto output_dim = symbolic_range_dim(op);
    assert(output_dim.has_value());
    auto attributes = op.to_value();
    auto target     = shape::dynamic_dimension{output_dim->sym_expr.subs(substitutions)};
    return make_op("dynamic_range",
                   {{"max_output", attributes.at("max_output")}, {"output_dim", to_value(target)}});
}

instruction_ref emit_static_dynamic_range(module& m,
                                          const operation& op,
                                          const std::vector<instruction_ref>& args,
                                          const freeze_map& freeze)
{
    assert(args.size() == 3);
    auto output_dim = symbolic_range_dim(op);
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

struct dynamic_range_family
{
    bool matches(const operation& op) const { return op.name() == "dynamic_range"; }

    op_semantics describe(const operation& op, const std::vector<shape>& inputs) const
    {
        if(inputs.size() != 3 or inputs.front().type() != shape::int64_type or
           not symbolic_range_dim(op).has_value())
            return {};
        return {[](io_ref, std::size_t) { return parallel_axis(); },
                symbolic_target_policy{
                    retarget_dynamic_range_to_optimal,
                    [](const operation& source, const freeze_map&) { return source; },
                    std::nullopt,
                    emit_static_dynamic_range}};
    }
};

std::vector<sym::expr>
substitute_expressions(const value& attributes, const std::string& key, const optimal_map& subs)
{
    auto expressions = from_value<std::vector<sym::expr>>(attributes.at(key));
    std::transform(expressions.begin(), expressions.end(), expressions.begin(), [&](const auto& e) {
        return e.subs(subs);
    });
    return expressions;
}

operation retarget_dyn_slice_to_optimal(const operation& op, const optimal_map& substitutions)
{
    auto attributes = op.to_value();
    auto starts     = substitute_expressions(attributes, "starts", substitutions);
    auto ends       = substitute_expressions(attributes, "ends", substitutions);
    return make_op(
        "dyn_slice",
        {{"axes", attributes.at("axes")}, {"starts", to_value(starts)}, {"ends", to_value(ends)}});
}

instruction_ref emit_static_dyn_slice(module& m,
                                      const operation& op,
                                      const std::vector<instruction_ref>& args,
                                      const freeze_map& freeze)
{
    assert(args.size() == 3);
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
        if(axis < 0 or axis >= input.ndim() or not sym::find_variables(starts.at(i)).empty())
            return false;
        const auto& end = ends.at(i);
        if(not sym::find_variables(end).empty() and not(end == input_dims.at(axis).sym_expr))
            return false;
    }
    return true;
}

struct slice_family
{
    bool matches(const operation& op) const
    {
        return op.name() == "slice" or op.name() == "dyn_slice";
    }

    op_semantics describe(const operation& op, const std::vector<shape>& inputs) const
    {
        if(op.name() == "slice")
        {
            if(inputs.size() != 1)
                return {};
            auto axes = op.to_value().at("axes").to_vector<int64_t>();
            for(auto& axis : axes)
            {
                axis = normalize_axis(axis, inputs.front().ndim());
                if(axis < 0 or axis >= inputs.front().ndim() or
                   is_variable_axis(inputs.front().dyn_dims().at(axis)))
                    return {};
            }
            return axis_semantics([](io_ref, std::size_t) { return parallel_axis(); });
        }

        if(inputs.size() != 3 or not is_prefix_stable_dyn_slice(op, inputs.front()))
            return {};
        return {[](io_ref, std::size_t) { return parallel_axis(); },
                symbolic_target_policy{
                    retarget_dyn_slice_to_optimal,
                    [](const operation& source, const freeze_map&) { return source; },
                    std::nullopt,
                    emit_static_dyn_slice}};
    }
};

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

instruction_ref emit_static_scatternd(module& m,
                                      const operation& op,
                                      const std::vector<instruction_ref>& args,
                                      const std::vector<scatter_prefix_axis>& prefix_axes)
{
    assert(args.size() == 3);
    assert(not prefix_axes.empty());

    auto data       = args.front();
    auto indices    = args.at(1);
    auto updates    = args.back();
    auto index_lens = indices->get_shape().lens();
    assert(not index_lens.empty());
    auto k = index_lens.back();
    index_lens.pop_back();

    auto valid       = make_scatter_prefix_mask(m, index_lens, prefix_axes);
    auto effective_k = k;
    if(k == 0)
    {
        data        = m.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), data);
        effective_k = 1;
    }

    auto data_lens = data->get_shape().lens();
    assert(effective_k <= data_lens.size());
    std::vector<int64_t> pads(data_lens.size() * 2, 0);
    std::fill(
        pads.begin() + data_lens.size(), pads.begin() + data_lens.size() + effective_k, int64_t{1});
    auto padded_data = m.add_instruction(make_op("pad", {{"pads", pads}}), data);

    auto rewritten_index_lens = index_lens;
    rewritten_index_lens.push_back(effective_k);
    auto condition =
        m.add_instruction(make_op("unsqueeze", {{"axes", {index_lens.size()}}}), valid);
    condition = m.add_instruction(make_op("multibroadcast", {{"out_lens", rewritten_index_lens}}),
                                  condition);

    std::vector<int64_t> sink_values(effective_k);
    std::transform(data_lens.begin(),
                   data_lens.begin() + effective_k,
                   sink_values.begin(),
                   [](auto x) { return static_cast<int64_t>(x); });
    auto sink = m.add_literal(literal{shape{shape::int64_type, {effective_k}}, sink_values});
    sink = m.add_instruction(make_op("multibroadcast", {{"out_lens", rewritten_index_lens}}), sink);

    if(k == 0)
    {
        std::vector<int64_t> zeros(effective_k, 0);
        indices = m.add_literal(literal{shape{shape::int64_type, {effective_k}}, zeros});
        indices = m.add_instruction(make_op("multibroadcast", {{"out_lens", rewritten_index_lens}}),
                                    indices);
    }
    indices = m.add_instruction(make_op("where"), condition, indices, sink);

    auto scattered = m.add_instruction(op, padded_data, indices, updates);
    std::vector<int64_t> axes(effective_k);
    std::iota(axes.begin(), axes.end(), int64_t{0});
    std::vector<int64_t> starts(effective_k, 0);
    std::vector<int64_t> ends(effective_k);
    std::transform(data_lens.begin(), data_lens.begin() + effective_k, ends.begin(), [](auto x) {
        return static_cast<int64_t>(x);
    });
    auto result = m.add_instruction(
        make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}), scattered);
    if(k == 0)
        result = m.add_instruction(make_op("squeeze", {{"axes", {0}}}), result);
    return m.add_instruction(make_op("contiguous"), result);
}

struct scatternd_family
{
    bool matches(const operation& op) const { return op.name() == "scatternd_none"; }

    op_semantics describe(const operation&, const std::vector<shape>& inputs) const
    {
        if(inputs.size() != 3)
            return {};
        const auto& indices = inputs.at(1);
        auto q              = indices.ndim();
        assert(q > 0);
        std::vector<scatter_prefix_axis> prefix_axes;
        const auto& index_dims = indices.dyn_dims();
        for(std::size_t axis = 0; axis + 1 < q; ++axis)
        {
            if(is_variable_axis(index_dims.at(axis)))
                prefix_axes.emplace_back(axis, index_dims.at(axis).sym_expr);
        }
        auto axes = [](io_ref, std::size_t) { return parallel_axis(); };
        if(prefix_axes.empty())
            return axis_semantics(std::move(axes));
        if(indices.type() != shape::int64_type or
           any_of(inputs, [](const auto& input) { return not input.standard(); }))
            return {};
        auto emitter = [prefix_axes = std::move(prefix_axes)](
                           module& m, const operation& op, const auto& args, const freeze_map&) {
            return emit_static_scatternd(m, op, args, prefix_axes);
        };
        return {std::move(axes),
                symbolic_target_policy{
                    [](const operation& source, const optimal_map&) { return source; },
                    [](const operation& source, const freeze_map&) { return source; },
                    std::nullopt,
                    std::move(emitter)}};
    }
};

struct softmax_family
{
    bool matches(const operation& op) const { return is_softmax(op); }

    op_semantics describe(const operation& op, const std::vector<shape>& inputs) const
    {
        if(inputs.size() != 1)
            return {};
        auto axis = normalize_axis(op.to_value().at("axis").to<int64_t>(), inputs.front().ndim());
        if(axis < 0 or axis >= inputs.front().ndim())
            return {};
        return axis_semantics([axis](io_ref io, std::size_t current_axis) {
            if(io.is_output)
                return parallel_axis();
            return current_axis == axis ? masked_axis(axis_kind::normalized, fill_kind::neg_inf)
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
        std::size_t spatial_dimensions = 0;
        if(op.name() == "convolution" or op.name() == "quant_convolution")
        {
            auto attributes = op.to_value();
            default_padding = attributes.at("padding_mode").to<op::padding_mode_t>() ==
                              op::padding_mode_t::default_;
            group              = attributes.at("group").to<std::size_t>();
            padding            = attributes.at("padding").to_vector<std::size_t>();
            spatial_dimensions = attributes.at("stride").to_vector<std::size_t>().size();
        }
        // data [N, C, spatial...]; weights [K, C, kernel...]; out [N, K, spatial...].
        return axis_semantics(
            [default_padding, group, padding = std::move(padding), spatial_dimensions](
                io_ref io, std::size_t axis) {
                auto spatial = [&](std::size_t spatial_axis) {
                    if(not default_padding)
                        return axis_desc{};
                    return axis_desc{axis_kind::windowed,
                                     {fill_kind::zero,
                                      windowed_zero_pad(padding, spatial_dimensions, spatial_axis),
                                      true},
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
                    return parallel_axis(); // output channels
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
        auto ceil_mode  = attributes.at("ceil_mode").to<bool>();
        fill_kind fill  = fill_kind::zero; // lpnorm, average count_include_pad
        if(mode == op::pooling_mode::max)
            fill = fill_kind::lowest;
        // Average pooling cannot absorb optimal padding when that padding would change
        // the divisor: either padding is excluded or ceil mode creates partial tail windows.
        else if(mode == op::pooling_mode::average and
                (not attributes.at("count_include_pad").to<bool>() or ceil_mode))
            fill = fill_kind::none;
        if(attributes.at("dyn_global").to<bool>() and mode == op::pooling_mode::average)
            fill = fill_kind::none;
        auto default_padding =
            attributes.at("padding_mode").to<op::padding_mode_t>() == op::padding_mode_t::default_;
        auto padding            = attributes.at("padding").to_vector<std::size_t>();
        auto spatial_dimensions = attributes.at("stride").to_vector<std::size_t>().size();
        return axis_semantics(
            [fill, default_padding, ceil_mode, padding = std::move(padding), spatial_dimensions](
                io_ref, std::size_t axis) {
                if(axis < 2)
                    return parallel_axis(); // N, C
                if(not default_padding or fill == fill_kind::none)
                    return axis_desc{};
                return axis_desc{
                    axis_kind::windowed,
                    {fill,
                     not ceil_mode and windowed_zero_pad(padding, spatial_dimensions, axis),
                     true},
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
              gather_family{},
              concat_family{},
              slice_family{},
              unit_axis_transform_family{},
              fill_family{},
              dynamic_range_family{},
              scatternd_family{},
              pointwise_family{},
              reduce_family{},
              dot_family{},
              broadcast_family{},
              allocate_family{},
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
    MIGRAPHX_THROW("SPLIT_SYM_DIM: unsupported fill kind");
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
    fill_kind fill         = fill_kind::none;
    const auto& dimensions = input.dyn_dims();
    std::size_t axis       = 0;
    for(const auto& dimension : dimensions)
    {
        auto current_axis = axis++;
        if(not is_variable_axis(dimension))
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
                {desc.kind, current_axis, desc.masking->fill, dimension.sym_expr});
            continue;
        }
        result.padding.coalesce_safe = result.padding.coalesce_safe and desc.padding.coalesce_safe;
        if(not desc.padding.coalesce_safe)
            result.padding.unsafe_axes.push_back(current_axis);
        if(desc.padding.fill == fill_kind::none)
            continue;
        if(fill != fill_kind::none and fill != desc.padding.fill)
            MIGRAPHX_THROW("SPLIT_SYM_DIM: conflicting padding fills on one operand");
        fill = desc.padding.fill;
    }
    result.padding.fill = fill_value(fill);
    return result;
}

// Fully symbolic: every axis carries a sym::expr (constants as sym::lit).
bool is_symbolic(const shape& s) { return s.symbolic(); }

bool has_symbolic_param(const module& m)
{
    auto param_shapes = m.get_parameter_shapes();
    return any_of(param_shapes, [](const auto& p) { return is_symbolic(p.second); });
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
            if(d.is_symbolic() and d.sym_expr.name() == "variable")
                result.emplace(sym::as_symbol(d.sym_expr), parameter);
    }
    return result;
}

void lower_symbolic_dimensions_of(module& m)
{
    auto root_sources = find_root_sources(m);
    auto sources      = m.get_parameters();
    std::vector<instruction_ref> instructions;
    auto instruction_range = iterator_for(m);
    std::copy(instruction_range.begin(), instruction_range.end(), std::back_inserter(instructions));
    for(auto ins : instructions)
    {
        if(ins->name() != "dimensions_of" or ins->inputs().size() != 1)
            continue;
        const auto& input_shape = ins->inputs().front()->get_shape();
        if(not input_shape.symbolic())
            continue;
        auto attributes = ins->get_operator().to_value();
        auto start      = attributes.at("start").to<std::size_t>();
        auto end        = attributes.at("end").to<std::size_t>();
        if(end > input_shape.ndim())
            continue;
        std::vector<sym::expr> expressions;
        std::transform(input_shape.dyn_dims().begin() + start,
                       input_shape.dyn_dims().begin() + end,
                       std::back_inserter(expressions),
                       [](const auto& d) { return d.sym_expr; });
        if(any_of(expressions, [&](const auto& expression) {
               auto variables = sym::find_variables(expression);
               return any_of(variables, [&](const auto& variable) {
                   return not contains(root_sources, variable);
               });
           }))
            continue;
        m.replace_instruction(
            ins,
            make_op("eval_expr_from_shape", {{"expressions", to_value(expressions)}}),
            sources);
    }
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
    std::vector<std::size_t> symbolic_axes;
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
        auto describe_axis     = std::move(semantics.axes);
        const auto& dimensions = info.output.dyn_dims();
        std::size_t axis       = 0;
        for(const auto& dimension : dimensions)
        {
            auto current_axis = axis++;
            if(not is_variable_axis(dimension))
                continue;
            auto desc = describe_axis(io_ref{true, 0}, current_axis);
            info.symbolic_axes.push_back(current_axis);
            info.paddable = info.paddable and desc.padding.supported;
            info.maskable = info.maskable and not desc.masking.has_value();
        }
        info.operands.resize(info.inputs.size());
        std::size_t operand_index = 0;
        for(auto&& [input, operand] : views::zip(info.inputs, info.operands))
        {
            auto current_operand = operand_index++;
            if(not retains_input(info.symbolic_target, current_operand))
                continue;
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
    sym::expr optimal_symbol;
    std::vector<std::size_t> optimal_values;
    std::vector<shape::dynamic_dimension::interval> subranges;
};

// Gather each distinct symbolic dimension used by the module parameters. For each dimension,
// validate its bounds and any supplied optimal sizes, create a unique internal symbol for the
// clone's target size, and divide the original range into subranges that select those targets at
// runtime. The interval bounds are always clone targets, even when no optimal values were supplied.
// Clone limits are checked later for each discovered block, not for the whole module.
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
    std::set<std::string> symbol_names;
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

        root.subranges.resize(root.optimal_values.size());
        auto lower = root.interval.min;
        std::transform(root.optimal_values.begin(),
                       root.optimal_values.end(),
                       root.subranges.begin(),
                       [&](auto upper) {
                           auto result = shape::dynamic_dimension::interval{lower, upper};
                           lower       = upper + 1;
                           return result;
                       });
    }
    return roots;
}

struct clone_pad
{
    std::size_t operand;
    float value;
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

struct optimal_rewrite_result
{
    // Padding is emitted only after clone parameters have been narrowed to their
    // dispatch subrange, so fixed_pad can infer its target from input.max_lens().
    std::unordered_map<instruction_ref, std::vector<clone_pad>> pad_plan;
    // The transient main graph stays real-sized. Clone inputs separately record
    // coalesced producer edges and omit shape-carrier inputs removed by target freezing.
    std::unordered_map<instruction_ref, std::vector<instruction_ref>> clone_inputs;
    std::unordered_map<instruction_ref, operation> clone_operations;
    // Masks are delayed until static clone construction.
    std::unordered_map<instruction_ref, std::vector<consumer_mask>> mask_plan;
    std::unordered_map<instruction_ref, static_op_freezer> static_op_freezers;
    std::unordered_map<instruction_ref, static_subgraph_emitter> static_subgraph_emitters;
    // The transient rewrite retains real-size operations. These planned optimal
    // shapes describe the values returned by select_module after specialization.
    std::unordered_map<instruction_ref, shape> dispatch_output_shapes;
    // Optimal-sized clone-body instructions owned by each planned block.
    std::unordered_map<const block_plan*, std::unordered_set<instruction_ref>>
        optimal_body_instructions;
};

struct slice_spec
{
    std::vector<int64_t> axes;
    std::vector<sym::expr> starts;
    std::vector<sym::expr> ends;
};

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
                auto variables = sym::find_variables(mask.extent);
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
        if(root.optimal_values.size() > std::numeric_limits<std::size_t>::max() / clone_count)
            return std::nullopt;
        clone_count *= root.optimal_values.size();
        if(max_clones != 0 and clone_count > max_clones)
            return std::nullopt;
        selected.push_back(root);
    }
    if(not required.empty() or selected.empty())
        return std::nullopt;
    return selected;
}

bool can_specialize(const symbolic_op_info& info)
{
    return not info.symbolic_axes.empty() and info.paddable and info.maskable and
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
        const auto& info          = *op;
        const auto& args          = info.ins->inputs();
        std::size_t operand_index = 0;
        for(auto&& [source, operand] : views::zip(args, info.operands))
        {
            auto current_operand = operand_index++;
            if(not retains_input(info.symbolic_target, current_operand))
                continue;
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
    std::unordered_map<instruction_ref, const symbolic_op_info*> specializable;
    for(const auto& info : infos)
    {
        if(can_specialize(info))
            specializable.emplace(info.ins, &info);
    }

    for(const auto& info : infos)
    {
        if(not contains(specializable, info.ins))
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
insert_dyn_slice(module& m, instruction_ref pos, instruction_ref input, const slice_spec& slice)
{
    auto sources = m.get_parameters();
    auto starts  = m.insert_instruction(
        pos, make_op("eval_expr_from_shape", {{"expressions", to_value(slice.starts)}}), sources);
    auto ends = m.insert_instruction(
        pos, make_op("eval_expr_from_shape", {{"expressions", to_value(slice.ends)}}), sources);
    return m.insert_instruction(pos,
                                make_op("dyn_slice",
                                        {{"axes", slice.axes},
                                         {"starts", to_value(slice.starts)},
                                         {"ends", to_value(slice.ends)}}),
                                input,
                                starts,
                                ends);
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

slice_spec retain_slice_axes(const value& attributes, const std::vector<std::size_t>& axes);

// Build a transient real-size graph with the topology that each static clone will
// use. Padding and target freezing are recorded in the plan and applied only while
// constructing a clone, after its parameters have been narrowed to a dispatch range.
optimal_rewrite_result rewrite_to_optimal(module& m,
                                          const std::vector<symbolic_op_info>& infos,
                                          const std::vector<block_plan>& blocks,
                                          const std::vector<root_spec>& roots)
{
    optimal_rewrite_result result;
    optimal_map optimal_substitutions;
    for(const auto& root : roots)
        optimal_substitutions.emplace(root.root, root.optimal_symbol);

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
    auto clone_rewired = [&](instruction_ref ins) {
        if(contains(replacements, ins))
            return;
        auto args    = ins->inputs();
        bool changed = false;
        for(auto& arg : args)
        {
            if(not contains(replacements, arg))
                continue;
            arg     = replacements.at(arg);
            changed = true;
        }
        if(not changed)
            return;

        instruction_ref rewired;
        try
        {
            rewired = m.insert_instruction(ins, ins->get_operator(), args, ins->module_inputs());
        }
        catch(const std::exception& e)
        {
            std::vector<shape> input_shapes;
            std::transform(args.begin(),
                           args.end(),
                           std::back_inserter(input_shapes),
                           [](auto arg) { return arg->get_shape(); });
            MIGRAPHX_THROW("SPLIT_SYM_DIM: failed to rewire " + ins->name() + " with inputs " +
                           to_string_range(input_shapes) + ": " + e.what());
        }
        if(not ins->get_debug_symbols().empty())
            m.add_debug_symbols(rewired, ins->get_debug_symbols());
        replacements.emplace(ins, rewired);
    };
    auto needs_fixed_retarget = [&](const shape& s) {
        if(not s.symbolic())
            return false;
        return any_of(s.dyn_dims(),
                      [&](const auto& d) {
                          return d.is_fixed() and
                                 d.sym_expr.subs(optimal_substitutions) != d.sym_expr;
                      }) or
               any_of(s.dyn_strides(), [&](const auto& stride) {
                   return stride.subs(optimal_substitutions) != stride;
               });
    };
    for(const auto& info : infos)
    {
        if(not contains(block_for_instruction, info.ins))
        {
            clone_rewired(info.ins);
            continue;
        }
        const auto* block               = block_for_instruction.at(info.ins);
        auto ins                        = info.ins;
        const auto& op                  = ins->get_operator();
        const auto& original_dimensions = info.output.dyn_dims();
        const auto& inputs              = info.inputs;

        auto args       = ins->inputs();
        auto clone_args = args;
        assert(inputs.size() == args.size());
        std::vector<clone_pad> pads;
        std::size_t operand_index = 0;
        for(auto&& [arg, plan, input] : views::zip(args, info.operands, inputs))
        {
            auto current_operand = operand_index++;
            if(not retains_input(info.symbolic_target, current_operand))
                continue;
            auto source = arg;
            if(contains(replacements, source))
                arg = replacements.at(source);
            clone_args.at(current_operand) = arg;

            bool retarget_fixed = needs_fixed_retarget(input);
            if(not plan.padding.required and not retarget_fixed)
                continue;
            auto operand  = arg;
            bool emit_pad = true;
            if(plan.padding.required and operand->name() == "dyn_slice" and
               contains(block_for_instruction, source) and
               (block_for_instruction.at(source) == block or not plan.padding.unsafe_axes.empty()))
            {
                assert(not operand->inputs().empty());
                auto producer   = operand->inputs().front();
                auto attributes = operand->get_operator().to_value();
                auto kept       = retain_slice_axes(attributes, plan.padding.unsafe_axes);
                if(kept.axes.empty())
                {
                    clone_args.at(current_operand) = producer;
                    emit_pad                       = false;
                }
                else if(kept.axes.size() != attributes.at("axes").size())
                {
                    arg                            = insert_dyn_slice(m, ins, producer, kept);
                    clone_args.at(current_operand) = arg;
                }
            }
            if(emit_pad)
            {
                auto retained_operand = current_operand;
                if(info.symbolic_target.has_value() and
                   info.symbolic_target->selected_inputs.has_value())
                {
                    const auto& indices = *info.symbolic_target->selected_inputs;
                    auto found = std::find(indices.begin(), indices.end(), current_operand);
                    assert(found != indices.end());
                    retained_operand = std::distance(indices.begin(), found);
                }
                pads.push_back({retained_operand, plan.padding.fill});
            }
        }

        bool has_pending_pads = not pads.empty();
        std::optional<operation> clone_op;
        if(info.symbolic_target.has_value())
            clone_op = info.symbolic_target->retarget_to_optimal(op, optimal_substitutions);

        instruction_ref optimal_ins;
        try
        {
            optimal_ins = m.insert_instruction(ins, op, args, ins->module_inputs());
        }
        catch(const std::exception& e)
        {
            MIGRAPHX_THROW("SPLIT_SYM_DIM: failed to plan " + op.name() +
                           " for specialization: " + e.what());
        }
        result.clone_inputs.emplace(
            optimal_ins,
            info.symbolic_target.has_value()
                ? select_inputs(clone_args, info.symbolic_target->selected_inputs)
                : std::move(clone_args));
        if(has_pending_pads)
            result.pad_plan.emplace(optimal_ins, std::move(pads));
        if(info.symbolic_target.has_value())
        {
            result.clone_operations.emplace(optimal_ins, std::move(*clone_op));
            if(info.symbolic_target->emit_static_subgraph.has_value())
                result.static_subgraph_emitters.emplace(
                    optimal_ins, *info.symbolic_target->emit_static_subgraph);
            else
                result.static_op_freezers.emplace(optimal_ins, info.symbolic_target->freeze_op);
        }
        result.dispatch_output_shapes.emplace(optimal_ins,
                                              substitute_shape(info.output, optimal_substitutions));
        result.optimal_body_instructions[block].insert(optimal_ins);
        std::size_t operand = 0;
        for(const auto& plan : info.operands)
        {
            auto current_operand = operand++;
            const auto& masking  = plan.masking;
            if(not masking.required)
                continue;
            assert(not masking.regions.empty());
            auto retained_operand = current_operand;
            if(info.symbolic_target.has_value() and
               info.symbolic_target->selected_inputs.has_value())
            {
                const auto& indices = *info.symbolic_target->selected_inputs;
                auto found          = std::find(indices.begin(), indices.end(), current_operand);
                if(found == indices.end())
                    MIGRAPHX_THROW(
                        "SPLIT_SYM_DIM: cannot remove an input that requires runtime masking");
                retained_operand = std::distance(indices.begin(), found);
            }
            for(const auto& mask : masking.regions)
                result.mask_plan[optimal_ins].push_back(
                    {retained_operand, mask.axis, mask.fill, mask.extent});
        }

        slice_spec slice;
        for(auto axis : info.symbolic_axes)
        {
            assert(axis < original_dimensions.size());
            slice.axes.push_back(axis);
            slice.starts.push_back(sym::lit(int64_t{0}));
            slice.ends.push_back(original_dimensions[axis].sym_expr);
        }
        auto sliced = insert_dyn_slice(m, ins, optimal_ins, slice);
        replacements.emplace(ins, sliced);
        if(not ins->get_debug_symbols().empty())
        {
            m.add_debug_symbols(optimal_ins, ins->get_debug_symbols());
            m.add_debug_symbols(sliced, ins->get_debug_symbols());
        }
    }

    // Values crossing out of a block use its real-size slice. Clone every original
    // operation left in main that consumes a replacement, including unsupported
    // symbolic operations and static consumers. Keeping the original chain intact
    // avoids transiently recomputing its dead block consumers with mixed real/optimal
    // shapes before dead-code elimination removes it.
    for(auto ins : original_instructions)
    {
        if(starts_with(ins->name(), "@") or contains(block_for_instruction, ins))
            continue;
        clone_rewired(ins);
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
        if(axis < 0 or not contains(axes, static_cast<std::size_t>(axis)))
            continue;
        result.axes.push_back(axis);
        result.starts.push_back(start);
        result.ends.push_back(end);
    }
    return result;
}

// A back-slice left by the optimal rewrite at a module output: a `dyn_slice`
// still producing a symbolic (real-size) shape. Its input is the optimal-sized
// body output.
bool is_back_slice(instruction_ref ins)
{
    return ins->name() == "dyn_slice" and is_symbolic(ins->get_shape());
}

// A clone's parameter shape: substitute each routed root with its dispatch
// subrange. This also narrows compound dimensions such as min(s - 2, opt_s - 2),
// allowing fixed_pad to infer the case target from input.max_lens().
shape clone_parameter_shape(
    const shape& s,
    const std::unordered_map<sym::expr, shape::dynamic_dimension::interval>& subranges)
{
    if(not s.symbolic())
        return s;
    optimal_map substitutions;
    for(const auto& [root, interval] : subranges)
    {
        substitutions.emplace(root, sym::var(root.to_string(), {interval.min, interval.max}));
    }
    auto result = substitute_shape(s, substitutions);
    if(s.is_fixed() and all_of(result.dyn_strides(), [](const auto& stride) {
           return sym::fixed_value(stride).has_value();
       }))
        return result.to_static();
    return result;
}

using clone_output_case = std::pair<freeze_map, std::vector<shape>>;

shape dispatch_shape_for_clones(const shape& planned,
                                const std::vector<clone_output_case>& clone_outputs,
                                std::size_t output_index)
{
    assert(planned.symbolic());
    assert(not clone_outputs.empty());
    auto matches = [&](const shape& candidate) {
        return all_of(clone_outputs, [&](const auto& clone_output) {
            const auto& [freeze, outputs] = clone_output;
            const auto& output            = outputs.at(output_index);
            auto expected                 = candidate.to_static(freeze);
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
            clone_output.second.at(output_index).with_lens(planned.type(), planned.dyn_dims());
        if(matches(actual_layout))
            return actual_layout;
    }

    std::vector<shape> outputs;
    std::transform(clone_outputs.begin(),
                   clone_outputs.end(),
                   std::back_inserter(outputs),
                   [&](const auto& clone_output) { return clone_output.second.at(output_index); });
    MIGRAPHX_THROW("SPLIT_SYM_DIM: planned dispatch shape " +
                   to_string_range(std::vector<shape>{planned}) +
                   " does not represent clone outputs " + to_string_range(outputs));
}

std::vector<instruction_ref> clone_inputs_for(const optimal_rewrite_result& rewrite,
                                              instruction_ref ins)
{
    auto found = rewrite.clone_inputs.find(ins);
    return found == rewrite.clone_inputs.end() ? ins->inputs() : found->second;
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
                                 const consumer_mask& mask,
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

// Clone each maximal block for its own Cartesian product of root optimal values. Values
// crossing a block boundary remain in main: select_module chooses an optimal-sized
// clone result, then the existing symbolic slice restores its runtime shape.
void specialize_blocks(module_pass_manager& mpm,
                       const std::vector<block_plan>& blocks,
                       const optimal_rewrite_result& optimal_rewrite)
{
    module& m = mpm.get_module();
    std::unordered_map<instruction_ref, std::string> parameter_names;
    for(const auto& name : m.get_parameter_names())
        parameter_names[m.get_parameter(name)] = name;
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
    auto logical_outputs = m.get_returns();
    std::unordered_set<instruction_ref> logical_inputs(logical_outputs.begin(),
                                                       logical_outputs.end());
    for(auto ins : iterator_for(m))
    {
        if(not contains(live, ins))
            continue;
        const auto& inputs = clone_inputs_for(optimal_rewrite, ins);
        logical_inputs.insert(inputs.begin(), inputs.end());
    }

    std::size_t next_block_number = 0;
    for(const auto& block : blocks)
    {
        auto block_number        = next_block_number++;
        const auto& optimal_body = optimal_rewrite.optimal_body_instructions.at(&block);
        std::vector<instruction_ref> stack;
        for(auto ins : iterator_for(m))
            if(contains(live, ins) and contains(logical_inputs, ins) and is_back_slice(ins) and
               not ins->inputs().empty() and contains(optimal_body, ins->inputs().front()))
                stack.push_back(ins->inputs().front());

        auto absorbable_dependency = [&](instruction_ref ins) {
            if(contains(optimal_body, ins))
                return true;
            if(starts_with(ins->name(), "@") or not ins->module_inputs().empty())
                return false;
            const auto& s = ins->get_shape();
            if(not s.dynamic())
                return true;
            if(not s.symbolic() or has_variable_axis(s))
                return false;
            return all_of(s.dyn_strides(),
                          [](const auto& stride) { return sym::fixed_value(stride).has_value(); });
        };
        std::unordered_set<instruction_ref> clone_body;
        while(not stack.empty())
        {
            auto ins = stack.back();
            stack.pop_back();
            if(not clone_body.insert(ins).second)
                continue;
            for(auto input : clone_inputs_for(optimal_rewrite, ins))
                if(absorbable_dependency(input))
                    stack.push_back(input);
        }
        if(clone_body.empty())
            continue;

        std::vector<instruction_ref> clone_body_instructions;
        std::vector<instruction_ref> boundary_outputs;
        std::unordered_set<instruction_ref> boundary;
        for(auto ins : iterator_for(m))
        {
            if(contains(clone_body, ins))
            {
                clone_body_instructions.push_back(ins);
                for(auto input : clone_inputs_for(optimal_rewrite, ins))
                    if(not contains(clone_body, input))
                        boundary.insert(input);
            }
            if(contains(live, ins) and contains(logical_inputs, ins) and is_back_slice(ins) and
               not ins->inputs().empty() and contains(clone_body, ins->inputs().front()))
                boundary_outputs.push_back(ins);
        }
        if(boundary_outputs.empty())
            continue;

        std::vector<instruction_ref> runtime_extent_inputs;
        for(const auto& root : block.roots)
        {
            if(not contains(root_sources, root.root))
                MIGRAPHX_THROW("SPLIT_SYM_DIM: no parameter resolves block root " + root.name);
            auto source = root_sources.at(root.root);
            boundary.insert(source);
            if(not contains(runtime_extent_inputs, source))
                runtime_extent_inputs.push_back(source);
        }

        std::set<std::string> used_names;
        for(const auto& name : m.get_parameter_names())
            used_names.insert(name);
        std::map<std::string, instruction_ref> input_sources;
        std::vector<instruction_ref> literal_sources;
        const std::string input_prefix = "#split_sym_dim_input_";
        std::size_t generated_suffix   = 0;
        for(auto ins : iterator_for(m))
        {
            if(not contains(boundary, ins))
                continue;
            if(contains(parameter_names, ins))
            {
                input_sources.emplace(parameter_names.at(ins), ins);
                continue;
            }
            if(ins->name() == "@literal")
            {
                literal_sources.push_back(ins);
                ++generated_suffix;
                continue;
            }
            auto name = input_prefix + std::to_string(block_number) + "_" +
                        std::to_string(generated_suffix++);
            while(not used_names.insert(name).second)
                name = input_prefix + std::to_string(block_number) + "_" +
                       std::to_string(generated_suffix++);
            input_sources.emplace(std::move(name), ins);
        }
        if(input_sources.size() + literal_sources.size() != boundary.size())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: failed to collect every block input");

        // Cartesian product of this block's root optimal values.
        const auto& block_roots = block.roots;
        optimal_map fixed_substitutions;
        for(const auto& root : block_roots)
            if(root.interval.min == root.interval.max)
                fixed_substitutions.emplace(root.root, root.optimal_symbol);
        std::vector<std::size_t> choices(block_roots.size(), 0);
        std::vector<module> clones;
        std::vector<clone_output_case> clone_output_shapes;
        std::size_t clone_index = 0;
        for(bool more_combinations = true; more_combinations;)
        {
            freeze_map freeze;
            std::unordered_map<sym::expr, shape::dynamic_dimension::interval> subranges;
            for(auto&& [root, choice] : views::zip(block_roots, choices))
            {
                auto target                 = root.optimal_values.at(choice);
                freeze[root.root]           = target;
                freeze[root.optimal_symbol] = target;
                subranges[root.root]        = root.subranges.at(choice);
            }

            module clone_module{m.name() + ":split_sym_dim_" + std::to_string(block_number) + "_" +
                                std::to_string(clone_index++)};
            std::unordered_map<instruction_ref, instruction_ref> clone_map;
            for(const auto& input : input_sources)
                clone_map[input.second] = clone_module.add_parameter(
                    input.first, clone_parameter_shape(input.second->get_shape(), subranges));
            for(auto literal_source : literal_sources)
                clone_map[literal_source] = clone_module.add_literal(literal_source->get_literal());

            auto record_clone = [&](instruction_ref source, instruction_ref clone) {
                if(clone->get_shape().dynamic())
                    MIGRAPHX_THROW("SPLIT_SYM_DIM: clone body is not fully static");
                clone_map[source] = clone;
                if(not source->get_debug_symbols().empty())
                    clone_module.add_debug_symbols(clone, source->get_debug_symbols());
            };

            std::vector<instruction_ref> runtime_extent_sources;
            std::transform(runtime_extent_inputs.begin(),
                           runtime_extent_inputs.end(),
                           std::back_inserter(runtime_extent_sources),
                           [&](instruction_ref source) { return clone_map.at(source); });
            runtime_cache cache;
            pad_cache reusable_pads;
            for(auto ins : clone_body_instructions)
            {
                std::vector<instruction_ref> args;
                auto source_inputs = clone_inputs_for(optimal_rewrite, ins);
                std::transform(source_inputs.begin(),
                               source_inputs.end(),
                               std::back_inserter(args),
                               [&](instruction_ref in) { return clone_map.at(in); });
                if(contains(optimal_rewrite.pad_plan, ins))
                    for(const auto& pad : optimal_rewrite.pad_plan.at(ins))
                        args.at(pad.operand) =
                            add_or_reuse_pad(clone_module,
                                             make_op("fixed_pad", {{"value", pad.value}}),
                                             args.at(pad.operand),
                                             reusable_pads);
                if(contains(optimal_rewrite.mask_plan, ins))
                    for(const auto& mask : optimal_rewrite.mask_plan.at(ins))
                        args.at(mask.operand) = add_runtime_mask(clone_module,
                                                                 args.at(mask.operand),
                                                                 mask,
                                                                 runtime_extent_sources,
                                                                 fixed_substitutions,
                                                                 cache);
                if(contains(optimal_rewrite.static_subgraph_emitters, ins))
                {
                    const auto& source_op = contains(optimal_rewrite.clone_operations, ins)
                                                ? optimal_rewrite.clone_operations.at(ins)
                                                : ins->get_operator();
                    record_clone(ins,
                                 optimal_rewrite.static_subgraph_emitters.at(ins)(
                                     clone_module, source_op, args, freeze));
                    continue;
                }
                auto op          = contains(optimal_rewrite.clone_operations, ins)
                                       ? optimal_rewrite.clone_operations.at(ins)
                                       : ins->get_operator();
                auto cloned_args = args;
                if(contains(optimal_rewrite.static_op_freezers, ins))
                    op = optimal_rewrite.static_op_freezers.at(ins)(op, freeze);
                else if(ins->get_shape().dynamic() and not contains(optimal_body, ins))
                {
                    std::vector<shape> input_shapes;
                    std::transform(ins->inputs().begin(),
                                   ins->inputs().end(),
                                   std::back_inserter(input_shapes),
                                   [](instruction_ref arg) { return arg->get_shape(); });
                    auto semantics = describe_op(op, input_shapes);
                    if(semantics.symbolic_target.has_value())
                    {
                        const auto& target = *semantics.symbolic_target;
                        cloned_args        = select_inputs(args, target.selected_inputs);
                        if(target.emit_static_subgraph.has_value())
                        {
                            record_clone(ins,
                                         (*target.emit_static_subgraph)(
                                             clone_module, op, cloned_args, freeze));
                            continue;
                        }
                        op = target.freeze_op(op, freeze);
                    }
                }
                record_clone(ins,
                             clone_module.add_instruction(op, cloned_args, ins->module_inputs()));
            }

            std::vector<instruction_ref> clone_outputs;
            std::transform(
                boundary_outputs.begin(),
                boundary_outputs.end(),
                std::back_inserter(clone_outputs),
                [&](instruction_ref out) { return clone_map.at(out->inputs().front()); });
            if(any_of(clone_outputs,
                      [](instruction_ref out) { return out->get_shape().dynamic(); }))
                MIGRAPHX_THROW("SPLIT_SYM_DIM: clone output is not fully static");
            std::vector<shape> output_shapes;
            std::transform(clone_outputs.begin(),
                           clone_outputs.end(),
                           std::back_inserter(output_shapes),
                           [](instruction_ref out) { return out->get_shape(); });
            clone_output_shapes.emplace_back(freeze, std::move(output_shapes));
            clone_module.add_return(clone_outputs);
            clones.push_back(std::move(clone_module));

            more_combinations = false;
            for(auto&& [root, choice] : views::zip(block_roots, choices))
            {
                if(++choice < root.optimal_values.size())
                {
                    more_combinations = true;
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

        std::vector<instruction_ref> selection_inputs;
        std::transform(input_sources.begin(),
                       input_sources.end(),
                       std::back_inserter(selection_inputs),
                       [](const auto& input) { return input.second; });
        std::vector<shape> body_output_shapes;
        for(std::size_t output_index = 0; output_index < boundary_outputs.size(); ++output_index)
        {
            auto source = boundary_outputs.at(output_index)->inputs().front();
            body_output_shapes.push_back(
                dispatch_shape_for_clones(optimal_rewrite.dispatch_output_shapes.at(source),
                                          clone_output_shapes,
                                          output_index));
        }
        auto selection = m.add_instruction(
            make_op("select_module", {{"output_dyn_shapes", to_value(shape{body_output_shapes})}}),
            selection_inputs,
            submodules);

        // Keep each boundary slice at its existing instruction_ref so downstream
        // blocks and unsupported consumers remain wired. Sorting after all blocks
        // moves the select/get_tuple_elem producers before these rewritten slices.
        std::size_t output_number = 0;
        for(auto output : boundary_outputs)
        {
            auto selected_output =
                m.add_instruction(make_op("get_tuple_elem", {{"index", output_number}}), selection);
            std::vector<instruction_ref> args = {selected_output};
            std::copy(std::next(output->inputs().begin()),
                      output->inputs().end(),
                      std::back_inserter(args));
            // The route guarantees the logical extent fits the selected buffer. Preserve those
            // dimensions, but use the selected buffer's strides and do not normalize them again.
            auto output_op                   = output->normalized_operator();
            const auto& logical_output_shape = output->get_shape();
            shape output_shape{logical_output_shape.type(),
                               logical_output_shape.dyn_dims(),
                               selected_output->get_shape().to_symbolic().dyn_strides()};
            instruction::replace(output, std::move(output_op), output_shape, std::move(args));
            output->set_normalized();
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

    lower_symbolic_dimensions_of(m);
    // Shape queries no longer need their model-value producers.
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

    auto optimal_rewrite = rewrite_to_optimal(m, infos, blocks, *roots);
    specialize_blocks(mpm, blocks, optimal_rewrite);
    run_passes(m, {dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
