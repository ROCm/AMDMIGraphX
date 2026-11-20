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
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/symbolic_shape_helpers.hpp>
#include <migraphx/value.hpp>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <optional>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

instruction_ref
make_scatter_prefix_mask(module& m,
                         const std::vector<std::size_t>& prefix_lens,
                         const std::vector<std::pair<std::size_t, sym::expr>>& prefix_axes)
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

// The symbols that have a single value in this clone: those that are exactly a dimension of a
// parameter whose range is one value.
std::unordered_map<sym::expr, sym::expr>
find_fixed_symbols(const std::vector<instruction_ref>& parameters)
{
    std::unordered_map<sym::expr, sym::expr> result;
    for(auto parameter : parameters)
    {
        const auto& parameter_shape = parameter->get_shape();
        if(not parameter_shape.symbolic())
            continue;
        for(const auto& dim : parameter_shape.dyn_dims())
        {
            auto variables = sym::find_variables(dim.sym_expr);
            if(dim.is_symbolic() and dim.is_fixed() and variables.size() == 1 and
               sym::same_symbol(dim.sym_expr, *variables.begin()))
                result.emplace(*variables.begin(),
                               sym::lit(static_cast<int64_t>(dim.get_interval().max)));
        }
    }
    return result;
}

// Constant indices with fixed extents are normalized here, when the pass runs. The same arithmetic
// left in the program is narrowed by the GPU data type pass, which leaves a convert of a literal
// behind.
std::optional<instruction_ref> fold_constant_indices(module& m,
                                                     instruction_ref indices,
                                                     const std::vector<sym::expr>& extents,
                                                     const std::vector<instruction_ref>& parameters)
{
    if(not indices->can_eval())
        return std::nullopt;
    auto fixed_symbols = find_fixed_symbols(parameters);
    std::vector<sym::expr> fixed_extents;
    std::transform(extents.begin(),
                   extents.end(),
                   std::back_inserter(fixed_extents),
                   [&](const auto& extent) { return extent.subs(fixed_symbols); });
    auto extent_values = sym::fixed_values<int64_t>(fixed_extents);
    if(not extent_values.has_value())
        return std::nullopt;

    std::vector<int64_t> values;
    indices->eval().visit([&](auto view) { values.assign(view.begin(), view.end()); });
    std::vector<std::size_t> positions(values.size());
    std::iota(positions.begin(), positions.end(), 0);
    std::transform(values.begin(),
                   values.end(),
                   positions.begin(),
                   values.begin(),
                   [&](int64_t value, std::size_t position) {
                       return value < 0 ? value + extent_values->at(position % extents.size())
                                        : value;
                   });
    return m.add_literal(literal{shape{shape::int64_type, indices->get_shape().lens()}, values});
}

// A negative index counts back from the end of its dimension. The clone pads the indexed axes of
// the data, so the index is normalized against the runtime extent of the unpadded data first.
instruction_ref
normalize_negative_indices(module& m, instruction_ref source, instruction_ref indices)
{
    const auto& index_shape = indices->get_shape();
    auto depth              = index_shape.lens().back();
    if(depth == 0 or shape::is_unsigned(index_shape.type()))
        return indices;
    auto data_dims = source->inputs().front()->get_shape().to_symbolic().dyn_dims();
    std::vector<sym::expr> extents;
    std::transform(data_dims.begin(),
                   data_dims.begin() + depth,
                   std::back_inserter(extents),
                   [](const auto& dim) { return dim.sym_expr; });
    auto parameters = m.get_parameters();
    // Leave the indices as they are if the clone cannot evaluate an extent
    if(find_unresolved_shape_expression(to_shapes(parameters), extents).has_value())
        return indices;
    if(auto folded = fold_constant_indices(m, indices, extents, parameters))
        return *folded;

    const auto lens = index_shape.lens();
    if(index_shape.type() != shape::int64_type)
        indices =
            m.add_instruction(make_op("convert", {{"target_type", shape::int64_type}}), indices);
    auto extent = m.add_instruction(
        make_op("eval_expr_from_shape", {{"expressions", to_value(extents)}}), parameters);
    extent        = m.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}), extent);
    auto zero     = m.add_literal(literal{shape{shape::int64_type, {1}}, {int64_t{0}}});
    zero          = m.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}), zero);
    auto negative = m.add_instruction(make_op("less"), indices, zero);
    auto wrapped  = m.add_instruction(make_op("add"), indices, extent);
    return m.add_instruction(make_op("where"), negative, wrapped, indices);
}

struct analyze_scatternd : analyzer<analyze_scatternd>
{
    bool matches(const operation& op) const { return op.name() == "scatternd_none"; }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        auto inputs = to_shapes(ins->inputs());
        if(inputs.size() != 3)
            return symbolic_op_info{ins};
        const auto& indices = inputs.at(1);
        auto index_rank     = indices.ndim();
        assert(index_rank > 0);
        auto index_dims      = indices.to_symbolic().dyn_dims();
        bool has_prefix_axis = any_of(range(index_rank - 1), [&](auto axis) {
            return is_variable_axis(index_dims.at(axis));
        });
        if(has_prefix_axis and
           (indices.type() != shape::int64_type or
            any_of(inputs, [](const auto& input) { return not input.standard(); })))
            return symbolic_op_info{ins};
        auto index_depth = sym::fixed_value(index_dims.back().sym_expr);
        auto data_dims   = inputs.front().to_symbolic().dyn_dims();
        // Padding an indexed axis of the data changes where a negative index points
        bool has_variable_indexed_axis =
            index_depth.has_value() and
            any_of(range(sym::to<std::size_t>(*index_depth)),
                   [&](auto axis) { return is_variable_axis(data_dims.at(axis)); });
        auto result = analyze_axes(ins);
        if(has_prefix_axis)
            result.freezer = freeze;
        else if(has_variable_indexed_axis)
            result.freezer = freeze_negative_indices;
        return result;
    }

    static instruction_ref
    freeze_negative_indices(module& m,
                            instruction_ref source,
                            const std::vector<instruction_ref>& args,
                            const std::unordered_map<sym::expr, std::size_t>&)
    {
        assert(args.size() == 3);
        auto indices = normalize_negative_indices(m, source, args.at(1));
        return m.add_instruction(source->get_operator(), args.front(), indices, args.back());
    }

    static instruction_ref freeze(module& m,
                                  instruction_ref source,
                                  const std::vector<instruction_ref>& args,
                                  const std::unordered_map<sym::expr, std::size_t>&)
    {
        assert(args.size() == 3);
        const auto& index_shape = source->inputs().at(1)->get_shape();
        auto index_rank         = index_shape.ndim();
        assert(index_rank > 0);
        std::vector<std::pair<std::size_t, sym::expr>> prefix_axes;
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
        // Before the padding below changes the extents of the indexed axes
        indices = normalize_negative_indices(m, source, indices);

        auto valid                 = make_scatter_prefix_mask(m, index_lens, prefix_axes);
        auto effective_index_depth = index_depth;
        if(index_depth == 0)
        {
            if(data->get_shape().scalar())
                data = m.add_instruction(make_op("reshape", {{"dims", {1, 1}}}), data);
            else
                data = m.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), data);
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
        condition = m.add_instruction(
            make_op("multibroadcast", {{"out_lens", rewritten_index_lens}}), condition);

        std::vector<int64_t> data_prefix_lens(data_lens.begin(),
                                              data_lens.begin() + effective_index_depth);
        auto sink = m.add_literal(
            literal{shape{shape::int64_type, {effective_index_depth}}, data_prefix_lens});
        sink = m.add_instruction(make_op("multibroadcast", {{"out_lens", rewritten_index_lens}}),
                                 sink);

        if(index_depth == 0)
        {
            std::vector<int64_t> zeros(effective_index_depth, 0);
            indices =
                m.add_literal(literal{shape{shape::int64_type, {effective_index_depth}}, zeros});
            indices = m.add_instruction(
                make_op("multibroadcast", {{"out_lens", rewritten_index_lens}}), indices);
        }
        indices = m.add_instruction(make_op("where"), condition, indices, sink);

        auto scattered = m.add_instruction(op, padded_data, indices, updates);
        std::vector<int64_t> axes(effective_index_depth);
        std::iota(axes.begin(), axes.end(), int64_t{0});
        std::vector<int64_t> starts(effective_index_depth, 0);
        auto result = m.add_instruction(
            make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", data_prefix_lens}}),
            scattered);
        if(index_depth == 0)
            result = m.add_instruction(make_op("squeeze", {{"axes", {0}}}), result);
        return m.add_instruction(make_op("contiguous"), result);
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
