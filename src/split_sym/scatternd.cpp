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
#include <migraphx/value.hpp>
#include <algorithm>
#include <numeric>
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
        const auto& index_dims = indices.dyn_dims();
        bool has_prefix_axis   = false;
        for(std::size_t axis = 0; axis + 1 < index_rank; ++axis)
            has_prefix_axis = has_prefix_axis or is_variable_axis(index_dims.at(axis));
        if(has_prefix_axis and
           (indices.type() != shape::int64_type or
            any_of(inputs, [](const auto& input) { return not input.standard(); })))
            return symbolic_op_info{ins};
        auto result = analyze_axes(ins);
        if(has_prefix_axis)
            result.freezer = freeze;
        return result;
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
        condition = m.add_instruction(
            make_op("multibroadcast", {{"out_lens", rewritten_index_lens}}), condition);

        std::vector<int64_t> sink_values(effective_index_depth);
        std::transform(data_lens.begin(),
                       data_lens.begin() + effective_index_depth,
                       sink_values.begin(),
                       [](auto x) { return static_cast<int64_t>(x); });
        auto sink =
            m.add_literal(literal{shape{shape::int64_type, {effective_index_depth}}, sink_values});
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
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
