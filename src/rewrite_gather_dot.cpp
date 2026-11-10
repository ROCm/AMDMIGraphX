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
 *
 */
#include <migraphx/rewrite_gather_dot.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape_transform_descriptor.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

/// The rows gathered with one index vector from tables of the same length.
/// Each token owns rows / tokens consecutive rows, so row r belongs to token
/// r / (rows / tokens).
struct row_group
{
    instruction_ref indices;
    std::size_t entries = 0;
    std::size_t rows    = 0;
    /// Zero until a dot reads rows that are broadcast over the rows of a token
    std::size_t tokens = 0;

    /// The lens of a row tensor {rows, rest...} computed for every entry
    std::vector<std::size_t> dense_lens(const std::vector<std::size_t>& lens, bool per_token) const
    {
        std::vector<std::size_t> result = {entries, per_token ? tokens : 1};
        result.insert(result.end(), lens.begin() + 1, lens.end());
        return result;
    }

    /// The lens of the row tensor {rows, rest...} computed as {entries, t, rest...}
    std::vector<std::size_t> row_lens(const std::vector<std::size_t>& dense) const
    {
        std::vector<std::size_t> result = {rows};
        result.insert(result.end(), dense.begin() + 2, dense.end());
        return result;
    }
};

/// A row tensor {rows, rest...} computed for every table entry instead of the
/// gathered one: {entries, tokens, rest...}, or {entries, 1, rest...} when
/// it is the same for every token
struct lifted_rows
{
    instruction_ref dense;
    instruction_ref indices;
    bool per_token = false;
    /// Whether a dot was computed over the whole table to get these rows
    bool has_dot = false;
    /// Views of the rows of dense that are applied together when the rows are read
    std::vector<operation> views = {};
};

/// Lifts the instructions computed from gathered rows to all the entries of
/// the table, as long as each row is computed from the same row of the inputs
struct row_lifter
{
    float min_rows_per_entry = 1.0f;
    std::unordered_map<instruction_ref, row_group> groups;
    std::unordered_map<instruction_ref, lifted_rows> lifted;

    /// The normalized axes shifted past the token axis, unless empty or
    /// including the row axis
    static optional<std::vector<std::int64_t>> shift_axes(std::vector<std::int64_t> result)
    {
        if(result.empty() or contains(result, 0))
            return nullopt;
        std::transform(
            result.begin(), result.end(), result.begin(), [](auto axis) { return axis + 1; });
        return result;
    }

    /// The views of a row tensor as one transform, which is empty unless the
    /// descriptor models them and they keep the rows apart
    static shape_transform_descriptor rows_transform(const std::vector<std::size_t>& lens,
                                                     const std::vector<operation>& views)
    {
        auto desc = shape_transform_descriptor::create(lens, views);
        if(desc.get_dst_axes_from_src(0) != std::vector<std::size_t>{0})
            return {};
        return desc;
    }

    /// The indices of the group the lifted inputs belong to
    optional<instruction_ref> find_group(instruction_ref ins) const
    {
        std::vector<instruction_ref> keys;
        transform_if(
            ins->inputs().begin(),
            ins->inputs().end(),
            std::back_inserter(keys),
            [&](instruction_ref input) { return contains(lifted, input); },
            [&](instruction_ref input) { return lifted.at(input).indices; });
        if(keys.empty())
            return nullopt;
        if(not std::all_of(
               keys.begin(), keys.end(), [&](instruction_ref key) { return key == keys.front(); }))
            return nullopt;
        return keys.front();
    }

    /// Whether the input is one row broadcast to all the rows
    static bool is_row_broadcast(instruction_ref input)
    {
        if(input->name() != "multibroadcast")
            return false;
        const auto& s = input->inputs().front()->get_shape();
        return s.ndim() < input->get_shape().ndim() or s.lens().front() == 1;
    }

    /// Whether any lifted input has the flag set
    bool any_lifted(instruction_ref ins, bool lifted_rows::* flag) const
    {
        const auto& inputs = ins->inputs();
        return std::any_of(inputs.begin(), inputs.end(), [&](instruction_ref input) {
            auto it = lifted.find(input);
            return it != lifted.end() and it->second.*flag;
        });
    }

    /// The rows of a lifted instruction for every entry, inserted before ins,
    /// with its views applied to the dense rows as one transform
    instruction_ref dense_rows(module& m, instruction_ref ins, instruction_ref x) const
    {
        const auto& rows  = lifted.at(x);
        const auto& group = groups.at(rows.indices);
        auto result       = rows.dense;
        if(not rows.views.empty())
        {
            const auto& lens = rows.dense->get_shape().lens();
            auto desc        = rows_transform(group.row_lens(lens), rows.views);
            assert(not desc.empty());
            // Split the rows axis into the entry and token axes of the dense rows
            desc.prepend_axis(lens.front());
            auto dense_desc = desc.rebase(lens);
            assert(not dense_desc.empty());
            result = insert_ops(m, ins, dense_desc.generate(), rows.dense);
        }
        assert(result->get_shape().lens() ==
               group.dense_lens(x->get_shape().lens(), rows.per_token));
        return result;
    }

    /// An input of a pointwise operator for every entry: either lifted rows,
    /// or one row broadcast to all the rows
    instruction_ref dense_input(module& m,
                                instruction_ref ins,
                                instruction_ref input,
                                const std::vector<std::size_t>& out_lens) const
    {
        auto bcast = make_op("multibroadcast", {{"out_lens", out_lens}});
        if(contains(lifted, input))
        {
            auto dense = dense_rows(m, ins, input);
            if(dense->get_shape().lens() == out_lens)
                return dense;
            return m.insert_instruction(ins, bcast, dense);
        }
        assert(is_row_broadcast(input));
        auto row = input->inputs().front();
        // Keep the row axis apart from the token axis
        if(row->get_shape().ndim() == input->get_shape().ndim())
            row = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {1}}}), row);
        return m.insert_instruction(ins, bcast, row);
    }

    optional<lifted_rows> lift_gather(module& m, instruction_ref ins)
    {
        auto data          = ins->inputs().front();
        auto indices       = ins->inputs().back();
        const auto& dshape = data->get_shape();
        const auto& ishape = indices->get_shape();
        if(dshape.dynamic() or ishape.dynamic() or ishape.ndim() != 1)
            return nullopt;
        if(ins->normalized_operator().to_value().at("axis").to<std::int64_t>() != 0)
            return nullopt;
        auto entries = dshape.lens().front();
        auto rows    = ishape.elements();
        // With too few rows per entry the gathered dots cost less than a dot over the table
        if(entries < 2 or rows < min_rows_per_entry * entries)
            return nullopt;
        auto it = groups.emplace(indices, row_group{indices, entries, rows}).first;
        if(it->second.entries != entries)
            return nullopt;
        auto dense = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {1}}}), data);
        return lifted_rows{dense, indices};
    }

    optional<lifted_rows>
    lift_pointwise(module& m, instruction_ref ins, const row_group& group) const
    {
        const auto& lens   = ins->get_shape().lens();
        const auto& inputs = ins->inputs();
        // Inputs that are not lifted must be the same for every row
        if(not std::all_of(inputs.begin(), inputs.end(), [&](instruction_ref input) {
               const auto& s = input->get_shape();
               if(s.dynamic() or s.lens() != lens)
                   return false;
               return contains(lifted, input) or is_row_broadcast(input);
           }))
            return nullopt;
        bool per_token = any_lifted(ins, &lifted_rows::per_token);
        bool has_dot   = any_lifted(ins, &lifted_rows::has_dot);
        auto out_lens  = group.dense_lens(lens, per_token);
        std::vector<instruction_ref> args;
        std::transform(inputs.begin(),
                       inputs.end(),
                       std::back_inserter(args),
                       [&](instruction_ref input) { return dense_input(m, ins, input, out_lens); });
        auto dense = m.insert_instruction(ins, ins->get_operator(), args);
        return lifted_rows{dense, group.indices, per_token, has_dot};
    }

    /// A view is only recorded, so that a chain of views is applied to the
    /// dense rows as one transform
    optional<lifted_rows> lift_view(instruction_ref ins, const row_group& group) const
    {
        auto rows = lifted.at(ins->inputs().front());
        rows.views.push_back(ins->get_operator());
        if(rows_transform(group.row_lens(rows.dense->get_shape().lens()), rows.views).empty())
            return nullopt;
        return rows;
    }

    /// A slice or an unpack, which the descriptor does not model, applied to
    /// the dense rows with its axes shifted past the token axis
    optional<lifted_rows> lift_axis_op(module& m, instruction_ref ins) const
    {
        auto v = ins->normalized_operator().to_value();
        if(ins->name() == "slice")
        {
            auto axes = shift_axes(v.at("axes").to_vector<std::int64_t>());
            if(not axes.has_value())
                return nullopt;
            v["axes"] = *axes;
        }
        else
        {
            auto axes = shift_axes({v.at("axis").to<std::int64_t>()});
            if(not axes.has_value())
                return nullopt;
            v["axis"] = axes->front();
        }
        auto input = ins->inputs().front();
        auto rows  = lifted.at(input);
        rows.dense = m.insert_instruction(ins, make_op(ins->name(), v), dense_rows(m, ins, input));
        rows.views.clear();
        return rows;
    }

    /// The rows of a dot input that repeat the same row for each row of a
    /// token, as one matrix {entries, tokens, n} for all the entries
    static optional<instruction_ref>
    token_matrix(module& m, instruction_ref ins, instruction_ref input, row_group& group)
    {
        if(input->name() != "reshape")
            return nullopt;
        auto bcast = input->inputs().front();
        if(bcast->name() != "multibroadcast")
            return nullopt;
        auto x            = bcast->inputs().front();
        const auto& blens = bcast->get_shape().lens();
        if(x->get_shape().ndim() != blens.size())
            return nullopt;
        // The leading axes that make up the rows
        std::vector<std::size_t> sizes(blens.size());
        std::partial_sum(blens.begin(), blens.end(), sizes.begin(), std::multiplies<>{});
        auto it = std::find(sizes.begin(), sizes.end(), group.rows);
        if(it == sizes.end())
            return nullopt;
        // The last of them must be the only broadcast, over the rows of a token
        auto axis  = std::distance(sizes.begin(), it);
        auto xlens = x->get_shape().lens();
        if(blens[axis] < 2 or xlens[axis] != 1)
            return nullopt;
        xlens[axis] = blens[axis];
        if(xlens != blens)
            return nullopt;
        auto tokens = group.rows / blens[axis];
        if(group.tokens != 0 and group.tokens != tokens)
            return nullopt;
        group.tokens                  = tokens;
        std::vector<std::size_t> lens = {1, tokens, input->get_shape().lens().back()};
        auto matrix  = m.insert_instruction(ins, make_op("reshape", {{"dims", lens}}), x);
        lens.front() = group.entries;
        return m.insert_instruction(ins, make_op("multibroadcast", {{"out_lens", lens}}), matrix);
    }

    /// A dot of one row vector per row with the matrix gathered for the row
    /// becomes a dot of the rows of all the tokens with every matrix
    optional<lifted_rows> lift_dot(module& m, instruction_ref ins, row_group& group)
    {
        auto a  = ins->inputs().front();
        auto b  = ins->inputs().back();
        auto ib = lifted.find(b);
        if(ib == lifted.end() or ib->second.per_token)
            return nullopt;
        const auto& alens = a->get_shape().lens();
        if(alens.size() != 3 or alens[1] != 1)
            return nullopt;
        bool per_token = true;
        instruction_ref a_matrix;
        auto ia = lifted.find(a);
        if(ia != lifted.end())
        {
            per_token = ia->second.per_token;
            a_matrix  = m.insert_instruction(
                ins, make_op("squeeze", {{"axes", {2}}}), dense_rows(m, ins, a));
        }
        else
        {
            auto matrix = token_matrix(m, ins, a, group);
            if(not matrix.has_value())
                return nullopt;
            a_matrix = *matrix;
        }
        auto b_matrix =
            m.insert_instruction(ins, make_op("squeeze", {{"axes", {1}}}), dense_rows(m, ins, b));
        auto dot   = m.insert_instruction(ins, make_op("dot"), a_matrix, b_matrix);
        auto dense = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {2}}}), dot);
        return lifted_rows{dense, group.indices, per_token, true};
    }

    optional<lifted_rows> lift(module& m, instruction_ref ins)
    {
        if(ins->name() == "gather")
            return lift_gather(m, ins);
        auto key = find_group(ins);
        if(not key.has_value() or not ins->module_inputs().empty())
            return nullopt;
        auto& group   = groups.at(*key);
        const auto& s = ins->get_shape();
        if(s.dynamic() or s.ndim() == 0 or s.lens().front() != group.rows)
            return nullopt;
        if(ins->name() == "dot")
            return lift_dot(m, ins, group);
        if(ins->get_operator().attributes().contains("pointwise"))
            return lift_pointwise(m, ins, group);
        if(contains({"slice", "unpack_int4"}, ins->name()))
            return lift_axis_op(m, ins);
        if(ins->inputs().size() == 1)
            return lift_view(ins, group);
        return nullopt;
    }

    /// Whether the rows are read by an instruction that was not lifted. A
    /// gather is lifted without reading its inputs for every entry.
    bool is_read(instruction_ref ins) const
    {
        const auto& outputs = ins->outputs();
        return outputs.empty() or
               std::any_of(outputs.begin(), outputs.end(), [&](instruction_ref output) {
                   return not contains(lifted, output) or output->name() == "gather";
               });
    }

    /// Gather the rows from the ones computed for every entry
    instruction_ref insert_select(module& m, instruction_ref ins) const
    {
        const auto& rows  = lifted.at(ins);
        const auto& group = groups.at(rows.indices);
        auto dense        = dense_rows(m, ins, ins);
        if(not rows.per_token)
        {
            auto table = m.insert_instruction(ins, make_op("squeeze", {{"axes", {1}}}), dense);
            return m.insert_instruction(
                ins, make_op("gather", {{"axis", 0}}), table, group.indices);
        }
        assert(group.tokens > 0 and group.rows % group.tokens == 0);
        shape token_rows{shape::uint64_type, {group.tokens, group.rows / group.tokens}};
        std::vector<std::size_t> token_ids(group.rows);
        std::iota(token_ids.begin(), token_ids.end(), 0);
        std::transform(token_ids.begin(), token_ids.end(), token_ids.begin(), [&](auto row) {
            return token_rows.multi(row).front();
        });
        auto tokens = m.add_literal(
            literal{shape{group.indices->get_shape().type(), {group.rows, 1}}, token_ids});
        auto entries =
            m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {1}}}), group.indices);
        auto indices = m.insert_instruction(ins, make_op("concat", {{"axis", 1}}), entries, tokens);
        return m.insert_instruction(ins, make_op("gathernd"), dense, indices);
    }

    void apply(module& m)
    {
        // Only the rows computed with a dot are worth computing for every entry
        std::vector<instruction_ref> computed;
        for(auto ins : iterator_for(m))
        {
            auto rows = lift(m, ins);
            if(not rows.has_value())
                continue;
            lifted.emplace(ins, *rows);
            if(rows->has_dot)
                computed.push_back(ins);
        }
        auto reads = find_all(computed, [&](instruction_ref ins) { return is_read(ins); });
        for(auto ins : reads)
            m.replace_instruction(ins, insert_select(m, ins));
    }
};

} // namespace

void rewrite_gather_dot::apply(module& m) const
{
    row_lifter{.min_rows_per_entry = min_rows_per_entry}.apply(m);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
