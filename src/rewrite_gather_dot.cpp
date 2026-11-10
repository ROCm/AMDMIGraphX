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
#include <algorithm>
#include <cassert>
#include <numeric>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

/// The rows gathered with one index vector from tables of the same length.
/// The rows may repeat over tokens, with the same number of rows for each
/// token: row r then belongs to token r / (rows / tokens).
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
};

/// Lifts the instructions computed from gathered rows to all the entries of
/// the table, as long as each row is computed from the same row of the inputs
struct row_lifter
{
    std::unordered_map<instruction_ref, row_group> groups;
    std::unordered_map<instruction_ref, lifted_rows> lifted;

    /// The axes after inserting the token axis, which is not one of them
    static optional<std::vector<std::int64_t>> shift_axes(std::vector<std::int64_t> result,
                                                          std::size_t ndim)
    {
        if(result.empty())
            return nullopt;
        std::transform(result.begin(), result.end(), result.begin(), [&](std::int64_t axis) {
            return axis < 0 ? axis + ndim : axis;
        });
        if(contains(result, 0))
            return nullopt;
        std::transform(
            result.begin(), result.end(), result.begin(), [](auto axis) { return axis + 1; });
        return result;
    }

    /// The operator computing a view of the rows for every entry, if the view
    /// keeps the rows apart
    static optional<operation> dense_view_op(instruction_ref ins,
                                             const std::vector<std::size_t>& out_lens)
    {
        auto input = ins->inputs().front();
        auto v     = ins->get_operator().to_value();
        if(ins->name() == "contiguous")
            return ins->get_operator();
        if(ins->name() == "reshape")
            return make_op("reshape", {{"dims", out_lens}});
        if(ins->name() == "multibroadcast")
        {
            if(input->get_shape().ndim() != ins->get_shape().ndim())
                return nullopt;
            return make_op("multibroadcast", {{"out_lens", out_lens}});
        }
        if(ins->name() == "broadcast")
        {
            if(v.at("axis").to<std::int64_t>() != 0)
                return nullopt;
            return make_op("broadcast", {{"axis", 0}, {"out_lens", out_lens}});
        }
        if(ins->name() == "transpose")
        {
            auto perm = v.at("permutation").to_vector<std::int64_t>();
            if(perm.front() != 0)
                return nullopt;
            std::transform(
                perm.begin(), perm.end(), perm.begin(), [](auto axis) { return axis + 1; });
            perm.insert(perm.begin(), 0);
            return make_op("transpose", {{"permutation", perm}});
        }
        if(contains({"squeeze", "unsqueeze", "slice"}, ins->name()))
        {
            // The axes of an unsqueeze are those of the output
            auto ndim =
                ins->name() == "unsqueeze" ? ins->get_shape().ndim() : input->get_shape().ndim();
            auto axes = shift_axes(v.at("axes").to_vector<std::int64_t>(), ndim);
            if(not axes.has_value())
                return nullopt;
            v["axes"] = *axes;
            return make_op(ins->name(), v);
        }
        if(ins->name() == "unpack_int4")
        {
            auto axes = shift_axes({v.at("axis").to<std::int64_t>()}, input->get_shape().ndim());
            if(not axes.has_value())
                return nullopt;
            v["axis"] = axes->front();
            return make_op(ins->name(), v);
        }
        return nullopt;
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
        if(s.dynamic())
            return false;
        return s.ndim() < input->get_shape().ndim() or s.lens().front() == 1;
    }

    /// An input of a pointwise operator for every entry: either lifted rows,
    /// or one row broadcast to all the rows
    instruction_ref dense_input(module& m,
                                instruction_ref ins,
                                instruction_ref input,
                                const std::vector<std::size_t>& out_lens) const
    {
        auto bcast = make_op("multibroadcast", {{"out_lens", out_lens}});
        auto it    = lifted.find(input);
        if(it != lifted.end())
        {
            auto dense = it->second.dense;
            if(dense->get_shape().lens() == out_lens)
                return dense;
            return m.insert_instruction(ins, bcast, dense);
        }
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
        auto axis = ins->get_operator().to_value().at("axis").to<std::int64_t>();
        if(axis < 0)
            axis += dshape.ndim();
        if(axis != 0)
            return nullopt;
        auto entries = dshape.lens().front();
        auto rows    = ishape.elements();
        // With fewer rows than entries the gathers read less than the table
        if(entries < 2 or rows < entries)
            return nullopt;
        auto it = groups.emplace(indices, row_group{indices, entries, rows}).first;
        if(it->second.entries != entries)
            return nullopt;
        auto dense = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {1}}}), data);
        return lifted_rows{dense, indices};
    }

    optional<lifted_rows> lift_pointwise(module& m, instruction_ref ins, const row_group& group)
    {
        const auto& lens   = ins->get_shape().lens();
        const auto& inputs = ins->inputs();
        // The other inputs must be the same for every row
        if(not std::all_of(inputs.begin(), inputs.end(), [&](instruction_ref input) {
               const auto& s = input->get_shape();
               if(s.dynamic() or s.lens() != lens)
                   return false;
               return contains(lifted, input) or is_row_broadcast(input);
           }))
            return nullopt;
        bool per_token = std::any_of(inputs.begin(), inputs.end(), [&](instruction_ref input) {
            return contains(lifted, input) and lifted.at(input).per_token;
        });
        bool has_dot   = std::any_of(inputs.begin(), inputs.end(), [&](instruction_ref input) {
            return contains(lifted, input) and lifted.at(input).has_dot;
        });
        auto out_lens  = group.dense_lens(lens, per_token);
        std::vector<instruction_ref> args;
        std::transform(inputs.begin(),
                       inputs.end(),
                       std::back_inserter(args),
                       [&](instruction_ref input) { return dense_input(m, ins, input, out_lens); });
        auto dense = m.insert_instruction(ins, ins->get_operator(), args);
        return lifted_rows{dense, group.indices, per_token, has_dot};
    }

    optional<lifted_rows> lift_view(module& m, instruction_ref ins, const row_group& group)
    {
        auto rows     = lifted.at(ins->inputs().front());
        auto out_lens = group.dense_lens(ins->get_shape().lens(), rows.per_token);
        auto op       = dense_view_op(ins, out_lens);
        if(not op.has_value())
            return nullopt;
        rows.dense = m.insert_instruction(ins, *op, rows.dense);
        assert(rows.dense->get_shape().lens() == out_lens);
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
        if(x->get_shape().dynamic() or x->get_shape().ndim() != blens.size())
            return nullopt;
        // The leading axes that make up the rows
        std::vector<std::size_t> sizes(blens.size());
        std::partial_sum(blens.begin(), blens.end(), sizes.begin(), std::multiplies<>{});
        auto it = std::find(sizes.begin(), sizes.end(), group.rows);
        if(it == sizes.end())
            return nullopt;
        // The last of them must be the only broadcast, over the rows of a token
        auto axis   = std::distance(sizes.begin(), it);
        auto xlens  = x->get_shape().lens();
        auto tokens = group.rows / blens[axis];
        if(blens[axis] < 2 or xlens[axis] != 1)
            return nullopt;
        xlens[axis] = blens[axis];
        if(xlens != blens)
            return nullopt;
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
        if(a->get_shape().dynamic() or alens.size() != 3 or alens[1] != 1)
            return nullopt;
        lifted_rows result{ins, group.indices, true, true};
        instruction_ref a_matrix;
        auto ia = lifted.find(a);
        if(ia != lifted.end())
        {
            result.per_token = ia->second.per_token;
            a_matrix =
                m.insert_instruction(ins, make_op("squeeze", {{"axes", {2}}}), ia->second.dense);
        }
        else
        {
            auto matrix = token_matrix(m, ins, a, group);
            if(not matrix.has_value())
                return nullopt;
            a_matrix = *matrix;
        }
        auto b_matrix =
            m.insert_instruction(ins, make_op("squeeze", {{"axes", {1}}}), ib->second.dense);
        auto dot     = m.insert_instruction(ins, make_op("dot"), a_matrix, b_matrix);
        result.dense = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {2}}}), dot);
        return result;
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
        if(ins->inputs().size() == 1)
            return lift_view(m, ins, group);
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
    instruction_ref insert_select(module& m, instruction_ref ins, const lifted_rows& rows) const
    {
        const auto& group = groups.at(rows.indices);
        if(not rows.per_token)
        {
            auto table = m.insert_instruction(ins, make_op("squeeze", {{"axes", {1}}}), rows.dense);
            return m.insert_instruction(
                ins, make_op("gather", {{"axis", 0}}), table, group.indices);
        }
        std::vector<std::size_t> token_ids(group.rows);
        std::iota(token_ids.begin(), token_ids.end(), 0);
        std::transform(token_ids.begin(), token_ids.end(), token_ids.begin(), [&](auto row) {
            return row / (group.rows / group.tokens);
        });
        auto tokens = m.add_literal(
            literal{shape{group.indices->get_shape().type(), {group.rows, 1}}, token_ids});
        auto entries =
            m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {1}}}), group.indices);
        auto indices = m.insert_instruction(ins, make_op("concat", {{"axis", 1}}), entries, tokens);
        return m.insert_instruction(ins, make_op("gathernd"), rows.dense, indices);
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
        std::vector<instruction_ref> reads;
        std::copy_if(computed.begin(),
                     computed.end(),
                     std::back_inserter(reads),
                     [&](instruction_ref ins) { return is_read(ins); });
        for(auto ins : reads)
            m.replace_instruction(ins, insert_select(m, ins, lifted.at(ins)));
    }
};

} // namespace

void rewrite_gather_dot::apply(module& m) const { row_lifter{}.apply(m); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
