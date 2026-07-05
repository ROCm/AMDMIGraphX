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
#include <migraphx/fuse_horizontal.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/functional.hpp>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <tuple>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// ---------------------------------------------------------------------------
// Horizontal fusion framework
//
// To add a new horizontal fusion, define a plain struct that implements:
//
//   std::size_t min_group_size() const          — minimum group size for fusion
//   bool is_candidate(instruction_ref) const    — does this instruction qualify?
//   auto group_key(instruction_ref) const       — grouping key (equality-comparable)
//   std::vector<instruction_ref>
//       fuse(module&, const std::vector<instruction_ref>&,
//            instruction_ref insert_pt) const
//       — fuse a group, return one replacement instruction per original op
//
// Then pass an instance to fuse_horizontal_ops().
// The framework handles scanning, grouping independent instructions by key,
// filtering inter-dependent instructions, dispatching to fuse(), and replacing
// originals with results.
// ---------------------------------------------------------------------------

template <class Finder>
static void apply_horizontal_finder(module& m, const Finder& finder)
{
    // Collect all candidate instructions and build position map.  Skip dead
    // instructions (no outputs): a finder leaves the originals it replaced in
    // place until the next dead_code_elimination pass, and those stale ops must
    // not be re-fused by a subsequent finder.
    std::vector<instruction_ref> candidates;
    copy_if(iterator_for(m), std::back_inserter(candidates), [&](auto ins) {
        return not ins->outputs().empty() and finder.is_candidate(ins);
    });
    std::unordered_map<instruction_ref, std::size_t> pos;
    std::size_t p = 0;
    for(auto ins : iterator_for(m))
    {
        pos[ins] = p++;
    }

    auto pred = [&](instruction_ref x, instruction_ref y) {
        if(x == y)
            return true;
        if(finder.group_key(x) != finder.group_key(y))
            return false;
        if(pos.at(x) < pos.at(y))
            return not reaches(x, y);
        return not reaches(y, x);
    };

    auto each = [&](auto start, auto last) {
        auto n = std::distance(start, last);
        if(n < finder.min_group_size())
            return;

        std::vector<instruction_ref> group(start, last);
        // Sort by position for consistent ordering
        std::sort(
            group.begin(), group.end(), [&](auto a, auto b) { return pos.at(a) < pos.at(b); });

        auto insert_pt    = std::next(group.back());
        auto replacements = finder.fuse(m, group, insert_pt);
        if(replacements.empty())
            return;

        assert(replacements.size() == group.size());

        // Move outputs of the original instructions to after the new instructions
        // so that replace_instruction's validity assertions hold.
        std::for_each(group.begin(), group.end(), [&](auto g) {
            m.move_output_instructions_after(g, replacements.back());
        });

        migraphx::for_each(group.begin(), group.end(), replacements.begin(), [&](auto g, auto r) {
            m.replace_instruction(g, r);
        });
    };

    group_by(candidates.begin(), candidates.end(), each, pred);
}

template <class... Finders>
static void fuse_horizontal_ops(module& m, Finders&&... finders)
{
    each_args([&](auto&& finder) { apply_horizontal_finder(m, finder); }, finders...);
}

// Slice a batched-gather result back into one row range per original gather,
// using each gather's index batch (first index dim) as the row count.
static std::vector<instruction_ref> slice_gather_rows(module& m,
                                                      instruction_ref batched_gather,
                                                      const std::vector<instruction_ref>& gathers,
                                                      instruction_ref insert_pt)
{
    // Inclusive prefix sum of each gather's index batch (first index dim) gives
    // the slice end offsets; shifting right and prepending 0 gives the starts.
    std::vector<std::size_t> slice_ends(gathers.size());
    transform_partial_sum(
        gathers.begin(), gathers.end(), slice_ends.begin(), std::plus<>{}, [](auto g) {
            return g->inputs().at(1)->get_shape().lens().front();
        });

    std::vector<std::size_t> slice_starts(gathers.size());
    slice_starts[0] = 0;
    std::copy(slice_ends.begin(), std::prev(slice_ends.end()), slice_starts.begin() + 1);

    std::vector<instruction_ref> results;
    results.reserve(gathers.size());
    migraphx::for_each(
        slice_starts.begin(), slice_starts.end(), slice_ends.begin(), [&](auto start, auto end) {
            results.push_back(m.insert_instruction(
                insert_pt,
                make_op("slice",
                        {{"axes", std::vector<int64_t>{0}},
                         {"starts", std::vector<int64_t>{static_cast<int64_t>(start)}},
                         {"ends", std::vector<int64_t>{static_cast<int64_t>(end)}}}),
                batched_gather));
        });

    return results;
}

// Fuse a group of gathers whose indices may have *different* shapes.  Each index is
// flattened to 1-D and concatenated, a single batched gather runs over `table`, then each
// element range is sliced back out and reshaped to the original gather's output shape
// (index dims + embedding dim).  indices[i] is the (already offset-adjusted) index for
// gathers[i]; table is the gather data operand (a single, possibly concatenated table).
static std::vector<instruction_ref>
fuse_gathers_flattened(module& m,
                       const std::vector<instruction_ref>& gathers,
                       const std::vector<instruction_ref>& indices,
                       instruction_ref table,
                       instruction_ref insert_pt)
{
    std::vector<instruction_ref> flat_inputs(indices.size());
    std::transform(indices.begin(), indices.end(), flat_inputs.begin(), [&](instruction_ref idx) {
        if(idx->get_shape().lens().size() == 1)
            return idx;
        std::int64_t n = idx->get_shape().elements();
        return m.insert_instruction(insert_pt, make_op("reshape", {{"dims", {n}}}), idx);
    });

    auto big_idx = m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), flat_inputs);
    auto batched_gather =
        m.insert_instruction(insert_pt, make_op("gather", {{"axis", 0}}), table, big_idx);

    // Inclusive prefix sum of element counts gives each range's end; shift right and
    // prepend 0 for the (exclusive) start offsets into the batched gather.
    std::vector<std::size_t> slice_ends(gathers.size());
    transform_partial_sum(
        gathers.begin(), gathers.end(), slice_ends.begin(), std::plus<>{}, [](auto g) {
            return g->inputs().at(1)->get_shape().elements();
        });
    std::vector<std::size_t> slice_starts(gathers.size());
    slice_starts[0] = 0;
    std::copy(slice_ends.begin(), std::prev(slice_ends.end()), slice_starts.begin() + 1);

    const std::size_t emb_dim = table->get_shape().lens().back();
    std::vector<instruction_ref> results(gathers.size());
    std::transform(gathers.begin(),
                   gathers.end(),
                   slice_starts.begin(),
                   results.begin(),
                   [&](instruction_ref g, std::size_t start) -> instruction_ref {
                       const auto& idx_lens = g->inputs().at(1)->get_shape().lens();
                       std::size_t n        = g->inputs().at(1)->get_shape().elements();
                       auto sliced          = m.insert_instruction(
                           insert_pt,
                           make_op("slice",
                                            {{"axes", {0}},
                                             {"starts", {static_cast<std::int64_t>(start)}},
                                             {"ends", {static_cast<std::int64_t>(start + n)}}}),
                           batched_gather);
                       // A 1-D index already yields {n, emb_dim}; only multi-dim indices need a
                       // reshape back to (index dims + embedding dim).
                       if(idx_lens.size() == 1)
                           return sliced;
                       std::vector<std::int64_t> out_dims(idx_lens.begin(), idx_lens.end());
                       out_dims.push_back(static_cast<std::int64_t>(emb_dim));
                       return m.insert_instruction(
                           insert_pt, make_op("reshape", {{"dims", out_dims}}), sliced);
                   });

    return results;
}

// ---------------------------------------------------------------------------
// Same-table gather horizontal fusion
//
// Candidates: gather(axis=0) with 2D constant embedding table and a non-scalar
//             index whose first dim is >= min_index_batch (worthwhile batch)
// Grouping:   by (table instruction, index type, index trailing dims) so only
//             gathers reading the *same* table are merged.  With merge_mixed_lengths
//             the trailing dims are dropped, so same-table gathers with differing
//             index shapes are merged too (via the flattened path).
// Fusion:     concatenate the indices, single batched gather, slice rows back.
//             No index offset adjustment is needed since the table is shared.
// ---------------------------------------------------------------------------

struct same_table_gather_horizontal_fusion
{
    // Minimum first index dim for the batched gather to be worthwhile.
    static constexpr std::size_t min_index_batch = 4;

    // When set, merge same-table gathers whose indices differ only in shape by flattening
    // each index (rather than requiring matching trailing dims).  Because the table is
    // shared, no index offset adjustment is introduced.
    bool merge_mixed_lengths = false;

    std::size_t min_group_size() const { return 2; }

    bool is_candidate(instruction_ref ins) const
    {
        if(ins->name() != "gather")
            return false;

        if(ins->get_operator().to_value()["axis"].to<int>() != 0)
            return false;

        auto data = ins->inputs().at(0);
        auto idx  = ins->inputs().at(1);

        // Embedding must be a 2D constant: {num_rows, embedding_dim}
        if(data->get_shape().lens().size() != 2)
            return false;
        if(not data->can_eval())
            return false;

        // Index must not be scalar
        if(idx->get_shape().scalar() or idx->get_shape().lens().empty())
            return false;

        // Require an index batch of at least min_index_batch to be worthwhile
        if(idx->get_shape().lens().front() < min_index_batch)
            return false;

        return true;
    }

    auto group_key(instruction_ref ins) const
    {
        auto data        = ins->inputs().at(0);
        auto idx         = ins->inputs().at(1);
        auto idx_type    = idx->get_shape().type();
        const auto& lens = idx->get_shape().lens();
        assert(not lens.empty());
        // Trailing index dims (all except first) — must match for concat on axis 0.  When
        // merging mixed lengths the indices are flattened, so trailing dims are ignored.
        // Keying on the data instruction itself restricts grouping to one table.
        std::vector<std::size_t> trailing;
        if(not merge_mixed_lengths)
            trailing.assign(lens.begin() + 1, lens.end());
        return std::make_tuple(data, idx_type, std::move(trailing));
    }

    std::vector<instruction_ref>
    fuse(module& m, const std::vector<instruction_ref>& gathers, instruction_ref insert_pt) const
    {
        assert(gathers.size() >= min_group_size());
        auto data = gathers.front()->inputs().at(0);
        assert(data->get_shape().lens().size() == 2);

        // Collect the per-gather indices (the table is shared, so no offset adjustment).
        std::vector<instruction_ref> idx_inputs(gathers.size());
        std::transform(gathers.begin(), gathers.end(), idx_inputs.begin(), [](auto g) {
            return g->inputs().at(1);
        });

        // Mixed index shapes cannot be concatenated on axis 0; flatten each instead.  Only
        // take the flattened path when the group is actually non-uniform, so uniform-shape
        // groups produce the same (cheaper, reshape-free) IR regardless of the flag.
        if(merge_mixed_lengths)
        {
            const auto& first_lens = gathers.front()->inputs().at(1)->get_shape().lens();
            bool uniform           = std::all_of(gathers.begin(), gathers.end(), [&](auto g) {
                return g->inputs().at(1)->get_shape().lens() == first_lens;
            });
            if(not uniform)
                return fuse_gathers_flattened(m, gathers, idx_inputs, data, insert_pt);
        }

        auto concat_idx =
            m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), idx_inputs);

        // Single batched gather over the shared table.
        auto batched_gather =
            m.insert_instruction(insert_pt, make_op("gather", {{"axis", 0}}), data, concat_idx);

        return slice_gather_rows(m, batched_gather, gathers, insert_pt);
    }
};

// ---------------------------------------------------------------------------
// Cross-embedding gather horizontal fusion
//
// Candidates: gather(axis=0) with 2D constant embedding table, static shapes,
//             non-scalar index
// Grouping:   by (embedding dimension, index type, index trailing dims)
// Fusion:     concatenate the *distinct* embedding tables (shared tables are kept
//             once), adjust indices with per-table offsets, single batched gather,
//             slice results back
// ---------------------------------------------------------------------------

struct gather_horizontal_fusion
{
    std::size_t min_group_size() const { return 4; }

    bool is_candidate(instruction_ref ins) const
    {
        if(ins->name() != "gather")
            return false;

        if(ins->get_operator().to_value()["axis"].to<int>() != 0)
            return false;

        auto data = ins->inputs().at(0);
        auto idx  = ins->inputs().at(1);

        // Embedding must be 2D: {num_rows, embedding_dim}
        if(data->get_shape().lens().size() != 2)
            return false;

        // Embedding must be constant (evaluable)
        if(not data->can_eval())
            return false;

        // Index must not be scalar
        if(idx->get_shape().scalar() or idx->get_shape().lens().empty())
            return false;

        return true;
    }

    auto group_key(instruction_ref ins) const
    {
        auto emb_dim     = ins->inputs().at(0)->get_shape().lens().back();
        auto idx         = ins->inputs().at(1);
        auto idx_type    = idx->get_shape().type();
        const auto& lens = idx->get_shape().lens();
        // Trailing index dims (all except first) — must match for concat on axis 0
        std::vector<std::size_t> trailing(lens.begin() + 1, lens.end());
        return std::make_tuple(emb_dim, idx_type, std::move(trailing));
    }

    std::vector<instruction_ref>
    fuse(module& m, const std::vector<instruction_ref>& gathers, instruction_ref insert_pt) const
    {
        auto idx_type = gathers.front()->inputs().at(1)->get_shape().type();

        // Deduplicate shared tables.  Several gathers in a group may read the *same*
        // table instruction (e.g. after same-table siblings are folded in).  Emitting
        // one table copy per gather would replicate that data, bloating both the
        // concatenated table literal and the batched gather.  Instead keep one copy per
        // distinct table (in first-appearance order) and record the row offset of each.
        std::vector<instruction_ref> unique_tables;
        std::unordered_map<instruction_ref, std::size_t> table_offset;
        std::size_t running_offset = 0;
        for(auto g : gathers)
        {
            auto data = g->inputs().at(0);
            if(table_offset.emplace(data, running_offset).second)
            {
                unique_tables.push_back(data);
                running_offset += data->get_shape().lens().front();
            }
        }

        // Concatenate the distinct tables (skip the concat when only one remains).
        auto concat_emb =
            unique_tables.size() == 1
                ? unique_tables.front()
                : m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), unique_tables);

        // Build adjusted indices (add each gather's table offset to shift into the
        // concatenated table).  Gathers whose table lands at offset 0 need no adjustment.
        std::vector<instruction_ref> adjusted_idx_inputs;
        adjusted_idx_inputs.reserve(gathers.size());
        std::transform(gathers.begin(),
                       gathers.end(),
                       std::back_inserter(adjusted_idx_inputs),
                       [&](auto g) -> instruction_ref {
                           auto idx    = g->inputs().at(1);
                           auto offset = table_offset.at(g->inputs().at(0));
                           if(offset == 0)
                               return idx;
                           auto offset_scalar = m.add_literal(literal{shape{idx_type}, {offset}});
                           auto offset_broadcast = m.insert_instruction(
                               insert_pt,
                               make_op("multibroadcast", {{"out_lens", idx->get_shape().lens()}}),
                               offset_scalar);
                           return m.insert_instruction(
                               insert_pt, make_op("add"), idx, offset_broadcast);
                       });

        // Concatenate adjusted indices
        auto concat_idx =
            m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), adjusted_idx_inputs);

        // Single batched gather
        auto batched_gather = m.insert_instruction(
            insert_pt, make_op("gather", {{"axis", 0}}), concat_emb, concat_idx);

        return slice_gather_rows(m, batched_gather, gathers, insert_pt);
    }
};

// ---------------------------------------------------------------------------
// Future: add more horizontal fusion finders here, e.g.
//
// struct pointwise_horizontal_fusion
// {
//     std::size_t min_group_size() const { return 2; }
//     bool is_candidate(instruction_ref ins) const { ... }
//     std::string group_key(instruction_ref ins) const { ... }
//     std::vector<instruction_ref>
//         fuse(module& m, const std::vector<instruction_ref>& ops,
//              instruction_ref insert_pt) const { ... }
// };
// ---------------------------------------------------------------------------

void fuse_horizontal::apply(module_pass_manager& mpm) const
{
    auto& m = mpm.get_module();

    // Run cross-embedding fusion first so a group of gathers sharing an embedding
    // dimension is bundled together even when it spans several tables (table dedup
    // keeps each distinct table once, so shared tables are not replicated).  Running
    // same-table fusion first would greedily collapse the same-table subset and strand
    // the remaining siblings below cross-table fusion's group-size threshold.  Same-table
    // fusion then mops up the smaller same-instance groups (size 2-3) that cross-table
    // fusion's larger threshold skips.  merge_mixed_lengths only affects same-table fusion:
    // it lets same-table gathers with differing index shapes merge (via a flattened index)
    // at no offset-add cost, without over-merging unrelated tables.
    fuse_horizontal_ops(
        m,
        gather_horizontal_fusion{},
        same_table_gather_horizontal_fusion{.merge_mixed_lengths = merge_mixed_lengths});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
