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
#include <iterator>

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

// Fuse same-table gathers whose indices differ in shape (not just first dim): flatten each
// index to 1-D, concatenate, run one batched gather over the shared table, then slice each
// element range back out and reshape it to that gather's output shape (index dims +
// embedding dim).  Results are returned in the same order as indices.  No index offset
// adjustment is needed since the table is shared.
static std::vector<instruction_ref>
fuse_gathers_flattened(module& m,
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

    // Inclusive prefix sum of per-index element counts gives each slice's end offset; the
    // start is that end minus the index's own element count.
    std::vector<std::size_t> slice_ends(indices.size());
    transform_partial_sum(
        indices.begin(), indices.end(), slice_ends.begin(), std::plus<>{}, [](instruction_ref idx) {
            return idx->get_shape().elements();
        });

    const std::int64_t emb_dim = table->get_shape().lens().back();
    std::vector<instruction_ref> results(indices.size());
    std::transform(indices.begin(),
                   indices.end(),
                   slice_ends.begin(),
                   results.begin(),
                   [&](instruction_ref idx, std::size_t end) -> instruction_ref {
                       const auto& lens = idx->get_shape().lens();
                       int64_t start    = end - idx->get_shape().elements();
                       auto sliced      = m.insert_instruction(
                           insert_pt,
                           make_op("slice", {{"axes", {0}}, {"starts", {start}}, {"ends", {end}}}),
                           batched_gather);
                       // A 1-D index already yields {n, emb_dim}; only multi-dim indices need
                       // a reshape back to (index dims + embedding dim).
                       if(lens.size() == 1)
                           return sliced;
                       auto out_dims = lens;
                       out_dims.push_back(emb_dim);
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
// Grouping:   by (table instruction, index type) so gathers reading the *same* table
//             are merged regardless of index shape.
// Fusion:     when the group's indices are concat-compatible (same rank + trailing dims)
//             concatenate them and slice rows back; otherwise flatten each index to 1-D and
//             gather+reshape back.  No index offset adjustment is needed (shared table).
// ---------------------------------------------------------------------------

struct same_table_gather_horizontal_fusion
{
    // Minimum first index dim for the batched gather to be worthwhile.
    static constexpr std::size_t min_index_batch = 4;

    std::size_t min_group_size() const { return 2; }

    bool is_candidate(instruction_ref ins) const
    {
        if(ins->name() != "gather")
            return false;

        if(ins->get_operator().to_value()["axis"].to<int>() != 0)
            return false;

        // Skip dynamic shapes: this fusion relies on static `lens()` on inputs.
        const auto& inputs = ins->inputs();
        if(std::any_of(inputs.begin(), inputs.end(), [](const auto& inp) {
               return inp->get_shape().dynamic();
           }))
            return false;

        auto data = inputs.at(0);
        auto idx  = inputs.at(1);

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
        auto data     = ins->inputs().at(0);
        auto idx_type = ins->inputs().at(1)->get_shape().type();
        assert(not ins->inputs().at(1)->get_shape().lens().empty());
        // Key on the table instruction + index type only, so same-table gathers merge
        // regardless of index shape.  Differing shapes are reconciled in fuse() by flattening.
        return std::make_tuple(data, idx_type);
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

        // The indices can be concatenated on axis 0 only when they share rank and trailing
        // dims.  If they do, use that cheaper (reshape-free) path; otherwise flatten each.
        const auto& ref_lens   = idx_inputs.front()->get_shape().lens();
        bool concat_compatible = std::all_of(idx_inputs.begin(), idx_inputs.end(), [&](auto idx) {
            const auto& l = idx->get_shape().lens();
            return l.size() == ref_lens.size() and
                   std::equal(l.begin() + 1, l.end(), ref_lens.begin() + 1);
        });
        if(not concat_compatible)
            return fuse_gathers_flattened(m, idx_inputs, data, insert_pt);

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
// Fusion:     concatenate the *distinct* embedding tables (shared tables kept once),
//             adjust indices with per-table offsets, single batched gather, slice back
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

        // Skip dynamic shapes: this fusion relies on static `lens()` on inputs.
        const auto& inputs = ins->inputs();
        if(std::any_of(inputs.begin(), inputs.end(), [](const auto& inp) {
               return inp->get_shape().dynamic();
           }))
            return false;

        auto data = inputs.at(0);
        auto idx  = inputs.at(1);

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

        // A group can contain several gathers that read the *same* 2D data buffer with
        // different indices.  Concatenate each distinct buffer only once (in first-appearance
        // order) to avoid replicating data, and record the row offset where it lands in the
        // concatenated result.
        std::vector<instruction_ref> unique_tables;
        std::transform(gathers.begin(),
                       gathers.end(),
                       std::back_inserter(unique_tables),
                       [](auto g) { return g->inputs().at(0); });
        unique_tables.erase(distinct(unique_tables.begin(), unique_tables.end()),
                            unique_tables.end());

        std::vector<std::size_t> offsets(unique_tables.size(), 0);
        transform_partial_sum(unique_tables.begin(),
                              std::prev(unique_tables.end()),
                              std::next(offsets.begin()),
                              std::plus<>{},
                              [](auto t) { return t->get_shape().lens().front(); });

        std::unordered_map<instruction_ref, std::size_t> table_offset;
        std::transform(unique_tables.begin(),
                       unique_tables.end(),
                       offsets.begin(),
                       std::inserter(table_offset, table_offset.end()),
                       [](auto t, std::size_t off) { return std::make_pair(t, off); });

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
// Generic dot horizontal fusion
//
// Batches structurally-identical dot operations into a single batched GEMM by
// stacking activations and weights along a new leading dimension (axis 0).  The
// batched dot output is sliced and squeezed back into the individual results.
// ---------------------------------------------------------------------------

// A dot whose sole consumer is a pointwise op gets that op folded into its GEMM
// epilogue by fuse_mlir/fuse_ops (e.g. mlir_dot_add, mlir_dot_add_sigmoid_mul).
// Horizontally batching such a dot inserts a slice+squeeze between the batched
// dot and the pointwise, which is a fusion boundary, so the epilogue would fall
// out as a separate kernel.  Skip these to avoid regressing epilogue fusion.
static bool feeds_fusable_pointwise(instruction_ref ins)
{
    if(ins->outputs().size() != 1)
        return false;
    return ins->outputs().front()->get_operator().attributes().contains("pointwise");
}

struct dot_horizontal_fusion
{
    // Batching adds N unsqueeze + 1 concat + N slice + N squeeze of glue, so only
    // pay it for groups large enough to be worthwhile.
    std::size_t min_group_size() const { return 3; }

    bool is_candidate(instruction_ref ins) const
    {
        if(ins->name() != "dot")
            return false;
        if(ins->get_shape().dynamic())
            return false;
        if(ins->get_shape().ndim() < 2)
            return false;
        // Don't break an existing GEMM-epilogue fusion (see helper).
        if(feeds_fusable_pointwise(ins))
            return false;
        // Only fold when the weight is a compile-time constant so the batched
        // weight tensor can be materialized.
        return ins->inputs().at(1)->can_eval();
    }

    auto group_key(instruction_ref ins) const
    {
        return std::make_tuple(ins->inputs().at(0)->get_shape().lens(),
                               ins->inputs().at(1)->get_shape().lens(),
                               ins->get_shape().type());
    }

    std::vector<instruction_ref>
    fuse(module& m, const std::vector<instruction_ref>& dots, instruction_ref insert_pt) const
    {
        // Stack input `input_idx` of every dot along a new leading axis.
        auto stack = [&](std::size_t input_idx) {
            std::vector<instruction_ref> unsqueezed(dots.size());
            std::transform(dots.begin(), dots.end(), unsqueezed.begin(), [&](auto d) {
                return m.insert_instruction(
                    insert_pt, make_op("unsqueeze", {{"axes", {0}}}), d->inputs().at(input_idx));
            });
            return m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), unsqueezed);
        };

        auto batched_act = stack(0);
        auto batched_wt  = stack(1);
        auto batched_dot = m.insert_instruction(insert_pt, make_op("dot"), batched_act, batched_wt);

        // Slice each original result back out of the batched dot.
        std::vector<instruction_ref> results(dots.size());
        for(std::size_t i = 0; i < dots.size(); ++i)
        {
            auto sliced = m.insert_instruction(
                insert_pt,
                make_op("slice", {{"axes", {0}}, {"starts", {i}}, {"ends", {i + 1}}}),
                batched_dot);
            results[i] =
                m.insert_instruction(insert_pt, make_op("squeeze", {{"axes", {0}}}), sliced);
        }
        return results;
    }
};

void fuse_horizontal::apply(module_pass_manager& mpm) const
{
    auto& m = mpm.get_module();

    // Fuse across distinct tables first, then same-table groups.  Running same-table
    // fusion first can shrink cross-table groups below their size threshold and miss it.
    fuse_horizontal_ops(m, gather_horizontal_fusion{}, same_table_gather_horizontal_fusion{}, dot_horizontal_fusion{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
