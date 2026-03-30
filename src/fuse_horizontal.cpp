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
#include <iostream>

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
    std::vector<instruction_ref> candidates;
    copy_if(iterator_for(m), std::back_inserter(candidates), [&](auto ins) {
        return finder.is_candidate(ins);
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

// Common predicate for gather fusion candidates:
// gather(axis=0) with 2D constant embedding table and non-scalar index.
static bool is_gather_fusion_candidate(instruction_ref ins)
{
    if(ins->name() != "gather")
        return false;

    if(ins->get_operator().to_value()["axis"].to<int>() != 0)
        return false;

    auto data = ins->inputs().at(0);
    auto idx  = ins->inputs().at(1);

    if(data->get_shape().lens().size() != 2)
        return false;

    if(not data->can_eval())
        return false;

    if(idx->get_shape().scalar() or idx->get_shape().lens().empty())
        return false;

    return true;
}

// ---------------------------------------------------------------------------
// Phase 1: Same-table gather deduplication
//
// When multiple gathers read from the identical constant embedding table,
// there is no need to concatenate tables or add index offsets.  We simply
// concat the index tensors, issue one gather on the original table, and
// slice the results back.
//
// This must run BEFORE cross-table fusion so that duplicate table references
// are collapsed first, keeping the subsequent cross-table combined table as
// small as possible.
// ---------------------------------------------------------------------------

struct gather_same_table_fusion
{
    std::size_t min_group_size() const { return 2; }

    bool is_candidate(instruction_ref ins) const { return is_gather_fusion_candidate(ins); }

    auto group_key(instruction_ref ins) const
    {
        auto data        = ins->inputs().at(0);
        auto idx         = ins->inputs().at(1);
        auto idx_type    = idx->get_shape().type();
        const auto& lens = idx->get_shape().lens();
        std::vector<std::size_t> trailing(lens.begin() + 1, lens.end());
        return std::make_tuple(data, idx_type, std::move(trailing));
    }

    std::vector<instruction_ref>
    fuse(module& m, const std::vector<instruction_ref>& gathers, instruction_ref insert_pt) const
    {
        auto data = gathers.front()->inputs().at(0);

        // Concat index tensors along axis 0 — no offsets needed
        std::vector<instruction_ref> idx_inputs;
        idx_inputs.reserve(gathers.size());
        for(const auto& g : gathers)
            idx_inputs.push_back(g->inputs().at(1));

        auto concat_idx =
            m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), idx_inputs);

        // Single gather on the original (unmodified) table
        auto batched_gather = m.insert_instruction(
            insert_pt, make_op("gather", {{"axis", 0}}), data, concat_idx);

        // Slice results back — one per original gather
        std::vector<instruction_ref> results;
        results.reserve(gathers.size());
        std::size_t offset = 0;
        for(const auto& g : gathers)
        {
            auto sz = g->inputs().at(1)->get_shape().lens().front();
            results.push_back(m.insert_instruction(
                insert_pt,
                make_op("slice",
                        {{"axes", std::vector<int64_t>{0}},
                         {"starts", std::vector<int64_t>{static_cast<int64_t>(offset)}},
                         {"ends", std::vector<int64_t>{static_cast<int64_t>(offset + sz)}}}),
                batched_gather));
            offset += sz;
        }

        return results;
    }
};

// ---------------------------------------------------------------------------
// Phase 2: Cross-embedding gather horizontal fusion
//
// After same-table dedup, the remaining gathers use distinct tables.
// We concatenate the different tables, adjust indices with per-table offsets,
// issue one gather on the combined table, and slice the results back.
//
// Candidates: gather(axis=0) with 2D constant embedding table, non-scalar index
// Grouping:   by (embedding dimension, index type, index trailing dims)
// ---------------------------------------------------------------------------

struct gather_horizontal_fusion
{
    std::size_t min_group_size() const { return 4; }

    bool is_candidate(instruction_ref ins) const { return is_gather_fusion_candidate(ins); }

    auto group_key(instruction_ref ins) const
    {
        auto emb_dim     = ins->inputs().at(0)->get_shape().lens().back();
        auto idx         = ins->inputs().at(1);
        auto idx_type    = idx->get_shape().type();
        const auto& lens = idx->get_shape().lens();
        // Include full index shape (first dim + trailing dims).  After
        // same-table dedup (phase 1) some indices have a larger first dim;
        // mixing them with first-dim-1 indices in the adjusted-index concat
        // triggers fused_concat GPU kernel failures (duplicate module refs
        // instantiated with incompatible tensor shapes).
        return std::make_tuple(emb_dim, idx_type, lens);
    }

    std::vector<instruction_ref>
    fuse(module& m, const std::vector<instruction_ref>& gathers, instruction_ref insert_pt) const
    {
        auto idx_type = gathers.front()->inputs().at(1)->get_shape().type();

        // Concatenate all embedding tables
        std::vector<instruction_ref> emb_inputs(gathers.size());
        std::transform(gathers.begin(), gathers.end(), emb_inputs.begin(), [](auto g) {
            return g->inputs().at(0);
        });
        auto concat_emb =
            m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), emb_inputs);

        // Compute cumulative embedding offsets using transform_partial_sum.
        // Inclusive partial sum gives end offsets; shift right and prepend 0
        // to get start (exclusive) offsets.
        std::vector<std::size_t> cum_sizes(gathers.size());
        transform_partial_sum(
            gathers.begin(), gathers.end(), cum_sizes.begin(), std::plus<>{}, [](auto g) {
                return g->inputs().at(0)->get_shape().lens().front();
            });

        // Exclusive offsets: [0, cum_sizes[0], cum_sizes[1], ...]
        std::vector<std::size_t> emb_offsets(gathers.size());
        emb_offsets[0] = 0;
        std::copy(cum_sizes.begin(), std::prev(cum_sizes.end()), emb_offsets.begin() + 1);

        // Build adjusted indices (add offset to shift into concatenated table)
        std::vector<instruction_ref> adjusted_idx_inputs;
        adjusted_idx_inputs.reserve(gathers.size());

        migraphx::for_each(
            gathers.begin(), gathers.end(), emb_offsets.begin(), [&](auto g, auto offset) {
                auto idx = g->inputs().at(1);
                if(offset == 0)
                {
                    adjusted_idx_inputs.push_back(idx);
                }
                else
                {
                    auto offset_scalar    = m.add_literal(literal{shape{idx_type}, {offset}});
                    auto offset_broadcast = m.insert_instruction(
                        insert_pt,
                        make_op("multibroadcast", {{"out_lens", idx->get_shape().lens()}}),
                        offset_scalar);
                    auto adjusted_idx =
                        m.insert_instruction(insert_pt, make_op("add"), idx, offset_broadcast);
                    adjusted_idx_inputs.push_back(adjusted_idx);
                }
            });

        // Concatenate adjusted indices
        auto concat_idx =
            m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), adjusted_idx_inputs);

        // Single batched gather
        auto batched_gather = m.insert_instruction(
            insert_pt, make_op("gather", {{"axis", 0}}), concat_emb, concat_idx);

        // Compute slice boundaries using partial_sum of index sizes
        std::vector<std::size_t> idx_sizes(gathers.size());
        std::transform(gathers.begin(), gathers.end(), idx_sizes.begin(), [](auto g) {
            return g->inputs().at(1)->get_shape().lens().front();
        });

        std::vector<std::size_t> slice_ends(gathers.size());
        std::partial_sum(idx_sizes.begin(), idx_sizes.end(), slice_ends.begin());

        // slice_starts = [0, slice_ends[0], slice_ends[1], ...]
        std::vector<std::size_t> slice_starts(gathers.size());
        slice_starts[0] = 0;
        std::copy(slice_ends.begin(), std::prev(slice_ends.end()), slice_starts.begin() + 1);

        // Slice results back — one per original gather
        std::vector<instruction_ref> results;
        results.reserve(gathers.size());

        migraphx::for_each(
            slice_starts.begin(),
            slice_starts.end(),
            slice_ends.begin(),
            [&](auto start, auto end) {
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
};

// ---------------------------------------------------------------------------
// SwiGLU expert head horizontal fusion
//
// Matches the pattern: x -> sigmoid(x) -> mul(x, sigmoid(x)) -> dot(weight) -> add(bias)
// where weight and bias are constants.
//
// Candidates: add instructions at the end of a SwiGLU-dot-add chain
// Grouping:   by (add output lens, output type, weight lens)
// Fusion:     stack all expert inputs along new batch axis 0, apply batched
//             sigmoid, mul, dot, add, then slice+squeeze results back
// ---------------------------------------------------------------------------

struct expert_head_horizontal_fusion
{
    struct pattern_info
    {
        instruction_ref add_ins;
        instruction_ref dot_ins;
        instruction_ref mul_ins;
        instruction_ref sig_ins;
        instruction_ref input_ins;
        instruction_ref weight_ins;
        instruction_ref bias_ins;
    };

    static bool try_match(instruction_ref add_ins, pattern_info& pm)
    {
        pm.add_ins = add_ins;

        if(add_ins->name() != "add" || add_ins->inputs().size() != 2)
            return false;
        if(add_ins->get_shape().dynamic())
            return false;

        instruction_ref dot_ins{};
        instruction_ref bias_ins{};
        for(const auto& in : add_ins->inputs())
        {
            if(in->name() == "dot")
                dot_ins = in;
            else
                bias_ins = in;
        }
        if(dot_ins == instruction_ref{} || bias_ins == instruction_ref{})
            return false;

        if(dot_ins->inputs().size() != 2)
            return false;
        if(not dot_ins->inputs().at(1)->can_eval())
            return false;
        if(not bias_ins->can_eval())
            return false;
        if(dot_ins->outputs().size() != 1)
            return false;

        auto a_ins = dot_ins->inputs().at(0);
        if(a_ins->name() != "mul" || a_ins->inputs().size() != 2)
            return false;
        if(a_ins->outputs().size() != 1)
            return false;

        instruction_ref sig_ins{};
        instruction_ref x_ins{};
        for(const auto& in : a_ins->inputs())
        {
            if(in->name() == "sigmoid")
                sig_ins = in;
            else
                x_ins = in;
        }
        if(sig_ins == instruction_ref{} || x_ins == instruction_ref{})
            return false;
        if(sig_ins->inputs().size() != 1 || sig_ins->inputs().at(0) != x_ins)
            return false;
        if(sig_ins->outputs().size() != 1)
            return false;

        pm.dot_ins    = dot_ins;
        pm.mul_ins    = a_ins;
        pm.sig_ins    = sig_ins;
        pm.input_ins  = x_ins;
        pm.weight_ins = dot_ins->inputs().at(1);
        pm.bias_ins   = bias_ins;
        return true;
    }

    std::size_t min_group_size() const { return 4; }

    bool is_candidate(instruction_ref ins) const
    {
        pattern_info pm;
        return try_match(ins, pm);
    }

    auto group_key(instruction_ref ins) const
    {
        pattern_info pm;
        try_match(ins, pm);
        auto out_lens = ins->get_shape().lens();
        auto out_type = ins->get_shape().type();
        auto w_lens   = pm.weight_ins->get_shape().lens();
        return std::make_tuple(out_lens, out_type, w_lens);
    }

    std::vector<instruction_ref>
    fuse(module& m,
         const std::vector<instruction_ref>& adds,
         instruction_ref insert_pt) const
    {
        auto n = adds.size();
        std::cerr << "[expert_horiz] FUSING group of " << n
                  << " expert heads, out=" << adds.front()->get_shape() << "\n";

        std::vector<instruction_ref> input_parts;
        std::vector<instruction_ref> weight_parts;
        std::vector<instruction_ref> bias_parts;
        input_parts.reserve(n);
        weight_parts.reserve(n);
        bias_parts.reserve(n);

        for(const auto& add_ins : adds)
        {
            pattern_info pm;
            try_match(add_ins, pm);

            input_parts.push_back(m.insert_instruction(
                insert_pt, make_op("unsqueeze", {{"axes", {0}}}), pm.input_ins));
            weight_parts.push_back(m.insert_instruction(
                insert_pt, make_op("unsqueeze", {{"axes", {0}}}), pm.weight_ins));
            bias_parts.push_back(m.insert_instruction(
                insert_pt, make_op("unsqueeze", {{"axes", {0}}}), pm.bias_ins));
        }

        auto stacked_x =
            m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), input_parts);
        auto stacked_w =
            m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), weight_parts);
        auto stacked_b =
            m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), bias_parts);

        auto batched_sig =
            m.insert_instruction(insert_pt, make_op("sigmoid"), stacked_x);
        auto batched_mul =
            m.insert_instruction(insert_pt, make_op("mul"), stacked_x, batched_sig);
        auto batched_dot =
            m.insert_instruction(insert_pt, make_op("dot"), batched_mul, stacked_w);
        auto batched_add =
            m.insert_instruction(insert_pt, make_op("add"), batched_dot, stacked_b);

        std::vector<instruction_ref> results;
        results.reserve(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            auto s = m.insert_instruction(
                insert_pt,
                make_op("slice",
                        {{"axes", std::vector<int64_t>{0}},
                         {"starts", std::vector<int64_t>{static_cast<int64_t>(i)}},
                         {"ends", std::vector<int64_t>{static_cast<int64_t>(i + 1)}}}),
                batched_add);
            results.push_back(
                m.insert_instruction(insert_pt, make_op("squeeze", {{"axes", {0}}}), s));
        }
        return results;
    }
};

// ---------------------------------------------------------------------------
// Dot horizontal fusion (guarded)
//
// Candidates: dot ops with a constant (evaluable) second input, static shapes,
//             and whose output does NOT feed into an add followed by activation
//             (those patterns are better handled by MLIR fusion)
// Grouping:   by (output lens, output type, weight lens)
// Fusion:     unsqueeze+concat inputs along new batch axis 0, single batched
//             dot, slice+squeeze results back per original op
// ---------------------------------------------------------------------------

struct dot_horizontal_fusion
{
    std::size_t min_group_size() const { return 2; }

    bool is_candidate(instruction_ref ins) const
    {
        if(ins->name() != "dot")
            return false;
        if(ins->get_shape().dynamic())
            return false;
        if(ins->inputs().size() != 2)
            return false;
        if(not ins->inputs().at(1)->can_eval())
            return false;

        // Skip dots that feed into add: MLIR fuses dot+add(+sigmoid+mul)
        // more efficiently than horizontal batching
        for(const auto& out : ins->outputs())
        {
            if(out->name() == "add")
                return false;
        }
        return true;
    }

    auto group_key(instruction_ref ins) const
    {
        auto out_lens = ins->get_shape().lens();
        auto out_type = ins->get_shape().type();
        auto b_lens   = ins->inputs().at(1)->get_shape().lens();
        return std::make_tuple(out_lens, out_type, b_lens);
    }

    std::vector<instruction_ref>
    fuse(module& m, const std::vector<instruction_ref>& dots, instruction_ref insert_pt) const
    {
        auto n = dots.size();

        std::vector<instruction_ref> a_parts;
        a_parts.reserve(n);
        std::vector<instruction_ref> b_parts;
        b_parts.reserve(n);

        for(const auto& d : dots)
        {
            a_parts.push_back(m.insert_instruction(
                insert_pt, make_op("unsqueeze", {{"axes", {0}}}), d->inputs().at(0)));
            b_parts.push_back(m.insert_instruction(
                insert_pt, make_op("unsqueeze", {{"axes", {0}}}), d->inputs().at(1)));
        }

        auto stacked_a =
            m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), a_parts);
        auto stacked_b =
            m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), b_parts);

        auto batched = m.insert_instruction(insert_pt, make_op("dot"), stacked_a, stacked_b);

        std::vector<instruction_ref> results;
        results.reserve(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            auto s = m.insert_instruction(
                insert_pt,
                make_op("slice",
                        {{"axes", std::vector<int64_t>{0}},
                         {"starts", std::vector<int64_t>{static_cast<int64_t>(i)}},
                         {"ends", std::vector<int64_t>{static_cast<int64_t>(i + 1)}}}),
                batched);
            results.push_back(
                m.insert_instruction(insert_pt, make_op("squeeze", {{"axes", {0}}}), s));
        }
        return results;
    }
};

void fuse_horizontal::apply(module_pass_manager& mpm) const
{
    auto& m = mpm.get_module();

    // Phase 1: collapse gathers that share the same constant table.
    // No table concatenation or index offset arithmetic — just concat indices,
    // one gather on the original table, slice results back.
    fuse_horizontal_ops(m, gather_same_table_fusion{});

    // Phase 2: fuse gathers across different tables that share the same
    // embedding dimension and index layout.  Runs on the reduced set of
    // gathers left after phase 1.
    fuse_horizontal_ops(m, gather_horizontal_fusion{});
    fuse_horizontal_ops(
        m,
        gather_horizontal_fusion{},
        expert_head_horizontal_fusion{},
        dot_horizontal_fusion{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
