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
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <vector>
#include <map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// ---------------------------------------------------------------------------
// Horizontal fusion framework
//
// To add a new horizontal fusion, define a plain struct that implements:
//
//   std::size_t min_group_size() const          — minimum group size for fusion
//   bool is_candidate(instruction_ref) const    — does this instruction qualify?
//   auto group_key(instruction_ref) const       — grouping key (comparable/orderable)
//   std::vector<instruction_ref>
//       fuse(module&, const std::vector<instruction_ref>&) const
//       — fuse a group, return one replacement instruction per original op
//
// Then pass an instance to fuse_horizontal_ops().
// The framework handles scanning, grouping by key, filtering inter-dependent
// instructions, dispatching to fuse(), and replacing originals with results.
// Groups are collected in module order (iterator_for traverses topologically).
// ---------------------------------------------------------------------------

/// Remove instructions that transitively depend on an earlier instruction
/// in the group, since horizontal fusion requires truly independent operations.
static void filter_dependent(std::vector<instruction_ref>& group)
{
    for(auto it = std::next(group.begin()); it != group.end();)
    {
        bool dependent =
            std::any_of(group.begin(), it, [&](auto j) { return reaches(j, *it); });
        it = dependent ? group.erase(it) : std::next(it);
    }
}

template <class Finder>
static void apply_horizontal_finder(module& m, const Finder& finder)
{
    using key_type =
        std::decay_t<decltype(finder.group_key(std::declval<instruction_ref>()))>;

    // Collect candidates and group by key (already in module order).
    std::map<key_type, std::vector<instruction_ref>> groups;
    for(auto ins : iterator_for(m))
    {
        if(finder.is_candidate(ins))
            groups[finder.group_key(ins)].push_back(ins);
    }

    for(auto& [key, group] : groups)
    {
        if(group.size() < finder.min_group_size())
            continue;
        filter_dependent(group);
        if(group.size() < finder.min_group_size())
            continue;

        auto replacements = finder.fuse(m, group);
        if(replacements.empty())
            continue;

        assert(replacements.size() == group.size());
        for(std::size_t i = 0; i < group.size(); i++)
        {
            m.replace_instruction(group[i], replacements[i]);
        }
    }
}

template <class... Finders>
void fuse_horizontal_ops(module& m, Finders&&... finders)
{
    (apply_horizontal_finder(m, finders), ...);
}

// ---------------------------------------------------------------------------
// Cross-embedding gather horizontal fusion
//
// Candidates: gather(axis=0) with 2D constant embedding table, static shapes
// Grouping:   by embedding dimension (last dim of the embedding table)
// Fusion:     concatenate embedding tables, adjust indices with offsets,
//             single batched gather, slice results back
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

        return true;
    }

    std::size_t group_key(instruction_ref ins) const
    {
        // Group by embedding dimension (last dim of embedding table)
        return ins->inputs().at(0)->get_shape().lens().back();
    }

    std::vector<instruction_ref> fuse(module& m,
                                      const std::vector<instruction_ref>& gathers) const
    {
        // Validate compatible index shapes for concatenation along axis 0
        auto first_idx       = gathers.front()->inputs().at(1);
        const auto& idx_lens = first_idx->get_shape().lens();
        auto idx_type        = first_idx->get_shape().type();

        // Skip scalar indices (empty lens)
        if(idx_lens.empty())
            return {};

        bool compatible = std::all_of(gathers.begin(), gathers.end(), [&](auto g) {
            auto idx        = g->inputs().at(1);
            const auto& ish = idx->get_shape();
            if(ish.type() != idx_type)
                return false;
            if(ish.lens().size() != idx_lens.size())
                return false;
            if(ish.lens().empty())
                return false;
            // All dimensions except the first must match (we concatenate along axis 0)
            for(std::size_t i = 1; i < ish.lens().size(); i++)
            {
                if(ish.lens()[i] != idx_lens[i])
                    return false;
            }
            return true;
        });

        if(not compatible)
            return {};

        // Insert after the last gather in the group (already sorted by position)
        auto insert_pt = std::next(gathers.back());

        // Collect embedding inputs and compute cumulative offsets
        std::vector<instruction_ref> emb_inputs;
        std::vector<std::size_t> emb_offsets;
        std::size_t cumulative_offset = 0;

        for(auto g : gathers)
        {
            auto emb = g->inputs().at(0);
            emb_inputs.push_back(emb);
            emb_offsets.push_back(cumulative_offset);
            cumulative_offset += emb->get_shape().lens().front();
        }

        // Concatenate all embedding tables
        auto concat_emb =
            m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), emb_inputs);

        // Build adjusted indices (add offset to shift into concatenated table)
        std::vector<instruction_ref> adjusted_idx_inputs;
        std::vector<std::size_t> idx_sizes;

        for(std::size_t i = 0; i < gathers.size(); i++)
        {
            auto g      = gathers[i];
            auto idx    = g->inputs().at(1);
            auto offset = emb_offsets[i];

            instruction_ref adjusted_idx;
            if(offset == 0)
            {
                adjusted_idx = idx;
            }
            else
            {
                auto offset_scalar = m.add_literal(literal{shape{idx_type}, {offset}});

                auto offset_broadcast = m.insert_instruction(
                    insert_pt,
                    make_op("multibroadcast", {{"out_lens", idx->get_shape().lens()}}),
                    offset_scalar);

                adjusted_idx = m.insert_instruction(
                    insert_pt, make_op("add"), idx, offset_broadcast);
            }

            adjusted_idx_inputs.push_back(adjusted_idx);
            idx_sizes.push_back(idx->get_shape().lens().front());
        }

        // Concatenate adjusted indices
        auto concat_idx = m.insert_instruction(
            insert_pt, make_op("concat", {{"axis", 0}}), adjusted_idx_inputs);

        // Single batched gather
        auto batched_gather = m.insert_instruction(
            insert_pt, make_op("gather", {{"axis", 0}}), concat_emb, concat_idx);

        // Slice results back — one per original gather
        std::vector<instruction_ref> results;
        std::size_t slice_offset = 0;
        for(std::size_t i = 0; i < gathers.size(); i++)
        {
            auto idx_size = idx_sizes[i];
            auto start    = static_cast<int64_t>(slice_offset);
            auto end      = static_cast<int64_t>(slice_offset + idx_size);

            results.push_back(m.insert_instruction(
                insert_pt,
                make_op("slice",
                        {{"axes", std::vector<int64_t>{0}},
                         {"starts", std::vector<int64_t>{start}},
                         {"ends", std::vector<int64_t>{end}}}),
                batched_gather));

            slice_offset += idx_size;
        }
        return results;
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
//         fuse(module& m, const std::vector<instruction_ref>& ops) const { ... }
// };
// ---------------------------------------------------------------------------

void fuse_horizontal::apply(module_pass_manager& mpm) const
{
    auto& m = mpm.get_module();

    fuse_horizontal_ops(m, gather_horizontal_fusion{});

    dead_code_elimination{}.apply(m);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
