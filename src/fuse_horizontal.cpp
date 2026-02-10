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

/// Move transitive outputs of `src` that are between `src` and `dst`
/// to just after `dst`, preserving their relative order.
/// This ensures consumers come after the fused instruction.
static void move_output_instructions_after(module& m, instruction_ref src, instruction_ref dst)
{
    auto d = std::distance(src, dst);
    std::vector<std::pair<std::size_t, instruction_ref>> instructions;
    fix([&](auto self, instruction_ref ins) {
        for(auto output : ins->outputs())
        {
            if(not m.has_instruction(output))
                continue;
            if(any_of(instructions, [&](const auto& p) { return p.second == output; }))
                continue;
            auto i = std::distance(src, output);
            if(i >= d)
                continue;
            instructions.emplace_back(i, output);
            self(output);
        }
    })(src);
    std::sort(instructions.begin(), instructions.end(), by(std::less<>{}, [](auto&& p) {
                  return p.first;
              }));
    auto loc = std::next(dst);
    for(auto& [i, ins] : instructions)
        m.move_instruction(ins, loc);
}

template <class Finder>
static void apply_horizontal_finder(module& m, const Finder& finder)
{
    // Collect all candidate instructions and build position map
    std::vector<instruction_ref> candidates;
    std::unordered_map<instruction_ref, std::size_t> pos;
    std::size_t p = 0;
    for(auto ins : iterator_for(m))
    {
        pos[ins] = p++;
        if(finder.is_candidate(ins))
            candidates.push_back(ins);
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
        if(n < static_cast<std::ptrdiff_t>(finder.min_group_size()))
            return;

        std::vector<instruction_ref> group(start, last);
        // Sort by position for consistent ordering
        std::sort(group.begin(), group.end(), [&](auto a, auto b) {
            return pos.at(a) < pos.at(b);
        });

        // Check that all outputs are in this module
        for(auto g : group)
        {
            if(not std::all_of(g->outputs().begin(), g->outputs().end(), [&](auto out) {
                   return m.has_instruction(out);
               }))
                return;
        }

        auto insert_pt    = std::next(group.back());
        auto replacements = finder.fuse(m, group, insert_pt);
        if(replacements.empty())
            return;

        assert(replacements.size() == group.size());

        // Move outputs of the original instructions to after the new instructions
        // so that replace_instruction's validity assertions hold.
        for(auto g : group)
        {
            move_output_instructions_after(m, g, replacements.back());
        }

        migraphx::for_each(group.begin(), group.end(), replacements.begin(), [&](auto g, auto r) {
            m.replace_instruction(g, r);
        });
    };

    group_by(candidates.begin(), candidates.end(), each, pred);
}

template <class... Finders>
void fuse_horizontal_ops(module& m, Finders&&... finders)
{
    (apply_horizontal_finder(m, finders), ...);
}

// ---------------------------------------------------------------------------
// Cross-embedding gather horizontal fusion
//
// Candidates: gather(axis=0) with 2D constant embedding table, static shapes,
//             non-scalar index
// Grouping:   by (embedding dimension, index type, index trailing dims)
// Fusion:     concatenate embedding tables, adjust indices with offsets,
//             single batched gather, slice results back
// ---------------------------------------------------------------------------

struct gather_horizontal_fusion
{
    // Key type for grouping: (emb_dim, idx_type, trailing_idx_dims)
    using key_type =
        std::tuple<std::size_t, shape::type_t, std::vector<std::size_t>>;

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
        if(idx->get_shape().lens().empty())
            return false;

        return true;
    }

    key_type group_key(instruction_ref ins) const
    {
        auto emb_dim  = ins->inputs().at(0)->get_shape().lens().back();
        auto idx      = ins->inputs().at(1);
        auto idx_type = idx->get_shape().type();
        auto& lens    = idx->get_shape().lens();
        // Trailing index dims (all except first) — must match for concat on axis 0
        std::vector<std::size_t> trailing(lens.begin() + 1, lens.end());
        return std::make_tuple(emb_dim, idx_type, std::move(trailing));
    }

    std::vector<instruction_ref> fuse(module& m,
                                      const std::vector<instruction_ref>& gathers,
                                      instruction_ref insert_pt) const
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
            gathers.begin(),
            gathers.end(),
            cum_sizes.begin(),
            std::plus<>{},
            [](auto g) { return g->inputs().at(0)->get_shape().lens().front(); });

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
                    auto offset_scalar =
                        m.add_literal(literal{shape{idx_type}, {offset}});
                    auto offset_broadcast = m.insert_instruction(
                        insert_pt,
                        make_op("multibroadcast", {{"out_lens", idx->get_shape().lens()}}),
                        offset_scalar);
                    auto adjusted_idx = m.insert_instruction(
                        insert_pt, make_op("add"), idx, offset_broadcast);
                    adjusted_idx_inputs.push_back(adjusted_idx);
                }
            });

        // Concatenate adjusted indices
        auto concat_idx = m.insert_instruction(
            insert_pt, make_op("concat", {{"axis", 0}}), adjusted_idx_inputs);

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

    fuse_horizontal_ops(m, gather_horizontal_fusion{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
