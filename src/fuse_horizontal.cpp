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
    // Collect all candidate instructions and build position map
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
// Generic dot horizontal fusion
//
// Batches structurally-identical dot operations into a single batched GEMM
// by stacking activations and weights along a new leading dimension (axis 0).
// The batched dot output is sliced and squeezed back to individual results.
//
// Two downstream patterns are absorbed directly:
//   1. Bias:  dot → add(dot, broadcast(const))  →  batched_dot + batched_add
//   2. SiLU:  (frontier) → sigmoid → mul(frontier, sigmoid)
//             →  batched sigmoid + batched mul on the full result
//
// These are handled eagerly because they are common in MLP prediction towers
// and absorbing them avoids relying on multiple simplify_reshapes iterations
// plus CSE to reconstruct the batched computation.
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
        if(ins->get_shape().ndim() < 2)
            return false;
        auto weight = ins->inputs().at(1);
        return weight->can_eval();
    }

    // Check whether a single dot's output (possibly through a bias add)
    // feeds into a SiLU pattern: target → sigmoid → mul(target, sigmoid).
    static bool has_downstream_silu(instruction_ref dot)
    {
        auto target = dot;
        auto outputs = target->outputs();
        if(outputs.size() == 1 and outputs.front()->name() == "add")
        {
            auto add_ins       = outputs.front();
            const auto& inputs = add_ins->inputs();
            auto other         = (inputs[0] == target) ? inputs[1] : inputs[0];
            if(other->name() == "broadcast" or other->name() == "multibroadcast")
                target = add_ins;
        }

        outputs = target->outputs();
        if(outputs.size() != 2)
            return false;
        instruction_ref sig{};
        instruction_ref mul{};
        for(auto out : outputs)
        {
            if(out->name() == "sigmoid" and out->inputs().size() == 1 and
               out->inputs().front() == target)
                sig = out;
            else if(out->name() == "mul")
                mul = out;
        }
        if(sig == instruction_ref{} or mul == instruction_ref{})
            return false;
        if(sig->outputs().size() != 1 or sig->outputs().front() != mul)
            return false;
        const auto& mul_inputs = mul->inputs();
        return (mul_inputs[0] == target and mul_inputs[1] == sig) or
               (mul_inputs[0] == sig and mul_inputs[1] == target);
    }

    auto group_key(instruction_ref ins) const
    {
        return std::make_tuple(ins->inputs().at(0)->get_shape().lens(),
                               ins->inputs().at(1)->get_shape().lens(),
                               ins->get_shape().type(),
                               has_downstream_silu(ins));
    }

    struct bias_detect_result
    {
        std::vector<instruction_ref> add_insts;
        std::vector<instruction_ref> bias_bcasts;
    };

    struct silu_detect_result
    {
        std::vector<instruction_ref> sig_insts;
        std::vector<instruction_ref> mul_insts;
    };

    // Check if every dot feeds into add(dot, broadcast(...)). If so, return
    // the add instructions and bias broadcast instructions.
    static bool detect_downstream_biases(const std::vector<instruction_ref>& dots,
                                         bias_detect_result& result)
    {
        for(const auto& d : dots)
        {
            auto outputs = d->outputs();
            if(outputs.size() != 1)
                return false;
            auto add_ins = outputs.front();
            if(add_ins->name() != "add")
                return false;
            const auto& add_inputs = add_ins->inputs();
            auto other             = (add_inputs[0] == d) ? add_inputs[1] : add_inputs[0];
            if(other->name() != "broadcast" and other->name() != "multibroadcast")
                return false;
            result.add_insts.push_back(add_ins);
            result.bias_bcasts.push_back(other);
        }
        return true;
    }

    // Check if every source instruction feeds into a SiLU pattern:
    //   source → sigmoid(source)
    //          → mul(source, sigmoid(source))
    static bool detect_downstream_silu(const std::vector<instruction_ref>& sources,
                                       silu_detect_result& result)
    {
        for(const auto& src : sources)
        {
            auto outputs = src->outputs();
            if(outputs.size() != 2)
                return false;

            instruction_ref sig_ins{};
            instruction_ref mul_ins{};
            bool found_sig = false;
            bool found_mul = false;

            for(auto out : outputs)
            {
                if(out->name() == "sigmoid" and out->inputs().size() == 1 and
                   out->inputs().front() == src)
                {
                    sig_ins   = out;
                    found_sig = true;
                }
                else if(out->name() == "mul")
                {
                    mul_ins   = out;
                    found_mul = true;
                }
            }

            if(not found_sig or not found_mul)
                return false;

            if(sig_ins->outputs().size() != 1 or sig_ins->outputs().front() != mul_ins)
                return false;

            const auto& mul_inputs = mul_ins->inputs();
            if(mul_inputs.size() != 2)
                return false;
            bool valid = (mul_inputs[0] == src and mul_inputs[1] == sig_ins) or
                         (mul_inputs[0] == sig_ins and mul_inputs[1] == src);
            if(not valid)
                return false;

            result.sig_insts.push_back(sig_ins);
            result.mul_insts.push_back(mul_ins);
        }
        return true;
    }

    std::vector<instruction_ref>
    fuse(module& m, const std::vector<instruction_ref>& dots, instruction_ref insert_pt) const
    {
        auto num = dots.size();

        // Detect bias absorption before inserting anything — we may need to
        // advance insert_pt past the add (and broadcast) instructions that
        // sit between the dots and the natural insertion point.
        auto advance_past = [&](instruction_ref target) {
            if(std::any_of(insert_pt, m.end(), [&](auto& i) { return &i == &*target; }))
                insert_pt = std::next(target);
        };

        bias_detect_result bias_info;
        bool absorb = detect_downstream_biases(dots, bias_info);
        if(absorb)
        {
            for(auto add : bias_info.add_insts)
                advance_past(add);
        }

        // The "frontier" is the furthest-downstream set of original instructions
        // that we will absorb.  Bias adds override the dots as frontier.
        const auto& frontier = absorb ? bias_info.add_insts : dots;

        // Detect SiLU: frontier → sigmoid → mul(frontier, sigmoid)
        silu_detect_result silu_info;
        bool has_silu = detect_downstream_silu(frontier, silu_info);
        if(has_silu)
        {
            for(std::size_t i = 0; i < num; ++i)
            {
                advance_past(silu_info.sig_insts[i]);
                advance_past(silu_info.mul_insts[i]);
            }
        }

        std::vector<instruction_ref> acts(num);
        std::transform(dots.begin(), dots.end(), acts.begin(), [&](auto d) {
            return m.insert_instruction(
                insert_pt, make_op("unsqueeze", {{"axes", {0}}}), d->inputs().at(0));
        });
        auto batched_act = m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), acts);

        std::vector<instruction_ref> wts(num);
        std::transform(dots.begin(), dots.end(), wts.begin(), [&](auto d) {
            return m.insert_instruction(
                insert_pt, make_op("unsqueeze", {{"axes", {0}}}), d->inputs().at(1));
        });
        auto batched_wt = m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), wts);

        auto bd = m.insert_instruction(insert_pt, make_op("dot"), batched_act, batched_wt);

        if(absorb)
        {
            std::vector<instruction_ref> unsqueezed_biases(num);
            for(std::size_t i = 0; i < num; ++i)
            {
                unsqueezed_biases[i] = m.insert_instruction(
                    insert_pt, make_op("unsqueeze", {{"axes", {0}}}), bias_info.bias_bcasts[i]);
            }
            auto stacked_bias = m.insert_instruction(
                insert_pt, make_op("concat", {{"axis", 0}}), unsqueezed_biases);
            bd = m.insert_instruction(insert_pt, make_op("add"), bd, stacked_bias);
        }

        if(has_silu)
        {
            auto sig_full = m.insert_instruction(insert_pt, make_op("sigmoid"), bd);
            bd            = m.insert_instruction(insert_pt, make_op("mul"), bd, sig_full);
        }

        std::vector<instruction_ref> results;
        results.reserve(num);
        for(int64_t i = 0; i < num; ++i)
        {
            auto sliced = m.insert_instruction(
                insert_pt,
                make_op("slice",
                        {{"axes", std::vector<int64_t>{0}},
                         {"starts", std::vector<int64_t>{i}},
                         {"ends", std::vector<int64_t>{i + 1}}}),
                bd);
            results.push_back(
                m.insert_instruction(insert_pt, make_op("squeeze", {{"axes", {0}}}), sliced));
        }

        // Replace the furthest-downstream absorbed instructions.  Anything
        // between the dots and the replaced instruction becomes dead and is
        // cleaned up by DCE.
        if(has_silu)
        {
            for(std::size_t i = 0; i < silu_info.mul_insts.size(); ++i)
                m.replace_instruction(silu_info.mul_insts[i], results[i]);
        }
        else if(absorb)
        {
            for(std::size_t i = 0; i < bias_info.add_insts.size(); ++i)
                m.replace_instruction(bias_info.add_insts[i], results[i]);
        }

        return results;
    }
};

void fuse_horizontal::apply(module_pass_manager& mpm) const
{
    auto& m = mpm.get_module();

    fuse_horizontal_ops(m, gather_horizontal_fusion{}, dot_horizontal_fusion{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
