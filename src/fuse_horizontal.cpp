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
// Chain-aware MLP tower horizontal fusion
//
// Batches structurally-identical MLP chains end-to-end, keeping the batch
// dimension through all layers including pointwise activations.
//
// Chain pattern:  dot → add(bias) → sigmoid → mul (SiLU)  → dot → ... → dot
// Each layer is:  dot(x, W) then optionally add(_, bias) → sigmoid → mul
// The chain ends at the last dot (or last SiLU if present on the final layer).
//
// The batch dimension is introduced once at the chain start (unsqueeze+concat
// activations and weights), carried through every layer (stacking biases and
// applying pointwise ops on the batched tensor), and split only at the chain
// end (slice+squeeze).  This avoids breaking MLIR vertical fusion.
// ---------------------------------------------------------------------------

struct mlp_layer
{
    instruction_ref dot;
    instruction_ref weight;
    bool silu_present = false;
    instruction_ref bias;
    instruction_ref add;
    instruction_ref sigmoid;
    instruction_ref mul;

    bool has_silu() const { return silu_present; }
    instruction_ref output() const { return silu_present ? mul : dot; }
};

struct mlp_chain
{
    instruction_ref root_input;
    std::vector<mlp_layer> layers;
    instruction_ref terminal() const { return layers.back().output(); }
};

static bool is_dot_with_constant_weight(instruction_ref ins)
{
    if(ins->name() != "dot")
        return false;
    if(ins->get_shape().dynamic())
        return false;
    if(ins->get_shape().lens().size() < 2)
        return false;
    auto weight = ins->inputs().at(1);
    return weight->can_eval() and weight->get_shape().lens().size() >= 2;
}

// Skip through multibroadcast/broadcast to find the source
static instruction_ref skip_broadcasts(instruction_ref ins)
{
    while(ins->name() == "multibroadcast" or ins->name() == "broadcast")
        ins = ins->inputs().at(0);
    return ins;
}

// Try to detect SiLU activation pattern after a dot:
//   add(dot_out, broadcast(bias)) → sigmoid → mul(sigmoid, add)
static bool detect_silu_after(instruction_ref dot_ins, mlp_layer& layer)
{
    const auto& dot_outs = dot_ins->outputs();
    if(dot_outs.size() != 1)
        return false;

    auto maybe_add = dot_outs.front();
    if(maybe_add->name() != "add")
        return false;

    // Identify bias vs dot input in the add
    auto add_in = maybe_add->inputs();
    instruction_ref bias_input;
    if(add_in.at(0) == dot_ins)
        bias_input = add_in.at(1);
    else if(add_in.at(1) == dot_ins)
        bias_input = add_in.at(0);
    else
        return false;

    auto bias_source = skip_broadcasts(bias_input);
    if(not bias_source->can_eval())
        return false;

    // add must have exactly 2 users: sigmoid and mul
    const auto& add_outs = maybe_add->outputs();
    if(add_outs.size() != 2)
        return false;

    instruction_ref sig_ins = add_outs.front();
    instruction_ref mul_ins = add_outs.back();
    bool found_sig          = false;
    bool found_mul          = false;
    for(auto out : add_outs)
    {
        if(out->name() == "sigmoid")
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

    // sigmoid's sole user must be the mul
    if(sig_ins->outputs().size() != 1 or sig_ins->outputs().front() != mul_ins)
        return false;

    // mul must take sigmoid and add as inputs (either order)
    const auto& mul_in = mul_ins->inputs();
    bool has_sig = (mul_in.at(0) == sig_ins or mul_in.at(1) == sig_ins);
    bool has_add = (mul_in.at(0) == maybe_add or mul_in.at(1) == maybe_add);
    if(not has_sig or not has_add)
        return false;

    layer.silu_present = true;
    layer.bias         = bias_source;
    layer.add          = maybe_add;
    layer.sigmoid      = sig_ins;
    layer.mul          = mul_ins;
    return true;
}

// Trace an MLP chain forward from a root dot
static mlp_chain trace_chain(instruction_ref first_dot)
{
    mlp_chain chain;
    chain.root_input = first_dot->inputs().at(0);

    auto current_dot = first_dot;
    while(true)
    {
        mlp_layer layer{};
        layer.dot    = current_dot;
        layer.weight = current_dot->inputs().at(1);

        detect_silu_after(current_dot, layer);
        chain.layers.push_back(layer);

        if(not layer.has_silu())
            break;

        // Check if the SiLU output feeds into exactly one next dot
        const auto& silu_outs = layer.mul->outputs();
        if(silu_outs.size() != 1)
            break;

        const auto& next = silu_outs.front();
        if(not is_dot_with_constant_weight(next))
            break;

        // Verify next dot takes the SiLU output as its activation (input 0)
        if(next->inputs().at(0) != layer.mul)
            break;

        current_dot = next;
    }

    return chain;
}

// Check whether an instruction is an interior dot of some chain (not a root)
static bool is_chain_interior_dot(instruction_ref ins)
{
    if(not is_dot_with_constant_weight(ins))
        return false;
    const auto& act = ins->inputs().at(0);
    // If the activation comes from a mul whose name is "mul" and that mul
    // is part of a SiLU from a preceding dot, this is an interior dot.
    if(act->name() != "mul")
        return false;
    const auto& mul_in = act->inputs();
    return std::any_of(mul_in.begin(), mul_in.end(), [](auto inp) {
        return inp->name() == "sigmoid" and inp->inputs().at(0)->name() == "add" and
               is_dot_with_constant_weight(inp->inputs().at(0)->inputs().at(0));
    });
}

static instruction_ref stack_along_dim0(module& m,
                                        instruction_ref insert_pt,
                                        const std::vector<instruction_ref>& items)
{
    std::vector<instruction_ref> unsqueezed(items.size());
    std::transform(items.begin(), items.end(), unsqueezed.begin(), [&](auto item) {
        return m.insert_instruction(insert_pt, make_op("unsqueeze", {{"axes", {0}}}), item);
    });
    return m.insert_instruction(insert_pt, make_op("concat", {{"axis", 0}}), std::move(unsqueezed));
}

// Build unsqueeze axes to pad a bias to the same rank as the batched dot output.
// The leading axes {0, 1, ..., n-1} are inserted so that the bias's original
// trailing dims stay right-aligned and the stacking dim ends up at axis 0.
//   e.g. bias {K} + target_ndim 4 → axes {0,1,2} → {1,1,1,K}
//        bias {1,1,K} + target_ndim 4 → axes {0} → {1,1,1,K}
static std::vector<int64_t> bias_unsqueeze_axes(std::size_t target_ndim,
                                                std::size_t bias_ndim)
{
    auto n = target_ndim - bias_ndim;
    std::vector<int64_t> axes(n);
    std::iota(axes.begin(), axes.end(), 0);
    return axes;
}

// 1. Find all chain roots (first dot not preceded by SiLU from another dot)
static std::vector<instruction_ref> find_chain_roots(module& m)
{
    std::vector<instruction_ref> roots;
    auto it = iterator_for(m);
    std::copy_if(it.begin(), it.end(), std::back_inserter(roots), [&](auto ins) {
        return is_dot_with_constant_weight(ins) and not is_chain_interior_dot(ins);
    });
    return roots;
}

// 2. Trace chains from each root, keeping only multi-layer chains.
//    Single-layer chains (standalone dots) are left alone so that MLIR
//    can still vertically fuse them with their downstream pointwise ops.
static std::vector<mlp_chain> trace_chains(const std::vector<instruction_ref>& roots)
{
    std::vector<mlp_chain> chains;
    for(const auto& root : roots)
    {
        auto chain = trace_chain(root);
        if(chain.layers.size() >= 2)
            chains.push_back(std::move(chain));
    }
    return chains;
}


// 3. Build position map for ordering and independence checks
static std::unordered_map<instruction_ref, std::size_t> build_position_map(module& m)
{
    std::unordered_map<instruction_ref, std::size_t> pos;
    std::size_t p = 0;
    for(auto ins : iterator_for(m))
        pos[ins] = p++;
    return pos;
}


// 4. Build a chain signature for grouping
static std::pair<std::vector<std::vector<std::size_t>>, int> chain_signature(const mlp_chain& c)
{
    std::vector<std::vector<std::size_t>> layer_shapes;
    for(const auto& l : c.layers)
    {
        layer_shapes.push_back(l.dot->get_shape().lens());
        auto k = l.dot->inputs().at(0)->get_shape().lens().back();
        layer_shapes.push_back({k, l.has_silu() ? std::size_t(1) : std::size_t(0)});
    }
    auto dtype = c.layers.front().dot->get_shape().type();
    return std::make_pair(layer_shapes, dtype);
}

// 5. Group chains by signature
static std::unordered_map<std::size_t, std::vector<std::size_t>> group_chains(const std::vector<mlp_chain>& chains)
{
    std::unordered_map<std::size_t, std::vector<std::size_t>> groups;
    for(std::size_t i = 0; i < chains.size(); ++i)
    {
        const auto& sig = chain_signature(chains[i]);
        std::size_t hash = 0;
        for(const auto& v : sig.first)
            for(const auto& d : v)
                hash ^= std::hash<std::size_t>{}(d) + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
        hash ^= std::hash<int>{}(static_cast<int>(sig.second));

        groups[hash].push_back(i);
    }
    return groups;
}

static void fuse_mlp_chains(module& m)
{
    auto roots = find_chain_roots(m);
    if(roots.empty())
        return;

    auto chains = trace_chains(roots);

    auto pos = build_position_map(m);

    auto groups = group_chains(chains);

    // 6. For each group with >= 2 chains, verify matching and fuse
    for(const auto& [hash, indices] : groups)
    {
        // Verify all chains in the bucket actually have the same signature
        const auto& ref_sig = chain_signature(chains[indices[0]]);
        std::vector<std::size_t> matching;

        for(const auto& idx : indices)
        {
            if(chain_signature(chains[idx]) == ref_sig)
                matching.push_back(idx);
        }

        if(matching.size() < 2)
            continue;

        // Verify chains are independent (no data dependency between them)
        std::vector<std::size_t> independent;
        for(const auto& idx : matching)
        {
            bool depends = false;
            for(const auto& prev : independent)
            {
                if(reaches(chains[prev].terminal(), chains[idx].layers.front().dot) or
                   reaches(chains[idx].terminal(), chains[prev].layers.front().dot))
                {
                    depends = true;
                    break;
                }
            }
            if(not depends)
                independent.push_back(idx);
        }

        if(independent.size() < 2)
            continue;

        // Sort by position for consistent ordering
        std::sort(independent.begin(), independent.end(), [&](auto a, auto b) {
            return pos.at(chains[a].layers.front().dot) < pos.at(chains[b].layers.front().dot);
        });

        auto num        = independent.size();
        auto& ref_chain = chains[independent[0]];
        auto num_layers = ref_chain.layers.size();
        auto insert_pt  = std::next(chains[independent.back()].terminal());

        // Stack root activations: [N, ...]
        std::vector<instruction_ref> root_acts;
        root_acts.reserve(num);
        for(const auto& idx : independent)
            root_acts.push_back(chains[idx].root_input);
        auto batched_act = stack_along_dim0(m, insert_pt, root_acts);

        auto current = batched_act;

        for(std::size_t l = 0; l < num_layers; ++l)
        {
            // Stack weights for this layer
            std::vector<instruction_ref> weights;
            weights.reserve(num);
            for(const auto& idx : independent)
                weights.push_back(chains[idx].layers[l].weight);
            auto batched_wt = stack_along_dim0(m, insert_pt, weights);

            // Batched dot
            current = m.insert_instruction(insert_pt, make_op("dot"), current, batched_wt);

            // Apply SiLU activation on the batched tensor (if this layer has it)
            if(ref_chain.layers[l].has_silu())
            {
                // Stack biases and reshape for broadcasting
                std::vector<instruction_ref> biases;
                biases.reserve(num);
                for(const auto& idx : independent)
                    biases.push_back(chains[idx].layers[l].bias);

                auto batched_dot_ndim = current->get_shape().lens().size();
                std::vector<instruction_ref> bias_expanded;
                bias_expanded.reserve(num);
                for(const auto& b : biases)
                {
                    auto axes = bias_unsqueeze_axes(batched_dot_ndim,
                                                    b->get_shape().lens().size());
                    if(axes.empty())
                    {
                        bias_expanded.push_back(b);
                    }
                    else
                    {
                        bias_expanded.push_back(m.insert_instruction(
                            insert_pt, make_op("unsqueeze", {{"axes", axes}}), b));
                    }
                }
                auto batched_bias = m.insert_instruction(
                    insert_pt, make_op("concat", {{"axis", 0}}), bias_expanded);

                auto bc_bias = m.insert_instruction(
                    insert_pt,
                    make_op("multibroadcast",
                            {{"out_lens", current->get_shape().lens()}}),
                    batched_bias);

                // add + sigmoid + mul (SiLU)
                auto added = m.insert_instruction(insert_pt, make_op("add"), current, bc_bias);
                auto sig   = m.insert_instruction(insert_pt, make_op("sigmoid"), added);
                current    = m.insert_instruction(insert_pt, make_op("mul"), sig, added);
            }
        }

        // Slice + squeeze the final batched output back to individual results
        for(std::size_t i = 0; i < independent.size(); ++i)
        {
            auto sliced = m.insert_instruction(
                insert_pt,
                make_op("slice",
                        {{"axes", std::vector<int64_t>{0}},
                         {"starts", std::vector<int64_t>{static_cast<int64_t>(i)}},
                         {"ends", std::vector<int64_t>{static_cast<int64_t>(i + 1)}}}),
                current);
            auto squeezed = m.insert_instruction(
                insert_pt, make_op("squeeze", {{"axes", {0}}}), sliced);

            auto chain_terminal = chains[independent[i]].terminal();
            m.move_output_instructions_after(chain_terminal, squeezed);
            m.replace_instruction(chain_terminal, squeezed);
        }
    }
}

void fuse_horizontal::apply(module_pass_manager& mpm) const
{
    auto& m = mpm.get_module();

    fuse_horizontal_ops(m, gather_horizontal_fusion{});
    fuse_mlp_chains(m);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
