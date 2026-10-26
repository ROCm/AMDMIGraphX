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
#include <migraphx/fuse_attention.hpp>
#include <migraphx/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/match/softmax.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/generic_float.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/split_factor.hpp>
#include <migraphx/tensor_view.hpp>
#include <migraphx/literal.hpp>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <optional>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

// env vars for flash decoding configuration
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_FLASH_DECODING_NUM_SPLITS);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_FLASH_DECODING_MIN_CHUNK_SIZE);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_FLASH_DECODING_MAX_SPLITS);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_FLASH_DECODING_THRESHOLD);

// Helper function to get config value with priority: struct member (if not default) > env var >
// default
template <typename EnvVar>
std::size_t get_config_value(std::size_t struct_value, std::size_t default_value, EnvVar env_var)
{
    // if struct member is not the default value, use it
    if(struct_value != default_value)
    {
        return struct_value;
    }

    // otherwise return env var value, or default if not set
    return value_of(env_var, default_value);
}

// Get num_splits with priority: struct member > env var > 0 (not set)
std::size_t get_num_splits(std::size_t member_num_splits)
{
    // if struct member is set (non-zero), use it
    if(member_num_splits > 0)
    {
        return member_num_splits;
    }

    // otherwise return env var value, or 0 if not set
    return value_of(MIGRAPHX_FLASH_DECODING_NUM_SPLITS{}, 0);
}

// calculate optimal flash decoding splits
inline std::size_t calculate_flash_decoding_splits(std::size_t sequence_length,
                                                   std::size_t min_chunk_size,
                                                   std::size_t max_splits)
{
    std::size_t r = sequence_length;
    return split_dim(r, min_chunk_size, max_splits);
}

// calculate the actual number of groups for flash decoding
// returns 0 if no splitting should be performed
inline std::size_t calculate_groups(std::size_t groups,
                                    std::size_t sequence_length,
                                    std::size_t threshold,
                                    std::size_t min_chunk_size,
                                    std::size_t max_splits)
{
    // if groups is explicitly set and valid, use it
    if(groups > 1)
        return groups;

    // if groups is 0, auto-calculate based on sequence length
    if(groups == 0)
    {
        // skip if sequence is too short
        if(sequence_length < threshold)
            return 0;

        std::size_t actual_groups =
            calculate_flash_decoding_splits(sequence_length, min_chunk_size, max_splits);

        // return 0 if auto-calculation determines no splitting needed
        if(actual_groups <= 1)
            return 0;

        return actual_groups;
    }

    // groups == 1 or invalid, no splitting
    return 0;
}

std::pair<instruction_ref, instruction_ref> get_gemms(module_ref submod)
{
    std::vector<instruction_ref> gemms;
    for(auto it = submod->begin(); it != submod->end(); ++it)
    {
        if(it->name() == "dot")
            gemms.push_back(it);
    }
    assert(gemms.size() == 2 and "Expected exactly 2 gemm operations in attention submodule");

    // gemms[0] is Q@K, gemms[1] is P@V
    // gemms are in order since we iterate from begin to end
    return {gemms[0], gemms[1]};
}

bool attention_submodule_rebuild_blocked(const module& source_mod,
                                         const std::vector<std::size_t>& scores_lens)
{
    // The rewrites below run from an inserter, which is not given the source instruction,
    // so anything that must inspect it is checked here, before target_mod is touched.
    auto unsupported = [&](const instruction& ins) {
        const auto name = ins.name();
        // "broadcast" pins an explicit axis, which shifts once gemm output is inserted
        if(name == "broadcast")
            return ins.get_shape().lens() == scores_lens;
        // reductions are rewritten onto the innermost axis of the split rank
        if(name == "reduce_max" or name == "reduce_sum")
        {
            const int64_t last_axis = ins.inputs().front()->get_shape().ndim() - 1;
            auto axes               = ins.get_operator().to_value()["axes"].to_vector<int64_t>();
            return axes.size() != 1 or (axes.front() != -1 and axes.front() != last_axis);
        }
        return false;
    };
    return std::any_of(source_mod.begin(), source_mod.end(), unsupported);
}

// Split a packed sequence vector, then broadcast it to the split scores.
// Broadcasting to the full scores and reshaping that view aliases a short
// buffer with packed strides.
instruction_ref broadcast_split_sequence(module& m,
                                         instruction_ref pos,
                                         instruction_ref input,
                                         std::size_t num_groups,
                                         const std::vector<std::size_t>& split_lens)
{
    auto lens          = input->get_shape().lens();
    const auto n_split = lens.back() / num_groups;
    lens.back()        = num_groups;
    lens.push_back(n_split);
    auto split      = m.insert_instruction(pos, make_op("reshape", {{"dims", lens}}), input);
    const auto axis = static_cast<int64_t>(split->get_shape().ndim() - 1);
    split           = m.insert_instruction(pos, make_op("unsqueeze", {{"axes", {axis}}}), split);
    return m.insert_instruction(pos, make_op("multibroadcast", {{"out_lens", split_lens}}), split);
}

// Apply ops in order, each to the result of the previous one.
instruction_ref insert_ops(module& mm,
                           instruction_ref insert_before,
                           const std::vector<operation>& ops,
                           instruction_ref input)
{
    return std::accumulate(
        ops.begin(), ops.end(), input, [&](instruction_ref x, const operation& op) {
            return mm.insert_instruction(insert_before, op, x);
        });
}

// Configuration from the fuse_attention pass. A non-default member wins over the
// matching MIGRAPHX_FLASH_DECODING_* environment variable.
struct flash_decoding_config
{
    std::size_t splits         = 0;
    std::size_t threshold      = 32;
    std::size_t max_splits     = 16;
    std::size_t min_chunk_size = 32;

    // Resolved split count for this sequence. 0 means the sequence is not split.
    std::size_t num_groups(std::size_t sequence_length) const
    {
        const auto groups = get_num_splits(splits);
        const auto resolved_threshold =
            get_config_value(threshold, 32, MIGRAPHX_FLASH_DECODING_THRESHOLD{});
        const auto resolved_min_chunk_size =
            get_config_value(min_chunk_size, 32, MIGRAPHX_FLASH_DECODING_MIN_CHUNK_SIZE{});
        const auto resolved_max_splits =
            get_config_value(max_splits, 16, MIGRAPHX_FLASH_DECODING_MAX_SPLITS{});
        return calculate_groups(groups,
                                sequence_length,
                                resolved_threshold,
                                resolved_min_chunk_size,
                                resolved_max_splits);
    }
};

// One sequence split. num_groups is G, and scores_lens is the pre-split QK shape.
struct flash_decoding_split
{
    std::size_t num_groups = 0;
    std::vector<std::size_t> scores_lens{};

    struct transformed_shapes_result
    {
        std::vector<size_t> q_shape;  // final Q shape: [B, G, M, k]
        std::vector<size_t> k_shape;  // final K shape: [B, G, k, N/G]
        std::vector<size_t> v_shape;  // final V shape: [B, G, N/G, D]
        std::vector<operation> q_ops; // unsqueeze + multibroadcast
        std::vector<operation> k_ops; // reshape + transpose
        std::vector<operation> v_ops; // reshape
        int64_t g_axis = 0;           // position of G in the split shapes
    };

    // Lens of a score-shaped tensor after its key axis is split:
    // [..., M, N] -> [..., G, M, N/G]
    std::vector<std::size_t> split_lens(const std::vector<std::size_t>& lens) const
    {
        assert(lens.size() >= 2);
        assert(num_groups > 0);
        assert(lens.back() % num_groups == 0);

        const auto ndim    = lens.size();
        const auto n_split = lens.back() / num_groups;
        std::vector<size_t> result(lens.begin(), lens.begin() + ndim - 2);
        result.push_back(num_groups);
        result.insert(result.end(), lens.begin() + ndim - 2, lens.end() - 1);
        result.push_back(n_split);
        return result;
    }

    // Ops splitting the key axis of a score-aligned tensor: [..., M, N] -> [..., G, M, N/G].
    // Reshaping straight to the split lens would regroup M with N, so reshape to
    // [..., M, G, N/G] and transpose G ahead of M instead.
    std::vector<operation> scores_split_ops(const std::vector<std::size_t>& lens) const
    {
        const auto ndim = lens.size();
        assert(ndim >= 2 and num_groups > 0 and lens.back() % num_groups == 0);

        std::vector<std::size_t> dims(lens.begin(), lens.end() - 1);
        dims.push_back(num_groups);
        dims.push_back(lens.back() / num_groups);

        std::vector<int64_t> perm(ndim + 1);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[ndim - 2], perm[ndim - 1]);

        return {make_op("reshape", {{"dims", dims}}),
                make_op("transpose", {{"permutation", perm}})};
    }

    instruction_ref
    split_scores(module& mm, instruction_ref ins, instruction_ref insert_before) const
    {
        return insert_ops(mm, insert_before, scores_split_ops(ins->get_shape().lens()), ins);
    }

    // Compile-time counterpart of split_scores for literals in the attention submodule,
    // like causal masks. Splits the literal's own layout (e.g. {1,1,M,N} -> {1,1,G,M,N/G}), so
    // the consumer's multibroadcast still expands the leading 1s. Broadcast-only constants
    // (scale, -inf) keep their original shape.
    literal split_literal(const literal& lit) const
    {
        const auto& input_lens = lit.get_shape().lens();
        if(input_lens.size() < 2 or not can_multibroadcast(input_lens, scores_lens))
            return lit;

        // N is the last axis; if it is 1 the literal only broadcasts along N, so nothing to split
        if(input_lens.back() == 1 or input_lens.back() % num_groups != 0)
            return lit;

        const auto split   = split_lens(input_lens);
        const auto ndim    = input_lens.size();
        const auto n_split = input_lens.back() / num_groups;

        literal result;
        lit.visit([&](auto in_view) {
            const auto& in_shape = in_view.get_shape();

            // Split element [..., g, m, j] reads [..., m, g * n_split + j], so the split layout
            // is a strided view of the same buffer: g steps a chunk of N, m keeps its stride.
            const auto& strides = in_shape.strides();
            std::vector<std::size_t> split_strides(strides.begin(), strides.end() - 2);
            split_strides.insert(split_strides.end(),
                                 {n_split * strides.back(), strides[ndim - 2], strides.back()});

            const shape split_view{in_shape.type(), split, split_strides};
            assert(split_view.element_space() <= in_shape.element_space() and
                   "split view must stay inside the literal's buffer");
            auto src = make_view(split_view, in_view.data());
            result   = literal{shape{in_shape.type(), split}, src.begin(), src.end()};
        });
        return result;
    }

    transformed_shapes_result transform_qkv(const shape& q, const shape& k, const shape& v) const
    {
        assert((q.lens().size() == 3 or q.lens().size() == 4) and
               "Expected 3D or 4D Q, K, V shapes");
        assert(k.lens().size() == q.lens().size() and v.lens().size() == q.lens().size() and
               "Q, K, V must have same number of dimensions");

        auto q_lens = q.lens();
        auto k_lens = k.lens();
        auto v_lens = v.lens();

        // 3D: Q_lens = [B, M, k]
        // 4D: Q_lens = [B, H, M, k]
        size_t ndim = q_lens.size();
        size_t n    = k_lens[ndim - 1];
        size_t g    = num_groups;

        assert(ndim >= 2 and k_lens.size() == ndim and v_lens.size() == ndim and
               "Q, K, V must have matching rank of at least 2");
        assert(n % g == 0 and
               "Key-value sequence length must be divisible by number of splits/groups");
        size_t n_split = n / g;

        transformed_shapes_result result;

        // batch_dims + G + spatial_dims
        auto insert_g = [&](const auto& lens) {
            std::vector<size_t> new_lens(lens.begin(), lens.begin() + ndim - 2);  // batch dims
            new_lens.push_back(g);                                                // insert G
            new_lens.insert(new_lens.end(), lens.begin() + ndim - 2, lens.end()); // last 2 dims
            return new_lens;
        };

        // Q: [B, M, k] -> [B, G, M, k] via unsqueeze + broadcast
        result.g_axis  = ndim - 2;
        result.q_shape = insert_g(q_lens);
        result.q_ops   = {make_op("unsqueeze", {{"axes", {result.g_axis}}}),
                          make_op("multibroadcast", {{"out_lens", result.q_shape}})};

        // K: [B, k, N] -> [B, G, k, N/G]. Splitting K's sequence axis is the same transform as
        // splitting a score-shaped tensor's key axis, so it reuses the same ops.
        result.k_shape = split_lens(k_lens);
        result.k_ops   = scores_split_ops(k_lens);

        // V: [B, N, D] -> [B, G, N/G, D] via direct reshape
        result.v_shape                            = insert_g(v_lens);
        result.v_shape[result.v_shape.size() - 2] = n_split;
        result.v_ops = {make_op("reshape", {{"dims", result.v_shape}})};

        return result;
    }

    // Both fd matchers copy the same softmax body. The KV fd also wraps the query,
    // skips the key/value layout it already rebuilt, and drops the views that
    // follow the second gemm
    bool
    rebuild(module& target_mod,
            const module& source_mod,
            const std::unordered_map<instruction_ref, instruction_ref>& param_map,
            const std::unordered_map<instruction_ref, std::vector<operation>>* qkv_splits = nullptr,
            const std::unordered_set<instruction_ref>* skip = nullptr) const
    {
        const auto split = split_lens(scores_lens);

        if(attention_submodule_rebuild_blocked(source_mod, scores_lens))
            return false;

        // map from instructions in the old module to the new ones in the target module
        std::unordered_map<instruction_ref, instruction_ref> map_old_to_new = param_map;

        // add_instructions copies literals verbatim and offers no hook for rewriting one, so
        // we split them up front and map them. This then skips literals in the rebuild below.
        for(auto ins : iterator_for(source_mod))
        {
            if(ins->name() != "@literal")
                continue;
            map_old_to_new[ins] = target_mod.add_literal(split_literal(ins->get_literal()));
        }

        // Rebuild the rest, transforming operators that depend on tensor shape/rank. The inserter
        // is never called for @param, @outline, or @return; add_instructions handles those.
        // Body instructions are visited in module order, so this cursor stays aligned with it.
        auto body            = source_mod.begin();
        const bool track_src = qkv_splits != nullptr or skip != nullptr;
        // add_instructions does not call the inserter for an instruction that is already
        // mapped. The prebuilt K and V splits are mapped that way, so this cursor skips
        // them too; otherwise a later view is copied onto the split-rank gemm.
        auto take_src = [&] {
            while(body != source_mod.end() and
                  (body->name() == "@param" or body->name() == "@literal" or
                   body->name() == "@outline" or contains(map_old_to_new, body)))
                ++body;
            auto cur = body;
            if(body != source_mod.end())
                ++body;
            return cur;
        };
        std::optional<instruction_ref> max_ins;
        std::optional<instruction_ref> sum_exp_ins;
        auto outputs = target_mod.add_instructions(
            &source_mod,
            &map_old_to_new,
            [&](module& m,
                instruction_ref pos,
                const operation& op,
                const std::vector<instruction_ref>& inputs,
                const std::vector<module_ref>& mod_args) {
                instruction_ref src{};
                if(track_src)
                    src = take_src();
                if(skip != nullptr and contains(*skip, src))
                    return inputs.front();

                const auto name = op.name();
                auto new_op     = op;
                if(name == "reduce_max" or name == "reduce_sum")
                {
                    new_op.from_value({{"axes", {inputs.front()->get_shape().ndim() - 1}}});
                }
                else if(name == "multibroadcast")
                {
                    auto value = op.to_value();
                    if(value["out_lens"].to_vector<std::size_t>() == scores_lens)
                    {
                        const auto& input_lens = inputs.front()->get_shape().lens();
                        // A packed sequence vector cannot broadcast to the split scores.
                        // Split that vector first. An input that already has the score
                        // rank is broadcast out to the original scores and then split.
                        if(not can_multibroadcast(input_lens, split) and
                           can_multibroadcast(input_lens, scores_lens))
                        {
                            if(inputs.front()->get_shape().standard() and
                               inputs.front()->get_shape().ndim() < scores_lens.size() and
                               input_lens.back() == scores_lens.back() and
                               input_lens.back() % num_groups == 0)
                                return broadcast_split_sequence(
                                    m, pos, inputs.front(), num_groups, split);
                            return split_scores(m, m.insert_instruction(pos, op, inputs), pos);
                        }

                        value["out_lens"] = split;
                        new_op.from_value(value);
                    }
                }

                auto new_ins = m.insert_instruction(pos, new_op, inputs, mod_args);
                if(qkv_splits != nullptr and contains(*qkv_splits, src))
                    new_ins = insert_ops(m, pos, qkv_splits->at(src), new_ins);

                // store key softmax components for LSE calculation
                if(name == "reduce_max")
                    max_ins = new_ins;
                if(name == "reduce_sum")
                    sum_exp_ins = new_ins;
                return new_ins;
            });

        assert(outputs.size() == 1 and "Attention submodule must have a single output");
        auto partial_output_o_prime = outputs.front();

        // calculate LSE = max(S) + log(sum(exp(S - max(S))))
        assert(max_ins.has_value() and sum_exp_ins.has_value() and
               "Softmax max and sum must be rebuilt");
        auto log_sum_exp = target_mod.add_instruction(make_op("log"), *sum_exp_ins);
        auto lse         = target_mod.add_instruction(make_op("add"), *max_ins, log_sum_exp);

        // return a tuple of {O', LSE}
        target_mod.add_return({partial_output_o_prime, lse});
        return true;
    }

    // kernel 2: combine using exp-normalize trick
    // O = sum(O' * exp(LSE - max)) / sum(exp(LSE - max))
    instruction_ref combine(module& mm,
                            instruction_ref pos,
                            instruction_ref o_prime,
                            instruction_ref lse,
                            int64_t g_axis,
                            const std::vector<operation>& output_views) const
    {
        // find max LSE across groups for numerical stability
        auto lse_max = mm.insert_instruction(pos, make_op("reduce_max", {{"axes", {g_axis}}}), lse);

        auto lse_max_bcast = mm.insert_instruction(
            pos, make_op("multibroadcast", {{"out_lens", lse->get_shape().lens()}}), lse_max);

        // compute unnormalized weights
        // exp(LSE - max)
        auto lse_sub = mm.insert_instruction(pos, make_op("sub"), lse, lse_max_bcast);

        auto lse_exp = mm.insert_instruction(pos, make_op("exp"), lse_sub);

        // broadcast weights to match O' shape
        // [B, G, M] -> [B, G, M, D]
        auto lse_exp_bcast = mm.insert_instruction(
            pos, make_op("multibroadcast", {{"out_lens", o_prime->get_shape().lens()}}), lse_exp);

        // convert weights to output type
        auto output_type = o_prime->get_shape().type();
        auto weights     = mm.insert_instruction(
            pos, make_op("convert", {{"target_type", output_type}}), lse_exp_bcast);

        // compute weighted sum: numerator = sum(O' * weights)
        auto weighted_o = mm.insert_instruction(pos, make_op("mul"), o_prime, weights);

        auto numerator =
            mm.insert_instruction(pos, make_op("reduce_sum", {{"axes", {g_axis}}}), weighted_o);

        // compute sum of weights: denominator = sum(weights)
        auto denominator =
            mm.insert_instruction(pos, make_op("reduce_sum", {{"axes", {g_axis}}}), weights);

        // final division: O = numerator / denominator
        auto final_output_o = mm.insert_instruction(pos, make_op("div"), numerator, denominator);

        // squeeze G to match the original output shape
        auto final_squeezed_o =
            mm.insert_instruction(pos, make_op("squeeze", {{"axes", {g_axis}}}), final_output_o);

        // Replay the views that followed the second gemm. Those views were written for the
        // unsplit rank, so they run only once the squeeze has restored that shape.
        return insert_ops(mm, pos, output_views, final_squeezed_o);
    }
};

bool is_range_literal(const literal& l)
{
    const auto& s = l.get_shape();
    if(s.elements() < 2 or not shape::is_computable(s.type()))
        return false;
    bool result = false;
    l.visit([&](auto x) {
        result = std::adjacent_find(x.begin(), x.end(), [](auto cur, auto next) {
                     return next <= cur or not float_equal(next - cur, 1);
                 }) == x.end();
    });
    return result;
}

bool is_inlinable_constant(instruction_ref ins)
{
    if(ins->name() != "@literal")
        return true;
    return ins->get_shape().elements() == 1 or is_range_literal(ins->get_literal());
}

// TODO: Write this in matcher.hpp as a general matcher for iterating through inputs
inline auto pointwise_inputs()
{
    return [](auto start, auto f) {
        std::unordered_set<instruction_ref> visited;
        fix([&](auto self, auto ins) {
            if(ins->can_eval())
                return;
            if(not visited.insert(ins).second)
                return;
            if(not ins->get_operator().attributes().contains("pointwise") and
               ins->get_operator().name() != "reshape")
            {
                f(ins);
                return;
            }
            for(auto input : ins->inputs())
                self(input);
        })(start);
    };
}

std::unordered_map<instruction_ref, instruction_ref>
invert_map_ins(const std::unordered_map<instruction_ref, instruction_ref>& map_ins)
{
    std::unordered_map<instruction_ref, instruction_ref> inverse_map;
    for(auto const& [key, value] : map_ins)
    {
        assert(not contains(inverse_map, value));
        inverse_map[value] = key;
    }
    return inverse_map;
}

std::vector<instruction_ref> find_outputs(const std::vector<instruction_ref>& inss)
{
    std::vector<instruction_ref> outputs;
    std::copy_if(inss.begin(), inss.end(), std::back_inserter(outputs), [&](auto i) {
        return not std::all_of(
            i->outputs().begin(), i->outputs().end(), [&](auto o) { return contains(inss, o); });
    });
    return outputs;
}

// {final_op, reduce_max, reduce_sum} outputs indicate the softmax statistics are
// consumed externally by an lse (log-sum-exp) computation
bool has_lse_outputs(const std::vector<instruction_ref>& outs, const std::string& final_op)
{
    auto count = [&](const std::string& name) {
        return std::count_if(outs.begin(), outs.end(), [&](auto o) { return o->name() == name; });
    };
    return outs.size() == 3 and count(final_op) == 1 and count("reduce_max") == 1 and
           count("reduce_sum") == 1;
}

std::vector<instruction_ref> get_lse_instructions(const std::vector<instruction_ref>& group_outs)
{
    std::vector<instruction_ref> lse_inss;
    auto rsum = std::find_if(
        group_outs.begin(), group_outs.end(), [](auto o) { return o->name() == "reduce_sum"; });
    if(rsum == group_outs.end())
        return lse_inss;
    auto rsum_outs = (*rsum)->outputs();
    auto log =
        std::find_if(rsum_outs.begin(), rsum_outs.end(), [](auto o) { return o->name() == "log"; });
    if(log == rsum_outs.end())
        return lse_inss;
    auto log_outs = (*log)->outputs();
    if(log_outs.size() != 1 or log_outs.front()->name() != "add")
        return lse_inss;
    auto add = log_outs.front();
    lse_inss.insert(lse_inss.end(), {*log, add});

    return lse_inss;
}

// find attention blocks that have been quantized and undo them
struct find_quant_attention
{
    auto matcher() const
    {
        auto gemm1 = match::opaque(
            match::name("dequantizelinear")(match::arg(0)(match::name("quant_dot").bind("qgemm1")))
                .bind("deq1"));
        auto softmax =
            match::opaque(match::softmax_input(match::skip(match::name("convert"))(gemm1)));
        auto softmax_cvt = match::opaque(match::skip(match::name("convert"))(softmax));
        auto probs       = match::opaque(match::name("quantizelinear")(match::arg(0)(softmax_cvt)));
        auto gemm2 = match::opaque(match::name("quant_dot")(match::arg(0)(probs)).bind("qgemm2"));
        return match::name("dequantizelinear")(match::arg(0)(gemm2)).bind("deq2");
    }

    // both inputs are quantizelinear
    static bool can_dequantize_gemm(instruction_ref qgemm)
    {
        return qgemm->inputs().at(0)->name() == "quantizelinear" and
               qgemm->inputs().at(1)->name() == "quantizelinear";
    }

    // removes the q/dq pairs from attention block gemms
    static void dequantize_gemm(module& m, instruction_ref qgemm, instruction_ref deq)
    {
        auto a            = qgemm->inputs().at(0)->inputs().front();
        auto b            = qgemm->inputs().at(1)->inputs().front();
        auto compute_type = b->get_shape().type();
        if(a->get_shape().type() != compute_type)
            a = m.insert_instruction(deq, make_op("convert", {{"target_type", compute_type}}), a);
        instruction_ref dot = m.insert_instruction(deq, make_op("dot"), a, b);
        if(compute_type != deq->get_shape().type())
            dot = m.insert_instruction(
                deq, make_op("convert", {{"target_type", deq->get_shape().type()}}), dot);
        m.replace_instruction(deq, dot);
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& m     = mpm.get_module();
        auto qgemm1 = r.instructions["qgemm1"];
        auto deq1   = r.instructions["deq1"];
        auto qgemm2 = r.instructions["qgemm2"];
        auto deq2   = r.instructions["deq2"];

        if(can_dequantize_gemm(qgemm1) and can_dequantize_gemm(qgemm2))
        {
            dequantize_gemm(m, qgemm1, deq1);
            dequantize_gemm(m, qgemm2, deq2);
        }
    }
};

// Rewrite attention sinks (an extra per-head logit column joining the softmax
// denominator, e.g. GPT-OSS) into a standard softmax so the attention matchers
// below (and rocMLIR) see a canonical gemm-softmax-gemm:
//   slice(softmax(concat(scores, sinks))) = softmax(scores) * sigmoid(lse - sinks)
// where lse = reduce_max(scores) + log(reduce_sum(exp(scores - max))). The
// sigmoid factor is constant along the reduction axis, so it commutes past the
// second gemm (and its transpose/reshape output tail) and is applied there as a
// small pointwise epilogue. The log/add lse chain hangs off the softmax
// reductions and becomes a second output of the fused attention group.
struct find_attention_sinks
{
    // ext and exp are each matched twice, so erase each level with opaque to
    // keep the nested matcher type from growing exponentially (gcc/msvc OOM)
    auto matcher() const
    {
        auto scores = match::any().bind("scores");
        auto sinks  = match::any().bind("sinks");
        auto ext  = match::opaque(match::name("concat")(match::args(scores, sinks)).bind("concat"));
        auto rmax = match::opaque(match::name("reduce_max")(match::arg(0)(ext)).bind("rmax"));
        auto bmax = match::skip_broadcasts(rmax);
        auto sub  = match::opaque(match::name("sub")(match::arg(0)(ext), match::arg(1)(bmax)));
        auto exp  = match::opaque(match::name("exp")(match::arg(0)(sub)));
        auto rsum = match::opaque(match::name("reduce_sum")(match::arg(0)(exp)).bind("rsum"));
        auto bsum = match::skip_broadcasts(rsum);
        auto sm   = match::opaque(match::name("div")(match::arg(0)(exp), match::arg(1)(bsum)));
        return match::name("slice")(match::arg(0)(match::skip(match::name("convert"))(sm)));
    }

    static std::vector<int64_t> normalized_axes(instruction_ref ins)
    {
        return ins->normalized_operator().to_value()["axes"].to_vector<int64_t>();
    }

    // The concat must append a single column on the softmax axis and the slice
    // must drop exactly that column again
    static bool is_sink_pattern(instruction_ref slc,
                                instruction_ref ext,
                                instruction_ref scores,
                                instruction_ref sinks,
                                instruction_ref rmax,
                                instruction_ref rsum)
    {
        if(slc->get_shape().dynamic() or scores->get_shape().dynamic() or
           sinks->get_shape().dynamic())
            return false;
        const auto& lens = scores->get_shape().lens();
        auto axis        = static_cast<int64_t>(lens.size()) - 1;
        if(sinks->get_shape().lens().back() != 1)
            return false;
        if(ext->normalized_operator().to_value()["axis"].to<int64_t>() != axis)
            return false;
        if(slc->get_shape().lens() != lens)
            return false;
        auto slc_val = slc->normalized_operator().to_value();
        if(slc_val["axes"].to_vector<int64_t>() != std::vector<int64_t>{axis})
            return false;
        if(slc_val["starts"].to_vector<int64_t>() != std::vector<int64_t>{0})
            return false;
        return normalized_axes(rmax) == std::vector<int64_t>{axis} and
               normalized_axes(rsum) == std::vector<int64_t>{axis};
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto slc    = r.result;
        auto ext    = r.instructions["concat"];
        auto scores = r.instructions["scores"];
        auto sinks  = r.instructions["sinks"];
        auto rmax   = r.instructions["rmax"];
        auto rsum   = r.instructions["rsum"];

        if(not is_sink_pattern(slc, ext, scores, sinks, rmax, rsum))
            return;

        const auto& lens = scores->get_shape().lens();
        auto bcast       = [&](instruction_ref ins) {
            if(ins->get_shape().lens() == lens)
                return ins;
            return m.insert_instruction(slc, make_op("multibroadcast", {{"out_lens", lens}}), ins);
        };
        auto convert_to = [&](instruction_ref ins, shape::type_t t, instruction_ref pos) {
            if(ins->get_shape().type() == t)
                return ins;
            return m.insert_instruction(pos, make_op("convert", {{"target_type", t}}), ins);
        };

        // softmax over the scores without the sink column
        auto new_rmax = m.insert_instruction(slc, rmax->get_operator(), scores);
        auto new_sub  = m.insert_instruction(slc, make_op("sub"), scores, bcast(new_rmax));
        auto new_exp  = m.insert_instruction(slc, make_op("exp"), new_sub);
        auto new_rsum = m.insert_instruction(slc, rsum->get_operator(), new_exp);
        auto probs    = m.insert_instruction(slc, make_op("div"), new_exp, bcast(new_rsum));
        probs         = convert_to(probs, slc->get_shape().type(), slc);

        // sigmoid(lse - sinks) scales each softmax row by sum/(sum + exp(sinks - max))
        auto log_sum  = m.insert_instruction(slc, make_op("log"), new_rsum);
        auto lse      = m.insert_instruction(slc, make_op("add"), new_rmax, log_sum);
        auto sink_val = convert_to(sinks, lse->get_shape().type(), slc);
        auto diff     = m.insert_instruction(slc, make_op("sub"), lse, sink_val);
        auto sigma    = m.insert_instruction(slc, make_op("sigmoid"), diff);

        // The row-wise correction commutes past the second gemm and its
        // transpose/reshape output tail; walk there so the multiply stays
        // outside the attention block matched below.
        std::vector<instruction_ref> tail;
        auto pos = slc;
        while(pos->outputs().size() == 1)
        {
            auto next    = pos->outputs().front();
            bool is_gemm = tail.empty() and next->name() == "dot" and next->inputs().front() == pos;
            bool is_view = not tail.empty() and contains({"transpose", "reshape"}, next->name());
            if(not is_gemm and not is_view)
                break;
            tail.push_back(next);
            pos = next;
        }

        if(tail.empty())
        {
            auto factor = bcast(convert_to(sigma, probs->get_shape().type(), slc));
            auto scaled = m.insert_instruction(slc, make_op("mul"), probs, factor);
            m.replace_instruction(slc, scaled);
            return;
        }
        m.replace_instruction(slc, probs);

        auto anchor = tail.back();
        auto factor = convert_to(sigma, anchor->get_shape().type(), anchor);
        for(auto ins : tail)
        {
            // the factor broadcasts unchanged over the gemm output columns
            if(ins->name() == "transpose")
            {
                factor = m.insert_instruction(anchor, ins->get_operator(), factor);
            }
            else if(ins->name() == "reshape")
            {
                const auto& in_lens = ins->inputs().front()->get_shape().lens();
                if(factor->get_shape().lens() != in_lens)
                    factor = m.insert_instruction(
                        anchor, make_op("multibroadcast", {{"out_lens", in_lens}}), factor);
                factor = m.insert_instruction(anchor, ins->get_operator(), factor);
            }
        }
        if(factor->get_shape().lens() != anchor->get_shape().lens())
            factor = m.insert_instruction(
                anchor,
                make_op("multibroadcast", {{"out_lens", anchor->get_shape().lens()}}),
                factor);
        auto new_anchor = m.insert_instruction(anchor, anchor->get_operator(), anchor->inputs());
        auto scaled     = m.insert_instruction(anchor, make_op("mul"), new_anchor, factor);
        m.replace_instruction(anchor, scaled);
    }
};

struct find_transposed_attention
{
    auto matcher() const
    {
        auto gemm1 =
            match::opaque(match::any_of[pointwise_inputs()](match::name("dot").bind("dot1")));
        auto softmax =
            match::opaque(match::skip(match::name("convert"))(match::softmax_input(gemm1)));
        auto swap_last_two = match::make_basic_pred_matcher([](instruction_ref ins) {
            auto perm = ins->get_operator().to_value()["permutation"].to_vector<int64_t>();
            if(perm.size() < 2)
                return false;
            return std::equal(
                       perm.begin(), perm.end() - 2, migraphx::range(perm.size() - 2).begin()) and
                   perm[perm.size() - 2] == static_cast<int64_t>(perm.size() - 1) and
                   perm[perm.size() - 1] == static_cast<int64_t>(perm.size() - 2);
        });
        auto transposed_softmax =
            match::opaque(match::name("transpose")(swap_last_two, match::arg(0)(softmax))
                              .bind("transposed_softmax"));
        auto input_of_dot2 = match::any().bind("input_of_dot2");
        return match::name("dot")(match::arg(0)(input_of_dot2), match::arg(1)(transposed_softmax));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& m                 = mpm.get_module();
        auto gemm2              = r.result;
        auto input_of_dot2      = r.instructions["input_of_dot2"];
        auto transposed_softmax = r.instructions["transposed_softmax"];
        auto softmax            = transposed_softmax->inputs().front();
        auto perm =
            transposed_softmax->get_operator().to_value()["permutation"].to_vector<int64_t>();

        if(input_of_dot2->get_shape().ndim() != perm.size())
            return;

        auto values = input_of_dot2;
        if(input_of_dot2->name() == "transpose")
        {
            auto input_perm =
                input_of_dot2->get_operator().to_value()["permutation"].to_vector<int64_t>();
            if(input_perm == perm)
                values = input_of_dot2->inputs().front();
        }
        if(values == input_of_dot2)
            values = m.insert_instruction(
                gemm2, make_op("transpose", {{"permutation", perm}}), input_of_dot2);

        auto canonical_gemm  = m.insert_instruction(gemm2, make_op("dot"), softmax, values);
        auto transposed_gemm = m.insert_instruction(
            gemm2, make_op("transpose", {{"permutation", perm}}), canonical_gemm);
        m.replace_instruction(gemm2, transposed_gemm);
    }
};

struct find_attention
{
    std::size_t* counter;

    auto matcher() const
    {
        auto gemm1 =
            match::opaque(match::any_of[pointwise_inputs()](match::name("dot").bind("dot1")));
        auto softmax =
            match::opaque(match::skip(match::name("convert"))(match::softmax_input(gemm1)));
        return match::name("dot")(match::arg(0)(softmax));
    }

    std::string get_count() const { return std::to_string((*counter)++); }

    std::vector<instruction_ref>
    get_attn_instructions(module& m, instruction_ref gemm1, instruction_ref gemm2) const
    {
        auto attn_inss = find_instructions_between(gemm1, gemm2, &m);

        // Pull in evaluable constants so MLIR can detect causal masks.
        auto expand = fix([&](auto self, auto ins) {
            for(auto input : ins->inputs())
            {
                if(not contains(attn_inss, input) and input->can_eval() and
                   is_inlinable_constant(input))
                {
                    attn_inss.insert(input);
                    self(input);
                }
            }
        });
        auto starts = attn_inss;
        for(auto ins : starts)
            expand(ins);

        std::vector<instruction_ref> sorted_inss(attn_inss.begin(), attn_inss.end());
        std::sort(
            sorted_inss.begin(), sorted_inss.end(), [&](instruction_ref x, instruction_ref y) {
                return std::distance(m.begin(), x) < std::distance(m.begin(), y);
            });

        return sorted_inss;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto gemm2         = r.result;
        auto gemm1         = r.instructions["dot1"];
        auto softmax_input = r.instructions["x"];

        // Capture all instructions part of the attention op
        auto attn_inss = get_attn_instructions(mpm.get_module(), gemm1, gemm2);

        // Add captured instructions to new submodule
        module m_attn;
        std::unordered_map<instruction_ref, instruction_ref> map_mm_to_mattn;
        m_attn.fuse(attn_inss, &map_mm_to_mattn);

        // Define outputs based on instructions that are used elsewhere in the graph.
        // Remove evaluable constants from the output list.
        auto all_outputs = find_outputs(attn_inss);
        std::vector<instruction_ref> required_outputs;
        std::copy_if(all_outputs.begin(),
                     all_outputs.end(),
                     std::back_inserter(required_outputs),
                     [](auto i) { return not i->can_eval(); });

        assert(not required_outputs.empty());

        // LSE case requires output from reduce_max and reduce_sum instructions
        if(has_lse_outputs(required_outputs, "dot"))
        {
            auto lse_inss = get_lse_instructions(required_outputs);
            m_attn.fuse(lse_inss, &map_mm_to_mattn);

            // Recompute required outputs after adding lse instructions
            attn_inss.insert(attn_inss.end(), lse_inss.begin(), lse_inss.end());
            required_outputs = find_outputs(attn_inss);

            // Final outputs should be the second gemm and add instruction
            // (shift lse to adjust for subtracting max during stable softmax computation)
            if(required_outputs.size() != 2)
                return;
        }
        else if(required_outputs.size() > 1)
        {
            return;
        }

        // Find corresponding output instructions in m_attn
        std::vector<instruction_ref> m_attn_outputs;
        std::transform(required_outputs.begin(),
                       required_outputs.end(),
                       std::back_inserter(m_attn_outputs),
                       [&](auto i) { return map_mm_to_mattn.at(i); });

        m_attn.add_return(m_attn_outputs);

        // Define inputs to m_attn
        auto map_mattn_to_mm = invert_map_ins(map_mm_to_mattn);
        auto new_inputs      = m_attn.get_inputs(map_mattn_to_mm);

        module_ref mpm_attn = mpm.create_module("attn" + get_count(), std::move(m_attn));
        mpm_attn->set_bypass();

        auto group_ins = mpm.get_module().insert_instruction(
            softmax_input, make_op("group", {{"tag", "attention"}}), new_inputs, {mpm_attn});

        if(m_attn_outputs.size() == 1)
        {
            mpm.get_module().replace_instruction(required_outputs.front(), group_ins);
        }
        else
        {
            for(std::size_t i = 0; i < required_outputs.size(); ++i)
            {
                mpm.get_module().replace_instruction(
                    required_outputs[i], make_op("get_tuple_elem", {{"index", i}}), group_ins);
            }
        }
    }
};

struct find_flash_decoding
{
    flash_decoding_config config{};

    auto matcher() const
    {
        return match::name("group")(match::has_op_value("tag", "attention")).bind("group");
    }

    struct flash_input_transform
    {
        instruction_ref main{};
        instruction_ref split_main{};
        shape submodule_param_shape{};
    };

    std::unordered_map<instruction_ref, instruction_ref>
    map_submod_params_to_inputs(module_ref submod,
                                const std::vector<instruction_ref>& group_inputs) const
    {
        auto map_param_to_main = submod->get_ins_param_map(group_inputs, true);
        assert(submod->get_inputs(map_param_to_main) == group_inputs and
               "Mapped inputs don't match group inputs");
        return map_param_to_main;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& mm            = mpm.get_module();
        auto attn_group_ins = r.instructions["group"];
        auto* submod        = attn_group_ins->module_inputs().front();

        // TODO: for this pass of flash decoding, if LSE attn, do not do flash decoding
        auto return_ins = std::prev(submod->end());
        assert(return_ins->name() == "@return" and
               "Last instruction must be a @return instruction");
        if(return_ins->inputs().size() > 1)
            return;

        // get gemm1 and gemm2
        auto [gemm1, gemm2] = get_gemms(submod);

        auto q_param = gemm1->inputs()[0];
        auto k_param = gemm1->inputs()[1];
        auto v_param = gemm2->inputs()[1];
        assert(q_param->name() == "@param" and "Q should be a parameter");
        assert(k_param->name() == "@param" and "K should be a parameter");
        assert(v_param->name() == "@param" and "V should be a parameter");

        // Get sequence length from K shape
        auto k_shape                = k_param->get_shape();
        std::size_t sequence_length = k_shape.lens().back();

        const auto actual_groups = config.num_groups(sequence_length);
        if(actual_groups == 0)
            return;

        // TODO: support uneven splits by padding K, V, and score-shaped inputs up to a
        // multiple of the split count
        if(sequence_length % actual_groups != 0)
            return;

        // create mapping from submodule params to main module inputs
        auto group_inputs      = attn_group_ins->inputs();
        auto map_param_to_main = map_submod_params_to_inputs(submod, group_inputs);

        // gemm1 output lens (Q@K attention scores): the lens whose key axis is split by G
        const auto& scores_lens = gemm1->get_shape().lens();
        const flash_decoding_split decoding{actual_groups, scores_lens};
        std::unordered_map<instruction_ref, flash_input_transform> param_transforms;
        const auto submod_params = submod->get_parameters();

        for(auto param : submod_params)
        {
            const auto main_ins     = map_param_to_main.at(param);
            param_transforms[param] = flash_input_transform{main_ins, {}, {}};
        }

        const auto q_main = map_param_to_main.at(q_param);
        const auto k_main = map_param_to_main.at(k_param);
        const auto v_main = map_param_to_main.at(v_param);
        const auto transform_info =
            decoding.transform_qkv(q_main->get_shape(), k_main->get_shape(), v_main->get_shape());

        // Split each submodule @param that needs it. Iterate submod_params rather than the
        // unordered param_transforms map so insertion order is deterministic.
        for(auto param : submod_params)
        {
            auto& transform = param_transforms.at(param);
            if(param == q_param)
            {
                transform.split_main =
                    insert_ops(mm, attn_group_ins, transform_info.q_ops, transform.main);
                transform.submodule_param_shape =
                    shape{q_main->get_shape().type(), transform_info.q_shape};
            }
            else if(param == k_param)
            {
                transform.split_main =
                    insert_ops(mm, attn_group_ins, transform_info.k_ops, transform.main);
                transform.submodule_param_shape =
                    shape{k_main->get_shape().type(), transform_info.k_shape};
            }
            else if(param == v_param)
            {
                transform.split_main =
                    insert_ops(mm, attn_group_ins, transform_info.v_ops, transform.main);
                transform.submodule_param_shape =
                    shape{v_main->get_shape().type(), transform_info.v_shape};
            }
            // extra @param with the same lens() as gemm1 (Q@K attention scores), e.g. a mask
            else if(param->get_shape().lens() == scores_lens)
            {
                transform.split_main = decoding.split_scores(mm, transform.main, attn_group_ins);
                transform.submodule_param_shape =
                    shape{param->get_shape().type(), decoding.split_lens(scores_lens)};
            }
            else
            {
                transform.split_main            = transform.main;
                transform.submodule_param_shape = param->get_shape();
            }
        }

        // Create new input list by replacing group inputs with split versions.
        std::unordered_map<instruction_ref, instruction_ref> main_to_split;
        for(const auto& entry : param_transforms)
            main_to_split[entry.second.main] = entry.second.split_main;

        std::vector<instruction_ref> new_group_inputs = group_inputs;
        for(auto& input : new_group_inputs)
        {
            if(contains(main_to_split, input))
                input = main_to_split.at(input);
        }

        // create new submodule for flash decoding
        module m_flash_decode;
        m_flash_decode.set_bypass();

        std::unordered_map<instruction_ref, instruction_ref> map_old_params_to_new;
        for(auto param : submod_params)
        {
            const auto& name = any_cast<builtin::param>(param->get_operator()).parameter;
            map_old_params_to_new[param] = m_flash_decode.add_parameter(
                name, param_transforms.at(param).submodule_param_shape);
        }

        // don't simply fuse previous attn submod, need to rebuild all the ops
        if(not decoding.rebuild(m_flash_decode, *submod, map_old_params_to_new))
            return;

        auto original_submod_name = attn_group_ins->module_inputs().front()->name();
        std::string new_mod_name  = original_submod_name + "_flash_decoding";

        module_ref mpm_flash_mod = mpm.create_module(new_mod_name, std::move(m_flash_decode));
        mpm_flash_mod->set_bypass();

        // insert the new group op, which returns a tuple of O' and LSE
        auto new_group_ins = mm.insert_instruction(attn_group_ins,
                                                   make_op("group", {{"tag", "attention"}}),
                                                   new_group_inputs,
                                                   {mpm_flash_mod});

        // unpack O' and LSE
        auto partial_output_o_prime = mm.insert_instruction(
            attn_group_ins, make_op("get_tuple_elem", {{"index", 0}}), new_group_ins);
        auto lse = mm.insert_instruction(
            attn_group_ins, make_op("get_tuple_elem", {{"index", 1}}), new_group_ins);

        auto combined = decoding.combine(
            mm, attn_group_ins, partial_output_o_prime, lse, transform_info.g_axis, {});
        mm.replace_instruction(attn_group_ins, combined);
    }
};

struct find_kv_cache_attention
{
    std::size_t* counter;

    auto matcher() const
    {
        static const std::unordered_set<std::string> skip_set = {
            "multibroadcast", "broadcast", "reshape", "unsqueeze", "squeeze"};

        auto keys = match::opaque(
            match::skip(match::name(skip_set))(match::name("concat_past_present")).bind("pres_k"));
        auto keys_transpose  = match::opaque(match::name("transpose")(match::arg(0)(keys)));
        auto k_transpose     = match::opaque(match::skip(match::name(skip_set))(keys_transpose));
        auto gemm1 = match::opaque(match::name("dot")(match::arg(1)(k_transpose)).bind("gemm1"));
        auto gemm1_maybe_cvt = match::opaque(match::skip(match::name("convert"))(gemm1));
        auto scale    = match::opaque(match::name("mul")(match::any_arg(0, 1)(gemm1_maybe_cvt)));
        auto constant = match::opaque(match::is_constant());
        auto broadcasted_const =
            match::opaque(match::name("multibroadcast")(match::arg(0)(constant)));
        auto attn_scores = match::opaque(match::any_of(scale, gemm1_maybe_cvt));
        auto causal_mask = match::opaque(
            match::name("where")(match::arg(0)(broadcasted_const), match::arg(2)(attn_scores)));
        auto grtr              = match::opaque(match::name("greater"));
        auto conv_grtr         = match::opaque(match::name("convert")(match::arg(0)(grtr)));
        auto local_window_comp = match::opaque(match::skip(match::name(skip_set))(conv_grtr));
        auto local_window_cond = match::opaque(match::any_of(local_window_comp, broadcasted_const));
        auto local_window_val  = match::opaque(match::any_of(causal_mask, scale, gemm1_maybe_cvt));
        auto local_window_mask = match::opaque(match::name("where")(
            match::arg(0)(local_window_cond), match::arg(2)(local_window_val)));
        auto greater =
            match::opaque(match::name("greater")(match::arg(1)(match::any().bind("total_sl"))));
        auto greater_cvt  = match::opaque(match::name("convert")(match::arg(0)(greater)));
        auto conv_greater = match::opaque(match::skip(match::name("unsqueeze"))(greater_cvt));
        auto bc_greater = match::opaque(match::name("multibroadcast")(match::arg(0)(conv_greater)));
        auto mask_val =
            match::opaque(match::any_of(local_window_mask, causal_mask, scale, gemm1_maybe_cvt));
        auto mask =
            match::opaque(match::name("where")(match::arg(0)(bc_greater), match::arg(2)(mask_val)));
        auto mask_cvt = match::opaque(match::skip(match::name("convert"))(mask));
        auto attn_probabilities =
            match::opaque(match::skip(match::name("convert"))(match::softmax_input(mask_cvt)));
        auto values = match::opaque(
            match::skip(match::name(skip_set))(match::name("concat_past_present")).bind("pres_v"));
        return match::name("dot")(match::arg(0)(attn_probabilities), match::arg(1)(values));
    }

    /// The view ops laying out the attention output after the second gemm,
    /// in whatever form the reshape simplifications left them
    static instruction_ref find_output_end(instruction_ref gemm2)
    {
        static const std::unordered_set<std::string> view_ops = {
            "transpose", "reshape", "unsqueeze", "squeeze"};
        auto end = gemm2;
        while(end->outputs().size() == 1 and contains(view_ops, end->outputs().front()->name()))
            end = end->outputs().front();
        return end;
    }

    std::string get_count() const { return std::to_string((*counter)++); }

    static bool is_valid_attn_op(instruction_ref i)
    {
        static const std::unordered_set<std::string> valid_attn_ops = {"softmax",
                                                                       "broadcast",
                                                                       "dot",
                                                                       "slice",
                                                                       "transpose",
                                                                       "greater",
                                                                       "convert",
                                                                       "where",
                                                                       "reshape",
                                                                       "reduce_sum",
                                                                       "reduce_max",
                                                                       "multibroadcast",
                                                                       "@literal",
                                                                       "unsqueeze",
                                                                       "squeeze"};
        return i->get_operator().attributes().get("pointwise", false) or
               contains(valid_attn_ops, i->get_operator().name());
    }

    // Outputs of the capture: captured instructions used elsewhere in the
    // graph that depend on the first gemm. A captured instruction independent
    // of it (eg the sequence-length mask chain shared by every layer) is a
    // common subexpression whose original stays alive outside, not an output.
    static std::vector<instruction_ref> find_required_outputs(
        const std::vector<instruction_ref>& attn_inss, instruction_ref gemm1, const_module_ref m)
    {
        auto outputs = find_outputs(attn_inss);
        // attn_inss is in program order, so captured instructions before gemm1
        // cannot depend on it (reaches also requires start to precede end)
        std::unordered_set<instruction_ref> before_gemm1(
            attn_inss.begin(), std::find(attn_inss.begin(), attn_inss.end(), gemm1));
        outputs.erase(std::remove_if(outputs.begin(),
                                     outputs.end(),
                                     [&](instruction_ref i) {
                                         return contains(before_gemm1, i) or
                                                not reaches(gemm1, i, m);
                                     }),
                      outputs.end());
        return outputs;
    }

    std::vector<instruction_ref>
    get_attn_instructions(module& m, instruction_ref start, instruction_ref end) const
    {
        // Start with instructions on data-dependency paths from start to end.
        auto inss = find_instructions_between(start, end, &m);
        // Expand by walking inputs of instructions already in the set.
        // An input is added when it is a valid attention op and all of
        // its outputs are already in the set. This pulls in constants,
        // broadcasts, and side inputs that feed exclusively into the
        // attention, while excluding ops with external consumers.
        // Reductions are only valid on the softmax path, so an upstream
        // reduce (e.g. a Q rmsnorm) is never absorbed as an input fusion.
        auto expand = fix([&](auto self, auto ins) {
            for(auto input : ins->inputs())
            {
                if(contains(inss, input))
                    continue;
                if(not is_valid_attn_op(input))
                    continue;
                if(input->get_operator().attributes().get("reduce", false))
                    continue;
                if(input->can_eval() or std::all_of(input->outputs().begin(),
                                                    input->outputs().end(),
                                                    [&](auto o) { return contains(inss, o); }))
                {
                    inss.insert(input);
                    self(input);
                }
            }
        });
        // Copy inss since we will be modifying it
        auto starts = inss;
        for(auto ins : starts)
        {
            expand(ins);
        }

        std::vector<instruction_ref> sorted_inss(inss.begin(), inss.end());
        std::sort(
            sorted_inss.begin(), sorted_inss.end(), [&](instruction_ref x, instruction_ref y) {
                return std::distance(m.begin(), x) < std::distance(m.begin(), y);
            });
        return sorted_inss;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto total_sl = r.instructions["total_sl"];
        auto gemm1    = r.instructions["gemm1"];
        auto end      = find_output_end(r.result);

        // Capture all instructions part of the attention op
        auto attn_inss = get_attn_instructions(mpm.get_module(), total_sl, end);

        // A capture that grew past the attention block (eg through live
        // instructions a previous fusion left behind) would put ops mlir
        // cannot compile into the module, so skip it
        if(not std::all_of(attn_inss.begin(), attn_inss.end(), &is_valid_attn_op))
            return;

        // Add captured instructions to new submodule
        module m_attn;
        std::unordered_map<instruction_ref, instruction_ref> map_mm_to_mattn;
        auto attn_outs = m_attn.fuse(attn_inss, &map_mm_to_mattn);

        for(auto ins : iterator_for(m_attn))
        {
            if(ins->can_eval())
            {
                auto lit_s   = ins->get_shape();
                auto strides = lit_s.strides();
                if(strides.size() == 4 and
                   std::all_of(
                       strides.begin(), strides.end() - 1, [](auto s) { return s == 0; }) and
                   strides.back() == 1)
                {
                    auto new_lit = m_attn.add_literal(literal::from_raw_buffer(
                        shape{lit_s.type(), {lit_s.lens().back()}}, ins->eval().data()));
                    m_attn.replace_instruction(
                        ins, make_op("multibroadcast", {{"out_lens", lit_s.lens()}}), {new_lit});
                }
            }
        }
        dead_code_elimination{}.apply(m_attn);

        // Define outputs based on instructions that are used elsewhere in the graph
        auto required_outputs = find_required_outputs(attn_inss, gemm1, &mpm.get_module());
        if(required_outputs.empty())
            return;

        // Attention sinks leave a log/add lse chain hanging off the softmax
        // reductions (see find_attention_sinks); fuse it in and return
        // {output, lse} as a tuple.
        bool has_lse = false;
        if(has_lse_outputs(required_outputs, "reshape"))
        {
            auto lse_inss = get_lse_instructions(required_outputs);
            if(not lse_inss.empty())
            {
                m_attn.fuse(lse_inss, &map_mm_to_mattn);
                attn_inss.insert(attn_inss.end(), lse_inss.begin(), lse_inss.end());
                required_outputs = find_required_outputs(attn_inss, gemm1, &mpm.get_module());
                if(required_outputs.size() != 2)
                    return;
                has_lse = true;
            }
        }
        if(not has_lse)
            required_outputs = {required_outputs.back()};

        // Find corresponding output instructions in m_attn
        std::vector<instruction_ref> m_attn_outputs;
        std::transform(required_outputs.begin(),
                       required_outputs.end(),
                       std::back_inserter(m_attn_outputs),
                       [&](auto i) { return map_mm_to_mattn.at(i); });
        m_attn.add_return(m_attn_outputs);

        // Define inputs to m_attn
        auto map_mattn_to_mm = invert_map_ins(map_mm_to_mattn);
        auto new_inputs      = m_attn.get_inputs(map_mattn_to_mm);

        module_ref mpm_attn = mpm.create_module("attn" + get_count(), std::move(m_attn));
        mpm_attn->set_bypass();

        // Construct group op with the attention module, inserted before the
        // earliest output; any inputs positioned later are fixed up by the
        // module sort in fuse_attention::apply.
        auto& mm       = mpm.get_module();
        auto insert_pt = *std::min_element(
            required_outputs.begin(), required_outputs.end(), [&](auto x, auto y) {
                return std::distance(mm.begin(), x) < std::distance(mm.begin(), y);
            });
        auto group_ins = mm.insert_instruction(
            insert_pt, make_op("group", {{"tag", "kv_cache_attention"}}), new_inputs, {mpm_attn});

        if(m_attn_outputs.size() == 1)
        {
            mm.replace_instruction(required_outputs.front(), group_ins);
        }
        else
        {
            for(std::size_t i = 0; i < required_outputs.size(); ++i)
            {
                mm.replace_instruction(
                    required_outputs[i], make_op("get_tuple_elem", {{"index", i}}), group_ins);
            }
        }
    }
};

// The dot inputs, the pre-split scores, and the views that follow the second gemm.
struct kv_fd_operands
{
    instruction_ref q{};
    instruction_ref k{};
    instruction_ref v{};
    std::vector<std::size_t> scores_lens{};
    std::size_t sequence_length = 0;
    std::vector<operation> output_views{};
    std::unordered_set<instruction_ref> skip_views{};

    static std::optional<kv_fd_operands> match(module& submod)
    {
        auto return_ins = std::prev(submod.end());
        if(return_ins->name() != "@return" or return_ins->inputs().size() != 1)
            return std::nullopt;

        const auto dots = std::count_if(
            submod.begin(), submod.end(), [](const auto& ins) { return ins.name() == "dot"; });
        if(dots != 2)
            return std::nullopt;

        const auto [gemm1, gemm2] = get_gemms(&submod);
        kv_fd_operands result;
        result.q           = gemm1->inputs().at(0);
        result.k           = gemm1->inputs().at(1);
        result.v           = gemm2->inputs().at(1);
        result.scores_lens = gemm1->get_shape().lens();
        if(result.q->get_shape().dynamic() or result.k->get_shape().dynamic() or
           result.v->get_shape().dynamic())
            return std::nullopt;

        const auto rank = result.q->get_shape().ndim();
        if((rank != 3 and rank != 4) or result.k->get_shape().ndim() != rank or
           result.v->get_shape().ndim() != rank)
            return std::nullopt;

        result.sequence_length = result.k->get_shape().lens().back();
        if(result.sequence_length == 0 or result.sequence_length != result.scores_lens.back() or
           result.v->get_shape().lens()[rank - 2] != result.sequence_length)
            return std::nullopt;

        // An axis-pinned broadcast does not follow the score split, so its consumers
        // disagree with the split range literal. multibroadcast is rewritten instead.
        const bool pinned_broadcast =
            std::any_of(submod.begin(), submod.end(), [](const auto& ins) {
                return ins.name() == "broadcast";
            });
        if(pinned_broadcast)
            return std::nullopt;

        const auto returned   = return_ins->inputs().front();
        const auto output_end = find_kv_cache_attention::find_output_end(gemm2);
        if(output_end != returned)
            return std::nullopt;

        auto view = gemm2;
        while(view != output_end)
        {
            view = view->outputs().front();
            result.output_views.push_back(view->get_operator());
            result.skip_views.insert(view);
        }
        return result;
    }
};

// How a packed cache is viewed at a dot: split its sequence axis, then repeat heads.
// Grouped heads are unsqueeze, an optional transpose that swaps the last two axes,
// broadcast, and a reshape that merges the repeat into the head axis. Without the
// repeat, K is that transpose of the cache.
struct kv_cache_layout
{
    struct head_repeat
    {
        int64_t axis      = 0;
        std::size_t count = 1;
    };

    instruction_ref cache{};
    std::size_t sequence_axis = 0;
    std::vector<instruction_ref> old_views{};
    std::optional<head_repeat> repeat{};
    bool transpose_k = false;
};

static bool is_kv_cache_layout_op(const std::string& name)
{
    return name == "reshape" or name == "multibroadcast" or name == "transpose" or
           name == "unsqueeze";
}

static bool is_last_two_transpose(const operation& op)
{
    const auto perm = op.to_value().at("permutation").to_vector<int64_t>();
    if(perm.size() < 2)
        return false;
    std::vector<int64_t> expect(perm.size());
    std::iota(expect.begin(), expect.end(), 0);
    std::swap(expect[expect.size() - 2], expect.back());
    return perm == expect;
}

// Fold the repeated-head axis into the axis before it.
static std::vector<std::size_t> merge_repeat_dims(const std::vector<std::size_t>& lens,
                                                  std::size_t axis)
{
    std::vector<std::size_t> dims(lens.begin(), lens.begin() + axis - 1);
    dims.push_back(lens[axis - 1] * lens[axis]);
    dims.insert(dims.end(), lens.begin() + axis + 1, lens.end());
    return dims;
}

static instruction_ref insert_last_two_transpose(module& mm, instruction_ref input)
{
    const auto rank = input->get_shape().ndim();
    std::vector<int64_t> perm(rank);
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[rank - 2], perm.back());
    return mm.insert_instruction(mm.end(), make_op("transpose", {{"permutation", perm}}), input);
}

static std::optional<kv_cache_layout> match_kv_cache_layout(instruction_ref dot_input,
                                                            std::size_t sequence_length)
{
    kv_cache_layout layout;
    auto cur = dot_input;
    while(cur->inputs().size() == 1 and cur->outputs().size() == 1 and
          is_kv_cache_layout_op(cur->name()))
    {
        layout.old_views.push_back(cur);
        cur = cur->inputs().front();
    }
    if(layout.old_views.empty() or cur->name() != "@param" or not cur->get_shape().standard())
        return std::nullopt;

    layout.cache     = cur;
    const auto& lens = layout.cache->get_shape().lens();
    if(std::count(lens.begin(), lens.end(), sequence_length) != 1)
        return std::nullopt;

    layout.sequence_axis = static_cast<std::size_t>(
        std::distance(lens.begin(), std::find(lens.begin(), lens.end(), sequence_length)));

    auto it         = layout.old_views.begin();
    const auto last = layout.old_views.end();
    std::vector<std::size_t> repeat_lens;
    if(it != last and (*it)->name() == "reshape")
    {
        ++it;
        if(it == last or (*it)->name() != "multibroadcast")
            return std::nullopt;
        repeat_lens   = (*it)->get_operator().to_value().at("out_lens").to_vector<std::size_t>();
        layout.repeat = kv_cache_layout::head_repeat{};
        ++it;
    }
    if(it != last and (*it)->name() == "transpose")
    {
        if(not is_last_two_transpose((*it)->get_operator()))
            return std::nullopt;
        layout.transpose_k = true;
        ++it;
    }
    if(it != last and (*it)->name() == "unsqueeze")
    {
        const auto axes = (*it)->get_operator().to_value().at("axes").to_vector<int64_t>();
        if(axes.size() != 1 or axes.front() <= 0)
            return std::nullopt;
        if(layout.repeat.has_value())
            layout.repeat->axis = axes.front();
        ++it;
    }
    if(it != last)
        return std::nullopt;

    if(not layout.repeat.has_value())
    {
        if(not layout.transpose_k)
            return std::nullopt;
        return layout;
    }

    if(layout.repeat->axis <= 0)
        return std::nullopt;
    const auto axis = static_cast<std::size_t>(layout.repeat->axis);
    if(axis >= repeat_lens.size())
        return std::nullopt;
    const auto merged =
        layout.old_views.front()->get_operator().to_value().at("dims").to_vector<std::size_t>();
    if(merged != merge_repeat_dims(repeat_lens, axis))
        return std::nullopt;
    if(layout.transpose_k and axis + 2 >= repeat_lens.size())
        return std::nullopt;

    layout.repeat->count = repeat_lens[axis];
    return layout;
}

// Split the packed cache, then rebuild the head repeat as a broadcast. Returns false
// without aliasing the old views when the built lens are not the expected split shape;
// the inserted ops are left for dead-code elimination.
static bool prebuild_kv_cache_split(module& mm,
                                    std::unordered_map<instruction_ref, instruction_ref>& params,
                                    const kv_cache_layout& layout,
                                    std::size_t groups,
                                    const std::vector<std::size_t>& expected)
{
    if(not contains(params, layout.cache))
        return false;

    const auto& lens = params.at(layout.cache)->get_shape().lens();
    std::vector<std::size_t> dims(lens.begin(), lens.begin() + layout.sequence_axis);
    dims.push_back(groups);
    dims.push_back(lens[layout.sequence_axis] / groups);
    dims.insert(dims.end(), lens.begin() + layout.sequence_axis + 1, lens.end());
    auto cur = mm.insert_instruction(
        mm.end(), make_op("reshape", {{"dims", dims}}), params.at(layout.cache));
    if(layout.repeat.has_value())
    {
        auto axis = layout.repeat->axis;
        if(axis > static_cast<int64_t>(layout.sequence_axis))
            ++axis;
        cur = mm.insert_instruction(mm.end(), make_op("unsqueeze", {{"axes", {axis}}}), cur);
        if(layout.transpose_k)
            cur = insert_last_two_transpose(mm, cur);
        auto out                            = cur->get_shape().lens();
        out[static_cast<std::size_t>(axis)] = layout.repeat->count;
        cur = mm.insert_instruction(mm.end(), make_op("multibroadcast", {{"out_lens", out}}), cur);
        cur = mm.insert_instruction(
            mm.end(), make_op("reshape", {{"dims", merge_repeat_dims(out, axis)}}), cur);
    }
    else if(layout.transpose_k)
    {
        cur = insert_last_two_transpose(mm, cur);
    }
    if(cur->get_shape().lens() != expected)
        return false;
    for(auto ins : layout.old_views)
        params[ins] = cur;
    return true;
}

// Generic split ops for one dot input. A parameter is rewritten immediately and any other
// instruction is wrapped when the softmax body is copied
struct dot_input_split
{
    instruction_ref input{};
    std::vector<std::size_t> expected{};
    std::vector<operation> ops{};

    void place(module& mm,
               std::unordered_map<instruction_ref, instruction_ref>& params,
               std::unordered_map<instruction_ref, std::vector<operation>>& splits,
               std::size_t sequence_length,
               std::size_t groups) const
    {
        const auto layout = match_kv_cache_layout(input, sequence_length);
        if(layout.has_value() and prebuild_kv_cache_split(mm, params, *layout, groups, expected))
            return;
        if(input->name() == "@param")
            params[input] = insert_ops(mm, mm.end(), ops, params.at(input));
        else
            splits.emplace(input, ops);
    }
};

struct find_kv_cache_flash_decoding
{
    flash_decoding_config config{};

    auto matcher() const
    {
        return match::name("group")(match::has_op_value("tag", "kv_cache_attention")).bind("group");
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto attn_group_ins = r.instructions["group"];
        auto* submod        = attn_group_ins->module_inputs().front();
        auto operands       = kv_fd_operands::match(*submod);
        if(not operands.has_value())
            return;

        const auto actual_groups = config.num_groups(operands->sequence_length);
        // TODO: support uneven splits by padding K, V, and score-shaped inputs
        if(actual_groups <= 1 or operands->sequence_length % actual_groups != 0)
            return;

        const flash_decoding_split decoding{actual_groups, operands->scores_lens};
        const auto transform = decoding.transform_qkv(
            operands->q->get_shape(), operands->k->get_shape(), operands->v->get_shape());

        module m_flash_decode;
        m_flash_decode.set_bypass();
        std::unordered_map<instruction_ref, instruction_ref> map_old_params_to_new;
        for(auto param : submod->get_parameters())
        {
            const auto& name = any_cast<builtin::param>(param->get_operator()).parameter;
            map_old_params_to_new[param] = m_flash_decode.add_parameter(name, param->get_shape());
        }
        // A packed cache is split on its sequence axis before the head repeat.
        // Q, and a K or V that is not that cache view, keep the generic split ops.
        std::unordered_map<instruction_ref, std::vector<operation>> qkv_splits;
        for(const auto& input : {dot_input_split{operands->q, transform.q_shape, transform.q_ops},
                                 dot_input_split{operands->k, transform.k_shape, transform.k_ops},
                                 dot_input_split{operands->v, transform.v_shape, transform.v_ops}})
            input.place(m_flash_decode,
                        map_old_params_to_new,
                        qkv_splits,
                        operands->sequence_length,
                        actual_groups);

        if(not decoding.rebuild(m_flash_decode,
                                *submod,
                                map_old_params_to_new,
                                qkv_splits.empty() ? nullptr : &qkv_splits,
                                operands->skip_views.empty() ? nullptr : &operands->skip_views))
            return;

        auto* flash_mod =
            mpm.create_module(submod->name() + "_kv_flash_decoding", std::move(m_flash_decode));
        flash_mod->set_bypass();

        auto& mm       = mpm.get_module();
        auto new_group = mm.insert_instruction(attn_group_ins,
                                               make_op("group", {{"tag", "kv_cache_attention"}}),
                                               attn_group_ins->inputs(),
                                               {flash_mod});
        auto o_prime   = mm.insert_instruction(
            attn_group_ins, make_op("get_tuple_elem", {{"index", 0}}), new_group);
        auto lse = mm.insert_instruction(
            attn_group_ins, make_op("get_tuple_elem", {{"index", 1}}), new_group);
        auto combined = decoding.combine(
            mm, attn_group_ins, o_prime, lse, transform.g_axis, operands->output_views);
        mm.replace_instruction(attn_group_ins, combined);
    }
};

} // namespace

void fuse_attention::apply(module_pass_manager& mpm) const
{
    std::size_t counter = 0;

    // Canonicalize attention sinks into a standard softmax with an lse-based
    // output correction so the matchers below can fuse the attention block
    match::find_matches(mpm.get_module(), find_attention_sinks{});
    mpm.run_pass(dead_code_elimination{});

    // Fuse kv-cache attention by default
    match::find_matches(mpm, find_kv_cache_attention{.counter = &counter});
    mpm.get_module().sort();
    mpm.run_pass(dead_code_elimination{});

    // Only fuse plain attention when requested
    if(attn_enabled)
    {
        // remove quantization from attention blocks so they can be fused; rocMLIR currently does
        // not support fp8 attention
        match::find_matches(mpm, find_quant_attention{});
        mpm.run_pass(dead_code_elimination{});

        match::find_matches(mpm, find_transposed_attention{});
        mpm.run_pass(dead_code_elimination{});

        match::find_matches(mpm, find_attention{.counter = &counter});
        mpm.get_module().sort();
        mpm.run_pass(dead_code_elimination{});
    }

    // enable flash decoding if splits configured or flash decoding is enabled
    std::size_t configured_splits = get_num_splits(flash_decoding_num_splits);
    if(configured_splits > 0 or flash_decoding_enabled)
    {
        const flash_decoding_config config{flash_decoding_num_splits,
                                           flash_decoding_threshold,
                                           flash_decoding_max_splits,
                                           flash_decoding_min_chunk_size};
        match::find_matches(mpm, find_flash_decoding{config});
        mpm.run_pass(dead_code_elimination{});
        match::find_matches(mpm, find_kv_cache_flash_decoding{config});
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
