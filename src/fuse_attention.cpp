/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/match/softmax.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/generic_float.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/split_factor.hpp>
#include <queue>
#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

// env vars for flash decoding configuration
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_FLASH_DECODING_ENABLED);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_FLASH_DECODING_NUM_SPLITS);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_FLASH_DECODING_MIN_CHUNK_SIZE);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_FLASH_DECODING_MAX_SPLITS);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_FLASH_DECODING_THRESHOLD);

bool is_flash_decoding_enabled() { return enabled(MIGRAPHX_FLASH_DECODING_ENABLED{}); }

// Get num_splits with priority: struct member > env var > 0 (not set)
std::size_t get_num_splits(const std::optional<std::size_t>& member_num_splits)
{
    // struct member var is used for testing
    if(member_num_splits.has_value())
    {
        return *member_num_splits;
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
inline std::size_t calculate_groups(std::size_t groups, std::size_t sequence_length, std::size_t threshold)
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

        // TODO: run experiments to find the optimal values for min_chunk and max_splits
        std::size_t min_chunk  = value_of(MIGRAPHX_FLASH_DECODING_MIN_CHUNK_SIZE{}, 32);
        std::size_t max_splits = value_of(MIGRAPHX_FLASH_DECODING_MAX_SPLITS{}, 16);

        std::size_t actual_groups = calculate_flash_decoding_splits(sequence_length, min_chunk, max_splits);

        // return 0 if auto-calculation determines no splitting needed
        if(actual_groups <= 1)
            return 0;

        return actual_groups;
    }

    // groups == 1 or invalid, no splitting
    return 0;
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

struct find_attention
{
    std::size_t* counter;

    auto matcher() const
    {
        auto gemm1   = match::any_of[pointwise_inputs()](match::name("dot").bind("dot1"));
        auto softmax = match::skip(match::name("convert"))(match::softmax_input(gemm1));
        return match::name("dot")(match::arg(0)(softmax));
    }

    std::string get_count() const { return std::to_string((*counter)++); }

    std::unordered_map<instruction_ref, instruction_ref>
    invert_map_ins(const std::unordered_map<instruction_ref, instruction_ref>& map_ins) const
    {
        std::unordered_map<instruction_ref, instruction_ref> inverse_map;
        for(auto const& [key, value] : map_ins)
        {
            assert(not contains(inverse_map, value));
            inverse_map[value] = key;
        }
        return inverse_map;
    }

    std::vector<instruction_ref>
    get_attn_instructions(module& m, instruction_ref gemm1, instruction_ref gemm2) const
    {
        auto attn_inss = find_instructions_between(gemm1, gemm2, &m);

        std::vector<instruction_ref> sorted_inss(attn_inss.begin(), attn_inss.end());
        std::sort(
            sorted_inss.begin(), sorted_inss.end(), [&](instruction_ref x, instruction_ref y) {
                return std::distance(m.begin(), x) < std::distance(m.begin(), y);
            });

        return sorted_inss;
    }

    static bool has_lse_out(std::vector<instruction_ref>& group_outs)
    {
        return (group_outs.size() == 3 and
                std::all_of(group_outs.begin(), group_outs.end(), [](auto o) {
                    return contains({"dot", "reduce_max", "reduce_sum"}, o->name());
                }));
    }

    std::vector<instruction_ref>
    get_lse_instructions(std::vector<instruction_ref>& group_outs) const
    {
        std::vector<instruction_ref> lse_inss;
        auto rsum = *std::find_if(
            group_outs.begin(), group_outs.end(), [](auto o) { return o->name() == "reduce_sum"; });
        auto rsum_outs = rsum->outputs();
        auto log       = std::find_if(
            rsum_outs.begin(), rsum_outs.end(), [](auto o) { return o->name() == "log"; });
        if(log == rsum_outs.end())
            return lse_inss;
        auto log_outs = (*log)->outputs();
        if(log_outs.size() != 1 or log_outs.front()->name() != "add")
            return lse_inss;
        auto add = log_outs.front();
        lse_inss.insert(lse_inss.end(), {*log, add});

        return lse_inss;
    }

    std::vector<instruction_ref> find_outputs(std::vector<instruction_ref> inss) const
    {
        std::vector<instruction_ref> outputs;
        std::copy_if(inss.begin(), inss.end(), std::back_inserter(outputs), [&](auto i) {
            return not std::all_of(i->outputs().begin(), i->outputs().end(), [&](auto o) {
                return contains(inss, o);
            });
        });
        return outputs;
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

        // Define outputs based on instructions that are used elsewhere in the graph
        auto required_outputs = find_outputs(attn_inss);

        assert(not required_outputs.empty());

        // LSE case requires output from reduce_max and reduce_sum instructions
        if(has_lse_out(required_outputs))
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
    // optional number of splits from fuse_attention pass config
    std::optional<std::size_t> configured_splits;

    auto matcher() const
    {
        return match::name("group")(match::has_op_value("tag", "attention")).bind("group");
    }

    std::pair<instruction_ref, instruction_ref> get_gemms(module_ref submod) const
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

    std::vector<shape> get_qkv_shapes(instruction_ref q, instruction_ref k, instruction_ref v) const
    {
        std::vector<shape> qkv_shapes;
        auto q_shape = q->get_shape();
        auto k_shape = k->get_shape();
        auto v_shape = v->get_shape();

        qkv_shapes.push_back(q_shape);
        qkv_shapes.push_back(k_shape);
        qkv_shapes.push_back(v_shape);

        assert((q_shape.lens().size() == 3 or q_shape.lens().size() == 4) and
               "Expected 3D or 4D Q, K, V shapes");
        assert(k_shape.lens().size() == q_shape.lens().size() and
               v_shape.lens().size() == q_shape.lens().size() and
               "Q, K, V must have same number of dimensions");
        return qkv_shapes;
    }

    struct transformed_shapes_result
    {
        std::vector<size_t> q_shape;           // final Q shape: [B, G, M, k]
        std::vector<size_t> k_intermediate;    // K intermediate: [B, k, G, N/G]
        std::vector<size_t> k_shape;           // final K shape: [B, G, k, N/G]
        std::vector<int64_t> k_transpose_perm; // permutation for K transpose
        std::vector<size_t> v_shape;           // final V shape: [B, G, N/G, D]
    };

    transformed_shapes_result get_transformed_shapes(const std::vector<shape>& input_shapes,
                                                     std::size_t num_groups) const
    {
        assert(input_shapes.size() == 3 and "Expected Q, K, V shapes");

        auto q_lens = input_shapes[0].lens();
        auto k_lens = input_shapes[1].lens();
        auto v_lens = input_shapes[2].lens();

        // 3D: Q_lens = [B, M, k]
        // 4D: Q_lens = [B, H, M, k]
        size_t ndim = q_lens.size();
        size_t n    = k_lens[ndim - 1];
        size_t g    = num_groups;

        // Note: sequence length may have been padded to be divisible by num_groups
        assert(n % g == 0 and "Key-value sequence length must be divisible by number of "
                              "splits/groups (after padding)");
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
        result.q_shape = insert_g(q_lens);

        // K: [B, k, N] -> [B, G, k, N/G] via reshape + transpose
        // intermediate shape for reshape: [B, k, G, N/G]
        result.k_intermediate.clear();
        for(size_t i = 0; i < k_lens.size() - 1; ++i)
        {
            result.k_intermediate.push_back(k_lens[i]);
        }
        result.k_intermediate.push_back(g);
        result.k_intermediate.push_back(n_split);

        // transpose permutation to get [B, G, k, N/G]
        result.k_transpose_perm.clear();
        for(size_t i = 0; i < k_lens.size() - 2; ++i)
        {
            result.k_transpose_perm.push_back(i); // batch dims stay in place
        }
        result.k_transpose_perm.push_back(k_lens.size() - 1); // G dimension
        result.k_transpose_perm.push_back(k_lens.size() - 2); // k dimension
        result.k_transpose_perm.push_back(k_lens.size());     // N/G dimension

        // final K shape after transpose
        result.k_shape                            = insert_g(k_lens);
        result.k_shape[result.k_shape.size() - 1] = n_split;

        // V: [B, N, D] -> [B, G, N/G, D] via direct reshape
        result.v_shape                            = insert_g(v_lens);
        result.v_shape[result.v_shape.size() - 2] = n_split;

        return result;
    }

    std::unordered_map<instruction_ref, instruction_ref>
    map_submod_params_to_inputs(module_ref submod,
                                const std::vector<instruction_ref>& group_inputs) const
    {
        auto map_param_to_main = submod->get_ins_param_map(group_inputs, true);
        // verify the mapping is correct
        auto expected_inputs = submod->get_inputs(map_param_to_main);
        assert(expected_inputs == group_inputs and "Mapped inputs don't match group inputs");
        return map_param_to_main;
    }

    void rebuild_attention_submodule(
        module& target_mod,
        const module& source_mod,
        const std::unordered_map<instruction_ref, instruction_ref>& param_map) const
    {
        // map from instructions in the old module to the new ones in the target module
        std::unordered_map<instruction_ref, instruction_ref> map_old_to_new = param_map;
        std::unordered_map<std::string, instruction_ref> softmax_parts;

        for(auto it = source_mod.begin(); it != source_mod.end(); ++it)
        {
            auto ins = it;
            if(ins->name() == "@param" or ins->name() == "@return")
                continue;

            // gather inputs for the new instruction
            std::vector<instruction_ref> new_inputs;
            std::transform(ins->inputs().begin(),
                           ins->inputs().end(),
                           std::back_inserter(new_inputs),
                           [&](auto i) {
                               assert(contains(map_old_to_new, i) and "Input not found in map");
                               return map_old_to_new.at(i);
                           });

            auto op = ins->get_operator();

            // transform operators that depend on tensor shape/rank
            // adjust reduction axes for the new rank
            if(op.name() == "reduce_max" or op.name() == "reduce_sum")
            {
                auto original_axes = op.to_value()["axes"].to_vector<int64_t>();
                assert(original_axes.size() == 1 and "Expected single axis for reduction");

                const auto& new_input_shape = new_inputs.front()->get_shape();
                assert(original_axes.front() ==
                           static_cast<int64_t>(ins->inputs().front()->get_shape().lens().size() -
                                                1) or
                       original_axes.front() == -1);
                op.from_value(
                    {{"axes", {static_cast<int64_t>(new_input_shape.lens().size() - 1)}}});
            }
            // TODO make less reliant on ops around it
            else if(op.name() == "multibroadcast")
            {
                // broadcast target shape is the shape of the
                // other input to the 'sub' or 'div' instruction.
                auto parent = ins->outputs().front();
                assert(parent->name() == "sub" or parent->name() == "div");

                // Find the sibling input that isn't the reduction result
                auto sibling = std::find_if(parent->inputs().begin(),
                                            parent->inputs().end(),
                                            [&](auto i) { return i != ins; });
                assert(sibling != parent->inputs().end() and
                       "Could not find sibling for broadcast target");

                const auto& target_shape = map_old_to_new.at(*sibling)->get_shape();
                op.from_value({{"out_lens", target_shape.lens()}});
            }

            auto new_ins        = target_mod.add_instruction(op, new_inputs);
            map_old_to_new[ins] = new_ins;

            // store key softmax components for LSE calculation
            if(op.name() == "reduce_max")
                softmax_parts["max"] = new_ins;
            if(op.name() == "reduce_sum")
                softmax_parts["sum_exp"] = new_ins;
        }

        // get the final partial output (O')
        auto orig_return_ins        = std::prev(source_mod.end())->inputs().front();
        auto partial_output_o_prime = map_old_to_new.at(orig_return_ins);

        // calculate LSE = max(S) + log(sum(exp(S - max(S))))
        assert(contains(softmax_parts, "max") and contains(softmax_parts, "sum_exp"));
        auto log_sum_exp = target_mod.add_instruction(make_op("log"), softmax_parts["sum_exp"]);
        auto lse = target_mod.add_instruction(make_op("add"), softmax_parts["max"], log_sum_exp);

        // return a tuple of {O', LSE}
        target_mod.add_return({partial_output_o_prime, lse});
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

        // TODO: for this first pass of flash decoding, assuming no input fusion / not supporting
        auto q_param = gemm1->inputs()[0];
        auto k_param = gemm1->inputs()[1];
        auto v_param = gemm2->inputs()[1];
        assert(q_param->name() == "@param" and "Q should be a parameter");
        assert(k_param->name() == "@param" and "K should be a parameter");
        assert(v_param->name() == "@param" and "V should be a parameter");

        // Get sequence length from K shape
        auto k_shape                = k_param->get_shape();
        std::size_t sequence_length = k_shape.lens().back();

        // read groups configuration from pass config or environment variable
        std::size_t groups = get_num_splits(configured_splits);
        std::size_t threshold = value_of(MIGRAPHX_FLASH_DECODING_THRESHOLD{}, 32);

        std::size_t actual_groups = calculate_groups(groups, sequence_length, threshold);
        if(actual_groups == 0)
            return;

        // calculate padding if sequence length not evenly divisible
        std::size_t padding_needed = 0;
        if(sequence_length % actual_groups != 0)
        {
            // round up to nearest multiple of actual_groups
            padding_needed            = ceil_mul_of(sequence_length, actual_groups) - sequence_length;
        }

        // create mapping from submodule params to main module inputs
        auto group_inputs      = attn_group_ins->inputs();
        auto map_param_to_main = map_submod_params_to_inputs(submod, group_inputs);

        // get actual Q, K, V instructions from main module
        auto q = map_param_to_main.at(q_param);
        auto k = map_param_to_main.at(k_param);
        auto v = map_param_to_main.at(v_param);

        // pad K and V if necessary
        if(padding_needed > 0)
        {
            // K shape: [B, k, N] or [B, H, k, N] for 4D. Padding on N
            auto k_ndim = k->get_shape().ndim();
            std::vector<std::size_t> k_pads(2 * k_ndim, 0);
            k_pads[k_ndim + k_ndim - 1] = padding_needed; // pad right on last dim
            k = mm.insert_instruction(attn_group_ins, make_op("pad", {{"pads", k_pads}}), k);

            // V shape: [B, N, D] or [B, H, N, D] for 4D
            auto v_ndim = v->get_shape().ndim();
            std::vector<std::size_t> v_pads(2 * v_ndim, 0);
            v_pads[v_ndim + v_ndim - 2] = padding_needed; // pad right on N dim
            v = mm.insert_instruction(attn_group_ins, make_op("pad", {{"pads", v_pads}}), v);
        }

        // get Q, K, V shapes (using potentially padded K and V)
        auto qkv_shapes = get_qkv_shapes(q, k, v);

        // check shapes are ok and get flash decoding transformed shapes (Q', V', K')
        auto transform_info = get_transformed_shapes(qkv_shapes, actual_groups);

        // insert reshape operations before group, for Q, K, V
        auto q_ndim    = q->get_shape().lens().size();
        int64_t g_axis = q_ndim - 2;

        // Q: [B, M, k] -> [B, G, M, k] via unsqueeze + broadcast
        auto q_unsqueeze =
            mm.insert_instruction(attn_group_ins, make_op("unsqueeze", {{"axes", {g_axis}}}), q);
        auto q_reshaped =
            mm.insert_instruction(attn_group_ins,
                                  make_op("multibroadcast", {{"out_lens", transform_info.q_shape}}),
                                  q_unsqueeze);

        // K: [B, k, N] -> [B, G, k, N/G] via reshape + transpose
        auto k_reshaped_intermediate = mm.insert_instruction(
            attn_group_ins, make_op("reshape", {{"dims", transform_info.k_intermediate}}), k);
        auto k_reshaped = mm.insert_instruction(
            attn_group_ins,
            make_op("transpose", {{"permutation", transform_info.k_transpose_perm}}),
            k_reshaped_intermediate);

        // V: [B, N, D] -> [B, G, N/G, D] via direct reshape
        auto v_reshaped = mm.insert_instruction(
            attn_group_ins, make_op("reshape", {{"dims", transform_info.v_shape}}), v);

        // create new input list by replacing Q, K, V with reshaped versions
        std::vector<instruction_ref> new_group_inputs = group_inputs;
        for(size_t i = 0; i < group_inputs.size(); ++i)
        {
            if(group_inputs[i] == q)
            {
                new_group_inputs[i] = q_reshaped;
            }
            else if(group_inputs[i] == k)
            {
                new_group_inputs[i] = k_reshaped;
            }
            else if(group_inputs[i] == v)
            {
                new_group_inputs[i] = v_reshaped;
            }
        }

        // create new submodule for flash decoding
        module m_flash_decode;
        m_flash_decode.set_bypass();

        // get parameter names
        auto q_name = q_param->get_operator().to_value()["parameter"].to<std::string>();
        auto k_name = k_param->get_operator().to_value()["parameter"].to<std::string>();
        auto v_name = v_param->get_operator().to_value()["parameter"].to<std::string>();

        // new params added first
        auto new_q_param = m_flash_decode.add_parameter(
            q_name, shape{qkv_shapes[0].type(), transform_info.q_shape});
        auto new_k_param = m_flash_decode.add_parameter(
            k_name, shape{qkv_shapes[1].type(), transform_info.k_shape});
        auto new_v_param = m_flash_decode.add_parameter(
            v_name, shape{qkv_shapes[2].type(), transform_info.v_shape});

        // build mapping for old params -> new params
        std::unordered_map<instruction_ref, instruction_ref> map_old_params_to_new;
        map_old_params_to_new[q_param] = new_q_param;
        map_old_params_to_new[k_param] = new_k_param;
        map_old_params_to_new[v_param] = new_v_param;

        // don't simply fuse previous attn submod, need to rebuild all the ops
        rebuild_attention_submodule(m_flash_decode, *submod, map_old_params_to_new);

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

        // kernel 2
        // the partial outputs O'[g] are already weighted by their group's softmax,
        // LSE[g] contains log(sum(exp(S[g]))) for each group
        // To combine: weight by exp(LSE[g]) / sum_g(exp(LSE[g']))

        // compute global max for numerical stability
        auto lse_max =
            mm.insert_instruction(attn_group_ins, make_op("reduce_max", {{"axes", {g_axis}}}), lse);
        auto lse_max_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", lse->get_shape().lens()}}),
            lse_max);

        // exp(LSE - max_LSE)
        auto lse_sub = mm.insert_instruction(attn_group_ins, make_op("sub"), lse, lse_max_bcast);
        auto lse_exp = mm.insert_instruction(attn_group_ins, make_op("exp"), lse_sub);

        // sum across groups
        auto lse_sum = mm.insert_instruction(
            attn_group_ins, make_op("reduce_sum", {{"axes", {g_axis}}}), lse_exp);
        auto lse_sum_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", lse_exp->get_shape().lens()}}),
            lse_sum);

        // scale factor: exp(LSE[g] - max_LSE) / sum(exp(LSE - max_LSE))
        auto scale = mm.insert_instruction(attn_group_ins, make_op("div"), lse_exp, lse_sum_bcast);

        auto scale_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", partial_output_o_prime->get_shape().lens()}}),
            scale);

        // convert scale to match the type of partial_output_o_prime
        auto output_type     = partial_output_o_prime->get_shape().type();
        auto scale_converted = mm.insert_instruction(
            attn_group_ins, make_op("convert", {{"target_type", output_type}}), scale_bcast);

        // R = mul(O', broadcasted_scale)
        auto scaled_r = mm.insert_instruction(
            attn_group_ins, make_op("mul"), partial_output_o_prime, scale_converted);

        // O = sum(R, axis=G_axis)
        auto final_output_o = mm.insert_instruction(
            attn_group_ins, make_op("reduce_sum", {{"axes", {g_axis}}}), scaled_r);

        // squeeze G to match the original output shape
        auto final_squeezed_o = mm.insert_instruction(
            attn_group_ins, make_op("squeeze", {{"axes", {g_axis}}}), final_output_o);

        // replace the original group instruction with the final result
        mm.replace_instruction(attn_group_ins, final_squeezed_o);
    }
};

struct find_kv_cache_attention
{
    std::size_t* counter;

    auto matcher() const
    {
        static const std::unordered_set<std::string> skip_set = {
            "multibroadcast", "reshape", "unsqueeze"};

        auto keys =
            match::skip(match::name(skip_set))(match::name("concat_past_present")).bind("pres_k");
        auto k_transpose =
            match::skip(match::name(skip_set))(match::name("transpose")(match::arg(0)(keys)));
        auto queries = match::name("slice");
        auto gemm1   = match::name("dot")(match::arg(0)(queries), match::arg(1)(k_transpose));
        auto scale   = match::name("mul")(match::any_arg(0, 1)(gemm1));
        auto broadcasted_const = match::name("multibroadcast")(match::arg(0)(match::is_constant()));
        auto attn_scores       = match::any_of(scale, gemm1);
        auto causal_mask =
            match::name("where")(match::arg(0)(broadcasted_const), match::arg(2)(attn_scores));
        auto greater = match::name("greater")(match::arg(1)(match::any().bind("total_sl")));
        auto conv_greater =
            match::skip(match::name("unsqueeze"))(match::name("convert")(match::arg(0)(greater)));
        auto bc_greater         = match::name("multibroadcast")(match::arg(0)(conv_greater));
        auto mask               = match::name("where")(match::arg(0)(bc_greater),
                                         match::arg(2)(match::any_of(causal_mask, scale, gemm1)));
        auto attn_probabilities = match::skip(match::name("convert"))(
            match::softmax_input(match::skip(match::name("convert"))(mask)));
        auto values =
            match::skip(match::name(skip_set))(match::name("concat_past_present")).bind("pres_v");
        auto gemm2 = match::name("dot")(match::arg(0)(attn_probabilities), match::arg(1)(values));
        auto transpose_out = match::name("transpose")(match::arg(0)(gemm2));
        return match::name("reshape")(match::arg(0)(transpose_out));
    }

    std::string get_count() const { return std::to_string((*counter)++); }

    std::unordered_map<instruction_ref, instruction_ref>
    invert_map_ins(const std::unordered_map<instruction_ref, instruction_ref>& map_ins) const
    {
        std::unordered_map<instruction_ref, instruction_ref> inverse_map;
        for(auto const& [key, value] : map_ins)
        {
            assert(not contains(inverse_map, value));
            inverse_map[value] = key;
        }
        return inverse_map;
    }

    std::vector<instruction_ref>
    get_attn_instructions(module& m, instruction_ref start, instruction_ref end) const
    {
        std::queue<instruction_ref> inputs;
        std::unordered_set<instruction_ref> inss;
        inputs.push(end);

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
                                                                       "broadcast",
                                                                       "multibroadcast",
                                                                       "@literal",
                                                                       "unsqueeze"};

        auto is_valid_attn_op = [&](auto i) {
            return i->get_operator().attributes().get("pointwise", false) or
                   contains(valid_attn_ops, i->get_operator().name()) or i == start or i == end;
        };

        while(not inputs.empty())
        {
            auto current_inp = inputs.front();
            inputs.pop();

            if(is_valid_attn_op(current_inp) and inss.insert(current_inp).second and
               current_inp != start)
            {
                for(auto i : current_inp->inputs())
                {
                    inputs.push(i);
                }
            }
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
        auto reshape  = r.result;

        // Capture all instructions part of the attention op
        auto attn_inss = get_attn_instructions(mpm.get_module(), total_sl, reshape);

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
                    auto new_lit = m_attn.add_literal(
                        literal{shape{lit_s.type(), {lit_s.lens().back()}}, ins->eval().data()});
                    m_attn.replace_instruction(
                        ins, make_op("multibroadcast", {{"out_lens", lit_s.lens()}}), {new_lit});
                }
            }
        }
        dead_code_elimination{}.apply(m_attn);

        // Define outputs based on instructions that are used elsewhere in the graph
        std::vector<instruction_ref> required_outputs;
        std::copy_if(
            attn_inss.begin(), attn_inss.end(), std::back_inserter(required_outputs), [&](auto i) {
                return not std::all_of(i->outputs().begin(), i->outputs().end(), [&](auto o) {
                    return contains(attn_inss, o);
                });
            });

        assert(not required_outputs.empty());

        // Find corresponding output instructions in m_attn
        std::vector<instruction_ref> m_attn_outputs;
        std::transform(required_outputs.begin(),
                       required_outputs.end(),
                       std::back_inserter(m_attn_outputs),
                       [&](auto i) { return map_mm_to_mattn.at(i); });
        m_attn.add_return({m_attn_outputs.back()});

        // Define inputs to m_attn
        auto map_mattn_to_mm = invert_map_ins(map_mm_to_mattn);
        auto new_inputs      = m_attn.get_inputs(map_mattn_to_mm);

        module_ref mpm_attn = mpm.create_module("attn" + get_count(), std::move(m_attn));
        mpm_attn->set_bypass();

        // Construct group op with the attention module
        auto group_ins =
            mpm.get_module().insert_instruction(required_outputs.back(),
                                                make_op("group", {{"tag", "kv_cache_attention"}}),
                                                new_inputs,
                                                {mpm_attn});

        mpm.get_module().replace_instruction(required_outputs.back(), group_ins);
    }
};

} // namespace

void fuse_attention::apply(module_pass_manager& mpm) const
{
    std::size_t counter = 0;

    // Fuse kv-cache attention by default
    match::find_matches(mpm, find_kv_cache_attention{.counter = &counter});
    mpm.get_module().sort();
    mpm.run_pass(dead_code_elimination{});

    // Only fuse plain attention when requested
    if(attn_enabled)
    {
        match::find_matches(mpm, find_attention{.counter = &counter});
        mpm.get_module().sort();
        mpm.run_pass(dead_code_elimination{});
    }

    // enable flash decoding if splits configured or explicitly enabled
    std::size_t configured_splits = get_num_splits(flash_decoding_num_splits);
    if(configured_splits > 0 or is_flash_decoding_enabled())
    {
        match::find_matches(mpm, find_flash_decoding{.configured_splits = flash_decoding_num_splits});
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
