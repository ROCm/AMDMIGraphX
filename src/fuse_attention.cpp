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

struct find_gqa_flash_decoding
{
    std::size_t groups;
    
    // Struct to hold all attention dimensions
    struct attention_dims
    {
        std::size_t batch_size;
        std::size_t num_heads;            // Q heads
        std::size_t kv_heads;             // K and V heads
        std::size_t concat_heads;         // total heads in QKV tensor
        std::size_t sequence_length;
        std::size_t max_seq_length;
        std::size_t head_dim;
        std::size_t seq_length_per_group; // sequence length per group after splitting max sequence length
        
        // constructor from parameters
        attention_dims(instruction_ref q_param, instruction_ref k_param, std::size_t num_groups)
        {
            auto q_shape = q_param->get_shape();
            auto k_shape = k_param->get_shape();
            
            batch_size = q_shape.lens()[0];
            concat_heads = q_shape.lens()[1];
            sequence_length = q_shape.lens()[2];
            head_dim = q_shape.lens()[3];
            
            kv_heads = k_shape.lens()[1];
            max_seq_length = k_shape.lens()[2];
            
            // calculate Q heads from concat_heads = num_heads + 2 * kv_heads
            num_heads = concat_heads - 2 * kv_heads;
            
            // calculate sequence length per group
            if(max_seq_length % num_groups != 0) {
                std::cout << "Max sequence length " << max_seq_length 
                          << " not divisible by " << num_groups << " groups" << std::endl;
                // TODO: Add padding support
                seq_length_per_group = 0;  // Set to 0 to indicate error
                return;
            }
            seq_length_per_group = max_seq_length / num_groups;
        }
    };

    auto matcher() const
    {
        return match::name("group")(match::has_op_value("tag", "kv_cache_attention")).bind("group");
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

    // Helper to extract Q, K, V parameters from the attention submodule's gemm inputs
    struct qkv_params {
        instruction_ref q_param;  // Parameter containing Q (full QKV tensor)
        instruction_ref k_param;  // Parameter for K (concat_past_present output)
        instruction_ref v_param;  // Parameter for V (concat_past_present output)
    };
    
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

    // rebuild GQA attention operations in flash decoding submodule
    // Helper to find early exit masking operations
    struct early_exit_mask_ops {
        instruction_ref pos_literal;      // Literal with position indices {0,1,2,3...}
        instruction_ref pos_broadcast;    // Broadcast of position literal
        instruction_ref seq_len_param;    // Sequence length parameter
        instruction_ref seq_multicast;    // Multibroadcast of seq_len
        instruction_ref greater_op;       // Greater comparison
        instruction_ref convert_op;       // Convert to bool
        instruction_ref unsqueeze_op;     // Unsqueeze mask
        instruction_ref mask_broadcast;   // Final multibroadcast of mask
        instruction_ref ninf_literal;     // -inf literal for masking
        instruction_ref ninf_broadcast;   // Multibroadcast of -inf
        instruction_ref where_op;         // Where operation applying mask
        
        // Flags to track which operations were found
        bool found = false;
        bool has_pos_literal = false;
        bool has_pos_broadcast = false;
        bool has_seq_len_param = false;
        bool has_seq_multicast = false;
        bool has_greater = false;
        bool has_convert = false;
        bool has_unsqueeze = false;
        bool has_mask_broadcast = false;
        bool has_ninf_literal = false;
        bool has_ninf_broadcast = false;
    };
    
    early_exit_mask_ops find_early_exit_masking_ops(
        const module& source_mod,
        instruction_ref scaled_scores,
        const std::unordered_map<instruction_ref, instruction_ref>& map_old_to_new) const
    {
        early_exit_mask_ops mask_ops;
        
        // Find the where operation that uses our scaled scores
        for(auto ins : iterator_for(source_mod)) {
            if(ins->name() == "where") {
                // Check if one of its inputs is our scaled scores (through the mapping)
                for(auto input : ins->inputs()) {
                    if(contains(map_old_to_new, input) && map_old_to_new.at(input) == scaled_scores) {
                        mask_ops.where_op = ins;
                        mask_ops.found = true;
                        break;
                    }
                }
                if(mask_ops.found) break;
            }
        }
        
        if(!mask_ops.found) {
            return mask_ops;
        }
        
        // Get the three inputs to where: mask, true_value (-inf), false_value (scores)
        auto mask_input = mask_ops.where_op->inputs()[0];
        mask_ops.ninf_broadcast = mask_ops.where_op->inputs()[1];
        
        // Trace back the mask to find multibroadcast -> unsqueeze -> convert -> greater
        instruction_ref current = mask_input;
        
        // Should be multibroadcast
        if(current->name() == "multibroadcast") {
            mask_ops.mask_broadcast = current;
            mask_ops.has_mask_broadcast = true;
            current = current->inputs()[0];
        }
        
        // Should be unsqueeze
        if(current->name() == "unsqueeze") {
            mask_ops.unsqueeze_op = current;
            mask_ops.has_unsqueeze = true;
            current = current->inputs()[0];
        }
        
        // Should be convert
        if(current->name() == "convert") {
            mask_ops.convert_op = current;
            mask_ops.has_convert = true;
            current = current->inputs()[0];
        }
        
        // Should be greater
        if(current->name() == "greater") {
            mask_ops.greater_op = current;
            mask_ops.has_greater = true;
            
            // Get inputs to greater
            auto pos_input = mask_ops.greater_op->inputs()[0];
            auto seq_input = mask_ops.greater_op->inputs()[1];
            
            // Position side: broadcast -> literal
            if(pos_input->name() == "broadcast") {
                mask_ops.pos_broadcast = pos_input;
                mask_ops.has_pos_broadcast = true;
                mask_ops.pos_literal = pos_input->inputs()[0];
                mask_ops.has_pos_literal = true;
            }
            
            // Sequence length side: multibroadcast -> param
            if(seq_input->name() == "multibroadcast") {
                mask_ops.seq_multicast = seq_input;
                mask_ops.has_seq_multicast = true;
                mask_ops.seq_len_param = seq_input->inputs()[0];
                mask_ops.has_seq_len_param = true;
            }
        }
        
        // Find the -inf literal source
        if(mask_ops.ninf_broadcast->name() == "multibroadcast") {
            mask_ops.has_ninf_broadcast = true;
            mask_ops.ninf_literal = mask_ops.ninf_broadcast->inputs()[0];
            mask_ops.has_ninf_literal = true;
        }
        
        return mask_ops;
    }

    void rebuild_gqa_attention(module& target_mod, 
                               const module& source_mod,
                               const std::unordered_map<instruction_ref, instruction_ref>& param_map,
                               instruction_ref q_param,
                               instruction_ref k_param,
                               instruction_ref v_param,
                               const attention_dims& dims,
                               std::size_t num_groups) const
    {
        // map from instructions in old module to new module
        std::unordered_map<instruction_ref, instruction_ref> map_old_to_new = param_map;
        
        // TODO can do this better, and also make it better for other flash decoding case
        // track softmax components for LSE calculation
        std::unordered_map<std::string, instruction_ref> softmax_parts;

        assert(contains(param_map, q_param) && "Q parameter must be mapped");
        assert(contains(param_map, k_param) && "K parameter must be mapped");
        assert(contains(param_map, v_param) && "V parameter must be mapped");
        (void)v_param; // Will be used later for V operations
        
        // handle Q extraction
        // since we slice on axis 1 (concat_heads) and groups are at axis 2, no change needed
        for(auto ins : iterator_for(source_mod)) {
            if(ins->name() == "slice" && ins->inputs()[0] == q_param) {
                auto op = ins->get_operator();
                auto new_q = map_old_to_new.at(q_param);
                auto sliced_q = target_mod.add_instruction(op, new_q);
                map_old_to_new[ins] = sliced_q;
                std::cout << "  Q slice created, shape: " << sliced_q->get_shape() << std::endl;
                break;
            }
        }
        
        // handle K transpose
        instruction_ref transposed_k;
        for(auto ins : iterator_for(source_mod)) {
            if(ins->name() == "transpose") {
                auto transpose_input = ins->inputs()[0];
                if(transpose_input == k_param) {
                    auto op = ins->get_operator();
                    auto perm = op.to_value()["permutation"].to_vector<int64_t>();
                    
                    // dims.batch_size, dims.kv_heads, groups, dims.seq_length_per_group, dims.head_dim}
                    // perm is now [0, 1, 2, 4, 3] for [B, H, G, D, S]
                    std::vector<int64_t> new_perm = {0, 1, 2, 4, 3};
                    auto new_transpose_op = make_op("transpose", {{"permutation", new_perm}});
                    auto new_k = map_old_to_new.at(k_param);
                    transposed_k = target_mod.add_instruction(new_transpose_op, new_k);
                    map_old_to_new[ins] = transposed_k;
                    
                    break;
                }
            }
        }
        
        // ninf is of shape
        // {batch_size, num_heads, sequence_length, max_seq_len}


        // handle literal constants and their broadcasts
        for(auto ins : iterator_for(source_mod)) {
            if(ins->name() == "@literal") {
                // copy literals directly
                auto lit_val = ins->get_literal();
                auto new_lit = target_mod.add_literal(lit_val);
                map_old_to_new[ins] = new_lit;
                std::cout << "  Added literal with shape: " << new_lit->get_shape() << std::endl;
            }
        }

        // TODO handle when kv_heads != num_heads
        // define expected broadcast shapes for literals
        std::vector<std::size_t> bnsm{dims.batch_size, dims.num_heads, dims.sequence_length, dims.max_seq_length};
        std::vector<std::size_t> bngsm{dims.batch_size, dims.num_heads, num_groups, dims.sequence_length, dims.seq_length_per_group};

        // update multibroadcast shapes for literals
        // if broadcasting to [B, N, S, M] shape, change to [B, N, G, S, M/G]
        // Keep track of specific broadcasts for -inf and scale
        instruction_ref ninf_broadcast;
        instruction_ref scale_broadcast;
        
        for(auto ins : iterator_for(source_mod)) {
            if(ins->name() == "multibroadcast" && 
               !contains(map_old_to_new, ins)) {
                auto input = ins->inputs()[0];
                
                // check if input is a literal and is already mapped
                if(contains(map_old_to_new, input) && input->name() == "@literal") {
                    auto op = ins->get_operator();
                    auto out_lens = op.to_value()["out_lens"].to_vector<std::size_t>();
                    
                    // check if the shape matches [B, N, S, M] pattern for attention scores
                    if(out_lens == bnsm) {
                        // use the pre-defined bngsm shape
                        auto new_op = make_op("multibroadcast", {{"out_lens", bngsm}});
                        auto new_input = map_old_to_new.at(input);
                        auto new_broadcast = target_mod.add_instruction(new_op, new_input);
                        map_old_to_new[ins] = new_broadcast;
                        
                        // Check the literal value to identify -inf or scale
                        auto lit = input->get_literal();
                        if(lit.get_shape().type() == migraphx::shape::half_type) {
                            // Get the literal value as a string for comparison
                            auto lit_str = lit.to_string();
                            if(lit_str == "-inf") {
                                ninf_broadcast = new_broadcast;
                                std::cout << "adjusted -inf multibroadcast from BNSM to BNGSM: " 
                                          << new_broadcast->get_shape() << std::endl;
                            } else {
                                // Assume it's the scale value
                                scale_broadcast = new_broadcast;
                                std::cout << "adjusted scale multibroadcast from BNSM to BNGSM: " 
                                          << new_broadcast->get_shape() << " (value: " << lit_str << ")" << std::endl;
                            }
                        }
                    } else {
                        // for other shapes, just copy the multibroadcast as-is
                        auto new_input = map_old_to_new.at(input);
                        auto new_broadcast = target_mod.add_instruction(op, new_input);
                        map_old_to_new[ins] = new_broadcast;
                    }
                }
            }
        }
        
        // Q slice [batch, num heads, groups, sl, max sl]
        // check if we found the specific broadcasts
        bool has_scale = false;
        bool has_ninf = false;
        for(auto ins : iterator_for(target_mod)) {
            if(ins == scale_broadcast) has_scale = true;
            if(ins == ninf_broadcast) has_ninf = true;
        }
        
        if(has_scale) {
            std::cout << "  found scale broadcast for attention scaling" << std::endl;
        }
        if(has_ninf) {
            std::cout << "  found -inf broadcast for masking" << std::endl;
        }
        
        // handle first dot product (Q @ K^T)
        std::cout << "rebuilding first dot product (Q @ K^T)..." << std::endl;
        instruction_ref dot1;
        for(auto ins : iterator_for(source_mod)) {
            if(ins->name() == "dot") {
                // check if this is the first dot (Q @ K^T)
                // it should have Q (or sliced Q) as first input and K transpose as second
                auto input0 = ins->inputs()[0];
                auto input1 = ins->inputs()[1];
                
                // check if we've already mapped these inputs (Q slice and K transpose)
                if(contains(map_old_to_new, input0) && contains(map_old_to_new, input1)) {
                    auto new_q = map_old_to_new.at(input0);
                    auto new_kt = map_old_to_new.at(input1);
                    
                    // create the dot product with transformed inputs
                    dot1 = target_mod.add_instruction(make_op("dot"), new_q, new_kt);
                    map_old_to_new[ins] = dot1;
                    
                    std::cout << "  created dot1 (Q @ K^T) with shape: " << dot1->get_shape() << std::endl;
                    break;  // assume first dot is Q @ K^T
                }
            }
        }
        
        // handle scaling (mul with scale factor)
        std::cout << "finding and rebuilding scale multiplication..." << std::endl;
        instruction_ref scaled_scores;
        
        // check if we have both dot1 and scale_broadcast
        bool has_dot1 = false;
        bool has_scale_bc = false;
        for(auto ins : iterator_for(target_mod)) {
            if(ins == dot1) has_dot1 = true;
            if(ins == scale_broadcast) has_scale_bc = true;
        }
        
        if(has_dot1 && has_scale_bc) {
            scaled_scores = target_mod.add_instruction(make_op("mul"), dot1, scale_broadcast);
            std::cout << "  created scaled scores with shape: " << scaled_scores->get_shape() << std::endl;
        } else if(has_dot1) {
            // ff we don't have scale_broadcast, try to find the mul in the original
            for(auto ins : iterator_for(source_mod)) {
                if(ins->name() == "mul") {
                    auto input0 = ins->inputs()[0];
                    auto input1 = ins->inputs()[1];
                    
                    if(contains(map_old_to_new, input0) && contains(map_old_to_new, input1)) {
                        bool input0_is_dot = (map_old_to_new.at(input0)->name() == "dot");
                        bool input1_is_dot = (map_old_to_new.at(input1)->name() == "dot");
                        
                        if(input0_is_dot || input1_is_dot) {
                            auto new_input0 = map_old_to_new.at(input0);
                            auto new_input1 = map_old_to_new.at(input1);
                            
                            scaled_scores = target_mod.add_instruction(make_op("mul"), new_input0, new_input1);
                            map_old_to_new[ins] = scaled_scores;
                            
                            std::cout << "  created scaled scores with shape: " << scaled_scores->get_shape() << std::endl;
                            break;
                        }
                    }
                }
            }
        }
        
        // For kv_cache_attention, rebuild early exit masking with modified broadcast shapes
        std::cout << "rebuilding early exit masking with adjusted broadcasts..." << std::endl;
        
        // Find the range literal (position indices like {0,1,2,3...})
        instruction_ref range_literal;
        bool found_range = false;
        for(auto ins : iterator_for(source_mod)) {
            if(ins->name() == "@literal") {
                auto shape = ins->get_literal().get_shape();
                if(shape.type() == migraphx::shape::int32_type && 
                   shape.lens().size() == 1 &&
                   shape.lens()[0] == dims.max_seq_length) {
                    range_literal = ins;
                    found_range = true;
                    std::cout << "  found range literal with shape: " << shape << std::endl;
                    break;
                }
            }
        }
        
        // Find the past_sl parameter (past sequence length)
        instruction_ref past_sl_param;
        bool found_past_sl = false;
        for(auto param : iterator_for(source_mod)) {
            if(param->name() == "@param") {
                auto shape = param->get_shape();
                // past_sl is int32 type with batch_size elements (e.g., [2,1])
                if(shape.type() == migraphx::shape::int32_type && 
                   shape.elements() == dims.batch_size) {
                    past_sl_param = param;
                    found_past_sl = true;
                    std::cout << "  found past_sl param with shape: " << shape << std::endl;
                    break;
                }
            }
        }
        
        if(!found_range) {
            std::cout << "  WARNING: Could not find range literal" << std::endl;
        }
        if(!found_past_sl) {
            std::cout << "  WARNING: Could not find past_sl parameter" << std::endl;
        }
        
        // create broadcast shape vector
        std::vector<std::size_t> broadcast_shape = {dims.batch_size, num_groups, dims.seq_length_per_group};
        std::cout << "  broadcast shape: [" << dims.batch_size << ", " << num_groups << ", " 
                  << dims.seq_length_per_group << "]" << std::endl;
        
        instruction_ref range_broadcast;
        instruction_ref past_sl_reshaped;
        
        if(found_range && found_past_sl && contains(param_map, past_sl_param)) {
            // range literal broadcast to [batch_size, num_groups, seq_length_per_group]
            if(!contains(map_old_to_new, range_literal)) {
                auto lit_val = range_literal->get_literal();
                auto new_lit = target_mod.add_literal(lit_val);
                map_old_to_new[range_literal] = new_lit;
            }
            
            // Broadcast range literal directly, matching original pattern
            // Use axis=1 to match the original pattern (broadcast[axis=1] adds batch dimension at front)
            std::vector<std::size_t> intermediate_bc_shape = {dims.batch_size, dims.max_seq_length};
            auto range_broadcast_intermediate = target_mod.add_instruction(
                make_op("broadcast", {{"axis", 1}, {"out_lens", intermediate_bc_shape}}),
                map_old_to_new.at(range_literal));
            std::cout << "  broadcasted range to: " << range_broadcast_intermediate->get_shape() << std::endl;
            
            // Then reshape to final shape [batch_size, num_groups, seq_length_per_group]
            range_broadcast = target_mod.add_instruction(
                make_op("reshape", {{"dims", broadcast_shape}}),
                range_broadcast_intermediate);
            std::cout << "  reshaped range to final shape: " << range_broadcast->get_shape() << std::endl;
            
            // past_sl param from [batch_size, 1] to [batch_size, max_seq_length]
            // then reshape to [batch_size, num_groups, seq_length_per_group]
            auto past_sl_new = param_map.at(past_sl_param);
            
            std::vector<std::size_t> intermediate_shape = {dims.batch_size, dims.max_seq_length};
            auto past_sl_broadcast = target_mod.add_instruction(
                make_op("multibroadcast", {{"out_lens", intermediate_shape}}),
                past_sl_new);
            std::cout << "  multibroadcasted past_sl to: " << past_sl_broadcast->get_shape() << std::endl;
            
            past_sl_reshaped = target_mod.add_instruction(
                make_op("reshape", {{"dims", broadcast_shape}}),
                past_sl_broadcast);
            std::cout << "  reshaped past_sl to: " << past_sl_reshaped->get_shape() << std::endl;
            
            auto greater = target_mod.add_instruction(
                make_op("greater"), range_broadcast, past_sl_reshaped);
            auto convert = target_mod.add_instruction(
                make_op("convert", {{"target_type", migraphx::shape::bool_type}}), greater);
            auto unsqueeze = target_mod.add_instruction(
                make_op("unsqueeze", {{"axes", {1, 3}}}), convert);
            auto multibroadcast = target_mod.add_instruction(
                make_op("multibroadcast", {{"out_lens", bngsm}}), unsqueeze);
            
            // Check if we have the ninf_broadcast before using it
            bool has_ninf_bc = false;
            for(auto ins : iterator_for(target_mod)) {
                if(ins == ninf_broadcast) {
                    has_ninf_bc = true;
                    break;
                }
            }
            
            if(has_ninf_bc) {
                auto where = target_mod.add_instruction(make_op("where"), multibroadcast, ninf_broadcast, scaled_scores);
                scaled_scores = where; // Update scaled_scores to the masked version
                
            } else {
                std::cout << "  WARNING: Could not find ninf_broadcast for where operation" << std::endl;
            }
        }
        
        // quick implementation of remaining ops for testing
        
        // convert to float for softmax computation
        auto convert_to_float = target_mod.add_instruction(
            make_op("convert", {{"target_type", migraphx::shape::float_type}}), scaled_scores);
        
        // Reduce max along last axis (axis 4 in BNGSM)
        auto reduce_max = target_mod.add_instruction(
            make_op("reduce_max", {{"axes", {4}}}), convert_to_float);
        
        // Broadcast max back to original shape
        auto max_broadcast = target_mod.add_instruction(
            make_op("multibroadcast", {{"out_lens", bngsm}}), reduce_max);
        
        // Subtract max for numerical stability
        auto sub = target_mod.add_instruction(
            make_op("sub"), convert_to_float, max_broadcast);
        
        // Exp
        auto exp_scores = target_mod.add_instruction(
            make_op("exp"), sub);
        
        // Reduce sum along last axis
        auto reduce_sum = target_mod.add_instruction(
            make_op("reduce_sum", {{"axes", {4}}}), exp_scores);
        
        // Broadcast sum back
        auto sum_broadcast = target_mod.add_instruction(
            make_op("multibroadcast", {{"out_lens", bngsm}}), reduce_sum);
        
        // Divide to get softmax
        auto softmax = target_mod.add_instruction(
            make_op("div"), exp_scores, sum_broadcast);
        
        // Convert back to half
        auto convert_to_half = target_mod.add_instruction(
            make_op("convert", {{"target_type", migraphx::shape::half_type}}), softmax);

        std::cout << "now doing the dot between the convert and v";
        std::cout << v_param->get_shape() << std::endl;

        // We need to use the mapped V parameter, not the original v_param
        // The v_param passed in is from the original submodule, we need the one in param_map
        auto v_mapped = param_map.at(v_param);
        std::cout << "V mapped shape: " << v_mapped->get_shape() << std::endl;
        
        // Dot with V
        auto dot2 = target_mod.add_instruction(
            make_op("dot"), convert_to_half, v_mapped);
        std::cout << "Dot2 shape: " << dot2->get_shape() << std::endl;
        

        
        // for flash decoding, we keep the group dimension and return it
        // kernel 2 will handle the LSE-weighted reduction
        // dot2 is currently [B, N, G, S, D]
        
        // transpose to [B, G, S, N, D] to match flash decoding output pattern
        auto transpose_out = target_mod.add_instruction(
            make_op("transpose", {{"permutation", {0, 2, 3, 1, 4}}}), dot2);
        std::cout << "Transpose shape: " << transpose_out->get_shape() << std::endl;
        
        // reshape to [B, G, S, N*D]
        std::vector<std::size_t> final_shape = {dims.batch_size, num_groups, dims.sequence_length, dims.num_heads * dims.head_dim};
        auto reshape_out = target_mod.add_instruction(
            make_op("reshape", {{"dims", final_shape}}), transpose_out);
        std::cout << "Final reshape shape (with groups): " << reshape_out->get_shape() << std::endl;
        
        // for LSE (log-sum-exp), we need log(sum_exp) + max
        // we already have reduce_max and reduce_sum from softmax computation
        // LSE shape is [B, N, G, S, 1] which is correct for flash decoding
        auto log_sum = target_mod.add_instruction(make_op("log"), reduce_sum);
        auto lse = target_mod.add_instruction(make_op("add"), reduce_max, log_sum);
        std::cout << "LSE shape: " << lse->get_shape() << std::endl;
        
        target_mod.add_return({reshape_out, lse});
        
        // print the complete submodule
        std::cout << "\n=== Complete GQA Flash Decoding Submodule ===" << std::endl;
        std::cout << target_mod << std::endl;
        std::cout << "=== End Submodule ===" << std::endl;
    }

    std::optional<qkv_params> extract_qkv_params(instruction_ref gemm1, instruction_ref gemm2) const
    {
        qkv_params result;
        
        // Q: gemm1's first input should be a slice from the QKV tensor
        auto q_input = gemm1->inputs()[0];
        if(q_input->name() == "slice") {
            // trace back from slice to find the parameter
            auto before_slice = q_input->inputs()[0];
            
            instruction_ref current = before_slice;
            while(current->name() != "@param") {
                if(current->inputs().empty()) {
                    std::cout << "Cannot trace Q back to parameter" << std::endl;
                    return std::nullopt;
                }
                current = current->inputs()[0];
            }
            result.q_param = current;
        } else {
            std::cout << "Expected Q to come from slice, got: " << q_input->name() << std::endl;
            return std::nullopt;
        }
        
        // K: gemm1's second input should be transposed K from concat_past_present
        auto k_input = gemm1->inputs()[1];
        if(k_input->name() == "transpose") {
            result.k_param = k_input->inputs()[0];
        } else {
            result.k_param = k_input;
        }
        
        if(result.k_param->name() != "@param") {
            std::cout << "Expected K to be a parameter, got: " << result.k_param->name() << std::endl;
            return std::nullopt;
        }
        
        // V: gemm2's second input should be V from concat_past_present
        result.v_param = gemm2->inputs()[1];
        if(result.v_param->name() != "@param") {
            std::cout << "Expected V to be a parameter, got: " << result.v_param->name() << std::endl;
            return std::nullopt;
        }
        
        return result;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& mm            = mpm.get_module();
        auto attn_group_ins = r.instructions["group"];
        auto* submod        = attn_group_ins->module_inputs().front();

        std::cout << "GQA flash decoding detected" << std::endl;

        // check multiple returns
        auto return_ins = std::prev(submod->end());
        assert(return_ins->name() == "@return" and
               "Last instruction must be a @return instruction");
        if(return_ins->inputs().size() > 1) {
            std::cout << "KV cache attention unexpected multiple returns" << std::endl;
            return;
        }

        // get gemm1 and gemm2
        auto [gemm1, gemm2] = get_gemms(submod);

        // Extract Q, K, V parameters from gemm inputs
        auto qkv_opt = extract_qkv_params(gemm1, gemm2);
        if(!qkv_opt) {
            std::cout << "Failed to extract Q, K, V parameters" << std::endl;
            return;
        }
        
        auto [q_param, k_param, v_param] = *qkv_opt;
        
        std::cout << "Q attn module param shape: " << q_param->get_shape() << std::endl;
        std::cout << "K attn module param shape: " << k_param->get_shape() << std::endl;
        std::cout << "V attn module param shape: " << v_param->get_shape() << std::endl;

        // derive dim values
        attention_dims dims(q_param, k_param, groups);
        
        std::cout << "Max sequence length: " << dims.max_seq_length << std::endl;

        if(groups <= 1) {
            std::cout << "No splitting requested (groups=" << groups << ")" << std::endl;
            return;
        }
        
        // check if dimensions were calculated successfully
        if(dims.seq_length_per_group == 0) {
            std::cout << "Failed to calculate sequence length per group, returning" << std::endl;
            return;
        }

        // map submodule params to main module inputs
        auto group_inputs      = attn_group_ins->inputs();
        auto map_param_to_main = map_submod_params_to_inputs(submod, group_inputs);
        
        // get actual Q, K, V instructions from main module
        auto q = map_param_to_main.at(q_param);  // maps to the QKV tensor
        auto k = map_param_to_main.at(k_param);  // maps to K concat_past_present output 
        auto v = map_param_to_main.at(v_param);  // maps to V concat_past_present output

        std::cout << "Main module Q shape: " << q->get_shape() << std::endl;
        std::cout << "Main module K shape: " << k->get_shape() << std::endl;
        std::cout << "Main module V shape: " << v->get_shape() << std::endl;
        
        // GQA flash decoding:
        // - Q (QKV tensor): broadcast across groups (no split)
        // - K: split sequence dimension into groups
        // - V: split sequence dimension into groups
        
        // shapes before group transformation
        auto q_shape = q->get_shape();
        auto k_shape_main = k->get_shape(); 
        auto v_shape_main = v->get_shape();
        
        // insert group dimension at position -2 for all tensors
        // K and V: [B, kv_heads, N, D] -> [B, kv_heads, G, N/G, D] (split)
        // build transformed shapes
        std::vector<std::size_t> q_transformed_shape;
        std::vector<std::size_t> k_transformed_shape;
        std::vector<std::size_t> v_transformed_shape;
        
        // Q shape transformation (broadcast group dimension)
        q_transformed_shape = {dims.batch_size, dims.concat_heads, groups, dims.sequence_length, dims.head_dim};
        k_transformed_shape = {dims.batch_size, dims.kv_heads, groups, dims.seq_length_per_group, dims.head_dim};
        v_transformed_shape = {dims.batch_size, dims.kv_heads, groups, dims.seq_length_per_group, dims.head_dim};

        std::cout << "Q transformed shape: ";
        for(auto d : q_transformed_shape) std::cout << d << " ";
        std::cout << std::endl;
        
        std::cout << "K transformed shape: ";
        for(auto d : k_transformed_shape) std::cout << d << " ";
        std::cout << std::endl;
        
        std::cout << "V transformed shape: ";
        for(auto d : v_transformed_shape) std::cout << d << " ";
        std::cout << std::endl;

        // insert reshape operations before the attention group
        // [B, concat_heads, seq, head_dim] -> [B, concat_heads, 1, seq, head_dim] -> [B, concat_heads, G, seq, head_dim]
        auto q_unsqueezed = mm.insert_instruction(
            attn_group_ins, 
            make_op("unsqueeze", {{"axes", {2}}}),
            q);
        
        auto q_reshaped = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", q_transformed_shape}}),
            q_unsqueezed);
        
        // K: reshape to split sequence dimension
        auto k_reshaped = mm.insert_instruction(
            attn_group_ins,
            make_op("reshape", {{"dims", k_transformed_shape}}),
            k);
        
        // V: reshape to split sequence dimension
        auto v_reshaped = mm.insert_instruction(
            attn_group_ins,
            make_op("reshape", {{"dims", v_transformed_shape}}),
            v);
        
        std::cout << "Q reshaped: " << q_reshaped->get_shape() << std::endl;
        std::cout << "K reshaped: " << k_reshaped->get_shape() << std::endl;
        std::cout << "V reshaped: " << v_reshaped->get_shape() << std::endl;
        
        // No need to handle positions outside the submodule
        // We'll adjust broadcast patterns inside for early exit masking

        // TODO can probably do this simpler
        // Create new input list by replacing Q, K, V with reshaped versions
        std::vector<instruction_ref> new_group_inputs = group_inputs;
        for(size_t i = 0; i < group_inputs.size(); ++i) {
            if(group_inputs[i] == q) {
                new_group_inputs[i] = q_reshaped;
            } else if(group_inputs[i] == k) {
                new_group_inputs[i] = k_reshaped;
            } else if(group_inputs[i] == v) {
                new_group_inputs[i] = v_reshaped;
            }
            // Other inputs (like seq_len) stay the same
        }

        // Create new flash decoding submodule
        module m_flash_decode;
        m_flash_decode.set_bypass();
        
        // Get parameter names from original submodule
        auto get_param_name = [](instruction_ref param) -> std::string {
            assert(param->name() == "@param");
            return param->get_operator().to_value()["parameter"].to<std::string>();
        };
        
        auto q_name = get_param_name(q_param);
        auto k_name = get_param_name(k_param);
        auto v_name = get_param_name(v_param);
        
        // Add parameters to new submodule with transformed shapes
        auto new_q_param = m_flash_decode.add_parameter(
            q_name, shape{q_shape.type(), q_transformed_shape});
        auto new_k_param = m_flash_decode.add_parameter(
            k_name, shape{k_shape_main.type(), k_transformed_shape});
        auto new_v_param = m_flash_decode.add_parameter(
            v_name, shape{v_shape_main.type(), v_transformed_shape});
        
        // Build mapping from old params to new params
        std::unordered_map<instruction_ref, instruction_ref> map_old_params_to_new;
        map_old_params_to_new[q_param] = new_q_param;
        map_old_params_to_new[k_param] = new_k_param;
        map_old_params_to_new[v_param] = new_v_param;
        
        // Add other parameters (like seq_len) that don't change shape
        for(auto param : iterator_for(*submod)) {
            if(param->name() == "@param") {
                if(param != q_param && param != k_param && param != v_param) {
                    auto param_name = get_param_name(param);
                    auto param_shape = param->get_shape();
                    auto new_param = m_flash_decode.add_parameter(param_name, param_shape);
                    map_old_params_to_new[param] = new_param;
                    std::cout << "Added unchanged param: " << param_name 
                              << " with shape: " << param_shape << std::endl;
                }
            }
        }
        
        // TODO all the param stuff before this can be simplified

        // rebuild the attention operations in the flash decode submodule
        std::cout << "Rebuilding GQA attention operations..." << std::endl;
        rebuild_gqa_attention(m_flash_decode, *submod, map_old_params_to_new, 
                             q_param, k_param, v_param, dims, groups);
        
        // create the module in the module pass manager
        auto original_submod_name = attn_group_ins->module_inputs().front()->name();
        std::string new_mod_name = original_submod_name + "_gqa_flash_decoding";
        
        module_ref mpm_flash_mod = mpm.create_module(new_mod_name, std::move(m_flash_decode));
        mpm_flash_mod->set_bypass();
        
        // insert the new group operation
        auto new_group_ins = mm.insert_instruction(
            attn_group_ins,
            make_op("group", {{"tag", "attention"}}),
            new_group_inputs,
            {mpm_flash_mod});
        
        std::cout << "Created GQA flash decoding group" << std::endl;
        std::cout << "Group output shape: " << new_group_ins->get_shape() << std::endl;
        
        // unpack O' and LSE
        auto partial_output_o_prime = mm.insert_instruction(
            attn_group_ins, make_op("get_tuple_elem", {{"index", 0}}), new_group_ins);
        auto lse = mm.insert_instruction(
            attn_group_ins, make_op("get_tuple_elem", {{"index", 1}}), new_group_ins);

        // LSE-weighted combination
        std::cout << "\n=== Kernel 2: LSE-weighted combination ===" << std::endl;
        std::cout << "Input LSE shape: " << lse->get_shape() << std::endl;  // [B, N, G, S, 1] = [2, 2, 2, 1, 1]
        std::cout << "Input O' shape: " << partial_output_o_prime->get_shape() << std::endl;  // [B, G, S, N*D] = [2, 2, 1, 4]
        
        // align LSE with O' for proper weighting
        // LSE is [B, N, G, S, 1], match group dimension of O' [B, G, S, N*D]
        
        // [B, N, G, S, 1] -> [B, G, N, S, 1]
        std::cout << "\n1. Transposing LSE to align group dimension..." << std::endl;
        auto lse_transposed = mm.insert_instruction(
            attn_group_ins, make_op("transpose", {{"permutation", {0, 2, 1, 3, 4}}}), lse);
        std::cout << "   LSE transposed shape: " << lse_transposed->get_shape() << std::endl;  // [2, 2, 2, 1, 1]
        
        // average across heads (N) since all heads in a group share the same weight
        // [B, G, N, S, 1] -> [B, G, S, 1] (reduce over axis 2, then squeeze)
        std::cout << "\n2. Averaging LSE across heads within each group..." << std::endl;
        auto lse_avg = mm.insert_instruction(
            attn_group_ins, make_op("reduce_mean", {{"axes", {2}}}), lse_transposed);
        std::cout << "   LSE averaged shape: " << lse_avg->get_shape() << std::endl;  // [2, 2, 1, 1, 1]
        
        // squeeze axes 2 and 4: [B, G, 1, S, 1] -> [B, G, S]
        auto lse_squeezed = mm.insert_instruction(
            attn_group_ins, make_op("squeeze", {{"axes", {2, 4}}}), lse_avg);
        std::cout << "   LSE squeezed shape: " << lse_squeezed->get_shape() << std::endl;  // [2, 2, 1]
        
        // softmax across groups for LSE weights
        std::cout << "\n3. Computing softmax of LSE across groups..." << std::endl;
        
        // find max across groups for numerical stability
        auto lse_max = mm.insert_instruction(
            attn_group_ins, make_op("reduce_max", {{"axes", {1}}}), lse_squeezed);
        std::cout << "   Max LSE shape: " << lse_max->get_shape() << std::endl;  // [2, 1, 1]
        
        // broadcast max back to original shape
        auto lse_max_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", lse_squeezed->get_shape().lens()}}),
            lse_max);
        
        // exp(LSE - max_LSE)
        auto lse_sub = mm.insert_instruction(attn_group_ins, make_op("sub"), lse_squeezed, lse_max_bcast);
        auto lse_exp = mm.insert_instruction(attn_group_ins, make_op("exp"), lse_sub);
        std::cout << "   Exp(LSE) shape: " << lse_exp->get_shape() << std::endl;  // [2, 2, 1]
        
        // sum exp across groups
        auto lse_sum = mm.insert_instruction(
            attn_group_ins, make_op("reduce_sum", {{"axes", {1}}}), lse_exp);
        std::cout << "   Sum exp shape: " << lse_sum->get_shape() << std::endl;  // [2, 1, 1]
        
        // broadcast sum back
        auto lse_sum_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", lse_exp->get_shape().lens()}}),
            lse_sum);
        
        auto weights = mm.insert_instruction(attn_group_ins, make_op("div"), lse_exp, lse_sum_bcast);
        std::cout << "   Softmax weights shape: " << weights->get_shape() << std::endl;  // [2, 2, 1]
        
        // weights is [B, G, S], O' is [B, G, S, N*D]
        // [B, G, S] -> [B, G, S, 1]
        std::cout << "\n4. Preparing weights for multiplication with O'..." << std::endl;
        auto weights_unsqueezed = mm.insert_instruction(
            attn_group_ins, make_op("unsqueeze", {{"axes", {3}}}), weights);
        std::cout << "   Weights unsqueezed shape: " << weights_unsqueezed->get_shape() << std::endl;  // [2, 2, 1, 1]
        
        // broadcast to match O' shape
        auto weights_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", partial_output_o_prime->get_shape().lens()}}),
            weights_unsqueezed);
        std::cout << "   Weights broadcast shape: " << weights_bcast->get_shape() << std::endl;  // [2, 2, 1, 4]
        
        // convert weights to match O' type
        auto output_type = partial_output_o_prime->get_shape().type();
        auto weights_converted = mm.insert_instruction(
            attn_group_ins, make_op("convert", {{"target_type", output_type}}), weights_bcast);
        
        // multiply O' by weights
        std::cout << "\n5. Multiplying O' by softmax weights..." << std::endl;
        auto weighted_output = mm.insert_instruction(
            attn_group_ins, make_op("mul"), partial_output_o_prime, weights_converted);
        std::cout << "   Weighted output shape: " << weighted_output->get_shape() << std::endl;  // [2, 2, 1, 4]
        
        // sum across groups to get final output
        std::cout << "\n6. Summing weighted outputs across groups..." << std::endl;
        auto final_output = mm.insert_instruction(
            attn_group_ins, make_op("reduce_sum", {{"axes", {1}}}), weighted_output);
        std::cout << "   Final output shape: " << final_output->get_shape() << std::endl;  // [2, 1, 1, 4]
        
        // squeeze the reduced group dimension
        auto final_squeezed = mm.insert_instruction(
            attn_group_ins, make_op("squeeze", {{"axes", {1}}}), final_output);
        std::cout << "   Final squeezed shape: " << final_squeezed->get_shape() << std::endl;  // [2, 1, 4]
        
        mm.replace_instruction(attn_group_ins, final_squeezed);
        
        std::cout << "\n=== Kernel 2 complete: LSE-weighted combination successful ===" << std::endl;
    }
};

struct find_flash_decoding
{
    // configuration from fuse_attention pass config
    std::size_t configured_splits;
    std::size_t configured_threshold;
    std::size_t configured_max_splits;
    std::size_t configured_min_chunk_size;

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

        // read configuration with priority: struct member (if not default) > env var > default
        std::size_t groups = get_num_splits(configured_splits);
        std::size_t threshold =
            get_config_value(configured_threshold, 32, MIGRAPHX_FLASH_DECODING_THRESHOLD{});
        std::size_t min_chunk_size = get_config_value(
            configured_min_chunk_size, 32, MIGRAPHX_FLASH_DECODING_MIN_CHUNK_SIZE{});
        std::size_t max_splits =
            get_config_value(configured_max_splits, 16, MIGRAPHX_FLASH_DECODING_MAX_SPLITS{});

        std::size_t actual_groups =
            calculate_groups(groups, sequence_length, threshold, min_chunk_size, max_splits);
        if(actual_groups == 0)
            return;

        // calculate padding if sequence length not evenly divisible
        std::size_t padding_needed = 0;
        if(sequence_length % actual_groups != 0)
        {
            // round up to nearest multiple of actual_groups
            padding_needed = ceil_mul_of(sequence_length, actual_groups) - sequence_length;
        }

        // create mapping from submodule params to main module inputs
        auto group_inputs      = attn_group_ins->inputs();
        auto map_param_to_main = map_submod_params_to_inputs(submod, group_inputs);

        // get actual Q, K, V instructions from main module
        auto q = map_param_to_main.at(q_param);
        auto k = map_param_to_main.at(k_param);
        auto v = map_param_to_main.at(v_param);

        // save original references before padding (needed for group_inputs replacement later)
        auto q_orig = q;
        auto k_orig = k;
        auto v_orig = v;

        // pad Q, K and V if necessary
        if(padding_needed > 0)
        {
            // Q shape: [B, M, k] or [B, H, M, k] for 4D. Padding on M (sequence length dim)
            auto q_ndim = q->get_shape().ndim();
            std::vector<std::size_t> q_pads(2 * q_ndim, 0);
            q_pads[q_ndim + q_ndim - 2] = padding_needed; // pad right on M dim (second to last)
            q = mm.insert_instruction(attn_group_ins, make_op("pad", {{"pads", q_pads}}), q);

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
        // use original references (before padding) for comparison
        std::vector<instruction_ref> new_group_inputs = group_inputs;
        for(size_t i = 0; i < group_inputs.size(); ++i)
        {
            if(group_inputs[i] == q_orig)
            {
                new_group_inputs[i] = q_reshaped;
            }
            else if(group_inputs[i] == k_orig)
            {
                new_group_inputs[i] = k_reshaped;
            }
            else if(group_inputs[i] == v_orig)
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

        // if padding was applied, slice to remove it
        instruction_ref final_result = final_squeezed_o;
        if(padding_needed > 0)
        {
            // need to slice the sequence dimension to remove padding
            // final_squeezed_o has shape like [B, M_padded, D], need to slice M back to original
            auto output_shape            = final_squeezed_o->get_shape();
            const auto& output_lens      = output_shape.lens();
            std::size_t seq_dim_idx      = output_lens.size() - 2; // sequence dim is second to last
            std::size_t original_seq_len = output_lens[seq_dim_idx] - padding_needed;

            final_result = mm.insert_instruction(
                attn_group_ins,
                make_op("slice",
                        {{"axes", {seq_dim_idx}}, {"starts", {0}}, {"ends", {original_seq_len}}}),
                final_squeezed_o);
        }

        // replace the original group instruction with the final result
        mm.replace_instruction(attn_group_ins, final_result);
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

    // enable flash decoding if splits configured or flash decoding is enabled
    std::size_t configured_splits = get_num_splits(flash_decoding_num_splits);
    if(configured_splits > 0 or flash_decoding_enabled)
    {
        // flash decoding for regular attention, single & multi-headed
        match::find_matches(
            mpm,
            find_flash_decoding{.configured_splits         = flash_decoding_num_splits,
                                .configured_threshold      = flash_decoding_threshold,
                                .configured_max_splits     = flash_decoding_max_splits,
                                .configured_min_chunk_size = flash_decoding_min_chunk_size});

        // flash decoding for GQA attention
        match::find_matches(
            mpm, find_gqa_flash_decoding{.groups = num_splits});
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
