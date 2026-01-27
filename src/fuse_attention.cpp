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
#include <sstream>
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

// Helper function to extract the two gemm operations from an attention submodule
// Returns {gemm1, gemm2} where gemm1 is Q@K and gemm2 is P@V
inline std::pair<instruction_ref, instruction_ref> get_attention_gemms(module_ref submod)
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

// Helper function to map submodule parameters to main module inputs
inline std::unordered_map<instruction_ref, instruction_ref>
map_submod_params_to_inputs(module_ref submod, const std::vector<instruction_ref>& group_inputs)
{
    auto map_param_to_main = submod->get_ins_param_map(group_inputs, true);
    // verify the mapping is correct
    auto expected_inputs = submod->get_inputs(map_param_to_main);
    assert(expected_inputs == group_inputs and "Mapped inputs don't match group inputs");
    return map_param_to_main;
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
                // TODO: add autosplitting (padding won't be needed)
                seq_length_per_group = 0;
            } else {
                seq_length_per_group = max_seq_length / num_groups;
            }
        }
    };

    // Adjust permutation when inserting a group dimension
    std::vector<int64_t> adjust_permutation(const std::vector<int64_t>& original_perm, 
                                                 int group_dim_pos) const
    {        
        // any dimension >= insert_pos shifts up by 1
        std::vector<int64_t> new_perm;
        for(auto idx : original_perm) {
            if(idx >= group_dim_pos) {
                new_perm.push_back(idx + 1);
            } else {
                new_perm.push_back(idx);
            }
        }
        
        // Insert the group dimension at its natural position in the output
        // The group dimension itself appears at position actual_insert_pos
        new_perm.insert(new_perm.begin() + group_dim_pos, group_dim_pos);
        return new_perm;
    }
    
    // Adjust axes when a group dimension is inserted
    std::vector<int64_t> adjust_axes(const std::vector<int64_t>& axes, int group_dim_pos) const
    {
        std::vector<int64_t> adjusted;
        for(auto axis : axes) {
            // If axis >= group_dim_pos, shift it by 1
            if(axis >= group_dim_pos) {
                adjusted.push_back(axis + 1);
            } else {
                adjusted.push_back(axis);
            }
        }
        return adjusted;
    }

    auto matcher() const
    {
        return match::name("group")(match::has_op_value("tag", "kv_cache_attention")).bind("group");
    }

    // Helper to extract Q, K, V parameters from the attention submodule's gemm inputs
    struct qkv_params {
        instruction_ref q_param;  // Parameter containing Q (full QKV tensor)
        instruction_ref k_param;  // Parameter for K (concat_past_present output)
        instruction_ref v_param;  // Parameter for V (concat_past_present output)

        // factory method to extract Q, K, V parameters from gemm operations
        static std::optional<qkv_params> from_gemms(instruction_ref gemm1, instruction_ref gemm2)
        {
            auto trace_back_to_param = [](instruction_ref ins) -> std::optional<instruction_ref> {
                instruction_ref current = ins;
                while(current->name() != "@param") {
                    if(current->inputs().empty()) {
                        return std::nullopt;
                    }
                    current = current->inputs()[0];
                }
                return current;
            };

            auto q_input = gemm1->inputs()[0];
            auto k_input = gemm1->inputs()[1];
            auto v_input = gemm2->inputs()[1];

            // trace back Q, K, V to find the parameters they originate from
            auto q_param_opt = trace_back_to_param(q_input);
            auto k_param_opt = trace_back_to_param(k_input);
            auto v_param_opt = trace_back_to_param(v_input);

            if(not q_param_opt or not k_param_opt or not v_param_opt) return std::nullopt;
            return qkv_params{*q_param_opt, *k_param_opt, *v_param_opt};
        }
    };

    void rebuild_gqa_attention(module& target_mod, 
                               const module& source_mod,
                               const std::unordered_map<instruction_ref, instruction_ref>& param_map,
                               instruction_ref gemm2,
                               const attention_dims& dims,
                               std::size_t num_groups) const
    {
        
        std::cout << "Rebuilding GQA attention with inserter..." << std::endl;
        std::cout << "Second gemm (will stop after): " << gemm2->name() << std::endl;
        
        int group_dim_pos = 2;
        
        // Define BNGSM and BNSM shapes
        std::vector<std::size_t> bngsm{dims.batch_size, dims.num_heads, num_groups, dims.sequence_length, dims.seq_length_per_group};
        std::vector<std::size_t> bnsm{dims.batch_size, dims.num_heads, dims.sequence_length, dims.max_seq_length};
        
        // need to track reduce operations for LSE calculation that's added to the submodule
        instruction_ref reduce_max_ref;
        instruction_ref reduce_sum_ref;
        bool found_reduce_max = false;
        bool found_reduce_sum = false;
        
        instruction_ref second_dot_result;
        
        // Create the inserter function that transforms operations
        auto inserter = [&](module& m,
                           instruction_ref ins,
                           const operation& op,
                           const std::vector<instruction_ref>& inputs,
                           const std::vector<module_ref>& mod_args) -> instruction_ref {
            
            auto op_name = op.name();
            
            // Helper to print shape
            auto print_shape = [](const std::vector<std::size_t>& lens) {
                std::cout << "{";
                for(size_t i = 0; i < lens.size(); ++i) {
                    std::cout << lens[i];
                    if(i < lens.size() - 1) std::cout << ",";
                }
                std::cout << "}";
            };
            
            auto print_output = [&print_shape](instruction_ref result) {
                std::cout << "    Output shape: ";
                print_shape(result->get_shape().lens());
                std::cout << std::endl;
            };
            
            // Helper to print operation attributes
            auto print_op_attrs = [](const operation& o) {
                try {
                    auto val = o.to_value();
                    if(!val.empty()) {
                        std::stringstream ss;
                        ss << val;
                        auto str = ss.str();
                        if(!str.empty()) {
                            std::cout << " " << str;
                        }
                    }
                } catch(...) {}
            };
            
            // Debug: print operation being processed
            std::cout << "\n>>> Processing op: " << op_name;
            print_op_attrs(op);
            std::cout << std::endl;
            std::cout << "    Input shapes: ";
            for(size_t i = 0; i < inputs.size(); ++i) {
                print_shape(inputs[i]->get_shape().lens());
                if(i < inputs.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
            
            // Transpose: adjust permutation
            if(op_name == "transpose") {
                auto perm = op.to_value()["permutation"].to_vector<int64_t>();
                auto new_perm = adjust_permutation(perm, group_dim_pos);
                
                std::cout << "    Adjusted perm: [";
                for(size_t i = 0; i < perm.size(); ++i) 
                    std::cout << perm[i] << (i < perm.size()-1 ? "," : "");
                std::cout << "] -> [";
                for(size_t i = 0; i < new_perm.size(); ++i)
                    std::cout << new_perm[i] << (i < new_perm.size()-1 ? "," : "");
                std::cout << "]" << std::endl;
                
                auto new_op = make_op("transpose", {{"permutation", new_perm}});
                std::cout << "    Creating: transpose";
                print_op_attrs(new_op);
                std::cout << std::endl;
                auto result = m.insert_instruction(ins, new_op, inputs, mod_args);
                print_output(result);
                return result;
            }
            
            // Reduce operations: adjust axes
            if(op_name == "reduce_max" || op_name == "reduce_sum") {
                auto axes = op.to_value()["axes"].to_vector<int64_t>();
                auto new_axes = adjust_axes(axes, group_dim_pos);
                
                std::cout << "    Adjusted axes: [" << axes[0] 
                          << "] -> [" << new_axes[0] << "]" << std::endl;
                
                auto new_op = make_op(op_name, {{"axes", new_axes}});
                std::cout << "    Creating: " << op_name;
                print_op_attrs(new_op);
                std::cout << std::endl;
                auto result = m.insert_instruction(ins, new_op, inputs, mod_args);
                
                // needed for LSE calculation that's added to the submodule
                if(op_name == "reduce_max") {
                    reduce_max_ref = result;
                    found_reduce_max = true;
                }
                if(op_name == "reduce_sum") {
                    reduce_sum_ref = result;
                    found_reduce_sum = true;
                }
                
                print_output(result);
                return result;
            }
            
            // TODO condense
            // Broadcast: adjust axis and out_lens for group dimension
            if(op_name == "broadcast") {
                auto op_val = op.to_value();
                auto out_lens = op_val["out_lens"].to_vector<std::size_t>();
                auto axis = op_val["axis"].to<uint64_t>();
                
                // if scalar output, or 1D output of len 1, can add group dim to out_lens directly
                if(out_lens.size() == 0 || (out_lens.size() == 1 && out_lens[0] == 1)) {
                    std::cout << "------------Broadcasting directly:\n\n\n";
                    out_lens.insert(out_lens.begin(), num_groups);
                    out_lens.push_back(dims.seq_length_per_group);
                    // axis needs to be adjusted if it's inserting in the new dimensions
                    uint64_t new_axis = (axis == 0) ? 0 : axis + 1;
                    auto new_op = make_op("broadcast", {{"axis", new_axis}, {"out_lens", out_lens}});
                    std::cout << "    Creating: broadcast";
                    print_op_attrs(new_op);
                    std::cout << std::endl;
                    auto result = m.insert_instruction(ins, new_op, inputs, mod_args);
                    print_output(result);
                    return result;
                } else if (inputs[0]->get_shape().lens().size() > out_lens.size()) {
                    // if the input rank does not match the length of out_lens,
                    // then that means the input has already been adjusted to have the
                    // group dim and seq_len_per_group, so we can edit the out_lens directly
                    std::vector<std::size_t> new_lens = {out_lens[0], out_lens[1], num_groups, out_lens[2], dims.seq_length_per_group};
                    // Adjust axis: if axis >= 2, shift by 1 to account for group dim insertion
                    uint64_t new_axis = (axis >= 2) ? axis + 1 : axis;
                    auto new_op = make_op("broadcast", {{"axis", new_axis}, {"out_lens", new_lens}});
                    std::cout << "    Creating: broadcast";
                    print_op_attrs(new_op);
                    std::cout << std::endl;
                    auto result = m.insert_instruction(ins, new_op, inputs, mod_args);
                    print_output(result);
                    return result;
                } else if(out_lens.size() == 2) {
                    // keep the broadcast as is
                    // follow up with a reshape to split the sequence dimension into groups
                    std::cout << "    broadcast unchanged - will add reshape" << std::endl;
                    auto result = m.insert_instruction(ins, op, inputs, mod_args);
                    std::cout << "    Creating: broadcast (unchanged)";   
                    print_op_attrs(op);
                    std::cout << std::endl;
                    print_output(result);
                    // Add reshape to split sequence dimension into groups
                    std::vector<std::size_t> reshape_dims = {dims.batch_size, num_groups, dims.seq_length_per_group};
                    std::cout << "\n>>> Auto-inserting reshape after 2D broadcast" << std::endl;
                    std::cout << "    Reshape dims: ";
                    print_shape(reshape_dims);
                    std::cout << std::endl;
                    
                    auto reshape_result = m.insert_instruction(ins, make_op("reshape", {{"dims", reshape_dims}}), result);
                    std::cout << "    Creating: reshape";
                    auto reshape_op = make_op("reshape", {{"dims", reshape_dims}});
                    print_op_attrs(reshape_op);
                    std::cout << std::endl;
                    print_output(reshape_result);
                    return reshape_result;
                } else if(out_lens.size() == 4) {
                    // keep the broadcast as is
                    // follow up with a reshape to split the sequence dimension into groups
                    std::cout << "    broadcast unchanged - will add reshape" << std::endl;
                    auto result = m.insert_instruction(ins, op, inputs, mod_args);
                    std::cout << "    Creating: broadcast (unchanged)";   
                    print_op_attrs(op);
                    std::cout << std::endl;
                    print_output(result);
                    // Add reshape to split sequence dimension into groups
                    auto result_lens = result->get_shape().lens();
                    std::vector<std::size_t> reshape_dims = result_lens;
                    reshape_dims.insert(reshape_dims.end() - 2, num_groups);
                    reshape_dims.back() = dims.seq_length_per_group;
                    std::cout << "\n>>> Auto-inserting reshape after 4D broadcast" << std::endl;
                    std::cout << "    Reshape dims: ";
                    print_shape(reshape_dims);
                    std::cout << std::endl;
                    
                    auto reshape_result = m.insert_instruction(ins, make_op("reshape", {{"dims", reshape_dims}}), result);
                    std::cout << "    Creating: reshape";
                    auto reshape_op = make_op("reshape", {{"dims", reshape_dims}});
                    print_op_attrs(reshape_op);
                    std::cout << std::endl;
                    print_output(reshape_result);
                    return reshape_result;
                }
                
                // Default: pass through unchanged
                std::cout << "    Creating: broadcast (unchanged)";
                print_op_attrs(op);
                std::cout << std::endl;
                auto result = m.insert_instruction(ins, op, inputs, mod_args);
                print_output(result);
                return result;
            }
            
            // TODO
            // Multibroadcast: adjust output shape if it matches BNSM pattern
            if(op_name == "multibroadcast") {
                auto out_lens = op.to_value()["out_lens"].to_vector<std::size_t>();

                // if scalar, or 1D of len 1, can add group dim to out_lens directly
                if(out_lens.size() == 0 || (out_lens.size() == 1 && out_lens[0] == 1)) {
                    std::cout << "------------Broadcasting directly:\n\n\n";
                    out_lens.insert(out_lens.begin(), num_groups);
                    out_lens.push_back(dims.seq_length_per_group);
                    auto new_op = make_op("multibroadcast", {{"out_lens", out_lens}});
                    std::cout << "    Creating: multibroadcast";
                    print_op_attrs(new_op);
                    std::cout << std::endl;
                    auto result = m.insert_instruction(ins, new_op, inputs, mod_args);
                    print_output(result);
                    return result;
                } else if (inputs[0]->get_shape().lens().size() > out_lens.size()) {
                    // if the input rank does not match the length of out_lens,
                    // then that means the input has already been adjusted to have the
                    // group dim and seq_len_per_group, so we can edit the out_lens directly
                    // TODO: remove magic nums, here and elsewhere
                    // new out_lens has groups at -3 and changes -1 from max_len to seq_len_per_group
                    std::vector<std::size_t> new_lens = {out_lens[0], out_lens[1], num_groups, out_lens[2], dims.seq_length_per_group};
                    auto new_op = make_op("multibroadcast", {{"out_lens", new_lens}});
                    std::cout << "    Creating: multibroadcast";
                    print_op_attrs(new_op);
                    std::cout << std::endl;
                    auto result = m.insert_instruction(ins, new_op, inputs, mod_args);
                    print_output(result);
                    return result;
                } else if(out_lens.size() == 2) {
                    // keep the multibroadcast as is
                    // follow up with a reshape to split the sequence dimension into groups
                    std::cout << "    multibroadcast unchanged - will add reshape" << std::endl;
                    auto result = m.insert_instruction(ins, op, inputs, mod_args);
                    std::cout << "    Creating: multibroadcast (unchanged)";   
                    print_op_attrs(op);
                    std::cout << std::endl;
                    print_output(result);
                    // Add reshape to split sequence dimension into groups
                    std::vector<std::size_t> reshape_dims = {dims.batch_size, num_groups, dims.seq_length_per_group};
                    std::cout << "\n>>> Auto-inserting reshape after 2D multibroadcast" << std::endl;
                    std::cout << "    Reshape dims: ";
                    print_shape(reshape_dims);
                    std::cout << std::endl;
                    
                    auto reshape_result = m.insert_instruction(ins, make_op("reshape", {{"dims", reshape_dims}}), result);
                    std::cout << "    Creating: reshape";
                    auto reshape_op = make_op("reshape", {{"dims", reshape_dims}});
                    print_op_attrs(reshape_op);
                    std::cout << std::endl;
                    print_output(reshape_result);
                    return reshape_result;
                } else if(out_lens.size() == 4) {
                    // keep the multibroadcast as is
                    // follow up with a reshape to split the sequence dimension into groups
                    std::cout << "    multibroadcast unchanged - will add reshape" << std::endl;
                    auto result = m.insert_instruction(ins, op, inputs, mod_args);
                    std::cout << "    Creating: multibroadcast (unchanged)";   
                    print_op_attrs(op);
                    std::cout << std::endl;
                    print_output(result);
                    // Add reshape to split sequence dimension into groups
                    std::vector<std::size_t> reshape_dims = {out_lens[0], out_lens[1], num_groups, out_lens[2], dims.seq_length_per_group};
                    std::cout << "\n>>> Auto-inserting reshape after 4D multibroadcast" << std::endl;
                    std::cout << "    Reshape dims: ";
                    print_shape(reshape_dims);
                    std::cout << std::endl;
                    
                    auto reshape_result = m.insert_instruction(ins, make_op("reshape", {{"dims", reshape_dims}}), result);
                    std::cout << "    Creating: reshape";
                    auto reshape_op = make_op("reshape", {{"dims", reshape_dims}});
                    print_op_attrs(reshape_op);
                    std::cout << std::endl;
                    print_output(reshape_result);
                    return reshape_result;
                } 
                // TODO else
            }
            
            // Unsqueeze: adjust axes
            if(op_name == "unsqueeze") {
                auto axes = op.to_value()["axes"].to_vector<int64_t>();
                auto new_axes = adjust_axes(axes, group_dim_pos);
                
                std::cout << "    Adjusted axes: [";
                for(size_t i = 0; i < axes.size(); ++i)
                    std::cout << axes[i] << (i < axes.size()-1 ? "," : "");
                std::cout << "] -> [";
                for(size_t i = 0; i < new_axes.size(); ++i)
                    std::cout << new_axes[i] << (i < new_axes.size()-1 ? "," : "");
                std::cout << "]" << std::endl;
                
                auto new_op = make_op("unsqueeze", {{"axes", new_axes}});
                std::cout << "    Creating: unsqueeze";
                print_op_attrs(new_op);
                std::cout << std::endl;
                auto result = m.insert_instruction(ins, new_op, inputs, mod_args);
                print_output(result);
                return result;
            }
            
            // Reshape: need to adjust dims if they span the group dimension
            // TODO
            if(op_name == "reshape") {
                auto dims_vec = op.to_value()["dims"].to_vector<std::size_t>();
                
                // Check if this is a mask reshape that needs to split the sequence dimension
                // e.g., {batch, max_seq_length} -> {batch, num_groups, seq_per_group}
                if(dims_vec.size() == 2 && dims_vec[1] == dims.max_seq_length) {
                    std::vector<std::size_t> new_dims = {dims.batch_size, num_groups, dims.seq_length_per_group};
                    std::cout << "    Adjusted dims: ";
                    print_shape(dims_vec);
                    std::cout << " -> ";
                    print_shape(new_dims);
                    std::cout << std::endl;
                    
                    auto new_op = make_op("reshape", {{"dims", new_dims}});
                    std::cout << "    Creating: reshape";
                    print_op_attrs(new_op);
                    std::cout << std::endl;
                    auto result = m.insert_instruction(ins, new_op, inputs, mod_args);
                    print_output(result);
                    return result;
                }
            }
            
            // Default: copy operation as-is
            std::cout << "    Creating: " << op_name << " (unchanged)";
            print_op_attrs(op);
            std::cout << std::endl;
            auto result = m.insert_instruction(ins, op, inputs, mod_args);
            print_output(result);
            return result;
        };
        
        // gemm2 should be the last ins transformed programmatically
        instruction_ref stop_point = std::next(gemm2);
        std::unordered_map<instruction_ref, instruction_ref> map_old_to_new = param_map;
        target_mod.add_instructions(source_mod.begin(), stop_point, &map_old_to_new, inserter);
        
        if(!contains(map_old_to_new, gemm2)) return;
        second_dot_result = map_old_to_new.at(gemm2);
        std::cout << "\n=== Adding final transpose and reshape for flash decoding ===" << std::endl;
        
        // final transpose and reshape need to be handled specially
        // transpose: {B, N, G, S, D} -> {B, G, S, N, D}
        std::cout << "Adding transpose with permutation {0, 2, 3, 1, 4}" << std::endl;
        auto transpose_out = target_mod.add_instruction(
            make_op("transpose", {{"permutation", {0, 2, 3, 1, 4}}}),
            second_dot_result);
        std::cout << "Transpose output shape: " << transpose_out->get_shape() << std::endl;
        
        // reshape: {B, G, S, N, D} -> {B, G, S, N*D}
        std::vector<std::size_t> final_shape = {dims.batch_size, num_groups, dims.sequence_length, dims.num_heads * dims.head_dim};
        std::cout << "Adding reshape with dims {";
        for(size_t i = 0; i < final_shape.size(); ++i) {
            std::cout << final_shape[i];
            if(i < final_shape.size() - 1) std::cout << ",";
        }
        std::cout << "}" << std::endl;
        auto reshape_out = target_mod.add_instruction(
            make_op("reshape", {{"dims", final_shape}}),
            transpose_out);
        std::cout << "Reshape output shape: " << reshape_out->get_shape() << std::endl;
        
        // Calculate LSE (log-sum-exp) from the tracked reduce operations
        // LSE = log(sum_exp) + max
        if(found_reduce_max && found_reduce_sum) {
            auto log_sum = target_mod.add_instruction(make_op("log"), reduce_sum_ref);
            auto lse = target_mod.add_instruction(make_op("add"), reduce_max_ref, log_sum);
            std::cout << "LSE shape: " << lse->get_shape() << std::endl;
            
            target_mod.add_return({reshape_out, lse});
        } else {
            return;
        }
        
        // print the complete submodule
        std::cout << "\n=== Complete GQA Flash Decoding Submodule ===" << std::endl;
        std::cout << target_mod << std::endl;
        std::cout << "=== End Submodule ===" << std::endl;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& mm            = mpm.get_module();
        auto attn_group_ins = r.instructions["group"];
        auto* submod        = attn_group_ins->module_inputs().front();

        std::cout << "GQA flash decoding detected, here is the submodule: " << std::endl;
        submod->debug_print();

        // extract Q, K, V parameters from gemm inputs
        auto [gemm1, gemm2] = get_attention_gemms(submod);
        auto qkv_opt = qkv_params::from_gemms(gemm1, gemm2);
        if(not qkv_opt) return;
        auto [q_param, k_param, v_param] = *qkv_opt;

        // derive attention dims from Q, K, V parameters
        attention_dims dims(q_param, k_param, groups);

        if(groups <= 1 or dims.seq_length_per_group == 0) {
            return;
        }

        // map submodule params to main module inputs
        auto group_inputs      = attn_group_ins->inputs();
        auto map_param_to_main = map_submod_params_to_inputs(submod, group_inputs);

        // get actual Q, K, V instructions from main module
        auto q = map_param_to_main.at(q_param);  // maps to the QKV tensor
        auto k = map_param_to_main.at(k_param);  // maps to K concat_past_present output 
        auto v = map_param_to_main.at(v_param);  // maps to V concat_past_present output

        // GQA flash decoding:
        // - Q (QKV tensor): add new group dim and broadcast
        // - K: split sequence dimension into groups
        // - V: split sequence dimension into groups
        auto q_type = q->get_shape().type();
        auto k_type = k->get_shape().type(); 
        auto v_type = v->get_shape().type();
        
        // insert group dimension at position -2 for all tensors
        // K and V: [B, kv_heads, N, D] -> [B, kv_heads, G, N/G, D]
        // build transformed shapes
        std::vector<std::size_t> q_transformed_shape;
        std::vector<std::size_t> k_transformed_shape;
        std::vector<std::size_t> v_transformed_shape;

        q_transformed_shape = {dims.batch_size, dims.concat_heads, groups, dims.sequence_length, dims.head_dim};
        k_transformed_shape = {dims.batch_size, dims.kv_heads, groups, dims.seq_length_per_group, dims.head_dim};
        v_transformed_shape = {dims.batch_size, dims.kv_heads, groups, dims.seq_length_per_group, dims.head_dim};

        // insert reshape operations before the attention group
        // [B, concat_heads, seq, head_dim] -> [B, concat_heads, 1, seq, head_dim] -> [B, concat_heads, G, seq, head_dim]
        auto q_unsqueezed = mm.insert_instruction(
            attn_group_ins, 
            make_op("unsqueeze", {{"axes", {-2}}}),
            q);
        
        auto q_reshaped = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", q_transformed_shape}}),
            q_unsqueezed);
        
        // [B, kv_heads, N, D] -> [B, kv_heads, G, N/G, D]
        auto k_reshaped = mm.insert_instruction(
            attn_group_ins,
            make_op("reshape", {{"dims", k_transformed_shape}}),
            k);
        
        // [B, kv_heads, N, D] -> [B, kv_heads, G, N/G, D]
        auto v_reshaped = mm.insert_instruction(
            attn_group_ins,
            make_op("reshape", {{"dims", v_transformed_shape}}),
            v);
        
        // No need to reshape additional inputs
        // We'll adjust broadcast patterns inside for masking

        // create new input list, starting with replacing Q, K, V with reshaped versions
        std::vector<instruction_ref> new_group_inputs = group_inputs;
        for(size_t i = 0; i < group_inputs.size(); ++i) {
            if(group_inputs[i] == q) {
                new_group_inputs[i] = q_reshaped;
            } else if(group_inputs[i] == k) {
                new_group_inputs[i] = k_reshaped;
            } else if(group_inputs[i] == v) {
                new_group_inputs[i] = v_reshaped;
            }
        }

        module m_flash_decode;
        m_flash_decode.set_bypass();
        
        // get parameter names from original submodule
        auto get_param_name = [](instruction_ref param) -> std::string {
            assert(param->name() == "@param");
            return param->get_operator().to_value()["parameter"].to<std::string>();
        };
        
        auto q_name = get_param_name(q_param);
        auto k_name = get_param_name(k_param);
        auto v_name = get_param_name(v_param);
        
        // Add parameters to new submodule with transformed shapes
        auto new_q_param = m_flash_decode.add_parameter(
            q_name, shape{q_type, q_transformed_shape});
        auto new_k_param = m_flash_decode.add_parameter(
            k_name, shape{k_type, k_transformed_shape});
        auto new_v_param = m_flash_decode.add_parameter(
            v_name, shape{v_type, v_transformed_shape});
        
        // Build mapping from old params to new params
        std::unordered_map<instruction_ref, instruction_ref> map_old_params_to_new;
        map_old_params_to_new[q_param] = new_q_param;
        map_old_params_to_new[k_param] = new_k_param;
        map_old_params_to_new[v_param] = new_v_param;
        
        // add the rest of the parameters
        for(auto param : iterator_for(*submod)) {
            if(param->name() == "@param") {
                if(not contains(map_old_params_to_new, param)) {
                    auto param_name = get_param_name(param);
                    auto param_shape = param->get_shape();
                    auto new_param = m_flash_decode.add_parameter(param_name, param_shape);
                    map_old_params_to_new[param] = new_param;
                }
            }
        }
        
        // rebuild the attention operations in the flash decode submodule
        rebuild_gqa_attention(m_flash_decode, *submod, map_old_params_to_new, 
                             gemm2, dims, groups);

        // create the module in the module pass manager and insert the new group operation
        auto orig_name = attn_group_ins->module_inputs().front()->name();
        std::string new_mod_name = orig_name + "_gqa_flash_decoding";
        
        module_ref mpm_flash_mod = mpm.create_module(new_mod_name, std::move(m_flash_decode));
        mpm_flash_mod->set_bypass();
        
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
        // each head has its own LSE values and needs its own weights
        // O' shape: [B, G, S, N*D] - heads concatenated
        // LSE shape: [B, N, G, S, 1] - per-head
        std::cout << "\n=== Kernel 2: LSE-weighted combination ===" << std::endl;
        std::cout << "Input LSE shape: " << lse->get_shape() << std::endl;
        std::cout << "Input O' shape: " << partial_output_o_prime->get_shape() << std::endl;
        
        auto output_type = partial_output_o_prime->get_shape().type();
        
        // reshape O' to separate heads: [B, G, S, N*D] -> [B, G, S, N, D]
        std::vector<std::size_t> o_prime_per_head_shape = {
            dims.batch_size, groups, dims.sequence_length, dims.num_heads, dims.head_dim};
        auto o_prime_reshaped = mm.insert_instruction(
            attn_group_ins,
            make_op("reshape", {{"dims", o_prime_per_head_shape}}),
            partial_output_o_prime);
        
        // in groups with all masked positions, softmax produces NaN (from -inf - (-inf))
        auto o_prime_is_nan = mm.insert_instruction(
            attn_group_ins, make_op("isnan"), o_prime_reshaped);
        auto zero_lit = mm.add_literal(literal{shape{output_type, {1}}, {0}});
        auto zero_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", o_prime_reshaped->get_shape().lens()}}),
            zero_lit);
        auto o_prime_cleaned = mm.insert_instruction(
            attn_group_ins, make_op("where"), o_prime_is_nan, zero_bcast, o_prime_reshaped);
        
        // [B, N, G, S, 1] -> [B, N, G, S]
        auto lse_squeezed = mm.insert_instruction(
            attn_group_ins, make_op("squeeze", {{"axes", {4}}}), lse);
        
        // compute weights per-head
        // [B, N, G, S] -> [B, N, 1, S]
        auto lse_max = mm.insert_instruction(
            attn_group_ins, make_op("reduce_max", {{"axes", {2}}}), lse_squeezed);
        
        // broadcast max back to original shape
        // [B, N, 1, S] -> [B, N, G, S]
        auto lse_max_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", lse_squeezed->get_shape().lens()}}),
            lse_max);
        
        // exp(LSE - max_LSE)
        auto lse_sub = mm.insert_instruction(attn_group_ins, make_op("sub"), lse_squeezed, lse_max_bcast);
        auto lse_exp = mm.insert_instruction(attn_group_ins, make_op("exp"), lse_sub);
        
        // sum exp across groups
        // [B, N, G, S] -> [B, N, 1, S]
        auto lse_sum = mm.insert_instruction(
            attn_group_ins, make_op("reduce_sum", {{"axes", {2}}}), lse_exp);
        
        // [B, N, 1, S] -> [B, N, G, S]
        auto lse_sum_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", lse_exp->get_shape().lens()}}),
            lse_sum);
        
        // weights per-head
        auto weights = mm.insert_instruction(attn_group_ins, make_op("div"), lse_exp, lse_sum_bcast);
        
        // transpose weights to align with O'
        // [B, N, G, S] -> [B, G, S, N]
        auto weights_transposed = mm.insert_instruction(
            attn_group_ins, make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), weights);
        
        // [B, G, S, N] -> [B, G, S, N, 1]
        auto weights_unsqueezed = mm.insert_instruction(
            attn_group_ins, make_op("unsqueeze", {{"axes", {4}}}), weights_transposed);
        
        // broadcast to match O' shape
        // [B, G, S, N, 1] -> [B, G, S, N, D]
        auto weights_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", o_prime_cleaned->get_shape().lens()}}),
            weights_unsqueezed);
        
        auto weights_converted = mm.insert_instruction(
            attn_group_ins, make_op("convert", {{"target_type", output_type}}), weights_bcast);
        
        // multiply O' by per-head weights
        // [B, G, S, N, D] -> [B, G, S, N, D]
        auto weighted_output = mm.insert_instruction(
            attn_group_ins, make_op("mul"), o_prime_cleaned, weights_converted);
        
        // sum across groups
        // [B, G, S, N, D] -> [B, 1, S, N, D]
        auto summed_output = mm.insert_instruction(
            attn_group_ins, make_op("reduce_sum", {{"axes", {1}}}), weighted_output);
        
        // squeeze the reduced group dimension
        // [B, 1, S, N, D] -> [B, S, N, D]
        auto squeezed_output = mm.insert_instruction(
            attn_group_ins, make_op("squeeze", {{"axes", {1}}}), summed_output);
        
        // reshape back to concatenated heads
        // [B, S, N, D] -> [B, S, N*D]
        std::vector<std::size_t> final_shape = {
            dims.batch_size, dims.sequence_length, dims.num_heads * dims.head_dim};
        auto final_squeezed = mm.insert_instruction(
            attn_group_ins, make_op("reshape", {{"dims", final_shape}}), squeezed_output);
        
        mm.replace_instruction(attn_group_ins, final_squeezed);
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

        // if LSE attn, do not do flash decoding
        auto return_ins = std::prev(submod->end());
        assert(return_ins->name() == "@return" and
               "Last instruction must be a @return instruction");
        if(return_ins->inputs().size() > 1)
            return;

        // get gemm1 and gemm2
        auto [gemm1, gemm2] = get_attention_gemms(submod);

        // no input fusion
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

        // kernel 2: combine using exp-normalize trick
        // O = sum(O' * exp(LSE - max)) / sum(exp(LSE - max))
        std::cout << "=== Kernel 2 shapes ===" << std::endl;
        std::cout << "partial_output_o_prime: " << partial_output_o_prime->get_shape() << std::endl;
        std::cout << "lse: " << lse->get_shape() << std::endl;
        std::cout << "g_axis: " << g_axis << std::endl;

        // find max LSE across groups for numerical stability
        auto lse_max = mm.insert_instruction(
            attn_group_ins, make_op("reduce_max", {{"axes", {g_axis}}}), lse);
        std::cout << "lse_max (reduce_max): " << lse_max->get_shape() << std::endl;

        auto lse_max_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", lse->get_shape().lens()}}),
            lse_max);
        std::cout << "lse_max_bcast (multibroadcast): " << lse_max_bcast->get_shape() << std::endl;

        // compute unnormalized weights
        // exp(LSE - max)
        auto lse_sub = mm.insert_instruction(
            attn_group_ins, make_op("sub"), lse, lse_max_bcast);
        std::cout << "lse_sub (sub): " << lse_sub->get_shape() << std::endl;

        auto lse_exp = mm.insert_instruction(
            attn_group_ins, make_op("exp"), lse_sub);
        std::cout << "lse_exp (exp): " << lse_exp->get_shape() << std::endl;

        // broadcast weights to match O' shape
        // [B, G, M] -> [B, G, M, D]
        auto lse_exp_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", partial_output_o_prime->get_shape().lens()}}),
            lse_exp);
        std::cout << "lse_exp_bcast (multibroadcast): " << lse_exp_bcast->get_shape() << std::endl;

        // convert weights to output type
        auto output_type = partial_output_o_prime->get_shape().type();
        auto weights = mm.insert_instruction(
            attn_group_ins, make_op("convert", {{"target_type", output_type}}), lse_exp_bcast);
        std::cout << "weights (convert): " << weights->get_shape() << std::endl;

        // compute weighted sum: numerator = sum(O' * weights)
        auto weighted_o = mm.insert_instruction(
            attn_group_ins, make_op("mul"), partial_output_o_prime, weights);
        std::cout << "weighted_o (mul): " << weighted_o->get_shape() << std::endl;

        auto numerator = mm.insert_instruction(
            attn_group_ins, make_op("reduce_sum", {{"axes", {g_axis}}}), weighted_o);
        std::cout << "numerator (reduce_sum): " << numerator->get_shape() << std::endl;

        // compute sum of weights: denominator = sum(weights)
        auto denominator = mm.insert_instruction(
            attn_group_ins, make_op("reduce_sum", {{"axes", {g_axis}}}), weights);
        std::cout << "denominator (reduce_sum): " << denominator->get_shape() << std::endl;

        // final division: O = numerator / denominator
        auto final_output_o = mm.insert_instruction(
            attn_group_ins, make_op("div"), numerator, denominator);
        std::cout << "final_output_o (div): " << final_output_o->get_shape() << std::endl;

        // squeeze G to match the original output shape
        auto final_squeezed_o = mm.insert_instruction(
            attn_group_ins, make_op("squeeze", {{"axes", {g_axis}}}), final_output_o);
        std::cout << "final_squeezed_o (squeeze): " << final_squeezed_o->get_shape() << std::endl;
        std::cout << "=== End Kernel 2 ===" << std::endl;

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
        std::cout << "module before flash decoding: " << std::endl;
        mpm.get_module().debug_print();
        // flash decoding for regular attention, single & multi-headed
        match::find_matches(
            mpm,
            find_flash_decoding{.configured_splits         = configured_splits,
                                .configured_threshold      = flash_decoding_threshold,
                                .configured_max_splits     = flash_decoding_max_splits,
                                .configured_min_chunk_size = flash_decoding_min_chunk_size});

        // flash decoding for GQA attention
        match::find_matches(
            mpm, find_gqa_flash_decoding{.groups = configured_splits});
        mpm.run_pass(dead_code_elimination{});
        std::cout << "module after flash decoding: " << std::endl;
        mpm.get_module().debug_print();
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
