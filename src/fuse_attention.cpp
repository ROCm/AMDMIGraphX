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
#include <migraphx/dead_code_elimination.hpp>
#include <queue>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

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

// Helper functions for flash decoding
namespace {

// Determine a good group size G for flash decoding
// We want to split N into G groups where each group has N/G elements
std::size_t choose_group_size(std::size_t N) {
    // Prefer powers of 2 for better memory alignment
    // Aim for group sizes between 16 and 128 for good performance
    std::vector<std::size_t> preferred_sizes = {128, 64, 32, 16, 8, 4, 2}; // Check largest first
    
    for (auto G : preferred_sizes) {
        if (N % G == 0 && N / G >= 16) {  // Each group should have at least 16 elements
            return G;
        }
    }
    
    // Fallback: find largest divisor that creates reasonable group sizes
    for (std::size_t G = std::min(N / 16, static_cast<std::size_t>(128)); G >= 2; --G) {
        if (N % G == 0) {
            return G;
        }
    }
    
    return 1; // No grouping beneficial
}

// Check if flash decoding would be beneficial for given shapes
bool should_use_flash_decoding(const std::vector<shape>& input_shapes) {
    // Need exactly 3 inputs: Q, K, V
    if (input_shapes.size() != 3) return false;
    
    const auto& q_shape = input_shapes[0];
    const auto& k_shape = input_shapes[1]; 
    const auto& v_shape = input_shapes[2];
    
    // All should have same number of dimensions (at least 3)
    if (q_shape.ndim() < 3 || k_shape.ndim() != q_shape.ndim() || v_shape.ndim() != q_shape.ndim())
        return false;
    
    std::size_t ndim = q_shape.ndim();
    
    // Check batch dimensions match
    for (std::size_t i = 0; i < ndim - 2; ++i) {
        if (q_shape.lens()[i] != k_shape.lens()[i] || q_shape.lens()[i] != v_shape.lens()[i])
            return false;
    }
    
    // For shapes [..., M, k] x [..., k, N] x [..., N, D]
    std::size_t M = q_shape.lens()[ndim - 2];
    std::size_t k_q = q_shape.lens()[ndim - 1];
    std::size_t k_k = k_shape.lens()[ndim - 2];
    std::size_t N = k_shape.lens()[ndim - 1];
    std::size_t N_v = v_shape.lens()[ndim - 2];
    std::size_t D = v_shape.lens()[ndim - 1];
    
    // Check dimension consistency
    if (k_q != k_k || N != N_v) return false;
    
    // Check if grouping would be beneficial
    std::size_t G = choose_group_size(N);
    return G > 1; // Only worthwhile if we can actually group
}

// Transform shapes for flash decoding by adding group dimension
std::vector<shape> transform_shapes_for_flash_decoding(const std::vector<shape>& input_shapes) {
    const auto& q_shape = input_shapes[0];
    const auto& k_shape = input_shapes[1];
    const auto& v_shape = input_shapes[2];
    
    std::size_t ndim = q_shape.ndim();
    std::size_t N = k_shape.lens()[ndim - 1];
    std::size_t G = choose_group_size(N);
    std::size_t N_per_group = N / G;
    
    // Transform Q: [..., M, k] -> [..., G, M, k] (G is broadcasted)
    auto q_lens = q_shape.lens();
    q_lens.insert(q_lens.end() - 2, G);
    shape new_q_shape{q_shape.type(), q_lens};
    
    // Transform K: [..., k, N] -> [..., G, k, N/G] 
    auto k_lens = k_shape.lens();
    k_lens[ndim - 1] = N_per_group;  // Change N to N/G
    k_lens.insert(k_lens.end() - 2, G);
    shape new_k_shape{k_shape.type(), k_lens};
    
    // Transform V: [..., N, D] -> [..., G, N/G, D]
    auto v_lens = v_shape.lens();
    v_lens[ndim - 2] = N_per_group;  // Change N to N/G  
    v_lens.insert(v_lens.end() - 2, G);
    shape new_v_shape{v_shape.type(), v_lens};
    
    return {new_q_shape, new_k_shape, new_v_shape};
}

} // namespace

struct find_flash_decoding
{
    std::size_t* counter;
    
    auto matcher() const
    {
        return match::name("group")(match::attribute("tag", "attention"));
    }
    
    std::string get_count() const { return std::to_string((*counter)++); }
    
    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto group_ins = r.result;
        auto inputs = group_ins->inputs();
        
        // Group instructions have regular inputs followed by module refs
        // We need to find where the module refs start
        std::vector<instruction_ref> regular_inputs;
        for (auto input : inputs) {
            if (input->get_operator().name() != "@module_ref") {
                regular_inputs.push_back(input);
            } else {
                break; // Module refs come at the end
            }
        }
        
        // Get input shapes
        std::vector<shape> input_shapes;
        for (auto input : regular_inputs) {
            input_shapes.push_back(input->get_shape());
        }
        
        // Check if this attention group should use flash decoding
        if (!should_use_flash_decoding(input_shapes)) {
            return; // Keep as regular attention
        }
        
        // Transform to flash decoding
        auto transformed_shapes = transform_shapes_for_flash_decoding(input_shapes);
        std::size_t G = choose_group_size(input_shapes[1].lens().back());
        
        // Create new flash decoding module
        module m_flash;
        
        // Add parameters with transformed shapes
        auto q_param = m_flash.add_parameter("q", transformed_shapes[0].as_standard());
        auto k_param = m_flash.add_parameter("k", transformed_shapes[1].as_standard()); 
        auto v_param = m_flash.add_parameter("v", transformed_shapes[2].as_standard());
        
        // First kernel: compute per-group attention with LSE
        auto s = m_flash.add_instruction(make_op("dot"), q_param, k_param);
        auto p = m_flash.add_instruction(make_op("softmax", {{"axis", -1}}), s);
        
        // Compute LSE: L = log(sum(exp(S), axis=-1))
        auto exp_s = m_flash.add_instruction(make_op("exp"), s);
        auto sum_exp = m_flash.add_instruction(make_op("reduce_sum", {{"axes", std::vector<int64_t>{-1}}}), exp_s);
        auto l = m_flash.add_instruction(make_op("log"), sum_exp);
        
        auto o_prime = m_flash.add_instruction(make_op("dot"), p, v_param);
        
        // Second kernel: scale and combine across groups
        auto scale = m_flash.add_instruction(make_op("softmax", {{"axis", static_cast<int64_t>(transformed_shapes[0].ndim() - 3)}}), l);
        auto scale_bc = m_flash.add_instruction(
            make_op("multibroadcast", {{"out_lens", o_prime->get_shape().lens()}}), scale);
        auto r = m_flash.add_instruction(make_op("mul"), o_prime, scale_bc);
        auto o = m_flash.add_instruction(make_op("reduce_sum", {{"axes", std::vector<int64_t>{static_cast<int64_t>(transformed_shapes[0].ndim() - 3)}}}), r);
        
        // Squeeze out the group dimension
        auto final_o = m_flash.add_instruction(make_op("squeeze", {{"axes", std::vector<int64_t>{static_cast<int64_t>(transformed_shapes[0].ndim() - 3)}}}), o);
        
        m_flash.add_return({final_o});
        
        // Create transformed input instructions
        std::vector<instruction_ref> new_inputs;
        
        // Transform Q: add broadcast dimension for G
        auto q_input = regular_inputs[0];
        auto q_unsqueeze = mpm.get_module().insert_instruction(
            group_ins, 
            make_op("unsqueeze", {{"axes", std::vector<int64_t>{static_cast<int64_t>(input_shapes[0].ndim() - 2)}}}), 
            q_input);
        new_inputs.push_back(mpm.get_module().insert_instruction(
            group_ins,
            make_op("multibroadcast", {{"out_lens", transformed_shapes[0].lens()}}),
            q_unsqueeze));
        
        // Transform K: reshape to add group dimension
        auto k_input = regular_inputs[1];
        new_inputs.push_back(mpm.get_module().insert_instruction(
            group_ins,
            make_op("reshape", {{"dims", transformed_shapes[1].lens()}}),
            k_input));
        
        // Transform V: reshape to add group dimension  
        auto v_input = regular_inputs[2];
        new_inputs.push_back(mpm.get_module().insert_instruction(
            group_ins,
            make_op("reshape", {{"dims", transformed_shapes[2].lens()}}),
            v_input));
        
        // Create flash decoding module reference
        module_ref mpm_flash = mpm.create_module("flash_decode" + get_count(), std::move(m_flash));
        mpm_flash->set_bypass();
        
        // Replace group with flash decoding group
        auto flash_group = mpm.get_module().insert_instruction(
            group_ins, 
            make_op("group", {{"tag", "flash_decoding"}}), 
            new_inputs, 
            {mpm_flash});
        
        mpm.get_module().replace_instruction(group_ins, flash_group);
    }
};

} // namespace

void fuse_attention::apply(module_pass_manager& mpm) const
{
    std::size_t counter = 0;
    // First, run regular attention fusion
    match::find_matches(mpm, find_attention{.counter = &counter});
    mpm.get_module().sort();
    
    // Then, look for attention groups that can be converted to flash decoding
    std::size_t flash_counter = 0;
    match::find_matches(mpm, find_flash_decoding{.counter = &flash_counter});
    mpm.get_module().sort();
    
    mpm.run_pass(dead_code_elimination{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
