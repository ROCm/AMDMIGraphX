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

// Helper to check if shapes match the flash decoding pattern
inline bool is_flash_decoding_pattern(const shape& q_shape, const shape& k_shape, const shape& v_shape)
{
    // Flash decoding pattern: Q -> [B, G, M, k], K -> [B, G, k, N/G], V -> [B, G, N/G, D]
    if(q_shape.ndim() != 4 or k_shape.ndim() != 4 or v_shape.ndim() != 4)
        return false;
    
    // Check batch and group dimensions match
    if(q_shape.lens()[0] != k_shape.lens()[0] or q_shape.lens()[0] != v_shape.lens()[0])
        return false;
    if(q_shape.lens()[1] != k_shape.lens()[1] or q_shape.lens()[1] != v_shape.lens()[1])
        return false;
    
    // Check that k dimension matches between Q and K
    if(q_shape.lens()[3] != k_shape.lens()[2])
        return false;
    
    // Check that N/G dimension matches between K and V
    if(k_shape.lens()[3] != v_shape.lens()[2])
        return false;
    
    return true;
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

        // Check if this should use flash decoding
        bool use_flash_decoding = false;
        if(attn_inss.size() >= 3) {
            auto q_input = gemm1->inputs()[0];
            auto k_input = gemm1->inputs()[1];
            auto v_input = gemm2->inputs()[1];
            use_flash_decoding = is_flash_decoding_pattern(q_input->get_shape(), k_input->get_shape(), v_input->get_shape());
        }

        // Add captured instructions to new submodule
        module m_attn;
        std::unordered_map<instruction_ref, instruction_ref> map_mm_to_mattn;
        
        if(use_flash_decoding) {
            // Create flash decoding module instead of regular attention
            auto q_input = gemm1->inputs()[0];
            auto k_input = gemm1->inputs()[1];
            auto v_input = gemm2->inputs()[1];
            
            // Add parameters
            auto q_param = m_attn.add_parameter("q", q_input->get_shape().as_standard());
            auto k_param = m_attn.add_parameter("k", k_input->get_shape().as_standard());
            auto v_param = m_attn.add_parameter("v", v_input->get_shape().as_standard());
            
            // Flash decoding computation
            auto s = m_attn.add_instruction(make_op("dot"), q_param, k_param);
            auto p = m_attn.add_instruction(make_op("softmax", {{"axis", -1}}), s);
            
            // Compute LSE properly: L = log(sum(exp(S), axis=-1))
            auto exp_s = m_attn.add_instruction(make_op("exp"), s);
            auto sum_exp = m_attn.add_instruction(make_op("reduce_sum", {{"axes", std::vector<int64_t>{-1}}}), exp_s);
            auto l = m_attn.add_instruction(make_op("log"), sum_exp);
            
            auto o_prime = m_attn.add_instruction(make_op("dot"), p, v_param);
            
            auto scale = m_attn.add_instruction(make_op("softmax", {{"axis", 1}}), l);
            auto scale_bc = m_attn.add_instruction(
                make_op("multibroadcast", {{"out_lens", o_prime->get_shape().lens()}}), scale);
            auto r = m_attn.add_instruction(make_op("mul"), o_prime, scale_bc);
            auto o = m_attn.add_instruction(make_op("sum", {{"axes", std::vector<int64_t>{1}}}), r);
            
            m_attn.add_return({o});
            
            module_ref mpm_attn = mpm.create_module("flash_decode" + get_count(), std::move(m_attn));
            mpm_attn->set_bypass();

            auto group_ins = mpm.get_module().insert_instruction(
                softmax_input, make_op("group", {{"tag", "flash_decoding"}}), {q_input, k_input, v_input}, {mpm_attn});

            mpm.get_module().replace_instruction(gemm2, group_ins);
            return;
        }

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

} // namespace

void fuse_attention::apply(module_pass_manager& mpm) const
{
    std::size_t counter = 0;
    match::find_matches(mpm, find_attention{.counter = &counter});
    mpm.get_module().sort();
    mpm.run_pass(dead_code_elimination{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
