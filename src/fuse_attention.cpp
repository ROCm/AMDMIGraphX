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

        for(auto ai: attn_inss)
        {
            if(ai->name() == "add")
            {
                for(auto inp : ai->inputs())
                {
                    if(inp->name() == "multibroadcast")
                    {
                        attn_inss.insert(inp);
                        attn_inss.insert(inp->inputs().front());
                        // for (auto inp2 : inp->inputs())
                        // {
                        //     if(inp2->name() == "where")
                        //     {
                        //         attn_inss.insert(inp);
                        //         attn_inss.insert(inp2);
                        //         // auto inp3 = inp2->inputs().front();
                        //         // if(inp3->name() == "convert")
                        //         // {
                        //         //     auto inp4 = inp3->inputs().front();
                        //         //     std::cout << "found where, 1st input: ";
                        //         //     std::cout << inp4->name() << std::endl;
                        //         //     if(inp4->name() == "equal")
                        //         //     {
                        //         //         attn_inss.insert(inp3);
                        //         //         attn_inss.insert(inp4);
                        //         //     }
                        //         // }
                        //     }
                        // }
                    }
                }
            }
        }

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
        std::cout << "Matched attention" << std::endl;
        auto gemm2         = r.result;
        auto gemm1         = r.instructions["dot1"];
        auto softmax_input = r.instructions["x"];

        // Capture all instructions part of the attention op
        auto attn_inss = get_attn_instructions(mpm.get_module(), gemm1, gemm2);

        for(auto ai : attn_inss)
        {
            ai->debug_print();
        }

        // for(auto ai: attn_inss)
        // {
        //     if(ai->name() == "add")
        //     {
        //         for(auto inp : ai->inputs())
        //         {
        //             if(inp->name() == "multibroadcast")
        //             {
        //                 for (auto inp2 : inp->inputs())
        //                 {
        //                     if(inp2->name() == "where")
        //                     {
        //                         attn_inss.push_back(inp2);
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

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
        // else if(required_outputs.size() > 1)
        // {
        //     return;
        // }

        // Find corresponding output instructions in m_attn
        std::vector<instruction_ref> m_attn_outputs;
        std::transform(required_outputs.begin(),
                       required_outputs.end(),
                       std::back_inserter(m_attn_outputs),
                       [&](auto i) { return map_mm_to_mattn.at(i); });
        std::cout << "Required outputs: " << std::endl;
        for (auto ro : required_outputs)
        {
            ro->debug_print();
        }

        m_attn.add_return({m_attn_outputs.back()});

        // Define inputs to m_attn
        auto map_mattn_to_mm = invert_map_ins(map_mm_to_mattn);
        auto new_inputs      = m_attn.get_inputs(map_mattn_to_mm);

        module_ref mpm_attn = mpm.create_module("attn" + get_count(), std::move(m_attn));
        mpm_attn->set_bypass();
        std::cout << "Attn module: " << std::endl;
        mpm_attn->debug_print();

        auto group_ins = mpm.get_module().insert_instruction(
            softmax_input, make_op("group", {{"tag", "attention"}}), new_inputs, {mpm_attn});

        if(m_attn_outputs.size() == 1)
        {
            mpm.get_module().replace_instruction(required_outputs.front(), group_ins);
        }
        else
            mpm.get_module().replace_instruction(required_outputs.back(), group_ins);
        // else
        // {
        //     for(std::size_t i = 0; i < required_outputs.size(); ++i)
        //     {
        //         mpm.get_module().replace_instruction(
        //             required_outputs[i], make_op("get_tuple_elem", {{"index", i}}), group_ins);
        //     }
        // }
    }
};

struct find_causal_attention
{
    std::size_t* counter;

    auto matcher() const
    {
        auto gemm1   = match::name("dot")(match::any_of[match::inputs()](match::pointwise()));
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
                                                                       "@literal"};

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
        std::cout << "Matched causal attn" << std::endl;
        auto pw = r.instructions["pointwise"];
        pw->debug_print();
        return;
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

        auto outs = m_attn.add_return({m_attn_outputs.back()});

        // Define inputs to m_attn
        auto map_mattn_to_mm = invert_map_ins(map_mm_to_mattn);
        auto new_inputs      = m_attn.get_inputs(map_mattn_to_mm);

        module_ref mpm_attn = mpm.create_module("attn" + get_count(), std::move(m_attn));
        mpm_attn->set_bypass();
        mpm_attn->debug_print();
        // Construct group op with the attention module
        auto group_ins =
            mpm.get_module().insert_instruction(required_outputs.back(),
                                                make_op("group", {{"tag", "attention"}}),
                                                new_inputs,
                                                {mpm_attn});

        mpm.get_module().replace_instruction(required_outputs.back(), group_ins);
    }
};

} // namespace

void fuse_attention::apply(module_pass_manager& mpm) const
{
    std::cout << "Fuse attention" << std::endl;
    mpm.get_module().debug_print();
    std::size_t counter = 0;
    match::find_matches(mpm, find_causal_attention{.counter = &counter});
    mpm.get_module().sort();
    mpm.run_pass(dead_code_elimination{});
    match::find_matches(mpm, find_attention{.counter = &counter});
    mpm.get_module().sort();
    mpm.run_pass(dead_code_elimination{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
