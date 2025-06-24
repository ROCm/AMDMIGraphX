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
#include <migraphx/fuse_special_ops.hpp>
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
        // auto gemm1 = match::inputs_of(match::pointwise())(match::name("dot").bind("gemm1"));
        auto gemm1 = match::any_of[pointwise_inputs()](match::name("dot").bind("dot1"));
        return match::name("dot")(match::arg(0)(match::softmax_input(gemm1).bind("div")));
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

    std::unordered_set<instruction_ref> get_attn_instructions(instruction_ref start,
                                                              instruction_ref end) const
    {
        std::queue<instruction_ref> inputs;
        std::unordered_set<instruction_ref> inss;
        inputs.push(end);

        auto is_valid_attn_op = [&](auto i) {
            std::unordered_set<std::string> valid_ops = {
                "reshape", "reduce_sum", "reduce_max", "broadcast", "multibroadcast", "@literal"};
            return i->get_operator().attributes().get("pointwise", false) or
                   contains(valid_ops, i->get_operator().name()) or i == start or i == end;
        };

        while(!inputs.empty())
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
        return inss;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto gemm2       = r.result;
        auto softmax_out = r.instructions["div"];
        auto softmax_inp = r.instructions["x"];
        auto gemm1       = r.instructions["dot1"];

        mpm.get_module().debug_print();
        mpm.get_module().debug_print(gemm2);
        mpm.get_module().debug_print(softmax_out);
        mpm.get_module().debug_print(softmax_inp);
        mpm.get_module().debug_print(gemm1);

        // Capture all instructions part of the attention op
        auto attn_inss = get_attn_instructions(gemm1, gemm2);

        // Transform unordered set to sorted vector to preserve original order
        std::vector<instruction_ref> attn_ins_vec;
        attn_ins_vec.assign(attn_inss.begin(), attn_inss.end());
        std::sort(
            attn_ins_vec.begin(), attn_ins_vec.end(), [&](instruction_ref x, instruction_ref y) {
                return std::distance(mpm.get_module().begin(), x) <
                       std::distance(mpm.get_module().begin(), y);
            });

        std::cout << "\n";
        mpm.get_module().debug_print(attn_ins_vec);

        // Add captured instructions to new submodule
        module m_attn;
        std::unordered_map<instruction_ref, instruction_ref> map_mm_to_mattn;
        auto attn_outs = m_attn.fuse(attn_ins_vec, &map_mm_to_mattn);

        std::cout << "\n";
        m_attn.debug_print();
        m_attn.debug_print(attn_outs);

        // Define outputs based on instructions that are used elsewhere in the graph
        std::vector<instruction_ref> required_outputs;
        std::copy_if(attn_ins_vec.begin(),
                     attn_ins_vec.end(),
                     std::back_inserter(required_outputs),
                     [&](auto i) {
                         return not std::all_of(i->outputs().begin(),
                                                i->outputs().end(),
                                                [&](auto o) { return contains(attn_ins_vec, o); });
                     });

        mpm.get_module().debug_print(required_outputs);

        assert(required_outputs.size() > 0);
        // Not supporting multi-out just yet - TODO: remove for lse support
        if(required_outputs.size() > 1)
            return;

        // Find corresponding output instructions in m_attn
        std::vector<instruction_ref> m_attn_outputs;
        std::transform(required_outputs.begin(),
                       required_outputs.end(),
                       std::back_inserter(m_attn_outputs),
                       [&](auto i) { return map_mm_to_mattn.at(i); });

        m_attn.debug_print(m_attn_outputs);

        m_attn.add_return(m_attn_outputs);

        // Define inputs to m_attn
        auto map_mattn_to_mm = invert_map_ins(map_mm_to_mattn);
        auto new_inputs      = m_attn.get_inputs(map_mattn_to_mm);

        module_ref mpm_attn = mpm.create_module("mlir_attn" + get_count(), std::move(m_attn));
        mpm_attn->set_bypass();

        // TODO: add replacement of multi-outputs using get_tuple_elem instructions

        // Construct group op with the attention module (for now this assumes single output)
        mpm.get_module().replace_instruction(required_outputs.front(),
                                             make_op("group", {{"tag", "attention"}}),
                                             new_inputs,
                                             {mpm_attn});

        // std::cout << "\n";
        // for (const auto& pair : map_mm_to_mattn)
        // {
        //     mpm.get_module().debug_print(pair.first);
        //     m_attn.debug_print(pair.second);
        //     std::cout << "\n";
        // }
    }
};

// struct find_attention
// {
//     auto matcher() const { return
//     match::name("dot")(match::arg(0)(match::softmax().bind("div"))); }

//     bool is_fusable_ins(instruction_ref ins)
//     {
//         return ins->get_operator().attributes().get("pointwise", false) or
//                ins->get_operator().name() == "reshape";
//     }

//     void apply(module_pass_manager& mpm, const match::matcher_result& r) const
//     {
//         auto gemm2       = r.result;
//         auto softmax_out = r.instructions["div"];
//         auto softmax_inp = r.instructions["x"];

//         mpm.get_module().debug_print();
//         mpm.get_module().debug_print(gemm2);
//         mpm.get_module().debug_print(softmax_out);
//         mpm.get_module().debug_print(softmax_inp);

//         std::vector<instruction_ref> attn_ins;
//         attn_ins.push_back(gemm2);
//     }
// };
} // namespace

void fuse_special_ops::apply(module_pass_manager& mpm) const
{
    std::size_t counter = 0;
    match::find_matches(mpm, find_attention{.counter = &counter});
    mpm.run_pass(dead_code_elimination{});

    mpm.get_module().debug_print();
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
