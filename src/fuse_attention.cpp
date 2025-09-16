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
        auto gemm1 = match::any_of[pointwise_inputs()](match::name("dot").bind("dot1"));
        return match::name("dot")(match::arg(0)(match::softmax_input(gemm1)));
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

        static const std::unordered_set<std::string> valid_attn_ops = {
            "reshape", "reduce_sum", "reduce_max", "broadcast", "multibroadcast", "@literal"};

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
        auto gemm2 = r.result;
        auto gemm1 = r.instructions["dot1"];

        // Capture all instructions part of the attention op
        auto attn_inss = get_attn_instructions(mpm.get_module(), gemm1, gemm2);

        // Add captured instructions to new submodule
        module m_attn;
        std::unordered_map<instruction_ref, instruction_ref> map_mm_to_mattn;
        auto attn_outs = m_attn.fuse(attn_inss, &map_mm_to_mattn);

        // Define outputs based on instructions that are used elsewhere in the graph
        std::vector<instruction_ref> required_outputs;
        std::copy_if(
            attn_inss.begin(), attn_inss.end(), std::back_inserter(required_outputs), [&](auto i) {
                return not std::all_of(i->outputs().begin(), i->outputs().end(), [&](auto o) {
                    return contains(attn_inss, o);
                });
            });

        assert(not required_outputs.empty());
        // Not supporting multi-out just yet - TODO: remove for lse support
        if(required_outputs.size() > 1)
            return;

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

        // Construct group op with the attention module
        mpm.get_module().replace_instruction(required_outputs.front(),
                                             make_op("group", {{"tag", "attention"}}),
                                             new_inputs,
                                             {mpm_attn});
    }
};

struct find_kv_cache_attention
{
    std::size_t* counter;

    auto matcher() const
    {
        auto slice =
            match::name("slice")(match::arg(0)(match::name("gqa_rotary_embedding").bind("query")));
        auto transpose1 = match::name("transpose")(
            match::arg(0)(match::name("concat_past_present").bind("pres_k")));
        auto gemm1       = match::name("dot")(match::arg(0)(slice), match::arg(1)(transpose1));
        auto scale       = match::name("mul")(match::any_arg(0, 1)(gemm1));
        auto causal_mask = match::name("where")(
            match::arg(0)(match::name("multibroadcast")(match::arg(0)(match::is_constant()))),
            match::arg(2)(scale));
        auto greater = match::name("multibroadcast")(match::arg(0)(match::name("convert")(
            match::arg(0)(match::name("greater")(match::arg(1)(match::any().bind("total_sl")))))));
        auto where   = match::name("where")(match::arg(0)(greater),
                                          match::arg(2)(match::any_of(causal_mask, scale)));
        auto gemm2 =
            match::name("dot")(match::arg(0)(match::softmax_input(where)),
                               match::arg(1)(match::name("concat_past_present").bind("pres_v")));
        auto transpose2 = match::name("transpose")(match::arg(0)(gemm2));
        auto reshape    = match::name("reshape")(match::arg(0)(transpose2));
        return reshape;
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
        std::cout << "matched kvca" << std::endl;
        auto query    = r.instructions["query"];
        auto pres_k   = r.instructions["pres_k"];
        auto pres_v   = r.instructions["pres_v"];
        auto total_sl = r.instructions["total_sl"];
        auto reshape  = r.result;

        // Capture all instructions part of the attention op
        auto attn_inss = get_attn_instructions(mpm.get_module(), total_sl, reshape);
        for(auto& ins : attn_inss)
        {
            ins->debug_print();
        }

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
                    std::cout << "non splat broadcasted: " << std::endl;
                    ins->debug_print();
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

        std::cout << "Required outputs" << std::endl;
        for(auto& ins : required_outputs)
        {
            ins->debug_print();
        }

        assert(not required_outputs.empty());
        // Not supporting multi-out just yet - TODO: remove for lse support
        // if(required_outputs.size() > 1)
        // {
        //     auto req = required_outputs.back();
        //     required_outputs.pop_back();
        //     required_outputs.insert(required_outputs.begin(), req);
        // }

        // Find corresponding output instructions in m_attn
        std::vector<instruction_ref> m_attn_outputs;
        std::transform(required_outputs.begin(),
                       required_outputs.end(),
                       std::back_inserter(m_attn_outputs),
                       [&](auto i) { return map_mm_to_mattn.at(i); });

        std::cout << "Attn outputs" << std::endl;
        for(auto& ins : m_attn_outputs)
        {
            ins->debug_print();
        }
        auto outs = m_attn.add_return({m_attn_outputs.back()});

        // Define inputs to m_attn
        auto map_mattn_to_mm = invert_map_ins(map_mm_to_mattn);
        auto new_inputs      = m_attn.get_inputs(map_mattn_to_mm);

        std::cout << "New inputs" << std::endl;
        for(auto& ins : new_inputs)
        {
            ins->debug_print();
        }

        module_ref mpm_attn = mpm.create_module("attn" + get_count(), std::move(m_attn));
        mpm_attn->set_bypass();

        // Construct group op with the attention module
        mpm_attn->debug_print();
        auto group_ins =
            mpm.get_module().insert_instruction(required_outputs.back(),
                                                make_op("group", {{"tag", "attention"}}),
                                                new_inputs,
                                                {mpm_attn});
        // if(m_attn_outputs.size() == 1)
        // {
        mpm.get_module().replace_instruction(required_outputs.back(), group_ins);
        // }
        // else
        // {
        //     for(std::size_t i = 0; i < required_outputs.size() - 1; ++i)
        //     {
        //         auto id = mpm.get_module().insert_instruction(
        //             std::next(group_ins), make_op("identity"), required_outputs[i]);
        //         mpm.get_module().replace_instruction(
        //             id, make_op("get_tuple_elem", {{"index", i}}), group_ins);
        //     }
        // mpm.get_module().replace_instruction(
        //             required_outputs[2], make_op("get_tuple_elem", {{"index", 2}}), group_ins);
        mpm.get_module().debug_print();
        // }
    }
};

} // namespace

void fuse_attention::apply(module_pass_manager& mpm) const
{
    std::cout << "Module @ fuse_attn" << std::endl;
    mpm.get_module().debug_print();
    std::size_t counter = 0;
    match::find_matches(mpm, find_kv_cache_attention{.counter = &counter});
    match::find_matches(mpm, find_attention{.counter = &counter});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
