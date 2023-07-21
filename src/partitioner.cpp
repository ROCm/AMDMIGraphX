/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cstddef>
#include <limits>
#include <iterator>
#include <unordered_set>

#include <migraphx/env.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/partitioner.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DEBUG_PARTITIONER)
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static literal get_scalar(instruction_ref ins)
{
    if(ins->name() == "contiguous")
        return get_scalar(ins->inputs().front());
    const auto& s = ins->get_shape();
    if(s.elements() != 1 && not(s.scalar()))
        return {};
    if(not ins->can_eval())
        return {};
    auto e = ins->eval();
    literal r{};
    // needed for bool as visit_at invokes as() which promotes bool to int8
    // Without this we'll break type checks for logical ops that are fused.
    if(e.get_shape().type() == shape::bool_type)
    {
        r = literal{e.at<bool>()};
    }
    else
    {
        e.visit_at([&](auto x) { r = literal{x}; });
    }
    return r;
}

void partition(migraphx::program& p, const target_assignments& tass)
{
    auto* mm = p.get_main_module();
    // sort the graph in reverse post order DFS order
    mm->sort();
    if(enabled(MIGRAPHX_DEBUG_PARTITIONER{}))
    {
        std::cout << "sorted program: \n";
        p.debug_print();
    }
    std::vector<instruction_ref> same_tid_ins_vec;
    std::unordered_set<instruction_ref> same_tid_ins_set;
    // walk the graph in reverse-DFS order
    std::unordered_map<std::size_t, std::size_t> tid_counter;
    size_t current_tid = std::numeric_limits<std::size_t>::max();
    std::unordered_set<instruction_ref> skip_ins;
    for(auto ins : iterator_for(*mm))
    {
        // gather instructions belonging to the same target_id
        // for now, make sure that all the inputs to the insturctions are also from the same
        // target_id, if not create another module
        // skip all the builtins
        if(enabled(MIGRAPHX_DEBUG_PARTITIONER{}))
        {
            std::cout << "currently processing: \n";
            ins->debug_print();
            std::cout << "\n";
        }
        if((starts_with(ins->name(), "@") and ins->name() != "@return") or skip_ins.count(ins) != 0)
        {
            continue;
        }
        else if(ins->name() != "@return" and current_tid == std::numeric_limits<std::size_t>::max())
        {
            current_tid              = tass.at(ins);
            tid_counter[current_tid] = 0;
            same_tid_ins_vec.push_back(ins);
            same_tid_ins_set.insert(ins);
        }
        else if(ins->name() != "@return" and tass.at(ins) == current_tid)
        {
            same_tid_ins_vec.push_back(ins);
            same_tid_ins_set.insert(ins);
        }
        else
        {
            // gather all parameters
            std::unordered_set<instruction_ref> params;
            // gather all return values
            std::unordered_set<instruction_ref> return_ins;
            for(auto tins : iterator_for(same_tid_ins_vec))
            {
                auto inputs  = (*tins)->inputs();
                auto outputs = (*tins)->outputs();
                transform_if(
                    inputs.cbegin(),
                    inputs.cend(),
                    std::inserter(params, params.end()),
                    [&](auto in_param) {
                        return (params.count(in_param) == 0 and
                                same_tid_ins_set.count(in_param) == 0);
                    },
                    [&](auto in_param) { return in_param; });
                if(std::any_of(outputs.begin(), outputs.end(), [&](const auto out_ins) {
                       return same_tid_ins_set.count(out_ins) == 0;
                   }))
                {
                    return_ins.insert(*tins);
                }
            }
            if(enabled(MIGRAPHX_DEBUG_PARTITIONER{}))
            {
                std::cout << "params ins: \n";
                for(auto tmp : iterator_for(params))
                {
                    (*tmp)->debug_print();
                }
                std::cout << "\n";
                std::cout << "return ins: \n";
                for(auto tmp : iterator_for(return_ins))
                {
                    (*tmp)->debug_print();
                }
                std::cout << "\n";
            }

            auto* tmod = p.create_module("target_mod:" + std::to_string(tid_counter[current_tid]));
            std::unordered_map<instruction_ref, instruction_ref> params_map;
            std::size_t param_counter = 0;
            std::vector<instruction_ref> rot_ins_params;
            for(auto pins : iterator_for(params))
            {
                auto scalar = get_scalar(*pins);
                if(scalar.empty())
                {
                    params_map[*pins] = tmod->add_parameter(
                        "param:" + std::to_string(param_counter++), (*pins)->get_shape());
                    rot_ins_params.push_back(*pins);
                }
                else
                {
                    params_map[*pins] = tmod->add_literal(scalar);
                }
            }
            // TODO: what if params_map is empty ?
            for(auto tins : iterator_for(same_tid_ins_vec))
            {
                auto inputs = (*tins)->inputs();
                std::vector<instruction_ref> new_inputs;
                std::transform(inputs.begin(),
                               inputs.end(),
                               std::back_inserter(new_inputs),
                               [&](auto input_ins) { return params_map.at(input_ins); });
                // [TODO]: what if it is has module args ?
                auto tmod_tins = tmod->add_instruction((*tins)->get_operator(), new_inputs);
                // add parameter to params map so that it can be looked up by other insturctions
                params_map[*tins] = tmod_tins;
            }
            std::vector<instruction_ref> rins;
            std::unordered_map<instruction_ref, std::size_t> return_ins_idx_map;
            for(auto ritr : iterator_for(return_ins))
            {
                rins.push_back(params_map.at(*ritr));
                return_ins_idx_map[*ritr] = std::distance(ritr, return_ins.begin());
            }
            tmod->add_return(rins);
            if(enabled(MIGRAPHX_DEBUG_PARTITIONER{}))
            {
                std::cout << "tmod: \n";
                tmod->debug_print();
            }
            // add run_on_target ins
            auto rot_ins =
                mm->insert_instruction(ins,
                                       make_op("run_on_target", {{"target_id", current_tid}}),
                                       rot_ins_params,
                                       {tmod});
            skip_ins.insert(rot_ins);
            // fetch return instructions from tuple
            for(auto mm_rins : iterator_for(return_ins))
            {
                auto tuple_elem_ins = mm->insert_instruction(
                    ins,
                    make_op("get_tuple_elem", {{"index", return_ins_idx_map.at(*mm_rins)}}),
                    rot_ins);
                skip_ins.insert(tuple_elem_ins);
                // replace returns from tmod in main module
                mm->replace_instruction(*mm_rins, tuple_elem_ins);
            }
            dead_code_elimination{}.apply(*mm);
            // update current_tid
            if(ins->name() != "@return")
            {
                current_tid = tass.at(ins);
                if(tid_counter.count(current_tid) == 0)
                {
                    tid_counter[current_tid] = 0;
                }
                tid_counter[current_tid]++;
                same_tid_ins_set.clear();
                same_tid_ins_vec.clear();
                same_tid_ins_set.insert(ins);
                same_tid_ins_vec.push_back(ins);
            }
            if(enabled(MIGRAPHX_DEBUG_PARTITIONER{}))
            {
                std::cout << "program after creation of tmod and rot: \n";
                p.debug_print();
            }
        }
    }
    dead_code_elimination{}.apply(p);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
