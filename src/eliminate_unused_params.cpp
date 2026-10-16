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
 */
#include <migraphx/eliminate_unused_params.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/rewrite_quantization.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static void rebuild_without_unused_params(module_pass_manager& mpm, instruction_ref ins)
{
    const_module_ref sm = ins->module_inputs().front();
    // The parameter is not dead yet - generate_pointwise folds on its own copy, after every
    // pass has run - so the fold has to happen here, where the parent's operand list can
    // still be corrected in the same step.
    module folded = *sm;
    folded.set_bypass(false);
    // Only absorbing an operand into a constant can orphan a parameter: find_zero_ops, or a
    // quantization rewrite dropping a zero point. Reshapes, converts, cse and constant
    // propagation cannot, so the rest of optimize_module would only cost time.
    run_passes(folded, {rewrite_quantization{}, simplify_algebra{}, dead_code_elimination{}});

    auto old_params = folded.get_parameters();
    if(std::none_of(old_params.begin(), old_params.end(), [](instruction_ref p) {
           return p->outputs().empty();
       }))
        return;
    sort_params(old_params);
    const auto& operands = ins->inputs();
    assert(old_params.size() == operands.size());

    std::vector<instruction_ref> kept_params;
    std::vector<instruction_ref> kept_operands;
    for(std::size_t i = 0; i < old_params.size(); i++)
    {
        if(old_params[i]->outputs().empty())
            continue;
        kept_params.push_back(old_params[i]);
        kept_operands.push_back(operands[i]);
    }

    if(kept_operands.empty())
    {
        // Every parameter went, so the folded body is constant and there is nothing for a
        // kernel to compute. Hoist it into the parent rather than rebuild a pointwise with no
        // inputs, which pointwise::compute_shape rejects.
        std::vector<instruction_ref> const_body;
        for(auto f_ins : iterator_for(folded))
            if(f_ins->name() != "@param" and f_ins->name() != "@return")
                const_body.push_back(f_ins);
        auto ret = std::prev(folded.end());
        if(ret->name() != "@return" or ret->inputs().size() != 1)
            return;
        if(ins->get_shape().dynamic())
            return;
        auto& parent = mpm.get_module();
        std::unordered_map<instruction_ref, instruction_ref> hoist;
        parent.insert_instructions(ins, const_body, &hoist);
        // the submodule body is written per-element, so its result is scalar-shaped while the
        // instruction it replaces carries the full tensor shape
        auto hoisted = hoist.at(ret->inputs().front());
        parent.replace_instruction(
            ins, make_op("multibroadcast", {{"out_lens", ins->get_shape().lens()}}), hoisted);
        return;
    }

    module new_sm;
    new_sm.set_bypass();
    std::unordered_map<instruction_ref, instruction_ref> map_ins;
    // identity transform: add_params' default as_standard() would turn a pointwise
    // param's scalar (0-stride) shape into a different (1-stride) one.
    new_sm.add_params(kept_params, &map_ins, [](const shape& s) { return s; });

    std::vector<instruction_ref> body;
    for(auto sm_ins : iterator_for(folded))
        if(sm_ins->name() != "@param")
            body.push_back(sm_ins);
    auto returns = new_sm.add_instructions(body, &map_ins);
    new_sm.add_return(returns);

    auto* new_pm = mpm.create_module(sm->name() + ":used", std::move(new_sm));
    new_pm->set_bypass();
    mpm.get_module().replace_instruction(ins, ins->get_operator(), kept_operands, {new_pm});
}

void eliminate_unused_params::apply(module_pass_manager& mpm) const
{
    for(auto ins : iterator_for(mpm.get_module()))
    {
        // Fusion containers only, whose submodule parameters are plain data operands. This
        // cannot be generalized by parameter/operand count: loop matches that shape, but its
        // parameters are positional and its output shape derives from the submodule.
        if(not contains({"pointwise", "fused_reduce", "split_fused_reduce"}, ins->name()))
            continue;
        rebuild_without_unused_params(mpm, ins);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
