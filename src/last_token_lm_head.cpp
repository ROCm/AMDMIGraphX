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
 */
#include <migraphx/last_token_lm_head.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void last_token_lm_head::apply(module& m) const
{
    auto last = std::prev(m.end());
    if(last->name() != "@return")
        return;

    auto outputs = last->inputs();
    if(outputs.empty())
        return;

    auto logits_ins = outputs[0];
    if(logits_ins->name() != "dot")
        return;

    auto logits_shape = logits_ins->get_shape();
    if(logits_shape.dynamic())
        return;

    auto logits_lens = logits_shape.lens();
    if(logits_lens.size() != 3 || logits_lens[1] <= 1 || logits_lens[2] < 1000)
        return;

    auto inputs      = logits_ins->inputs();
    auto hidden_ins  = inputs[0];
    auto weight_ins  = inputs[1];
    auto hidden_lens = hidden_ins->get_shape().lens();

    if(hidden_lens.size() != 3 || hidden_lens[1] <= 1)
        return;

    // Find reduce_sum that computes sequence length ({1,1} shape)
    instruction_ref seq_len_ins = m.end();
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "reduce_sum")
            continue;
        auto rs_shape = ins->get_shape();
        auto rs_lens = rs_shape.lens();
        // Look for scalar-like output {1,1}
        if(rs_lens.size() == 2 && rs_lens[0] == 1 && rs_lens[1] == 1)
        {
            seq_len_ins = ins;
            break;
        }
    }

    if(seq_len_ins == m.end())
        return;

    auto seq_len_shape = seq_len_ins->get_shape();
    auto seq_len_type  = seq_len_shape.type();
    
    // index = seq_len - 1
    auto one_lit = m.add_literal(literal{shape{seq_len_type, seq_len_shape.lens()}, {1}});
    auto idx_ins = m.insert_instruction(logits_ins, make_op("sub"), seq_len_ins, one_lit);
    
    // Squeeze to get scalar index: {1,1} -> {}
    auto idx_squeeze = m.insert_instruction(logits_ins, make_op("squeeze", {{"axes", {0, 1}}}), idx_ins);
    
    // Reshape to {1} for gather
    auto idx_reshape = m.insert_instruction(logits_ins, make_op("reshape", {{"dims", {1}}}), idx_squeeze);
    
    // gather(hidden_states, index, axis=1) -> {1, 1, hidden_dim}
    auto gather_ins = m.insert_instruction(
        logits_ins, make_op("gather", {{"axis", 1}}), hidden_ins, idx_reshape);

    auto new_dot = m.insert_instruction(logits_ins, make_op("dot"), gather_ins, weight_ins);
    m.replace_instruction(logits_ins, new_dot);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
