/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/rewrite_batched_gemms.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/broadcast.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/mul.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/dfor.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void rewrite_batched_gemms::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "dot")
            continue;
        //std::cout << "Rewrite Batched GEMMS" << std::endl;
        //ins->debug_print();
        //m.debug_print();
        //return;
        auto inputs = ins->inputs();
        auto a_mat = inputs.front();
        auto b_mat = inputs.at(1); //.back()?
        auto a_lens = a_mat->get_shape().lens();
        auto b_lens = b_mat->get_shape().lens();
        if (a_lens.size() > 2)
        {
            auto batch_size = std::accumulate(
                a_lens.rbegin() + 2, a_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());
            auto reshape_a = m.insert_instruction(ins, make_op("reshape", {{"dims", {batch_size * a_lens[a_lens.size() - 2], a_lens.back()}}}), a_mat);
            //reshape_a->debug_print();
            //std::cout << b_mat->get_operator().name() << std::endl;
            instruction_ref unbc_b;
            if (b_mat->get_operator().name() == "concat")
            {
                auto concat_inputs = b_mat->inputs();
                std::vector<instruction_ref> concat_lits;
                int concat_axis = 1;
                bool return_early = false;
                for (auto c : concat_inputs)
                {
                    if (c->get_operator().name() == "contiguous")
                        c = c->inputs().front();
                    //std::cout << c->get_operator().name() << ", " << c->get_shape() << ", " << c->get_shape().broadcasted() <<std::endl;
                    if (c->get_shape().broadcasted())
                    {
                        //std::cout << c->inputs().front()->get_operator() <<std::endl;
                        auto lit = c->inputs().front();
                        auto lit_dims = lit->get_shape().lens().size();
                        if (lit_dims > 2)
                            return_early = true;
                        concat_axis = lit_dims - 1;
                        concat_lits.push_back(lit);
                    }
                }
                if (return_early)
                    continue;
                unbc_b = m.insert_instruction(ins, make_op("concat", {{"axis", concat_axis}}), concat_lits);
            }
            else if (b_mat->get_operator().name() == "contiguous")
            {
                //std::cout << "Contiguous B" <<std::endl;
                //b_mat->debug_print();
                auto b_input = b_mat->inputs().front();
                //std::cout << b_input->get_operator().name() << ", " << b_input->get_shape().broadcasted() << ", " << b_input->can_eval() << std::endl;
                if (b_input->get_shape().broadcasted())
                {
                    auto lit = b_input->inputs().front();
                    auto lit_dims = lit->get_shape().lens().size();
                    if (lit_dims > 2)
                        continue;
                    unbc_b = lit;
                    //unbc_b->debug_print();
                }
                else
                    continue;
            }
            else 
            {
                //std::cout << "Else" << std::endl;
                continue;
            }
            auto new_dot = m.insert_instruction(ins, make_op("dot"), reshape_a, unbc_b);
            auto out_lens = a_lens;
            out_lens.pop_back();
            out_lens.push_back(b_lens.back());

            //std::cout << std::next(ins)->get_operator().name() << std::endl;
            auto next_ins = std::next(ins);
            if (next_ins->get_operator().name() == "add")
            {
                auto add_in = next_ins->inputs().back() == ins ? next_ins->inputs().front() : next_ins->inputs().back();
                //add_in->debug_print();
                auto reshape_add = m.insert_instruction(next_ins, make_op("reshape", {{"dims", {batch_size * a_lens[a_lens.size() - 2], b_lens.back()}}}), add_in);
                new_dot = m.replace_instruction(next_ins, make_op("add"), reshape_add, new_dot);
            }
            //std::cout << "here" <<std::endl;
            m.replace_instruction(ins, make_op("reshape", {{"dims", out_lens}}), new_dot);
            
        }



        //m.debug_print();
        
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
