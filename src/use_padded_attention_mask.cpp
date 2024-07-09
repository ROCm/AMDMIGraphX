/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/use_padded_attention_mask.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/im2col.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/op/pad.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void use_padded_attention_mask::apply(module& m) const
{
    std::cout << "use_padded_attention_mask::apply" << std::endl;
    //m.debug_print();
    instruction_ref attn_mask;
    instruction_ref reduce_sum;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "reduce_sum")
        {
            reduce_sum = ins;
            attn_mask = reduce_sum->inputs()[0];
            if(attn_mask->name() == "@param")
            {
                break;
            }
        }
        else if(ins == std::prev(m.end()))
        {
            MIGRAPHX_THROW("Could not find attention_mask param");
        }
    }
    for(auto ins : iterator_for(m))
    {   
        if(ins->name() == "convert")
        {
            auto convert_input = ins->inputs()[0];
            if(convert_input->name() == "gather")
            {
                auto gather_inputs = convert_input->inputs();
                if(std::all_of(gather_inputs.begin(), gather_inputs.end(), [](auto i){ return i->can_eval(); }))
                {
                    m.replace_instruction(ins, ins->get_operator(), reduce_sum);
                }
            }
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
