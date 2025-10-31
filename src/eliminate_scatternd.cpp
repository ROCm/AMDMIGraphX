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
#include <migraphx/eliminate_scatternd.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct find_where_scatternd
{
    auto matcher() const { return match::name("scatternd_none")(match::arg(2)(match::name("where").bind("where"))); }

    void apply(module& m, const match::matcher_result& mr) const
    {
        std::cout << "Matched where->scatternd" <<std::endl;
        auto where = mr.instructions["where"];
        auto scatternd = mr.result;

        auto comp = where->inputs().at(0);
        comp->debug_print();
        auto zeros = comp->inputs().front();
        zeros->debug_print();
        auto bc_ninf = where->inputs().at(1);
        bc_ninf->debug_print();
        auto full_mask = where->inputs().at(2);
        full_mask->debug_print();
        
        m.replace_instruction(scatternd, full_mask);
        return; 

        auto raw_mask = m.insert_instruction(scatternd, make_op("div"), full_mask, bc_ninf);

        // auto raw_mask = full_mask->inputs().front();
        // raw_mask->debug_print();
        // while(raw_mask->name() != "mul")
        // {
        //     raw_mask = raw_mask->inputs().front();
        //     raw_mask->debug_print();
        // }
        // raw_mask->debug_print();
        // // one more mul, then first arg
        // raw_mask = raw_mask->inputs().front();
        // raw_mask->debug_print();
        // raw_mask = raw_mask->inputs().front();
        // raw_mask->debug_print();

        m.replace_instruction(scatternd, make_op("where"), raw_mask, bc_ninf, zeros);


        // auto mask = where->inputs().back();
        // if(mask->can_eval())
        // {
        //     m.replace_instruction(scatternd, mask);
        // }
        
        // To do: more robust check

        // m.replace_instruction(scatternd, where);
        std::cout << "Replaced" << std::endl;
        // m.remove_instruction(scatternd);
        m.debug_print();
    }
};

void eliminate_scatternd::apply(module& m) const
{
    // std::cout << "Eliminate scatternd" << std::endl;
    // m.debug_print();
    match::find_matches(m, find_where_scatternd{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
