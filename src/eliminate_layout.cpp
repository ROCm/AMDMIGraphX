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
#include "migraphx/instruction_ref.hpp"
#include <cstdio>
#include <migraphx/eliminate_layout.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <unordered_set>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// template <class Predicate>
// std::vector<instruction_ref> find_lasts(const module& m, Predicate pred)
// {
//     std::vector<instruction_ref> result;
//     fix([&](auto self, auto ins) {
//         if(pred(ins))
//         {
//             result.push_back(ins);
//             return;
//         }
//         for(auto input : ins->inputs())
//             self(input);
//     })(std::prev(m.end()));
//     return result;
// }

// std::unordered_set<instruction_ref> preserve_output_layout(module& m)
// {
//     std::unordered_set<instruction_ref> result;
//     std::vector<instruction_ref> outputs =
//         find_lasts(m, [](auto ins) { return ins->get_shape().lens().size() == 4; });
//     for(auto output : outputs)
//     {
//         auto permutation = find_permutation(output->get_shape());

//         auto layout_ins = m.insert_instruction(
//             std::next(output), make_op("layout", {{"permutation", permutation}}), output);

//         auto output1 = m.insert_instruction(
//             layout_ins, make_op("allocate", {{"shape", to_value(layout_ins->get_shape())}}));
//         std::vector<instruction_ref> refs = layout_ins->inputs();
//         refs.push_back(output1);

//         auto layout = m.replace_instruction(
//             layout_ins,
//             make_op("gpu::precompile_op", {{"op", to_value(layout_ins->get_operator())}}),
//             refs,
//             layout_ins->module_inputs());

//         result.insert(layout);
//     }
//     return result;
// }

void remove_layout(module& m)
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "gpu::precompile_op")
            continue;

        auto precompile_op = ins->get_operator();
        auto val           = precompile_op.to_value();

        if(val["op"].at("name").to<std::string>() != "layout")
        {
            continue;
        }
        if(ins->get_shape() != ins->inputs().front()->get_shape())
        {
            continue;
        }
        // if(contains(output_layouts, ins))
        //     continue;

        m.replace_instruction(ins, ins->inputs().front());
    }
}

void eliminate_layout::apply(module_pass_manager& mpm) const
{
    // std::unordered_set<instruction_ref> output_layouts =
    // preserve_output_layout(mpm.get_module());
    remove_layout(mpm.get_module());
    mpm.run_pass(dead_code_elimination{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
