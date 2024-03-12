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
#include <migraphx/layout_nhwc.hpp>
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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Predicate>
std::vector<instruction_ref> find_lasts(const module& m, Predicate pred)
{
    std::vector<instruction_ref> result;
    std::unordered_set<instruction_ref> visited;
    fix([&](auto self, auto ins) {
        if(contains(visited, ins))
            return;
        visited.emplace(ins);
        if(pred(ins))
        {
            result.push_back(ins);
            return;
        }
        for(auto input : ins->inputs())
            self(input);
    })(std::prev(m.end()));
    return result;
}

void preserve_output_layout(module& m)
{
    std::vector<instruction_ref> outputs = find_lasts(m, [](auto ins) {
        return ins->name() == "convolution" and ins->get_shape().lens().size() == 4;
    });
    for(auto output : outputs)
    {
        auto permutation = find_permutation(output->get_shape());
        auto layout      = m.insert_instruction(
            std::next(output), make_op("layout", {{"permutation", permutation}}), output);
        m.replace_instruction(output, layout);
    }
}

void transform_convolutions(module& m)
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "convolution")
            continue;
        if(ins->get_shape().lens().size() != 4)
            continue;
        auto v = ins->get_operator().to_value();
        if(v.at("group").to<int>() > 1)
            continue;
        auto args = ins->inputs();
        std::transform(args.begin(), args.end(), args.begin(), [&](const auto& i) {
            return m.insert_instruction(ins, make_op("layout", {{"permutation", {0, 2, 3, 1}}}), i);
        });
        auto conv = m.insert_instruction(ins, ins->get_operator(), args);
        auto c    = m.insert_instruction(ins, make_op("contiguous"), conv);
        m.replace_instruction(ins, c);
    }
}

void layout_nhwc::apply(module_pass_manager& mpm) const
{
    module& m = mpm.get_module();
    preserve_output_layout(m);
    transform_convolutions(m);
    mpm.run_pass(dead_code_elimination{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
