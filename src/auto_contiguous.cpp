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
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

const auto& reshaper_op_names()
{
    // clang-format off
    static const std::unordered_set<std::string> names = {
        "flatten",
        "reshape",
        "contiguous",
        "squeeze",
        "unsqueeze"
        "transpose",
        "multibroadcast",
        "broadcast"
    };
    // clang-format on
    return names;
}

bool is_reshaper_op(instruction_ref ins) { return contains(reshaper_op_names(), ins->name()); }

void auto_contiguous::apply(module& m) const
{
    std::string key = "require_std_shape";
    for(auto ins : reverse_iterator_for(m))
    {
        auto&& attr = ins->get_operator().attributes();
        if((attr.get(key, false)))
        {
            auto args     = ins->inputs();
            auto new_args = args;
            std::transform(args.begin(), args.end(), new_args.begin(), [&](auto in) {
                if(in->name() == "contiguous")
                {
                    return in;
                }
                return m.insert_instruction(ins, make_op("contiguous"), in);
            });

            if(new_args != args)
            {
                m.replace_instruction(ins, ins->get_operator(), new_args);
            }
        }
    }

    auto last = std::prev(m.end());
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "layout")
            continue;
        // for last instruction that is NOT a return
        if(ins->outputs().empty() and ins != last)
            continue;

        // perform a pass to insert contiguous for every reshape (without reshaper) before
        // determining if aliasing can be performed
        if(ins->name() == "reshape" and not is_reshaper_op(std::next(ins)))
        {
            m.insert_instruction(std::next(ins), make_op("contiguous"), ins);
        }

        shape s = ins->get_shape();
        if(not s.dynamic() and not s.standard() and s.elements() != 0)
        {
            auto c = m.insert_instruction(std::next(ins), make_op("contiguous"), ins);
            m.replace_instruction(ins, c);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
