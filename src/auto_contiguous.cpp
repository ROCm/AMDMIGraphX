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
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {
// A tensor whose layout matters. Broadcasts and scalars have no meaningful layout,
// so they are never normalized.
bool needs_standard(instruction_ref in)
{
    const shape& s = in->get_shape();
    return not s.dynamic() and not s.broadcasted() and s.elements() > 1 and not s.standard();
}

// A reshape aliases its input, so a non-standard input yields a non-standard view.
// Give it a standard input so the output is a standard shape.
void contiguous_reshape_inputs(module& m)
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "reshape" or ins->inputs().size() != 1)
            continue;
        auto input = ins->inputs().front();
        if(not needs_standard(input))
            continue;
        m.replace_instruction(
            ins, ins->get_operator(), {m.insert_instruction(ins, make_op("contiguous"), input)});
    }
}

// When an op already has a contiguous input but its inputs are not all standard,
// make the non-standard ones contiguous so they share the standard layout (backends
// such as MIOpen require all inputs to match).
void contiguous_mixed_layout_inputs(module& m)
{
    for(auto ins : iterator_for(m))
    {
        auto args           = ins->inputs();
        bool has_contiguous = std::any_of(args.begin(), args.end(), [](instruction_ref in) {
            return in->name() == "contiguous";
        });
        if(not has_contiguous or std::none_of(args.begin(), args.end(), needs_standard))
            continue;
        auto new_args = args;
        std::transform(args.begin(), args.end(), new_args.begin(), [&](instruction_ref in) {
            return needs_standard(in) ? m.insert_instruction(ins, make_op("contiguous"), in) : in;
        });
        m.replace_instruction(ins, ins->get_operator(), new_args, ins->module_inputs());
    }
}
} // namespace

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
        if(contains({"layout", "@return"}, ins->name()))
            continue;
        // for last instruction that is NOT a return
        if(ins->outputs().empty() and ins != last)
            continue;
        shape s = ins->get_shape();
        // If s is not standard layout or has out of sequence strides, insert "contiguous" op
        // to make a standard shape
        if(not s.dynamic() and (not s.standard() or s.normalize_standard() != s) and
           s.elements() > 1)
        {
            auto c = m.insert_instruction(std::next(ins), make_op("contiguous"), ins);
            m.replace_instruction(ins, c);
        }
    }

    // Both run after the loop above: the mixed-layout pass keys off the contiguous
    // instructions that loop inserts.
    contiguous_reshape_inputs(m);
    contiguous_mixed_layout_inputs(m);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
