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
#include <migraphx/workgroup_reversal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void reverse_wg(module& m, instruction_ref ins)
{
    if(ins->name() == "gpu::precompile_op")
    {
        auto precomp_op_val       = ins->get_operator().to_value();
        precomp_op_val["reverse"] = true;

        m.replace_instruction(
            ins, make_op(ins->name(), precomp_op_val), ins->inputs(), ins->module_inputs());
    }
}

void workgroup_reversal::apply(module& m) const
{
    std::unordered_set<std::string> blas_names = {"gpu::gemm", "gpu::convolution"};

    std::vector<instruction_ref> precomp_instrs;
    auto m_it = iterator_for(m);
    std::copy_if(m_it.begin(), m_it.end(), std::back_inserter(precomp_instrs), [&](auto ins) {
        return (blas_names.find(ins->name()) != blas_names.end() or
                ins->name() == "gpu::precompile_op");
    });

    // Apply reversals around non-mlir gemms and convolutions
    for(auto i : iterator_for(precomp_instrs))
    {
        if(i == precomp_instrs.begin() or i == precomp_instrs.end())
            continue;

        instruction_ref ins      = *i;
        instruction_ref prev_ins = *std::prev(i);
        instruction_ref next_ins = *std::next(i);

        if(blas_names.find(ins->name()) != blas_names.end())
        {
            reverse_wg(m, prev_ins);
            reverse_wg(m, next_ins);
        }
    }

    // Apply reversals for remaining precompile ops
    for(auto i : iterator_for(precomp_instrs))
    {
        if(i == precomp_instrs.begin() or i == precomp_instrs.end())
            continue;

        if(std::all_of(std::prev(i), std::next(i), [](auto ins) {
               return (ins->name() == "gpu::precompile_op" and
                       not from_value<bool>(ins->get_operator().to_value()["reverse"]));
           }))
        {
            reverse_wg(m, *i);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
