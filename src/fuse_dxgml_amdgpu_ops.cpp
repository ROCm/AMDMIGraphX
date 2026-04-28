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
#include <migraphx/fuse_dxgml_amdgpu_ops.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static bool is_rank2(instruction_ref ins) { return ins->get_shape().ndim() == 2; }

// Try to fuse add(dot_ins, bias_ins) or add(bias_ins, dot_ins) into a fused AMD GPU op.
// Returns true if the add instruction was replaced.
static bool try_fuse_add(module& m,
                         instruction_ref add_ins,
                         instruction_ref dot_ins,
                         instruction_ref bias_ins,
                         const std::string& registry_file)
{
    if(dot_ins->name() != "dot" or dot_ins->inputs().size() != 2)
        return false;

    value attrs = {{"kernel_registry", registry_file}};

    auto dot_a = dot_ins->inputs().at(0);
    auto dot_b = dot_ins->inputs().at(1);

    // Pattern: add(dot(dot(A, B), D), E) -> amdgpu::gemm_gemm_add(A, B, D, E)
    if(dot_a->name() == "dot" and dot_a->inputs().size() == 2)
    {
        auto a = dot_a->inputs().at(0);
        auto b = dot_a->inputs().at(1);
        auto d = dot_b;
        auto e = bias_ins;
        if(is_rank2(a) and is_rank2(b) and is_rank2(d) and is_rank2(e))
        {
            auto fused = m.insert_instruction(
                add_ins, make_op("amdgpu::gemm_gemm_add", attrs), {a, b, d, e});
            m.replace_instruction(add_ins, fused);
            return true;
        }
    }

    // Pattern: add(dot(A, B), C) -> amdgpu::gemm_add(A, B, C)
    if(is_rank2(dot_a) and is_rank2(dot_b) and is_rank2(bias_ins))
    {
        auto fused = m.insert_instruction(
            add_ins, make_op("amdgpu::gemm_add", attrs), {dot_a, dot_b, bias_ins});
        m.replace_instruction(add_ins, fused);
        return true;
    }

    return false;
}

void fuse_dxgml_amdgpu_ops::apply(module& m) const
{
    if(kernel_registry_file.empty())
        return;

    // Collect add instructions first to avoid iterator invalidation.
    std::vector<instruction_ref> adds;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "add" and ins->inputs().size() == 2)
            adds.push_back(ins);
    }

    for(auto add_ins : adds)
    {
        auto lhs = add_ins->inputs().at(0);
        auto rhs = add_ins->inputs().at(1);

        // Try lhs as the dot chain, rhs as bias; then the reverse.
        if(try_fuse_add(m, add_ins, lhs, rhs, kernel_registry_file))
            continue;
        try_fuse_add(m, add_ins, rhs, lhs, kernel_registry_file);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
