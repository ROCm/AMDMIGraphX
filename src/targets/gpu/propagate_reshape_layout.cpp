/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/propagate_reshape_layout.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/reshape_dims.hpp>
#include <migraphx/value.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

void propagate_reshape_layout::apply(module& m) const
{
    std::vector<instruction_ref> reshapes;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "reshape_lazy")
            reshapes.push_back(ins);
    }

    for(auto rl : reshapes)
    {
        auto c = rl->inputs().front();
        // eliminate_contiguous only leaves a standardizing gpu::contiguous in front of
        // reshape_lazy when it could not alias the input directly; that is the only case
        // where a permutation was discarded.
        if(c->name() != "gpu::contiguous")
            continue;
        auto input = c->inputs().front();
        auto s     = input->get_shape();
        // A standard input carries no permutation to propagate.
        if(s.dynamic() or s.standard())
            continue;

        auto rdims = rl->get_shape().lens();
        // The permuted, packed output the original reshape would have produced from the
        // real (non-standard) input. reshape_dims does not verify the element count, so guard
        // it here the same way reshape_lazy::compute_shape does.
        auto permuted = reshape_dims(s, rdims, {.lazy = false});
        if(not permuted or permuted->standard() or permuted->elements() != s.elements())
            continue;
        // The packed layout that reshape_lazy can alias straight to that output.
        auto relayout = reshape_dims(*permuted, s.lens(), {.lazy = true});
        if(not relayout)
            continue;

        auto layout_op    = make_op("layout", {{"permutation", find_permutation(*relayout)}});
        auto layout_shape = layout_op.compute_shape(std::vector<shape>{s});
        auto alloc =
            m.insert_instruction(rl, make_op("allocate", {{"shape", to_value(layout_shape)}}));
        auto layout = m.insert_instruction(
            rl, make_op("gpu::precompile_op", {{"op", to_value(layout_op)}}), input, alloc);
        instruction::replace_argument(rl, c, layout);
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
