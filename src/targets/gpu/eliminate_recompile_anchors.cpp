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
#include <migraphx/gpu/eliminate_recompile_anchors.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/module.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/shape.hpp>
#include <iterator>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

/// Takes the unused instructions of a compiled module as inputs, so dead code elimination keeps
/// them while the module is recompiled. Kernels that write into part of a buffer, such as an
/// input of an eliminated concat, have no users. Added by restore_recipes and removed by
/// eliminate_recompile_anchors.
struct recompile_anchor
{
    std::string name() const { return "gpu::recompile_anchor"; }
    shape compute_shape(const std::vector<shape>&) const { return {}; }
};
MIGRAPHX_REGISTER_OP(recompile_anchor);

void eliminate_recompile_anchors::apply(module& m) const
{
    std::vector<instruction_ref> anchors;
    std::copy_if(iterator_for(m).begin(),
                 iterator_for(m).end(),
                 std::back_inserter(anchors),
                 [](instruction_ref ins) { return ins->name() == "gpu::recompile_anchor"; });
    for(auto ins : anchors)
        m.remove_instruction(ins);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
