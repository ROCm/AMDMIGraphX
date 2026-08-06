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
#include <migraphx/gpu/lower_device_ops.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {
operation precompiled(instruction_ref ins)
{
    // gpu::contiguous appends a separate output allocation (additional_args == 1) and compiles as
    // the "contiguous" kernel; hip::fill/hip::copy already include their output buffer as an input
    if(ins->name() == "gpu::contiguous")
        return make_op("gpu::precompile_op",
                       {{"op", to_value(make_op("contiguous"))}, {"additional_args", 1}});
    return make_op("gpu::precompile_op",
                   {{"op", to_value(ins->get_operator())}, {"additional_args", 0}});
}

struct find_device_memory_op
{
    auto matcher() const { return match::name("hip::fill", "hip::copy", "gpu::contiguous"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        if(ins->get_shape().dynamic())
            return;
        m.replace_instruction(ins, precompiled(ins), ins->inputs());
    }
};
} // namespace

void lower_device_ops::apply(module& m) const { match::find_matches(m, find_device_memory_op{}); }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
