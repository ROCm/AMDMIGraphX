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
#include <migraphx/gpu/dyn_concat.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_dyn_concat::compute_shape(std::vector<shape> inputs) const
{
    if(inputs.size() < 3)
        MIGRAPHX_THROW("GPU_DYN_CONCAT: invalid input list");
    inputs.resize(inputs.size() - 1);
    return op.normalize_compute_shape(inputs);
}

argument
hip_dyn_concat::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    if(args.size() < 3)
        MIGRAPHX_THROW("GPU_DYN_CONCAT: invalid input list");
    const auto n = args.size() - 2;
    std::vector<argument> inputs(args.begin() + 1, args.begin() + 1 + n);
    const auto& counts = args.front();
    const auto& output = args.back();
    auto outputs       = output.get_sub_objects();
    device::dyn_concat(
        ctx.get_stream().get(), op.axis, counts, inputs, outputs.front(), outputs.back());
    return output;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
