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
#include <migraphx/gpu/int8_gemm_pack.hpp>
#include <migraphx/gpu/device/int8_gemm_pack.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_int8_gemm_pack_a::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{{inputs.at(0)}, *this}.has(1).not_broadcasted().packed();
    return inputs.at(0);
}

argument
hip_int8_gemm_pack_a::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    device::int8_gemm_pack_a(ctx.get_stream().get(), args[1], args[0]);
    return args[1];
}

shape hip_int8_gemm_pack_b::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{{inputs.at(0)}, *this}.has(1).not_broadcasted().packed();
    return inputs.at(0);
}

argument
hip_int8_gemm_pack_b::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    device::int8_gemm_pack_b(ctx.get_stream().get(), args[1], args[0]);
    return args[1];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
