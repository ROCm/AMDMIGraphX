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
#include <migraphx/gpu/int8_conv_pack.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape pack_int8_shape(const shape& s)
{
    if(s.type() != shape::int8_type)
    {
        MIGRAPHX_THROW("PACK_INT8_ARGS: only process int8_type");
    }

    auto lens    = s.lens();
    auto strides = s.strides();
    lens[1]      = (lens[1] + 3) / 4 * 4;
    strides[0]   = strides[1] * lens[1];

    return {s.type(), lens, strides};
}

shape miopen_int8_conv_pack::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{{inputs.at(0)}, *this}.has(1).standard();
    return pack_int8_shape(inputs.at(0));
}

argument
miopen_int8_conv_pack::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    auto arg_desc      = make_tensor(args[0].get_shape());
    auto arg_desc_vec4 = make_tensor(args[0].get_shape(), true);

    float alpha = 1;
    float beta  = 0;
    // pack input to vec4 format
    auto status = miopenTransformTensor(ctx.get_stream().get_miopen(),
                                        &alpha,
                                        arg_desc.get(),
                                        args[0].implicit(),
                                        &beta,
                                        arg_desc_vec4.get(),
                                        args[1].implicit());
    if(status != miopenStatusSuccess)
    {
        MIGRAPHX_THROW("INT8_CONV_PACK: transform input tensor failed");
    }

    return args[1];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
