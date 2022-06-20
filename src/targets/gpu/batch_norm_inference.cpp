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
#include <migraphx/gpu/batch_norm_inference.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape miopen_batch_norm_inference::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(6);
    check_shapes{inputs.data(), inputs.data() + 1, *this}.same_ndims().max_ndims(5);
    return op.compute_shape({inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3), inputs.at(4)});
}

inline shape reshape_to_2d(const shape& input)
{
    auto dims = input.lens();
    if(dims.size() >= 4)
        return input;

    std::vector<size_t> new_dims(dims.begin(), dims.end());
    std::size_t num = 4 - dims.size();
    new_dims.insert(new_dims.end(), num, 1);
    return {input.type(), new_dims};
}

argument miopen_batch_norm_inference::compute(context& ctx,
                                              const shape& output_shape,
                                              const std::vector<argument>& args) const
{
    shape x_shape  = args[0].get_shape();
    shape y_shape  = output_shape;
    shape bn_shape = args[3].get_shape();

    auto x_desc  = make_tensor(reshape_to_2d(x_shape));
    auto y_desc  = make_tensor(reshape_to_2d(y_shape));
    auto bn_desc = make_tensor(reshape_to_2d(bn_shape));

    float alpha = 1.0;
    float beta  = 0.0f;

    miopenBatchNormalizationForwardInference(ctx.get_stream().get_miopen(),
                                             miopenBatchNormMode_t(op.bn_mode),
                                             &alpha,
                                             &beta,
                                             x_desc.get(),
                                             args[0].implicit(),
                                             y_desc.get(),
                                             args[5].implicit(),
                                             bn_desc.get(),
                                             args[1].implicit(),
                                             args[2].implicit(),
                                             args[3].implicit(),
                                             args[4].implicit(),
                                             op.epsilon);

    return args[5];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
