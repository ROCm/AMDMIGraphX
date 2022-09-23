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
#include <migraphx/gpu/quant_convolution.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/generate.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape miopen_quant_convolution::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(4).standard();
    return op.normalize_compute_shape({inputs.at(0), inputs.at(1)});
}
argument miopen_quant_convolution::compute(context& ctx,
                                           const shape& output_shape,
                                           const std::vector<argument>& args) const
{
    auto x_desc = make_tensor(args[0].get_shape(), int8_x4_format);
    auto w_desc = make_tensor(args[1].get_shape(), int8_x4_format);
    auto y_desc = make_tensor(output_shape);

    float alpha = 1;
    float beta  = 0;

    auto status = miopenConvolutionForward(ctx.get_stream().get_miopen(),
                                           &alpha,
                                           x_desc.get(),
                                           args[0].implicit(),
                                           w_desc.get(),
                                           args[1].implicit(),
                                           cd.get(),
                                           algo,
                                           &beta,
                                           y_desc.get(),
                                           args[3].implicit(),
                                           args[2].implicit(),
                                           args[2].get_shape().bytes());
    if(status != miopenStatusSuccess)
    {
        MIGRAPHX_THROW("QUANT_CONVOLUTION: run convolution forward failed");
    }

    return args[3];
}

shape miopen_quant_convolution::find(context& ctx,
                                     const shape& output_shape,
                                     std::vector<shape> inputs)
{
    shape workspace_shape{};
    auto x_desc = make_tensor(inputs[0], int8_x4_format);
    auto w_desc = make_tensor(inputs[1], int8_x4_format);
    auto y_desc = make_tensor(output_shape);

    std::size_t workspace_size = 0;
    miopenConvolutionForwardGetWorkSpaceSize(ctx.get_stream().get_miopen(),
                                             w_desc.get(),
                                             x_desc.get(),
                                             cd.get(),
                                             y_desc.get(),
                                             &workspace_size);
    workspace_shape = shape{shape::int8_type, {workspace_size}};

    auto x_shape = inputs[0];
    auto w_shape = inputs[1];
    if(int8_x4_format)
    {
        x_shape = pack_int8_shape(x_shape);
        w_shape = pack_int8_shape(w_shape);
    }
    auto x         = to_gpu(generate_argument(x_shape));
    auto w         = to_gpu(generate_argument(w_shape));
    auto y         = allocate_gpu(output_shape);
    auto workspace = allocate_gpu(workspace_shape);

    int algo_count = 1;
    miopenConvAlgoPerf_t perf;
    auto status = miopenFindConvolutionForwardAlgorithm(ctx.get_stream().get_miopen(),
                                                        x_desc.get(),
                                                        x.implicit(),
                                                        w_desc.get(),
                                                        w.implicit(),
                                                        cd.get(),
                                                        y_desc.get(),
                                                        y.implicit(),
                                                        1,
                                                        &algo_count,
                                                        &perf,
                                                        workspace.implicit(),
                                                        workspace_size,
                                                        false);
    if(status != miopenStatusSuccess)
        MIGRAPHX_THROW("MIOpen Quant Convolution: find convolution failed");
    algo = perf.fwd_algo;

    size_t solution_count;

    status = miopenConvolutionForwardGetSolutionCount(ctx.get_stream().get_miopen(),
                                                      w_desc.get(),
                                                      x_desc.get(),
                                                      cd.get(),
                                                      y_desc.get(),
                                                      &solution_count);
    if(status != miopenStatusSuccess)
        MIGRAPHX_THROW("MIOpen Quant Convolution: get solution count failed");

    std::vector<miopenConvSolution_t> solutions(solution_count);

    status = miopenConvolutionForwardGetSolution(ctx.get_stream().get_miopen(),
                                                 w_desc.get(),
                                                 x_desc.get(),
                                                 cd.get(),
                                                 y_desc.get(),
                                                 solution_count,
                                                 &solution_count,
                                                 solutions.data());
    if(status != miopenStatusSuccess)
        MIGRAPHX_THROW("MIOpen Quant Convolution: get solution failed");

    solution_id = solutions.front().solution_id;

    return shape{shape::int8_type, {perf.memory}};
}

void miopen_quant_convolution::finalize(context& ctx,
                                        const shape& output_shape,
                                        std::vector<shape> inputs)
{
    if(cd == nullptr)
        cd = make_conv(op);
    if(solution_id == 0)
    {
        // Check that workspace hasn't changed
        auto size = inputs.at(2).bytes();
        auto ws   = find(ctx, output_shape, inputs);
        if(ws.bytes() > size)
            MIGRAPHX_THROW("MIOpen Quant Convolution: workspace has changed during finalization.");
    }

    auto x_desc = make_tensor(inputs[0], int8_x4_format);
    auto w_desc = make_tensor(inputs[1], int8_x4_format);
    auto y_desc = make_tensor(output_shape);

    auto status = miopenConvolutionForwardCompileSolution(ctx.get_stream().get_miopen(),
                                                          w_desc.get(),
                                                          x_desc.get(),
                                                          cd.get(),
                                                          y_desc.get(),
                                                          solution_id);
    if(status != miopenStatusSuccess)
        MIGRAPHX_THROW("MIOpen Quant Convolution: compile solution failed");
}

shape miopen_quant_convolution::pack_int8_shape(const shape& s) const
{
    if(s.type() != shape::int8_type)
    {
        MIGRAPHX_THROW("PACK_INT8_SHAPE: only process int8_type");
    }

    auto lens    = s.lens();
    auto strides = s.strides();
    lens[1]      = (lens[1] + 3) / 4 * 4;
    strides[0]   = strides[1] * lens[1];

    return {s.type(), lens, strides};
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
