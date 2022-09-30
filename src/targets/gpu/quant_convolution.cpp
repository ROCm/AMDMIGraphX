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
#include <migraphx/gpu/device/convert.hpp>
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
    auto* miopen_stream_handle = ctx.get_stream().get_miopen();
    auto workspace_size = args[2].get_shape().bytes();
#ifdef MIGRAPHX_HAS_FIND_2_API
 {
        const miopenTensorArgument_t tensor_args[3] = {
            {miopenTensorConvolutionX, nullptr, args[0].implicit()},
            {miopenTensorConvolutionW, nullptr, args[1].implicit()},
            {miopenTensorConvolutionY, nullptr, args[3].implicit()},
        };

        if(solution_ptr.get() == nullptr)
            MIGRAPHX_THROW("MIOpen Quant Convolution : Load MIOpen Solution before running it");

        auto status = miopenRunSolution(miopen_stream_handle,
                                        solution_ptr.get(),
                                        3,
                                        tensor_args,
                                        args[2].implicit(),
                                        workspace_size);
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("MIOpen Quant Convolution: running convolution using find_2.0 failed");

        return args[3];
    }
#else
    float alpha = 1;
    float beta  = 0;

    auto status = miopenConvolutionForward(miopen_stream_handle,
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
                                           workspace_size);
    if(status != miopenStatusSuccess)
    {
        MIGRAPHX_THROW("MIOpen Quant Convolution: run convolution forward failed");
    }
    return args[3];
#endif
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
#ifdef MIGRAPHX_HAS_FIND_2_API
 {      auto conv_problem = make_obj<miopen_problem>(
            &miopenCreateConvProblem, cd.get(), miopenProblemDirectionForward);

        set_tensor_descriptor(miopenTensorConvolutionX, x_desc, conv_problem);
        set_tensor_descriptor(miopenTensorConvolutionW, w_desc, conv_problem);
        set_tensor_descriptor(miopenTensorConvolutionY, y_desc, conv_problem);

        auto* miopen_stream_handle = ctx.get_stream().get_miopen();

        solution_ptr = find_solution(miopen_stream_handle, conv_problem.get());

        auto status = miopenGetSolutionWorkspaceSize(solution_ptr.get(), &workspace_size);
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("MIOpen Quant Convolution : failed to get solution's workspace size");

        std::size_t solution_size;
        status = miopenGetSolutionSize(solution_ptr.get(), &solution_size);
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("MIOpen Quant Convolution: Failed to fetch solution size");

        auto solution_binary = std::vector<char>{};
        solution_binary.resize(solution_size);

        status = miopenSaveSolution(solution_ptr.get(), solution_binary.data());
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("MIOpen Quant Convolution: Saving solution failed");
        solution_object = value::binary{solution_binary.data(), solution_size};
        return shape{shape::int8_type, {workspace_size}};

 }
#else 
    miopenConvolutionForwardGetWorkSpaceSize(ctx.get_stream().get_miopen(),
                                             w_desc.get(),
                                             x_desc.get(),
                                             cd.get(),
                                             y_desc.get(),
                                             &workspace_size);
    if(status != miopenSuccess)
        MIGRAPHX_THROW("MIOpen Quant Convolution Failed to get forward workspace size");

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
#endif
}

void miopen_quant_convolution::finalize(context& ctx,
                                        const shape& output_shape,
                                        const std::vector<shape>& inputs)
{
#ifdef MIGRAPHX_HAS_FIND_2_API
    (void)(ctx); // avoid warnings
    (void)(output_shape);
    (void)(inputs);
    // load solution
    if(solution_ptr == nullptr)
    {
        miopenSolution_t ptr;
        auto status  = miopenLoadSolution(&ptr,
                                            reinterpret_cast<const char*>(solution_object.data()),
                                            solution_object.size());
        solution_ptr = miopen_solution{ptr};
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("MIOpen Quant Convolution: loading convolution solution failed");
    }
#else
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
#endif
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
