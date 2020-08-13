#include <migraphx/gpu/deconvolution.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/generate.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape miopen_deconvolution::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(4).standard();
    std::vector<shape> conv_inputs(inputs.begin(), inputs.begin() + 2);
    check_shapes{conv_inputs, *this}.max_ndims(5);
    return op.compute_shape(conv_inputs);
}

inline shape reshape_if_1d(const shape& input)
{
    shape new_shape{input};
    auto dims = new_shape.lens();

    if(dims.size() == 3)
    {
        std::vector<size_t> new_dims = dims;
        new_dims.insert(new_dims.begin() + 2, 1);
        new_shape = shape{input.type(), new_dims};
    }
    return new_shape;
}

argument miopen_deconvolution::compute(context& ctx,
                                       const shape& output_shape,
                                       const std::vector<argument>& args) const
{
    auto x_desc = make_tensor(reshape_if_1d(args[0].get_shape()));
    auto w_desc = make_tensor(reshape_if_1d(args[1].get_shape()));
    auto y_desc = make_tensor(reshape_if_1d(output_shape));

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
        MIGRAPHX_THROW("Running deconvolution failed");
    return args[3];
}

shape miopen_deconvolution::compile(context& ctx,
                                    const shape& output_shape,
                                    std::vector<shape> inputs)
{
    shape workspace_shape{};
    auto x_desc = make_tensor(reshape_if_1d(inputs[0]));
    auto w_desc = make_tensor(reshape_if_1d(inputs[1]));
    auto y_desc = make_tensor(reshape_if_1d(output_shape));

    std::size_t workspace_size = 0;
    miopenConvolutionForwardGetWorkSpaceSize(ctx.get_stream().get_miopen(),
                                             w_desc.get(),
                                             x_desc.get(),
                                             cd.get(),
                                             y_desc.get(),
                                             &workspace_size);
    workspace_shape = shape{shape::int8_type, {workspace_size}};

    auto x         = to_gpu(generate_argument(inputs[0]));
    auto w         = to_gpu(generate_argument(inputs[1]));
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
        MIGRAPHX_THROW("Find deconvolution failed");
    handle = ctx.get_stream().get_miopen();
    algo   = perf.fwd_algo;
    return shape{shape::int8_type, {perf.memory}};
}

void miopen_deconvolution::finalize(context& ctx,
                                    const shape& output_shape,
                                    std::vector<shape> inputs)
{
    if(handle == ctx.get_stream().get_miopen())
        return;
    // Check that workspace hasn't changed
    auto size = inputs.at(2).bytes();
    auto ws   = compile(ctx, output_shape, std::move(inputs));
    if(ws.bytes() > size)
        MIGRAPHX_THROW("Workspace has changed during finalization.");
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
