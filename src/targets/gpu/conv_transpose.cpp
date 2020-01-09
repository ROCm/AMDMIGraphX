#include <migraphx/gpu/conv_transpose.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/generate.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape miopen_conv_transpose::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(4).standard();
    return op.compute_shape({inputs.at(0), inputs.at(1)});
}
argument miopen_conv_transpose::compute(context& ctx,
                                     const shape& output_shape,
                                     const std::vector<argument>& args) const
{
    auto x_desc = make_tensor(args[0].get_shape());
    auto w_desc = make_tensor(args[1].get_shape());
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
        MIGRAPHX_THROW("Running conv_transpose failed");
    return args[3];
}

shape miopen_conv_transpose::compile(context& ctx,
                                  const shape& output_shape,
                                  std::vector<shape> inputs)
{
    shape workspace_shape{};
    auto x_desc = make_tensor(inputs[0]);
    auto w_desc = make_tensor(inputs[1]);
    auto y_desc = make_tensor(output_shape);

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
        MIGRAPHX_THROW("Find conv_transpose failed");
    handle = ctx.get_stream().get_miopen();
    algo   = perf.fwd_algo;
    return shape{shape::int8_type, {perf.memory}};
}

void miopen_conv_transpose::finalize(context& ctx,
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
