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
    return op.compute_shape({inputs.at(0), inputs.at(1)});
}
argument miopen_quant_convolution::compute(context& ctx,
                                           const shape& output_shape,
                                           const std::vector<argument>& args) const
{
    auto x_desc = make_tensor(args[0].get_shape(), true);
    auto w_desc = make_tensor(args[1].get_shape(), true);
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

shape miopen_quant_convolution::compile(context& ctx,
                                        const shape& output_shape,
                                        std::vector<shape> inputs)
{
    shape workspace_shape{};
    auto x_desc = make_tensor(inputs[0], true);
    auto w_desc = make_tensor(inputs[1], true);
    auto y_desc = make_tensor(output_shape);

    std::size_t workspace_size = 0;
    miopenConvolutionForwardGetWorkSpaceSize(ctx.get_stream().get_miopen(),
                                             w_desc.get(),
                                             x_desc.get(),
                                             cd.get(),
                                             y_desc.get(),
                                             &workspace_size);
    workspace_shape = shape{shape::int8_type, {workspace_size}};

    auto arg_vec4_x = to_gpu(generate_argument(pack_int8_shape(inputs[0])));
    auto arg_vec4_w = to_gpu(generate_argument(pack_int8_shape(inputs[1])));
    auto y          = allocate_gpu(output_shape);
    auto workspace  = allocate_gpu(workspace_shape);

    int algo_count = 1;
    miopenConvAlgoPerf_t perf;
    auto status = miopenFindConvolutionForwardAlgorithm(ctx.get_stream().get_miopen(),
                                                        x_desc.get(),
                                                        arg_vec4_x.implicit(),
                                                        w_desc.get(),
                                                        arg_vec4_w.implicit(),
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
    {
        MIGRAPHX_THROW("QUANT_CONVOLUTION: find convolution failed");
    }
    handle = ctx.get_stream().get_miopen();
    algo   = perf.fwd_algo;
    return shape{shape::int8_type, {perf.memory}};
}

void miopen_quant_convolution::finalize(context& ctx,
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
