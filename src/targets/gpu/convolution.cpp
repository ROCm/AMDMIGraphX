#include <migraphx/gpu/convolution.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

shape miopen_convolution::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(4).standard();
    return op.compute_shape({inputs.at(0), inputs.at(1)});
}
argument miopen_convolution::compute(context& ctx,
                                     const shape& output_shape,
                                     const std::vector<argument>& args) const
{
    auto x_desc = make_tensor(args[0].get_shape());
    auto w_desc = make_tensor(args[1].get_shape());
    auto y_desc = make_tensor(output_shape);

    float alpha = 1, beta = 0;
    miopenConvolutionForward(ctx.get_stream().get_miopen(),
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
    return args[3];
}

shape miopen_convolution::compile(context& ctx,
                                  const shape& output_shape,
                                  std::vector<instruction_ref> inputs)
{
    shape workspace_shape{};
    auto x_desc = make_tensor(inputs[0]->get_shape());
    auto w_desc = make_tensor(inputs[1]->get_shape());
    auto y_desc = make_tensor(output_shape);

    std::size_t workspace_size = 0;
    miopenConvolutionForwardGetWorkSpaceSize(ctx.get_stream().get_miopen(),
                                             w_desc.get(),
                                             x_desc.get(),
                                             cd.get(),
                                             y_desc.get(),
                                             &workspace_size);
    workspace_shape = shape{shape::int8_type, {workspace_size}};

    auto x         = to_gpu(generate_argument(inputs[0]->get_shape()));
    auto w         = to_gpu(generate_argument(inputs[1]->get_shape()));
    auto y         = allocate_gpu(output_shape);
    auto workspace = allocate_gpu(workspace_shape);

    int algo_count = 1;
    miopenConvAlgoPerf_t perf;
    miopenFindConvolutionForwardAlgorithm(ctx.get_stream().get_miopen(),
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
    algo = perf.fwd_algo;
    return shape{shape::int8_type, {perf.memory}};
}

} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx
