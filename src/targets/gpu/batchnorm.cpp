#include <migraphx/gpu/batchnorm.hpp>
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

template <int N>
inline void reshape_to_nd(shape& input)
{
    auto dims = input.lens();

    // not nd input, reshape to nd (total is (n + 2)d)
    if(dims.size() != N + 2)
    {
        std::size_t reshape_loc = N + 1;
        auto num                = std::accumulate(
            dims.begin() + reshape_loc, dims.end(), 1, std::multiplies<std::size_t>());
        std::vector<size_t> new_dims(dims.begin(), dims.begin() + reshape_loc);
        new_dims.push_back(num);
        input = shape{input.type(), new_dims};
    }
}

argument miopen_batch_norm_inference::compute(context& ctx,
                                              const shape& output_shape,
                                              const std::vector<argument>& args) const
{
    shape x_shape  = args[0].get_shape();
    shape y_shape  = output_shape;
    shape bn_shape = args[3].get_shape();

    // reshape_to_nd<2>(x_shape);
    // reshape_to_nd<2>(y_shape);
    // if(op.bn_mode == op::batch_norm_inference::per_activation)
    // {
    //     reshape_to_nd<1>(bn_shape);
    // }

    auto x_desc  = make_tensor(x_shape);
    auto y_desc  = make_tensor(y_shape);
    auto bn_desc = make_tensor(bn_shape);

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
