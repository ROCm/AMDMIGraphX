#include <migraphx/gpu/pooling.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape miopen_pooling::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2).standard();
    std::vector<shape> pooling_input = {inputs.at(0)};
    check_shapes{pooling_input, *this}.max_ndims(5);
    return op.compute_shape(pooling_input);
}

inline void recompute_shape_to_2d(shape& input)
{
    auto dims = input.lens();

    if(dims.size() == 3)
    {
        std::vector<size_t> new_dims = dims;
        new_dims.insert(new_dims.begin() + 2, 1);
        input = shape{input.type(), new_dims};
    }
}

argument miopen_pooling::compute(context& ctx,
                                 const shape& output_shape,
                                 const std::vector<argument>& args) const
{
    shape x_shape = args[0].get_shape();
    shape y_shape = output_shape;

    recompute_shape_to_2d(x_shape);
    recompute_shape_to_2d(y_shape);

    auto x_desc = make_tensor(x_shape);
    auto y_desc = make_tensor(y_shape);

    float alpha = 1;
    float beta  = 0;

    miopenPoolingForward(ctx.get_stream().get_miopen(),
                         pd.get(),
                         &alpha,
                         x_desc.get(),
                         args[0].implicit(),
                         &beta,
                         y_desc.get(),
                         args[1].implicit(),
                         false,
                         nullptr,
                         0);

    return args[1];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
