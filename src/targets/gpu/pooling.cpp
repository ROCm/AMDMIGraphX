#include <migraph/gpu/pooling.hpp>
#include <migraph/operators.hpp>
#include <migraph/manage_ptr.hpp>
#include <migraph/gpu/miopen.hpp>
#include <utility>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

shape miopen_pooling::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2).standard();
    return op.compute_shape({inputs.at(0)});
}
argument miopen_pooling::compute(context& ctx,
                                 const shape& output_shape,
                                 const std::vector<argument>& args) const
{
    auto x_desc = make_tensor(args[0].get_shape());
    auto y_desc = make_tensor(output_shape);

    float alpha = 1, beta = 0;

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
} // inline namespace MIGRAPH_INLINE_NS
} // namespace migraph
