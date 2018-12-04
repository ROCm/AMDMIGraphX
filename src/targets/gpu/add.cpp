#include <migraphx/gpu/add.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape miopen_add::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(3).not_broadcasted();
    return inputs.at(0);
}

argument miopen_add::compute(context& ctx,
                             const shape& output_shape,
                             const std::vector<argument>& args) const
{
    float alpha = 1, beta = 0;
    auto a_desc = make_tensor(args[0].get_shape());
    auto b_desc = make_tensor(args[1].get_shape());
    auto c_desc = make_tensor(output_shape);
    miopenOpTensor(ctx.get_stream().get_miopen(),
                   miopenTensorOpAdd,
                   &alpha,
                   a_desc.get(),
                   args[0].implicit(),
                   &alpha,
                   b_desc.get(),
                   args[1].implicit(),
                   &beta,
                   c_desc.get(),
                   args[2].implicit());
    return args[2];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
