#include <migraphx/gpu/abs.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape miopen_abs::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2).packed();
    return inputs.at(0);
}

argument miopen_abs::compute(context& ctx,
                             const shape& output_shape,
                             const std::vector<argument>& args) const
{
    float alpha = 1;
    float beta  = 0;
    auto x_desc = make_tensor(args[0].get_shape());
    auto y_desc = make_tensor(output_shape);
    miopenActivationForward(ctx.get_stream().get_miopen(),
                            ad.get(),
                            &alpha,
                            x_desc.get(),
                            args[0].implicit(),
                            &beta,
                            y_desc.get(),
                            args[1].implicit());

    return args[1];
}

void miopen_abs::finalize(context&, const shape&, const std::vector<shape>&) { ad = make_abs(); }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
