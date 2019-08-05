#include <migraphx/gpu/int8_conv_pack.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape miopen_int8_conv_pack::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{{inputs.at(0)}, *this}.has(1).standard();
    return inputs.at(0);
}

argument miopen_int8_conv_pack::compute(context& ctx,
                                        const shape&,
                                        const std::vector<argument>& args) const
{
    auto arg_desc      = make_tensor(args[0].get_shape());
    auto arg_desc_vec4 = make_tensor(args[0].get_shape(), true);

    float alpha = 1;
    float beta  = 0;
    // pack input to vec4 format
    auto status = miopenTransformTensor(ctx.get_stream().get_miopen(),
                                        &alpha,
                                        arg_desc.get(),
                                        args[0].implicit(),
                                        &beta,
                                        arg_desc_vec4.get(),
                                        args[1].implicit());
    if(status != miopenStatusSuccess)
    {
        MIGRAPHX_THROW("INT8_CONV_PACK: transform input tensor failed");
    }

    return args[1];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
