#include <migraphx/gpu/multinomial.hpp>
#include <migraphx/gpu/device/multinomial.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/tune_axis.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_multinomial::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(3).only_dims(2);
    size_t sample_size = inputs.back().lens().back();

    if(op.dtype == 6)
        return {shape::int32_type, {inputs[0].lens()[0], sample_size}};
    if(op.dtype == 7)
        return {shape::int64_type, {inputs[0].lens()[0], sample_size}};
    else
        MIGRAPHX_THROW("Invalid output type: " + std::to_string(op.dtype) +
                       ". Valid types are 6 (INT32) and 7 (INT64).");
}

argument
hip_multinomial::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    device::multinomial(ctx.get_stream().get(), args.back(), args.front(), args[1]);
    return args.back();
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
