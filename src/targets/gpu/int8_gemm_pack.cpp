#include <migraphx/gpu/int8_gemm_pack.hpp>
#include <migraphx/gpu/device/int8_gemm_pack.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_int8_gemm_pack_a::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{{inputs.at(0)}, *this}.has(1).not_broadcasted().packed();
    return inputs.at(0);
}

argument
hip_int8_gemm_pack_a::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    device::int8_gemm_pack_a(ctx.get_stream().get(), args[1], args[0]);
    return args[1];
}

shape hip_int8_gemm_pack_b::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{{inputs.at(0)}, *this}.has(1).not_broadcasted().packed();
    return inputs.at(0);
}

argument
hip_int8_gemm_pack_b::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    device::int8_gemm_pack_b(ctx.get_stream().get(), args[1], args[0]);
    return args[1];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
