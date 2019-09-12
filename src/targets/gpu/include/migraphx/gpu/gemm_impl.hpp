#ifndef MIGRAPHX_GUARD_RTGLIB_GEMM_IMPL_HPP
#define MIGRAPHX_GUARD_RTGLIB_GEMM_IMPL_HPP

#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

void gemm(context& ctx, const shape& output_shape, const std::vector<argument>& args, float alpha, float beta);
void gemm(context& ctx, const shape& output_shape, const std::vector<argument>& args, int32_t alpha, int32_t beta);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
