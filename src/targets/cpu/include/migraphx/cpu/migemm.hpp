#ifndef MIGRAPHX_GUARD_RTGLIB_CPU_GEMM_HPP
#define MIGRAPHX_GUARD_RTGLIB_CPU_GEMM_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

void migemm(
    const argument& c_arg, const argument& a_arg, const argument& b_arg, float alpha, float beta);
void migemm(const argument& c_arg,
            const argument& a_arg,
            const argument& b_arg,
            int32_t alpha,
            int32_t beta);

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
