#ifndef MIGRAPH_GUARD_RTGLIB_CPU_GEMM_HPP
#define MIGRAPH_GUARD_RTGLIB_CPU_GEMM_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace cpu {

void migemm(
    const argument& c_arg, const argument& a_arg, const argument& b_arg, float alpha, float beta);

} // namespace cpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
