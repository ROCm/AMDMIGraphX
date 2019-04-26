#ifndef MIGRAPHX_GUARD_RTGLIB_CPU_GEMM_HPP
#define MIGRAPHX_GUARD_RTGLIB_CPU_GEMM_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

template<class T>
void migemm(
    const argument& c_arg, const argument& a_arg, const argument& b_arg, T alpha, T beta);

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
