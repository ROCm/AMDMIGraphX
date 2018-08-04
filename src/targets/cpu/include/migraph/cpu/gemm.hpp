#ifndef MIGRAPH_GUARD_RTGLIB_CPU_GEMM_HPP
#define MIGRAPH_GUARD_RTGLIB_CPU_GEMM_HPP

#include <migraph/argument.hpp>

namespace migraph {
namespace cpu {

void migemm(
    const argument& c_arg, const argument& a_arg, const argument& b_arg, float alpha, float beta);

} // namespace cpu

} // namespace migraph

#endif
