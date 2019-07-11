#ifndef MIGRAPHX_GUARD_RTGLIB_POW_HPP
#define MIGRAPHX_GUARD_RTGLIB_POW_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/pow.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_pow : binary_device<hip_pow, device::pow>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
