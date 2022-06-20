#ifndef MIGRAPHX_GUARD_RTGLIB_SIGN_HPP
#define MIGRAPHX_GUARD_RTGLIB_SIGN_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/sign.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_sign : unary_device<hip_sign, device::sign>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
