#ifndef MIGRAPHX_GUARD_RTGLIB_ASIN_HPP
#define MIGRAPHX_GUARD_RTGLIB_ASIN_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/asin.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_asin : unary_device<hip_asin, device::asin>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
