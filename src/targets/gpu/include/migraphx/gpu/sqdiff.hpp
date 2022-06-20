#ifndef MIGRAPHX_GUARD_RTGLIB_SQDIFF_HPP
#define MIGRAPHX_GUARD_RTGLIB_SQDIFF_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/sqdiff.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_sqdiff : binary_device<hip_sqdiff, device::sqdiff>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
