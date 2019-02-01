#ifndef MIGRAPHX_GUARD_RTGLIB_MAX_HPP
#define MIGRAPHX_GUARD_RTGLIB_MAX_HPP

#include <migraphx/gpu/lowering.hpp>
#include <migraphx/gpu/oper.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/contiguous.hpp>
#include <migraphx/gpu/device/max.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/gpu/context.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_max : binary_device<hip_max, device::max>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
