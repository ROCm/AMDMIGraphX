#ifndef MIGRAPHX_GUARD_RTGLIB_ADD_HPP
#define MIGRAPHX_GUARD_RTGLIB_ADD_HPP

#include <migraphx/gpu/lowering.hpp>
#include <migraphx/gpu/oper.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/contiguous.hpp>
#include <migraphx/gpu/device/add.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/config.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_add : binary_device<hip_add, device::add>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
