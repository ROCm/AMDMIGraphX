#ifndef MIGRAPHX_GUARD_RTGLIB_REDUCE_MAX_HPP
#define MIGRAPHX_GUARD_RTGLIB_REDUCE_MAX_HPP

#include <migraphx/op/reduce_max.hpp>
#include <migraphx/gpu/reduce_op.hpp>
#include <migraphx/gpu/device/reduce_max.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_reduce_max : reduce_op<hip_reduce_max, op::reduce_max, device::reduce_max>
{
    hip_reduce_max() {}
    hip_reduce_max(const op::reduce_max& op_ref) : reduce_op(op_ref) {}
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
