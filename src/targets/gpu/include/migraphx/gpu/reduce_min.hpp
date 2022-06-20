#ifndef MIGRAPHX_GUARD_RTGLIB_REDUCE_MIN_HPP
#define MIGRAPHX_GUARD_RTGLIB_REDUCE_MIN_HPP

#include <migraphx/op/reduce_min.hpp>
#include <migraphx/gpu/reduce_op.hpp>
#include <migraphx/gpu/device/reduce_min.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_reduce_min : reduce_op<hip_reduce_min, op::reduce_min, device::reduce_min>
{
    hip_reduce_min() {}
    hip_reduce_min(const op::reduce_min& op_ref) : reduce_op(op_ref) {}
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
