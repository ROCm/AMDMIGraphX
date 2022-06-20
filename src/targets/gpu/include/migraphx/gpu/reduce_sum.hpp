#ifndef MIGRAPHX_GUARD_RTGLIB_REDUCE_SUM_HPP
#define MIGRAPHX_GUARD_RTGLIB_REDUCE_SUM_HPP

#include <migraphx/op/reduce_sum.hpp>
#include <migraphx/gpu/reduce_op.hpp>
#include <migraphx/gpu/device/reduce_sum.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_reduce_sum : reduce_op<hip_reduce_sum, op::reduce_sum, device::reduce_sum>
{
    hip_reduce_sum() {}
    hip_reduce_sum(const op::reduce_sum& op_ref) : reduce_op(op_ref) {}
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
