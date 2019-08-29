#ifndef MIGRAPHX_GUARD_RTGLIB_REDUCE_MEAN_HPP
#define MIGRAPHX_GUARD_RTGLIB_REDUCE_MEAN_HPP

#include <migraphx/op/reduce_mean.hpp>
#include <migraphx/gpu/reduce_op.hpp>
#include <migraphx/gpu/device/reduce_mean.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_reduce_mean : reduce_op<hip_reduce_mean, op::reduce_mean, device::reduce_mean>
{
    hip_reduce_mean() { }
    hip_reduce_mean(const op::reduce_mean& op_ref) : reduce_op(op_ref) { }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
