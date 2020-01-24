#ifndef MIGRAPHX_GUARD_RTGLIB_REDUCE_PROD_HPP
#define MIGRAPHX_GUARD_RTGLIB_REDUCE_PROD_HPP

#include <migraphx/op/reduce_prod.hpp>
#include <migraphx/gpu/reduce_op.hpp>
#include <migraphx/gpu/device/reduce_prod.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_reduce_prod : reduce_op<hip_reduce_prod, op::reduce_prod, device::reduce_prod>
{
    hip_reduce_prod() {}
    hip_reduce_prod(const op::reduce_prod& op_ref) : reduce_op(op_ref) {}
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
