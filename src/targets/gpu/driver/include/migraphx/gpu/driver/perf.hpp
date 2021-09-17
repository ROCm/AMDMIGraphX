#ifndef MIGRAPHX_GUARD_GPU_DRIVER_PERF_HPP
#define MIGRAPHX_GUARD_GPU_DRIVER_PERF_HPP

#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/operation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace driver {

double time_op(context& ctx, operation op, const std::vector<shape>& inputs, int n = 100);

} // namespace driver
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_DRIVER_PERF_HPP
