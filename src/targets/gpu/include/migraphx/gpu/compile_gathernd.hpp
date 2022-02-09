#ifndef MIGRAPHX_GUARD_GPU_COMPILE_GATHERND_HPP
#define MIGRAPHX_GUARD_GPU_COMPILE_GATHERND_HPP

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;
operation compile_gathernd(context& ctx, const std::vector<shape>& io_shapes, const value& val);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_COMPILE_GATHERND_HPP
