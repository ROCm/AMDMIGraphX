#ifndef MIGRAPHX_GUARD_RTGLIB_CONCAT_GPU_OPT_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONCAT_GPU_OPT_HPP

#include <migraphx/gpu/concat.hpp>

namespace migraphx {
namespace gpu {

struct concat_gpu_optimization
{
    std::string name() const { return "gpu::concat"; }
    std::string allocate() const { return "hip::allocate"; }
    migraphx::op::concat get_concat(const migraphx::operation& op) const
    {
        return migraphx::any_cast<migraphx::gpu::hip_concat>(op).op;
    }
};

} // namespace gpu

} // namespace migraphx

#endif
