#ifndef MIGRAPH_GUARD_RTGLIB_CONCAT_GPU_OPT_HPP
#define MIGRAPH_GUARD_RTGLIB_CONCAT_GPU_OPT_HPP

#include <migraph/gpu/concat.hpp>

namespace migraph {
namespace gpu {

struct concat_gpu_optimization
{
    std::string name() const { return "gpu::concat"; }
    std::string allocate() const { return "hip::allocate"; }
    migraph::op::concat get_concat(const migraph::operation& op) const
    {
        return migraph::any_cast<migraph::gpu::hip_concat>(op).op;
    }
};

} // namespace gpu

} // namespace migraph

#endif
