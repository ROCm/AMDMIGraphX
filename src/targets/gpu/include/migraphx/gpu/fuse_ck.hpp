#ifndef MIGRAPHX_GUARD_GPU_FUSE_CK_HPP
#define MIGRAPHX_GUARD_GPU_FUSE_CK_HPP

#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;

namespace gpu {

struct fuse_ck
{
    context* ctx = nullptr;
    std::string name() const { return "gpu::fuse_ck"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_FUSE_CK_HPP
