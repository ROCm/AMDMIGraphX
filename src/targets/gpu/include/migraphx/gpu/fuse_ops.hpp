#ifndef MIGRAPHX_GUARD_RTGLIB_FUSE_OPS_HPP
#define MIGRAPHX_GUARD_RTGLIB_FUSE_OPS_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace gpu {

struct fuse_ops
{
    context* ctx = nullptr;
    std::string name() const { return "gpu::fuse_ops"; }
    void apply(program& p) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
