#ifndef MIGRAPH_GUARD_RTGLIB_FUSE_OPS_HPP
#define MIGRAPH_GUARD_RTGLIB_FUSE_OPS_HPP

#include <migraph/program.hpp>
#include <migraph/config.hpp>
#include <migraph/gpu/context.hpp>

namespace migraph { inline namespace MIGRAPH_INLINE_NS {

namespace gpu {

struct fuse_ops
{
    context* ctx = nullptr;
    std::string name() const { return "gpu::fuse_ops"; }
    void apply(program& p) const;
};

} // namespace gpu
} // inline namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
