#ifndef MIGRAPH_GUARD_RTGLIB_FUSE_OPS_HPP
#define MIGRAPH_GUARD_RTGLIB_FUSE_OPS_HPP

#include <migraph/program.hpp>
#include <migraph/gpu/context.hpp>

namespace migraph {

namespace gpu {

struct fuse_ops
{
    std::string name() const { return "gpu::fuse_ops"; }
    void apply(program& p) const;
};

} // namespace gpu

} // namespace migraph

#endif
