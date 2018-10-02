#ifndef MIGRAPH_GUARD_RTGLIB_MIOPEN_LOWERING_MEMORY_COLORING_HPP
#define MIGRAPH_GUARD_RTGLIB_MIOPEN_LOWERING_MEMORY_COLORING_HPP

#include <migraph/program.hpp>
#include <migraph/gpu/context.hpp>

namespace migraph {

namespace gpu {

struct lowering_memory_coloring
{
    context* ctx = nullptr;
    std::string name() const { return "gpu::lowering_memory_coloring"; }

    void apply(program& p) const;
};
} // namespace gpu
} // namespace migraph

#endif
