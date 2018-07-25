#ifndef MIGRAPH_GUARD_RTGLIB_MIOPEN_LOWERING_HPP
#define MIGRAPH_GUARD_RTGLIB_MIOPEN_LOWERING_HPP

#include <migraph/program.hpp>

namespace migraph {
namespace gpu {

struct lowering
{
    std::string name() const { return "gpu::lowering"; }
    void apply(program& p) const;
};

} // namespace gpu

} // namespace migraph

#endif
