#ifndef MIGRAPH_GUARD_RTGLIB_MIOPEN_WRITE_LITERALS_HPP
#define MIGRAPH_GUARD_RTGLIB_MIOPEN_WRITE_LITERALS_HPP

#include <migraph/program.hpp>
#include <migraph/gpu/context.hpp>

namespace migraph {

namespace gpu {

struct write_literals
{
    context * ctx = nullptr;
    std::string name() const { return "gpu::write_literals"; }

    void apply(program& p) const;
};

} // namespace gpu

} // namespace migraph

#endif
