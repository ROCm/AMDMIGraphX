#ifndef MIGRAPH_GUARD_RTGLIB_ELIMINATE_WORKSPACE_HPP
#define MIGRAPH_GUARD_RTGLIB_ELIMINATE_WORKSPACE_HPP

#include <string>
#include <migraph/instruction_ref.hpp>

namespace migraph {
struct program;

namespace gpu {

struct eliminate_workspace
{
    std::string name() const { return "eliminate_workspace"; }
    void apply(program& p) const;
};
} // namespace gpu
} // namespace migraph

#endif
