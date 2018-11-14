#ifndef MIGRAPH_GUARD_RTGLIB_ELIMINATE_WORKSPACE_HPP
#define MIGRAPH_GUARD_RTGLIB_ELIMINATE_WORKSPACE_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
struct program;

namespace gpu {

struct eliminate_workspace
{
    std::string name() const { return "eliminate_workspace"; }
    void apply(program& p) const;
};
} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
