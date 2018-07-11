#ifndef MIGRAPH_GUARD_MIGRAPHLIB_CPU_TARGET_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_CPU_TARGET_HPP

#include <migraph/program.hpp>

namespace migraph {
namespace cpu {

struct cpu_target
{
    std::string name() const;
    std::vector<pass> get_passes(context& ctx) const;
    context get_context() const { return {}; }
};

} // namespace cpu

} // namespace migraph

#endif
