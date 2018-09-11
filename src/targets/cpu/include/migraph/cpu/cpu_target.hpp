#ifndef MIGRAPH_GUARD_MIGRAPHLIB_CPU_TARGET_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_CPU_TARGET_HPP

#include <migraph/program.hpp>
#include <migraph/cpu/context.hpp>

namespace migraph {
namespace cpu {

struct cpu_target
{
    std::string name() const;
    std::vector<pass> get_passes(migraph::context& ctx) const;
    migraph::context get_context(parameter_map params = parameter_map()) const { return context{params}; }
};

} // namespace cpu

} // namespace migraph

#endif
