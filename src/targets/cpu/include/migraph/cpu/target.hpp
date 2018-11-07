#ifndef MIGRAPH_GUARD_MIGRAPHLIB_CPU_TARGET_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_CPU_TARGET_HPP

#include <migraph/program.hpp>
#include <migraph/cpu/context.hpp>
#include <migraph/config.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {
namespace cpu {

struct target
{
    std::string name() const;
    std::vector<pass> get_passes(migraph::context& ctx) const;
    migraph::context get_context() const { return context{}; }
};

} // namespace cpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
