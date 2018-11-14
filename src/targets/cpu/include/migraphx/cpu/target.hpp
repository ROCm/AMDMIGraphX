#ifndef MIGRAPH_GUARD_MIGRAPHLIB_CPU_TARGET_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_CPU_TARGET_HPP

#include <migraphx/program.hpp>
#include <migraphx/cpu/context.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace cpu {

struct target
{
    std::string name() const;
    std::vector<pass> get_passes(migraphx::context& ctx) const;
    migraphx::context get_context() const { return context{}; }
};

} // namespace cpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
