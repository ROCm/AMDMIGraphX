#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_CPU_TARGET_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_CPU_TARGET_HPP

#include <migraphx/program.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/cpu/context.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct pass;
namespace cpu {

struct target
{
    std::string name() const;
    std::vector<pass> get_passes(migraphx::context& ctx, const compile_options&) const;
    migraphx::context get_context() const { return context{}; }

    argument copy_to(const argument& arg) const { return arg; }
    argument copy_from(const argument& arg) const { return arg; }
    argument allocate(const shape& s) const;
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
