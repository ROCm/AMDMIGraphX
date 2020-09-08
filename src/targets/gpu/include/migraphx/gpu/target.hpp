#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_MIOPEN_TARGET_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_MIOPEN_TARGET_HPP

#include <migraphx/program.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct target
{
    std::string name() const;
    std::vector<pass> get_passes(migraphx::context& gctx, const compile_options& options) const;
    migraphx::context get_context() const;

    argument copy_to(const argument& arg) const;
    argument copy_from(const argument& arg) const;
    argument allocate(const shape& s) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
