#ifndef MIGRAPH_GUARD_MIGRAPHLIB_MIOPEN_TARGET_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_MIOPEN_TARGET_HPP

#include <migraph/program.hpp>
#include <migraph/config.hpp>

namespace migraph { inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

struct target
{
    std::string name() const;
    std::vector<pass> get_passes(migraph::context& gctx) const;
    migraph::context get_context() const;
};

} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
