#ifndef MIGRAPH_GUARD_MIGRAPHLIB_MIOPEN_TARGET_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_MIOPEN_TARGET_HPP

#include <migraph/program.hpp>

namespace migraph {
namespace miopen {

struct target
{
    std::string name() const;
    std::vector<pass> get_passes(context& ctx) const;
    context get_context() const;
};

} // namespace miopen

} // namespace migraph

#endif
