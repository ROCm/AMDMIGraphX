#ifndef MIGRAPH_GUARD_MIGRAPHLIB_MIOPEN_TARGET_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_MIOPEN_TARGET_HPP

#include <migraph/program.hpp>

namespace migraph {
namespace miopen {

struct miopen_target
{
    std::string name() const;
    void apply(program& p) const;
    context get_context() const;
};

} // namespace miopen

} // namespace migraph

#endif
