#ifndef MIGRAPH_GUARD_RTGLIB_SIMPLIFY_RESHAPES_HPP
#define MIGRAPH_GUARD_RTGLIB_SIMPLIFY_RESHAPES_HPP

#include <string>
#include <migraph/instruction_ref.hpp>
#include <migraph/config.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {

struct program;

/**
 * Eliminate redundant reshapes.
 */
struct simplify_reshapes
{
    std::string name() const { return "simplify_reshapes"; }
    void apply(program& p) const;
};

} // namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
