#ifndef MIGRAPH_GUARD_RTGLIB_CONSTANT_PROPAGATE_HPP
#define MIGRAPH_GUARD_RTGLIB_CONSTANT_PROPAGATE_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

struct program;

/**
 * Replace instructions which take all literals with a literal of the computation.
 */
struct constant_propagate
{
    std::string name() const { return "constant_propagate"; }
    void apply(program& p) const;
};

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
