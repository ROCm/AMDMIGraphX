#ifndef MIGRAPHX_GUARD_RTGLIB_PROPAGATE_CONSTANT_HPP
#define MIGRAPHX_GUARD_RTGLIB_PROPAGATE_CONSTANT_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

/**
 * Replace instructions which take all literals with a literal of the computation.
 */
struct propagate_constant
{
    std::string name() const { return "propagate_constant"; }
    void apply(program& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
