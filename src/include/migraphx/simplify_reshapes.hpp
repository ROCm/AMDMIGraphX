#ifndef MIGRAPHX_GUARD_RTGLIB_SIMPLIFY_RESHAPES_HPP
#define MIGRAPHX_GUARD_RTGLIB_SIMPLIFY_RESHAPES_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

/**
 * Eliminate redundant reshapes.
 */
struct simplify_reshapes
{
    std::string name() const { return "simplify_reshapes"; }
    void apply(program& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
