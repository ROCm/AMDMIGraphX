#ifndef MIGRAPHX_GUARD_RTGLIB_SIMPLIFY_ALGEBRA_HPP
#define MIGRAPHX_GUARD_RTGLIB_SIMPLIFY_ALGEBRA_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

/**
 * Simplify many algebraic instructions to more efficient versions.
 */
struct simplify_algebra
{
    std::string name() const { return "simplify_algebra"; }
    void apply(program& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
