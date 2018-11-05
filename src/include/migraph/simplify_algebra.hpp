#ifndef MIGRAPH_GUARD_RTGLIB_SIMPLIFY_ALGEBRA_HPP
#define MIGRAPH_GUARD_RTGLIB_SIMPLIFY_ALGEBRA_HPP

#include <string>
#include <migraph/config.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {

struct program;

struct simplify_algebra
{
    std::string name() const { return "simplify_algebra"; }
    void apply(program& p) const;
};

} // namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
