#ifndef MIGRAPH_GUARD_RTGLIB_SIMPLIFY_ALGEBRA_HPP
#define MIGRAPH_GUARD_RTGLIB_SIMPLIFY_ALGEBRA_HPP

#include <string>

namespace migraph {

struct program;

struct simplify_algebra
{
    std::string name() const { return "simplify_algebra"; }
    void apply(program& p) const;
};

} // namespace migraph

#endif
