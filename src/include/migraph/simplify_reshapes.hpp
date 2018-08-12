#ifndef MIGRAPH_GUARD_RTGLIB_SIMPLIFY_RESHAPES_HPP
#define MIGRAPH_GUARD_RTGLIB_SIMPLIFY_RESHAPES_HPP

#include <string>
#include <migraph/instruction_ref.hpp>

namespace migraph {

struct program;

struct simplify_reshapes
{
    std::string name() const { return "simplify_reshapes"; }
    void apply(program& p) const;
};

} // namespace migraph

#endif
